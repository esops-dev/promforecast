"""Ad-hoc forecast preview endpoint.

``POST /forecast/preview`` is the operator-facing what-if knob:

* Pick an existing query by ``query_id``.
* Optionally override regressor values, ``horizon``, or ``lookback``.
* Receive a JSON forecast — point values + per-level bands — without
  touching the YAML config or polluting the scheduled-fit pipeline.

Stateless: nothing about the response leaks into ``/metrics``, no
snapshot is mutated, and the underlying executor pool stays untouched
beyond the single sample fit. Per-request CPU is bounded by
``safety.fit_timeout`` so a pathological request can't starve scheduled
fits.

HA mode: any replica may serve the endpoint. The response carries a
``served_by`` field so clients can correlate, but the path itself is
read-only and not subject to leader election.

Auth is intentionally absent — same network-level access-control model
as ``/-/reload``. A shared HTTP-control-endpoint auth scheme is tracked
under future considerations in the roadmap.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

from . import models as model_registry
from . import regressors as regressors_mod
from .config import (
    Config,
    QueryConfig,
    effective_season_length,
    effective_step,
)

if TYPE_CHECKING:
    import pandas as pd

    from .source import PromSource

logger = structlog.get_logger(__name__)


class PreviewError(RuntimeError):
    """Raised by :func:`run_preview` for any operator-visible failure.

    Carries a short ``reason`` token (``unknown_query`` | ``query_failed`` |
    ``fit_failed`` | ``no_data``) plus an HTTP status code so the FastAPI
    handler can render an actionable response.
    """

    def __init__(self, reason: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.reason = reason
        self.status_code = status_code


@dataclass(frozen=True)
class PreviewRequest:
    """Parsed JSON body of a ``POST /forecast/preview`` request.

    Every override is optional. ``regressor_overrides`` accepts either a
    scalar (broadcast across the in-sample + future horizon) or a list of
    floats (per-step values, length must equal the resulting series
    length). All keys map to regressor IDs declared in the query's
    ``regressors:`` block — overrides for unknown IDs are ignored so a
    config change doesn't break a stale dashboard.

    ``model`` optionally selects which model the preview uses; defaults
    to the first model in ``config.defaults.models``. Passing a name not
    present in the configured models fails the request — the preview is
    intended to mirror what the scheduled fit would produce, not let the
    operator silently try arbitrary models. ``series_selector`` picks
    one frame when the PromQL returns multiple series; without it a
    multi-frame response is rejected.
    """

    query_id: str
    regressor_overrides: dict[str, float | list[float]]
    horizon_seconds: float | None
    lookback_seconds: float | None
    model: str | None = None
    series_selector: dict[str, str] | None = None


def parse_request(payload: object) -> PreviewRequest:
    """Validate an inbound JSON body and return a :class:`PreviewRequest`.

    Raises :class:`PreviewError` with status 400 on any schema mismatch.
    Kept separate from :func:`run_preview` so unit tests can exercise the
    parsing without spinning up the full preview pipeline.
    """
    if not isinstance(payload, dict):
        raise PreviewError("invalid_body", "request body must be a JSON object", status_code=400)
    query_id_obj = payload.get("query_id")
    if not isinstance(query_id_obj, str) or not query_id_obj:
        raise PreviewError("invalid_body", "request requires a string 'query_id'", status_code=400)
    overrides_raw = payload.get("regressor_overrides") or {}
    if not isinstance(overrides_raw, dict):
        raise PreviewError(
            "invalid_body", "'regressor_overrides' must be a JSON object", status_code=400
        )
    overrides: dict[str, float | list[float]] = {}
    for key, value in overrides_raw.items():
        if isinstance(value, (int, float)):
            overrides[str(key)] = float(value)
        elif isinstance(value, list):
            try:
                overrides[str(key)] = [float(v) for v in value]
            except (TypeError, ValueError) as exc:
                raise PreviewError(
                    "invalid_body",
                    f"regressor_overrides[{key!r}] must be numeric or list of numbers",
                    status_code=400,
                ) from exc
        else:
            raise PreviewError(
                "invalid_body",
                f"regressor_overrides[{key!r}] must be numeric or list of numbers",
                status_code=400,
            )
    horizon = payload.get("horizon")
    lookback = payload.get("lookback")
    horizon_s = _parse_optional_duration(horizon, field="horizon")
    lookback_s = _parse_optional_duration(lookback, field="lookback")
    model_raw = payload.get("model")
    if model_raw is not None and not isinstance(model_raw, str):
        raise PreviewError("invalid_body", "'model' must be a string", status_code=400)
    selector_raw = payload.get("series_selector")
    selector: dict[str, str] | None
    if selector_raw is None:
        selector = None
    elif isinstance(selector_raw, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in selector_raw.items()
    ):
        selector = {str(k): str(v) for k, v in selector_raw.items()}
    else:
        raise PreviewError(
            "invalid_body",
            "'series_selector' must be a mapping of label -> value strings",
            status_code=400,
        )
    return PreviewRequest(
        query_id=query_id_obj,
        regressor_overrides=overrides,
        horizon_seconds=horizon_s,
        lookback_seconds=lookback_s,
        model=model_raw,
        series_selector=selector,
    )


def _parse_optional_duration(value: object, *, field: str) -> float | None:
    """Accept seconds (number) or Prometheus duration strings."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds <= 0:
            raise PreviewError("invalid_body", f"'{field}' must be positive", status_code=400)
        return seconds
    if isinstance(value, str):
        try:
            # Reuse the public schema duration parser so the YAML
            # syntax (``5m``, ``1h``, ``14d``) is accepted here too.
            from .config import parse_duration  # noqa: PLC0415

            parsed = parse_duration(value)
        except Exception as exc:
            raise PreviewError(
                "invalid_body", f"'{field}' duration parse failed: {exc}", status_code=400
            ) from exc
        if not isinstance(parsed, timedelta):
            raise PreviewError(
                "invalid_body",
                f"'{field}' must be a duration string like '6h' or a number of seconds",
                status_code=400,
            )
        if parsed.total_seconds() <= 0:
            raise PreviewError("invalid_body", f"'{field}' must be positive", status_code=400)
        return parsed.total_seconds()
    raise PreviewError(
        "invalid_body",
        f"'{field}' must be a duration string or numeric seconds",
        status_code=400,
    )


def find_query(config: Config, query_id: str) -> tuple[QueryConfig, str]:
    """Locate ``query_id`` in any group of ``config``.

    Returns ``(query, group_name)``. Raises :class:`PreviewError` with
    status 404 if no group has a matching query. The first match wins —
    the config schema doesn't enforce id uniqueness across groups, but
    duplicates are rare in practice and unambiguous as long as the
    operator knows which group they meant.
    """
    for group in config.groups:
        for query in group.queries:
            if query.id == query_id:
                return query, group.name
    raise PreviewError(
        "unknown_query",
        f"no query with id {query_id!r} in current config",
        status_code=404,
    )


async def run_preview(
    *,
    config: Config,
    source: PromSource,
    request: PreviewRequest,
    served_by: str,
) -> dict[str, Any]:
    """Execute the preview fit and return a JSON-shaped forecast.

    Doesn't touch the runner's snapshot or the global executor — the fit
    runs synchronously on the calling event loop's default executor, so a
    pathological body cannot starve scheduled work that holds the
    runner's executor. ``safety.fit_timeout`` still bounds the per-fit
    CPU.
    """
    import pandas as pd  # noqa: PLC0415

    query, group_name = find_query(config, request.query_id)
    eff_step = effective_step(query, config.defaults)
    eff_season_length = effective_season_length(query, config.defaults, step=eff_step)
    lookback_seconds = (
        request.lookback_seconds
        if request.lookback_seconds is not None
        else config.defaults.lookback.total_seconds()
    )
    horizon_seconds = (
        request.horizon_seconds
        if request.horizon_seconds is not None
        else config.defaults.horizon.total_seconds()
    )
    end = datetime.now(tz=UTC)
    start = end - timedelta(seconds=lookback_seconds)
    step_seconds = max(1, int(eff_step.total_seconds()))
    timeout_seconds = config.safety.fit_timeout.total_seconds()
    horizon_steps = max(1, int(horizon_seconds // max(1, step_seconds)))
    levels = list(config.defaults.confidence_levels)

    try:
        frames = await asyncio.wait_for(
            source.query_range(query.promql, start, end, step_seconds),
            timeout=config.safety.query_timeout.total_seconds(),
        )
    except TimeoutError as exc:
        raise PreviewError(
            "query_timeout",
            f"preview query timed out after {config.safety.query_timeout.total_seconds()}s",
            status_code=504,
        ) from exc
    except Exception as exc:
        raise PreviewError("query_failed", f"preview query failed: {exc}", status_code=502) from exc

    if not frames:
        raise PreviewError("no_data", "preview query returned no series", status_code=404)
    frame = _pick_frame(frames, request.series_selector)
    if not frame.values:
        raise PreviewError("no_data", "preview query returned an empty series", status_code=404)

    df = pd.DataFrame(
        {
            "unique_id": ["s"] * len(frame.values),
            "ds": pd.to_datetime(frame.timestamps),
            "y": frame.values,
        }
    )

    regressors_df = _build_preview_regressors(
        query=query,
        timestamps=frame.timestamps,
        overrides=request.regressor_overrides,
    )

    spec = _pick_model_spec(config, request.model)
    model = model_registry.build(spec.name, season_length=eff_season_length, params=spec.params)
    freq = _pandas_freq(step_seconds)
    loop = asyncio.get_running_loop()
    try:
        forecast = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                model.fit_predict,
                df,
                horizon_steps,
                freq,
                levels,
                regressors_df,
            ),
            timeout=timeout_seconds,
        )
    except TimeoutError as exc:
        raise PreviewError(
            "fit_timeout",
            f"preview fit timed out after {timeout_seconds}s",
            status_code=504,
        ) from exc
    except Exception as exc:
        raise PreviewError("fit_failed", f"preview fit failed: {exc}", status_code=500) from exc

    return _serialise_forecast(
        forecast=forecast,
        levels=levels,
        query_id=query.id,
        group_name=group_name,
        model_name=spec.name,
        served_by=served_by,
    )


def _build_preview_regressors(
    *,
    query: QueryConfig,
    timestamps: list[datetime],
    overrides: dict[str, float | list[float]],
) -> pd.DataFrame | None:
    """Build the regressor frame from query config + per-request overrides.

    Calendar regressors are derived from the timestamps directly (no
    overrides honoured — they're deterministic by definition). Custom +
    holiday regressors honour ``regressor_overrides`` when set so an
    operator can ask "what if deploys doubled?" without writing PromQL.

    Returns ``None`` when no regressors are configured *and* no overrides
    apply, so the downstream model sees an unconditional series.
    """
    import pandas as pd  # noqa: PLC0415

    if not query.regressors and not overrides:
        return None
    pf = regressors_mod.PromQLPrefetch(fetched={}, failures=[])
    outcome = regressors_mod.align_for_series(
        regressors=query.regressors, prefetch=pf, timestamps=timestamps
    )
    frame = (
        outcome.frame
        if outcome.frame is not None
        else pd.DataFrame({"ds": pd.to_datetime(timestamps)})
    )
    for reg_id, value in overrides.items():
        if isinstance(value, list):
            if len(value) != len(timestamps):
                raise PreviewError(
                    "invalid_body",
                    f"regressor_overrides[{reg_id!r}] length {len(value)} does not "
                    f"match series length {len(timestamps)}",
                    status_code=400,
                )
            frame[reg_id] = value
        else:
            frame[reg_id] = [float(value)] * len(timestamps)
    if len(frame.columns) <= 1:
        return None
    return frame


def _serialise_forecast(
    *,
    forecast: pd.DataFrame,
    levels: list[int],
    query_id: str,
    group_name: str,
    model_name: str,
    served_by: str,
) -> dict[str, Any]:
    """Project a forecast frame into the JSON shape /preview returns.

    Excludes ``unique_id`` (an internal statsforecast detail) and emits
    timestamps as ISO-8601 strings so clients don't have to guess the
    format. NaN values become JSON ``null`` via float-cast inside the
    list comprehensions (FastAPI's JSONResponse renders ``nan`` -> null
    only when wrapped in ``json.dumps(default=str)``; rendering ``None``
    explicitly avoids that footgun).
    """
    timestamps = [str(ts) for ts in forecast.get("ds", [])]
    yhat = _column_as_list(forecast, "yhat")
    bands: dict[str, dict[str, list[float | None]]] = {}
    for level in levels:
        lo_col = f"yhat_lower_{level}"
        hi_col = f"yhat_upper_{level}"
        if lo_col not in forecast.columns or hi_col not in forecast.columns:
            continue
        bands[str(level)] = {
            "lower": _column_as_list(forecast, lo_col),
            "upper": _column_as_list(forecast, hi_col),
        }
    return {
        "query_id": query_id,
        "group": group_name,
        "model": model_name,
        "served_by": served_by,
        "timestamps": timestamps,
        "yhat": yhat,
        "bands": bands,
    }


def _column_as_list(forecast: pd.DataFrame, col: str) -> list[float | None]:
    """Return ``forecast[col]`` as a list with NaN -> None for JSON safety."""
    import math  # noqa: PLC0415

    if col not in forecast.columns:
        return []
    out: list[float | None] = []
    for v in forecast[col].tolist():
        try:
            f = float(v)
        except (TypeError, ValueError):
            out.append(None)
            continue
        out.append(f if math.isfinite(f) else None)
    return out


def _pandas_freq(step_seconds: int) -> str:
    if step_seconds % 3600 == 0:
        return f"{step_seconds // 3600}h"
    if step_seconds % 60 == 0:
        return f"{step_seconds // 60}min"
    return f"{step_seconds}s"


def _pick_frame(
    frames: list[Any],
    selector: dict[str, str] | None,
) -> Any:
    """Choose one series frame from a multi-series query result.

    With one frame the choice is trivial. With multiple frames an explicit
    ``series_selector`` is required so the preview doesn't silently
    return forecasts for an arbitrary (and possibly unrelated) series.
    """
    if len(frames) == 1:
        return frames[0]
    if selector is None:
        raise PreviewError(
            "ambiguous_series",
            (
                f"preview query returned {len(frames)} series; pass a 'series_selector' "
                "(mapping of label -> value) to pick exactly one"
            ),
            status_code=400,
        )
    matches = [f for f in frames if all(f.labels.get(k) == v for k, v in selector.items())]
    if not matches:
        raise PreviewError(
            "no_data",
            f"'series_selector' {selector!r} matched no series among {len(frames)} returned",
            status_code=404,
        )
    if len(matches) > 1:
        raise PreviewError(
            "ambiguous_series",
            (
                f"'series_selector' {selector!r} matched {len(matches)} series; "
                "tighten it so exactly one frame is selected"
            ),
            status_code=400,
        )
    return matches[0]


def _pick_model_spec(config: Config, model_name: str | None) -> Any:
    """Resolve the configured ``ModelSpec`` the preview should fit.

    Defaults to the first entry in ``config.defaults.models`` — that's
    the conventional "first choice" model for the deployment. An
    explicit ``model_name`` must match one of the configured models so
    the preview reflects an actually-deployable choice; an unknown name
    fails the request rather than spinning up a stranger.
    """
    available = list(config.defaults.models)
    if not available:
        raise PreviewError(
            "fit_failed",
            "config.defaults.models is empty; cannot run a preview without a model",
            status_code=500,
        )
    if model_name is None:
        return available[0]
    for spec in available:
        if spec.name == model_name:
            return spec
    names = ", ".join(spec.name for spec in available)
    raise PreviewError(
        "invalid_body",
        f"model {model_name!r} not in configured models ({names})",
        status_code=400,
    )
