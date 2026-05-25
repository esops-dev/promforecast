"""Forecast inspector — debug "why is this one series weird?".

Pulls a single (series, model) tuple from the configured TSDB, runs
the configured fit pipeline exactly as the live forecaster would, and
prints the inputs, the fit, the forecast curve, the per-component
decomposition (when available), and the most recent backtest scores.
Optionally surfaces the per-fold backtest predictions vs. actuals so
"this fit backtests well but predicts badly" becomes a concrete two-
column comparison instead of a single summary MAPE.

The CLI talks to VM directly via the same :class:`PromSource` the
runner uses — no new HTTP surface, no shared scheduler state, no
risk of starving scheduled work. Use it from a bastion / debugging
pod or locally with a port-forwarded TSDB.

The output is a compact text table by default; ``--json`` switches
to a machine-friendly envelope for piping into ``jq`` or saving as
a CI artifact alongside a flaky forecast for later analysis.
"""

from __future__ import annotations

import asyncio
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import structlog

from . import accuracy as accuracy_mod
from . import config as config_module
from . import models as model_registry
from . import preprocess as preprocess_mod
from .config import Config, effective_emission_safeguards, effective_season_length, effective_step
from .preview import _pandas_freq, find_query
from .runner import _apply_safeguards_to_forecast_frame
from .source import PromSource, SeriesFrame

if TYPE_CHECKING:
    import pandas as pd

logger = structlog.get_logger(__name__)


# Default number of forecast curve rows to render in the table when
# ``--json`` is off. The full curve is always available via ``--json``;
# the table cap keeps the human view scannable on an 80-column terminal.
_TABLE_HEAD_ROWS = 10


class InspectError(RuntimeError):
    """Raised for any operator-visible inspector failure.

    Carries an integer exit code so :func:`run_inspect` can hand it
    straight to ``sys.exit`` without a second try/except.
    """

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class InputStats:
    """Summary statistics for the inspected series's lookback window."""

    point_count: int
    gap_count: int
    largest_gap_seconds: float
    lookback_start: str
    lookback_end: str


@dataclass(frozen=True)
class BacktestFoldRow:
    """One per-step row from the backtest hold-out comparison.

    Carries the ``ds`` timestamp, the held-out ``actual``, and the
    model's ``yhat`` / lower / upper. Bands are dict-shaped so a
    multi-level config emits them all without changing the row's
    schema; missing levels are simply absent rather than NaN.
    """

    ds: str
    actual: float
    yhat: float
    bands: dict[str, dict[str, float]]


@dataclass(frozen=True)
class BacktestSummary:
    """Compact per-fold backtest summary for the inspector output."""

    model: str
    n_holdout: int
    mape: float
    mase: float
    rows: list[BacktestFoldRow] = field(default_factory=list)


@dataclass(frozen=True)
class InspectReport:
    """Full inspector report — every section the operator may want."""

    query_id: str
    group: str
    model: str
    series_labels: dict[str, str]
    step_seconds: int
    season_length: int
    horizon_steps: int
    input_stats: InputStats
    forecast_timestamps: list[str]
    forecast_yhat: list[float]
    forecast_bands: dict[str, dict[str, list[float]]]
    preprocessing: dict[str, Any]
    backtest: BacktestSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "query_id": self.query_id,
            "group": self.group,
            "model": self.model,
            "series_labels": dict(self.series_labels),
            "step_seconds": self.step_seconds,
            "season_length": self.season_length,
            "horizon_steps": self.horizon_steps,
            "input_stats": {
                "point_count": self.input_stats.point_count,
                "gap_count": self.input_stats.gap_count,
                "largest_gap_seconds": self.input_stats.largest_gap_seconds,
                "lookback_start": self.input_stats.lookback_start,
                "lookback_end": self.input_stats.lookback_end,
            },
            "forecast": {
                "timestamps": list(self.forecast_timestamps),
                "yhat": list(self.forecast_yhat),
                "bands": {
                    level: {
                        "lower": list(payload["lower"]),
                        "upper": list(payload["upper"]),
                    }
                    for level, payload in self.forecast_bands.items()
                },
            },
            "preprocessing": dict(self.preprocessing),
        }
        if self.backtest is not None:
            out["backtest"] = {
                "model": self.backtest.model,
                "n_holdout": self.backtest.n_holdout,
                "mape": _json_value(self.backtest.mape),
                "mase": _json_value(self.backtest.mase),
                "rows": [
                    {
                        "ds": row.ds,
                        "actual": _json_value(row.actual),
                        "yhat": _json_value(row.yhat),
                        "bands": {
                            level: {
                                "lower": _json_value(values["lower"]),
                                "upper": _json_value(values["upper"]),
                            }
                            for level, values in row.bands.items()
                        },
                    }
                    for row in self.backtest.rows
                ],
            }
        return out


def run_inspect(
    *,
    config_path: Path,
    query_id: str,
    series_filter: dict[str, str] | None,
    horizon_seconds: float | None,
    lookback_seconds: float | None,
    model: str | None,
    backtest: bool,
    json_output: bool,
    out: IO[str] | None = None,
) -> int:
    """Top-level inspector entrypoint called by ``main.cli``.

    Returns a process exit code:

    * ``0`` on success;
    * ``1`` on inspector-level failure (unknown query, ambiguous
      series, fit error, etc.);
    * ``2`` on datasource / I/O failure.
    """
    stream: IO[str] = out if out is not None else sys.stdout

    try:
        cfg = config_module.load(config_path)
    except FileNotFoundError:
        sys.stderr.write(f"error: config not found: {config_path}\n")
        return 2
    except Exception as exc:
        sys.stderr.write(f"error: config validation failed: {exc}\n")
        return 1

    try:
        report = asyncio.run(
            _build_report(
                config=cfg,
                query_id=query_id,
                series_filter=series_filter,
                horizon_seconds=horizon_seconds,
                lookback_seconds=lookback_seconds,
                model_override=model,
                with_backtest=backtest,
            )
        )
    except InspectError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exc.exit_code
    except Exception as exc:
        sys.stderr.write(f"error: unexpected inspector failure: {exc}\n")
        return 2

    if json_output:
        import json  # noqa: PLC0415

        stream.write(json.dumps(report.to_dict(), indent=2) + "\n")
    else:
        stream.write(render_text(report))
    return 0


async def _build_report(
    *,
    config: Config,
    query_id: str,
    series_filter: dict[str, str] | None,
    horizon_seconds: float | None,
    lookback_seconds: float | None,
    model_override: str | None,
    with_backtest: bool,
) -> InspectReport:
    """Fetch + fit + (optionally) backtest, returning a populated report."""
    import pandas as pd  # noqa: PLC0415

    try:
        query, group_name = find_query(config, query_id)
    except Exception as exc:
        raise InspectError(f"unknown query {query_id!r}: {exc}") from exc

    eff_step = effective_step(query, config.defaults)
    eff_season_length = effective_season_length(query, config.defaults, step=eff_step)
    lookback = (
        timedelta(seconds=lookback_seconds)
        if lookback_seconds is not None
        else config.defaults.lookback
    )
    horizon_secs = (
        horizon_seconds if horizon_seconds is not None else config.defaults.horizon.total_seconds()
    )
    horizon_steps = max(1, int(horizon_secs // max(1, eff_step.total_seconds())))
    levels = list(config.defaults.confidence_levels)

    source = PromSource(
        url=config.datasource.url,
        timeout_seconds=config.datasource.timeout.total_seconds(),
        max_concurrency=config.datasource.max_concurrent_requests,
        max_keepalive_connections=config.datasource.max_keepalive_connections,
    )
    try:
        end = datetime.now(tz=UTC)
        start = end - lookback
        step_seconds = max(1, int(eff_step.total_seconds()))
        try:
            frames = await asyncio.wait_for(
                source.query_range(query.promql, start, end, step_seconds),
                timeout=config.safety.query_timeout.total_seconds(),
            )
        except TimeoutError as exc:
            raise InspectError("query timed out", exit_code=2) from exc
        except Exception as exc:
            raise InspectError(f"query failed: {exc}", exit_code=2) from exc

        frame = _pick_frame(frames, series_filter)

        cleaned = preprocess_mod.preprocess_series(
            timestamps=list(frame.timestamps),
            values=list(frame.values),
            step=eff_step,
            config=config.groups[_group_index(config, group_name)].preprocess,
        )
        if cleaned.dropped:
            raise InspectError("series was dropped by the preprocessor (too many gaps)")

        df = pd.DataFrame(
            {
                "unique_id": ["s"] * len(cleaned.values),
                "ds": pd.to_datetime(cleaned.timestamps),
                "y": cleaned.values,
            }
        )

        spec = _resolve_model_spec(config, model_override)
        model = model_registry.build(spec.name, season_length=eff_season_length, params=spec.params)

        loop = asyncio.get_running_loop()
        freq = _pandas_freq(step_seconds)
        try:
            forecast = await asyncio.wait_for(
                loop.run_in_executor(
                    None, model.fit_predict, df, horizon_steps, freq, levels, None
                ),
                timeout=config.safety.fit_timeout.total_seconds(),
            )
        except TimeoutError as exc:
            raise InspectError("fit timed out", exit_code=2) from exc
        except Exception as exc:
            raise InspectError(f"fit failed: {exc}") from exc

        # Apply count-aware safeguards (``clip_non_negative`` /
        # ``round_to_integer``) so the inspector mirrors what the live
        # runner actually emits. Without this, a ``data_profile: count``
        # query inspected by an operator would show raw model output
        # — possibly negative or fractional values — that the live
        # exporter would have clipped or rounded. The inspector exists
        # to debug "what the live forecaster would do right now"; the
        # safeguards are part of that contract.
        clip, rounder = effective_emission_safeguards(query)
        if clip or rounder:
            forecast = _apply_safeguards_to_forecast_frame(
                forecast, levels=levels, clip=clip, rounder=rounder
            )

        backtest_summary = None
        if with_backtest:
            backtest_summary = await _run_backtest(
                config=config,
                model=model,
                df=df,
                freq=freq,
                levels=levels,
                model_name=spec.name,
                season_length=eff_season_length,
                step_seconds=step_seconds,
            )

        return InspectReport(
            query_id=query.id,
            group=group_name,
            model=spec.name,
            series_labels={k: v for k, v in frame.labels.items() if k != "__name__"},
            step_seconds=step_seconds,
            season_length=eff_season_length,
            horizon_steps=horizon_steps,
            input_stats=InputStats(
                point_count=len(cleaned.values),
                gap_count=sum(cleaned.gaps_filled.values()),
                largest_gap_seconds=cleaned.max_gap_seconds,
                lookback_start=cleaned.timestamps[0].isoformat() if cleaned.timestamps else "",
                lookback_end=cleaned.timestamps[-1].isoformat() if cleaned.timestamps else "",
            ),
            forecast_timestamps=[str(ts) for ts in forecast.get("ds", [])],
            forecast_yhat=_column_as_floats(forecast, "yhat"),
            forecast_bands=_extract_bands(forecast, levels),
            preprocessing={
                "gaps_filled": dict(cleaned.gaps_filled),
                "outliers_replaced": cleaned.outliers_replaced,
                "change_points": cleaned.change_points,
                "imputed_fraction": cleaned.imputed_fraction,
                "effective_lookback_seconds": cleaned.effective_lookback_seconds,
            },
            backtest=backtest_summary,
        )
    finally:
        await source.aclose()


async def _run_backtest(
    *,
    config: Config,
    model: Any,
    df: pd.DataFrame,
    freq: str,
    levels: list[int],
    model_name: str,
    season_length: int,
    step_seconds: int,
) -> BacktestSummary | None:
    """Per-fold backtest summary — actuals vs. predicted with bands.

    Mirrors the live runner's hold-out sizing exactly:
    ``max(h.steps for h in horizons)`` — large enough to score every
    configured horizon from a single fit. Using the smallest horizon
    here would silently produce different MAPE/MASE numbers from the
    ones the live runner publishes, exactly the kind of "looks good
    in backtest, looks wrong in production" confusion the inspector
    exists to debug.
    """
    horizons = accuracy_mod.horizons_from_durations(
        list(config.defaults.accuracy.horizons), timedelta(seconds=step_seconds)
    )
    if not horizons:
        return None
    horizon_steps = max(h.steps for h in horizons)
    loop = asyncio.get_running_loop()
    try:
        outcome = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: accuracy_mod.backtest(
                    model=model,
                    df=df,
                    horizon_steps=horizon_steps,
                    freq=freq,
                    levels=levels,
                ),
            ),
            timeout=config.safety.fit_timeout.total_seconds(),
        )
    except TimeoutError:
        return BacktestSummary(model=model_name, n_holdout=0, mape=math.nan, mase=math.nan, rows=[])
    except Exception:
        return BacktestSummary(model=model_name, n_holdout=0, mape=math.nan, mase=math.nan, rows=[])
    if outcome is None:
        return None
    test, predictions = outcome

    n = min(len(test), len(predictions))
    if n == 0:
        return BacktestSummary(model=model_name, n_holdout=0, mape=math.nan, mase=math.nan, rows=[])

    results = accuracy_mod.evaluate(
        test=test,
        predictions=predictions,
        train=df.iloc[:-horizon_steps].reset_index(drop=True),
        season_length=season_length,
        horizons=horizons[:1],
        model_name=model_name,
    )
    mape = results[0].mape if results else math.nan
    mase = results[0].mase if results else math.nan
    rows = _materialise_backtest_rows(test, predictions, n=n, levels=levels)
    return BacktestSummary(model=model_name, n_holdout=n, mape=mape, mase=mase, rows=rows)


def _materialise_backtest_rows(
    test: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    n: int,
    levels: list[int],
) -> list[BacktestFoldRow]:
    """Stitch the hold-out actuals and per-step predictions into rows."""
    rows: list[BacktestFoldRow] = []
    yhat_col = predictions["yhat"].tolist() if "yhat" in predictions.columns else []
    actuals = test["y"].tolist() if "y" in test.columns else []
    timestamps = test["ds"].tolist() if "ds" in test.columns else []
    band_cols: dict[str, dict[str, list[float]]] = {}
    for level in levels:
        lo = f"yhat_lower_{level}"
        hi = f"yhat_upper_{level}"
        if lo in predictions.columns and hi in predictions.columns:
            band_cols[str(level)] = {
                "lower": predictions[lo].tolist(),
                "upper": predictions[hi].tolist(),
            }
    for i in range(n):
        bands_at_step: dict[str, dict[str, float]] = {}
        for level_label, values in band_cols.items():
            bands_at_step[level_label] = {
                "lower": float(values["lower"][i]),
                "upper": float(values["upper"][i]),
            }
        rows.append(
            BacktestFoldRow(
                ds=str(timestamps[i]) if i < len(timestamps) else "",
                actual=float(actuals[i]) if i < len(actuals) else math.nan,
                yhat=float(yhat_col[i]) if i < len(yhat_col) else math.nan,
                bands=bands_at_step,
            )
        )
    return rows


def _group_index(config: Config, group_name: str) -> int:
    for i, g in enumerate(config.groups):
        if g.name == group_name:
            return i
    raise InspectError(f"group {group_name!r} disappeared between find_query and preprocess")


def _resolve_model_spec(
    config: Config,
    override: str | None,
) -> Any:
    """Pick the model spec for the inspector fit.

    Mirrors :func:`promforecast.preview._pick_model_spec` shape: an
    explicit override must match one of the configured models so the
    inspector always reflects an actually-deployable choice. Falls
    back to the first configured model when no override is given.
    """
    available = list(config.defaults.models)
    if not available:
        raise InspectError("config.defaults.models is empty; cannot inspect without a model")
    if override is None:
        return available[0]
    for spec in available:
        if spec.name == override:
            return spec
    names = ", ".join(spec.name for spec in available)
    raise InspectError(f"model {override!r} not in configured models ({names})")


def _pick_frame(
    frames: list[SeriesFrame],
    selector: dict[str, str] | None,
) -> SeriesFrame:
    """Pick one frame from the source response, matching :func:`series_filter`.

    Same multi-frame disambiguation contract as the preview endpoint:
    a single frame returns trivially, multiple frames require a
    selector that picks exactly one.
    """
    if not frames:
        raise InspectError("query returned no series", exit_code=1)
    if selector is None:
        if len(frames) == 1:
            return frames[0]
        raise InspectError(f"query returned {len(frames)} series; pass --series-filter to pick one")
    matches = [f for f in frames if all(f.labels.get(k) == v for k, v in selector.items())]
    if not matches:
        raise InspectError(f"series-filter {selector!r} matched no series among {len(frames)}")
    if len(matches) > 1:
        raise InspectError(
            f"series-filter {selector!r} matched {len(matches)} series; tighten the filter"
        )
    return matches[0]


def parse_series_filter(value: str | None) -> dict[str, str] | None:
    """Parse ``--series-filter '{instance="a",device="sda1"}'`` into a dict.

    The shape mirrors PromQL label selectors (``label="value"`` pairs
    inside braces) so an operator can copy a selector straight out of
    a Grafana panel or a recording-rule body. Only the equality
    operator is supported — anything richer should be expressed
    server-side via the query's PromQL.
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    body = stripped[1:-1] if stripped.startswith("{") and stripped.endswith("}") else stripped
    out: dict[str, str] = {}
    if not body.strip():
        return None
    for part in _split_top_level_commas(body):
        if "=" not in part:
            raise InspectError(
                f"--series-filter token {part!r} is missing an '=' — use label=\"value\""
            )
        label, _, raw = part.partition("=")
        out[label.strip()] = raw.strip().strip('"').strip("'")
    return out


def _split_top_level_commas(body: str) -> Iterable[str]:
    depth = 0
    current: list[str] = []
    for ch in body:
        if ch == "," and depth == 0:
            piece = "".join(current).strip()
            if piece:
                yield piece
            current = []
            continue
        if ch == '"':
            depth ^= 1
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        yield tail


def _column_as_floats(forecast: pd.DataFrame, col: str) -> list[float]:
    if col not in forecast.columns:
        return []
    out: list[float] = []
    for v in forecast[col].tolist():
        try:
            f = float(v)
        except (TypeError, ValueError):
            out.append(math.nan)
            continue
        out.append(f)
    return out


def _extract_bands(forecast: pd.DataFrame, levels: list[int]) -> dict[str, dict[str, list[float]]]:
    bands: dict[str, dict[str, list[float]]] = {}
    for level in levels:
        lo = f"yhat_lower_{level}"
        hi = f"yhat_upper_{level}"
        if lo not in forecast.columns or hi not in forecast.columns:
            continue
        bands[str(level)] = {
            "lower": _column_as_floats(forecast, lo),
            "upper": _column_as_floats(forecast, hi),
        }
    return bands


def render_text(report: InspectReport) -> str:
    """Human-friendly inspector output for the CLI default mode."""
    lines: list[str] = []
    lines.append(f"Query:        {report.query_id} (group {report.group})")
    lines.append(f"Model:        {report.model}")
    label_str = (
        ", ".join(f"{k}={v}" for k, v in sorted(report.series_labels.items()))
        if report.series_labels
        else "(none)"
    )
    lines.append(f"Series:       {label_str}")
    lines.append(
        f"Step / season / horizon: {report.step_seconds}s / "
        f"{report.season_length} / {report.horizon_steps} steps"
    )
    lines.append("")
    lines.append("Input stats:")
    lines.append(f"  point_count        : {report.input_stats.point_count}")
    lines.append(f"  gaps_filled        : {report.input_stats.gap_count}")
    lines.append(f"  largest_gap_seconds: {report.input_stats.largest_gap_seconds:.1f}")
    lines.append(f"  lookback_start     : {report.input_stats.lookback_start}")
    lines.append(f"  lookback_end       : {report.input_stats.lookback_end}")
    lines.append("")
    lines.append("Preprocessing applied:")
    for key, value in report.preprocessing.items():
        lines.append(f"  {key:<26} : {value}")
    lines.append("")
    head_n = min(_TABLE_HEAD_ROWS, len(report.forecast_yhat))
    lines.append(f"Forecast (first {head_n} of {len(report.forecast_yhat)} rows):")
    if head_n == 0:
        lines.append("  (no forecast rows)")
    else:
        for i in range(head_n):
            ts = report.forecast_timestamps[i] if i < len(report.forecast_timestamps) else ""
            yhat = report.forecast_yhat[i]
            band_summary = ""
            if report.forecast_bands:
                first_level = sorted(report.forecast_bands.keys())[0]
                lo = report.forecast_bands[first_level]["lower"][i]
                hi = report.forecast_bands[first_level]["upper"][i]
                band_summary = f"  [{first_level}: {lo:.4f}..{hi:.4f}]"
            lines.append(f"  {ts}  yhat={yhat:.4f}{band_summary}")
    if report.backtest is not None:
        lines.append("")
        lines.append("Backtest (most-recent hold-out):")
        lines.append(f"  model    : {report.backtest.model}")
        lines.append(f"  n_holdout: {report.backtest.n_holdout}")
        lines.append(f"  MAPE     : {_format_metric(report.backtest.mape)}%")
        lines.append(f"  MASE     : {_format_metric(report.backtest.mase)}")
        lines.append(f"  per-fold rows (first {min(_TABLE_HEAD_ROWS, len(report.backtest.rows))}):")
        for row in report.backtest.rows[:_TABLE_HEAD_ROWS]:
            band_summary = ""
            if row.bands:
                first_level = sorted(row.bands.keys())[0]
                band_summary = (
                    f"  [{first_level}: {row.bands[first_level]['lower']:.4f}.."
                    f"{row.bands[first_level]['upper']:.4f}]"
                )
            lines.append(
                f"    {row.ds}  actual={_format_metric(row.actual)} "
                f"yhat={_format_metric(row.yhat)}{band_summary}"
            )
    return "\n".join(lines) + "\n"


def _format_metric(value: float) -> str:
    return f"{value:.4f}" if math.isfinite(value) else "NaN"


def _json_value(value: float) -> float | None:
    return value if math.isfinite(value) else None
