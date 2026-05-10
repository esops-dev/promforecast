"""``promforecast validate`` — schema check + optional dry-run sample fit.

``validate`` is the first thing CI runs against a config PR. It must:

* Fail fast and clearly on bad YAML or schema errors (so the GitHub Actions
  log points at the offending config instead of a stack trace).
* Estimate output cardinality so a reviewer can spot a runaway ``models +
  confidence_levels`` combination *before* it ships to prod.
* Optionally run a sample query + fit (``--dry-run``) so the reviewer sees
  the first forecast points and per-model MAPE preview the change will
  produce. The sample is intentionally tiny — it's a smoke test, not a
  benchmark.

The function returns a process exit code so it composes cleanly with
``argparse`` in ``main.cli`` and with shell pipelines in CI.
"""

from __future__ import annotations

import asyncio
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING

from . import config as config_module
from .accuracy import horizons_from_durations

if TYPE_CHECKING:
    from .config import Config
    from .source import SeriesFrame

Writer = Callable[[str], None]


@dataclass(frozen=True)
class CardinalityEstimate:
    """A coarse "what will this config emit?" number for review.

    The estimate is an upper bound assuming every query returns
    ``max_series_per_query`` series; users with low-cardinality queries will
    emit fewer, but the worst case is what trips the global cap.
    """

    series_per_query: int
    queries: int
    models: int
    confidence_levels: int
    forecast_lines: int
    accuracy_lines: int
    deviation_lines: int
    quality_lines: int
    operational_lines: int

    @property
    def total(self) -> int:
        return (
            self.forecast_lines
            + self.accuracy_lines
            + self.deviation_lines
            + self.quality_lines
            + self.operational_lines
        )


def run_validate(
    config_path: Path,
    *,
    dry_run: bool = False,
    datasource_url_override: str | None = None,
    out: IO[str] | None = None,
) -> int:
    """Validate ``config_path`` and optionally run a sample fit.

    Returns 0 on success, 1 on validation failure, 2 on dry-run failure.
    Output is plain text by default; ``out`` lets tests capture it.
    """
    stream: IO[str] = out if out is not None else sys.stdout

    def write(line: str) -> None:
        stream.write(line + "\n")

    def err(line: str) -> None:
        sys.stderr.write(line + "\n")

    try:
        cfg = config_module.load(config_path)
    except FileNotFoundError:
        err(f"error: config not found: {config_path}")
        return 1
    except Exception as exc:
        err(f"error: config validation failed: {exc}")
        return 1

    write(f"OK  config valid: {config_path}")
    write(f"    groups:  {len(cfg.groups)}")
    queries = sum(len(g.queries) for g in cfg.groups)
    write(f"    queries: {queries}")
    write(f"    models:  {', '.join(cfg.defaults.models)}")

    estimate = _estimate_cardinality(cfg)
    write("")
    write("Estimated cardinality (worst case at max_series_per_query):")
    write(
        f"  forecast lines     {estimate.forecast_lines:>8}  "
        f"= series({estimate.series_per_query}) * models({estimate.models}) "
        f"* (1 + 2*levels({estimate.confidence_levels})) * queries({estimate.queries})"
    )
    write(f"  accuracy lines     {estimate.accuracy_lines:>8}")
    write(f"  deviation lines    {estimate.deviation_lines:>8}")
    write(f"  quality lines      {estimate.quality_lines:>8}")
    write(f"  operational lines  {estimate.operational_lines:>8}")
    write(f"  total              {estimate.total:>8}")
    if cfg.safety.max_total_series and estimate.total > cfg.safety.max_total_series:
        write(
            f"  WARN: total exceeds safety.max_total_series ({cfg.safety.max_total_series}); "
            "series_overflow will drop the excess"
        )

    if not dry_run:
        return 0

    write("")
    write("Dry-run: running sample query + fit on first query of first group...")
    try:
        rc = asyncio.run(_dry_run(cfg, datasource_url_override, write))
    except Exception as exc:
        err(f"error: dry-run failed: {exc}")
        return 2
    return rc


def _estimate_cardinality(cfg: Config) -> CardinalityEstimate:
    """Worst-case forecast/accuracy/deviation/quality/operational line counts.

    Lower-bounding (e.g. accounting for ``min_points`` drops) would require
    real data — the point of this estimate is to catch *configurations*
    that will trip the global cap, so we deliberately overstate.
    """
    queries = sum(len(g.queries) for g in cfg.groups)
    series_cap = cfg.safety.max_series_per_query
    n_models = max(1, len(cfg.defaults.models))
    n_levels = len(cfg.defaults.confidence_levels)
    n_horizons = max(1, len(cfg.defaults.accuracy_horizons))

    # Each forecast: 1 yhat row + 2 (lower/upper) per confidence level.
    per_series_forecast = n_models * (1 + 2 * n_levels)
    # Auto-select emits a single winner; explicit emits all configured models.
    if cfg.defaults.auto_select and cfg.defaults.auto_select != "ensemble":
        per_series_forecast = 1 + 2 * n_levels

    forecast_lines = queries * series_cap * per_series_forecast
    accuracy_lines = (
        queries * series_cap * n_models * n_horizons * 2  # MAPE + MASE
        if cfg.defaults.evaluate
        else 0
    )
    # Deviation: one per (series, model, level) on the *emitted* models —
    # in auto-select mode that's 1 model; in explicit mode it's n_models.
    emitted_models = 1 if cfg.defaults.auto_select else n_models
    deviation_lines = (
        queries * series_cap * emitted_models * n_levels * 2  # ratio + outside_band
        if cfg.defaults.emit_deviation
        else 0
    )
    quality_lines = queries * series_cap * emitted_models if cfg.defaults.emit_quality else 0
    # Operational: 7 gauges + N reasons per group; we ballpark 7 per group.
    operational_lines = 7 * len(cfg.groups)

    return CardinalityEstimate(
        series_per_query=series_cap,
        queries=queries,
        models=n_models,
        confidence_levels=n_levels,
        forecast_lines=forecast_lines,
        accuracy_lines=accuracy_lines,
        deviation_lines=deviation_lines,
        quality_lines=quality_lines,
        operational_lines=operational_lines,
    )


async def _dry_run(
    cfg: Config,
    datasource_url_override: str | None,
    write: Writer,
) -> int:
    """Sample-query the configured datasource and run a single fit.

    Picks the first query of the first group, queries the configured
    lookback window, runs the first configured model on the first returned
    series, and prints the leading 10 forecast points + a tiny backtest
    preview. Failures are surfaced as a non-zero return code so CI can fail
    on a broken sample without forcing the operator to read logs.
    """
    from .source import PromSource  # noqa: PLC0415

    if not cfg.groups or not cfg.groups[0].queries:
        write("    (no queries to dry-run)")
        return 0

    group = cfg.groups[0]
    query = group.queries[0]
    url = datasource_url_override or cfg.datasource.url

    write(f"  group:  {group.name}")
    write(f"  query:  {query.id}  ({query.promql.strip()[:60]}...)")
    write(f"  source: {url}")

    source = PromSource(url=url, timeout_seconds=cfg.datasource.timeout.total_seconds())
    try:
        end = datetime.now(tz=UTC)
        start = end - cfg.defaults.lookback
        step_seconds = max(1, int(cfg.defaults.step.total_seconds()))
        t0 = time.monotonic()
        try:
            frames = await source.query_range(query.promql, start, end, step_seconds)
        except Exception as exc:
            write(f"    FAIL  query failed: {exc}")
            return 2
        query_seconds = time.monotonic() - t0

        write(f"    query returned {len(frames)} series in {query_seconds:.2f}s")
        if not frames:
            write("    (empty result — nothing to fit)")
            return 0
        frame = frames[0]
        if len(frame.values) < cfg.defaults.min_points:
            write(
                f"    WARN: first series has {len(frame.values)} points; "
                f"min_points is {cfg.defaults.min_points} — would be dropped at runtime"
            )

        fit_rc = _fit_sample(cfg, frame, write)
        return fit_rc
    finally:
        await source.aclose()


def _fit_sample(cfg: Config, frame: SeriesFrame, write: Writer) -> int:
    """Fit the first configured model and print a 10-line forecast preview.

    Imports of heavy modelling deps are lazy so ``promforecast validate``
    on a config-only PR doesn't pay the import cost when ``--dry-run`` is
    not requested.
    """
    import pandas as pd  # noqa: PLC0415

    from . import models as model_registry  # noqa: PLC0415

    model_name = cfg.defaults.models[0]
    write(f"    fitting first model: {model_name}")

    df = pd.DataFrame(
        {
            "unique_id": ["s"] * len(frame.values),
            "ds": pd.to_datetime(frame.timestamps),
            "y": frame.values,
        }
    )
    freq = _pandas_freq_seconds(int(cfg.defaults.step.total_seconds()))
    horizon = max(
        1,
        int(cfg.defaults.horizon.total_seconds() // cfg.defaults.step.total_seconds()),
    )
    levels = list(cfg.defaults.confidence_levels)
    try:
        model = model_registry.build(model_name, season_length=cfg.defaults.season_length)
        t0 = time.monotonic()
        forecast = model.fit_predict(df=df, horizon=horizon, freq=freq, levels=levels)
        fit_seconds = time.monotonic() - t0
    except Exception as exc:
        write(f"    FAIL  fit failed: {exc}")
        return 2

    write(f"    fit completed in {fit_seconds:.2f}s, {len(forecast)} forecast rows")
    write("    first 10 forecast lines:")
    head = forecast.head(10)
    for _, row in head.iterrows():
        ts = row.get("ds")
        yhat = row.get("yhat")
        yhat_str = f"{yhat:.4f}" if yhat is not None and math.isfinite(float(yhat)) else "NaN"
        write(f"      {ts}  yhat={yhat_str}")

    horizons = horizons_from_durations(list(cfg.defaults.accuracy_horizons), cfg.defaults.step)
    if horizons and len(df) > horizons[0].steps + 1:
        rc = _backtest_preview(model, df, horizons[0].steps, freq, levels, write)
        return rc
    return 0


def _backtest_preview(
    model: object,
    df: object,
    horizon_steps: int,
    freq: str,
    levels: list[int],
    write: Writer,
) -> int:
    """Print a single-horizon MAPE preview from the smallest accuracy horizon.

    A full multi-horizon backtest would multiply the ``--dry-run`` runtime;
    a single horizon is enough to tell "the fit produced sensible numbers"
    from "the fit produced garbage".
    """
    from .accuracy import _mape, backtest  # noqa: PLC0415

    try:
        result = backtest(
            model=model,  # type: ignore[arg-type]
            df=df,  # type: ignore[arg-type]
            horizon_steps=horizon_steps,
            freq=freq,
            levels=levels,
        )
    except Exception as exc:
        write(f"    backtest skipped: {exc}")
        return 0
    if result is None:
        write("    backtest skipped: series too short for a hold-out split")
        return 0
    test, predictions = result
    n = min(len(test), len(predictions))
    if n < 1:
        write("    backtest skipped: empty hold-out")
        return 0
    y_true = test["y"].to_numpy(dtype=float)[:n]
    y_pred = predictions["yhat"].to_numpy(dtype=float)[:n]
    mape = _mape(y_true, y_pred)
    write(f"    backtest preview: MAPE={mape:.2f}% over {n} hold-out points")
    return 0


def _pandas_freq_seconds(seconds: int) -> str:
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"
