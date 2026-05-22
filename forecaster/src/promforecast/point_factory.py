"""Per-series snapshot point assembly.

Each per-series fit produces up to four metric families: forecast points
(value + confidence band), accuracy backtest scores, deviation from the
backtested band, and a composite quality score. They share the same shape:
``base_labels`` (from the source frame) plus ``id``/``group`` plus a few
metric-specific labels.

:class:`SeriesPointFactory` is bound to one series and exposes one method
per metric family. It owns the ``common_labels`` derivation so callers
don't keep recomputing it, and it captures the fan-out logic (per-level,
per-model, per-horizon) so the runner can stay focused on orchestration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import deviation as deviation_mod
from . import quality as quality_mod
from .accuracy import overall_mape, overall_mase
from .exporter import (
    AccuracyPoint,
    CalibrationOffsetPoint,
    ContributionPoint,
    DeviationPoint,
    EnsembleWeightPoint,
    ForecastPoint,
    GrowthRatePoint,
    QualityPoint,
    ThresholdEtaPoint,
)
from .sink import ForecastCurvePoint

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from .accuracy import AccuracyResult
    from .capacity import GrowthRate, ThresholdEta
    from .contribution import ContributionResult
    from .selector import SelectionResult


def _format_window(seconds: float) -> str:
    """Format a window duration as a stable Prometheus-style label.

    The growth-rate ``window`` label is operator-facing — it appears on
    every panel and alert tuned against these metrics — so we render it
    in the same compact form the config accepts (``1d``, ``6h``,
    ``15m``, ``45s``). We snap to the largest natural unit that divides
    the value cleanly to match what the operator wrote in YAML; mixed
    durations (e.g. ``90m``) fall back to seconds so the label is still
    well-defined.
    """
    s = round(seconds)
    if s <= 0:
        return "0s"
    if s % 86_400 == 0:
        return f"{s // 86_400}d"
    if s % 3_600 == 0:
        return f"{s // 3_600}h"
    if s % 60 == 0:
        return f"{s // 60}m"
    return f"{s}s"


@dataclass(frozen=True)
class BandSummary:
    """Per-level (lower, upper) band from the trailing backtest row."""

    lower: float
    upper: float

    @property
    def width(self) -> float:
        return abs(self.upper - self.lower)


@dataclass(frozen=True)
class BacktestOutcome:
    """Per-(model, horizon) scores plus a compact summary of the trailing row.

    Storing only the last (actual, predicted, band) tuple — not the full
    ``test``/``predictions`` frames — keeps memory bounded for high-cardinality
    groups.
    """

    results: list[AccuracyResult]
    last_actual: float
    last_yhat: float
    last_bands: dict[int, BandSummary]


class SeriesPointFactory:
    """Assemble snapshot points for a single (query, series).

    The factory is constructed once per series and reused for each metric
    family. Common label derivation (everything except ``__name__``) happens
    in ``__init__`` so per-family helpers stay slim.
    """

    __slots__ = ("_base_common", "_group", "_levels", "_metric_id")

    def __init__(
        self,
        *,
        metric_id: str,
        base_labels: dict[str, str],
        group: str,
        levels: list[int],
    ) -> None:
        self._metric_id = metric_id
        self._group = group
        self._levels = levels
        self._base_common = {k: v for k, v in base_labels.items() if k != "__name__"}

    def forecast_points(
        self,
        forecast: pd.DataFrame,
        model_name: str,
        selection: SelectionResult | None,
    ) -> Iterable[ForecastPoint]:
        """Convert the *end-of-horizon* forecast row into snapshot points.

        /metrics is a snapshot endpoint, so we expose a single forecast
        value per (series, model, level). The exposed value is the
        **last** row of the forecast frame — i.e. the value at horizon
        steps from now (``defaults.horizon`` after the most recent
        sample). The historical label ``horizon="next"`` reads "the
        next snapshot value to be served", not "the next time step";
        the full per-step curve is what the optional remote_write sink
        carries. See :meth:`forecast_curve_points`.

        This convention matters for capacity-style alerts: an alert
        like ``node_filesystem_avail_bytes_forecast < 0`` answers
        "will the disk be empty at the configured horizon", not "will
        it be empty at the next 5-minute step". The v1.2 ETA family
        (``<id>_forecast_time_to_threshold_seconds``) is the
        complementary read for "*when* does the forecast first cross
        the threshold".

        When ``selection`` is set (auto-select mode), additional labels
        ``best_model`` and ``model_fallback`` are attached so dashboards can
        distinguish "this model won the horse race" from "this is the fallback
        after the winner blew its MAPE budget".

        Conformal calibration's ``calibrated="true"`` label is added post-hoc
        by ``runner._apply_calibration_offsets`` after this method returns,
        so callers don't need to thread that flag through here.
        """
        if forecast.empty:
            return
        last = forecast.iloc[-1]
        common = self._build_common_labels(model_name, selection)
        yield ForecastPoint(
            metric=f"{self._metric_id}_forecast",
            labels=common,
            value=float(last["yhat"]),
        )
        for level in self._levels:
            lo_col = f"yhat_lower_{level}"
            hi_col = f"yhat_upper_{level}"
            if lo_col in forecast.columns:
                yield ForecastPoint(
                    metric=f"{self._metric_id}_forecast_lower",
                    labels={**common, "level": str(level)},
                    value=float(last[lo_col]),
                )
            if hi_col in forecast.columns:
                yield ForecastPoint(
                    metric=f"{self._metric_id}_forecast_upper",
                    labels={**common, "level": str(level)},
                    value=float(last[hi_col]),
                )

    def calibration_offset_points(
        self,
        model_name: str,
        offsets: dict[int, float],
    ) -> Iterable[CalibrationOffsetPoint]:
        """One ``forecast_band_calibration_offset`` per (model, level).

        Emitted only when split-conformal calibration ran for this series.
        ``offsets`` is the per-level additive correction returned by
        :mod:`promforecast.conformal`; non-finite values are skipped.
        """
        common = self._common_with_id()
        for level, offset in offsets.items():
            if not math.isfinite(offset):
                continue
            yield CalibrationOffsetPoint(
                labels={**common, "model": model_name, "level": str(level)},
                offset=float(offset),
            )

    def forecast_curve_points(
        self,
        forecast: pd.DataFrame,
        model_name: str,
        selection: SelectionResult | None,
    ) -> Iterable[ForecastCurvePoint]:
        """Yield every horizon step as a (timestamp, value) push point.

        Used by the remote_write sink to ingest the full curve into a TSDB.
        Each row in ``forecast`` produces one ``yhat`` point plus a per-level
        ``lower``/``upper`` pair, all sharing the same ``timestamp_ms``.

        Rows with non-finite timestamps or values are dropped — TSDBs reject
        NaN samples, and emitting them would just generate noise in the
        sink's failure counter.
        """
        if forecast.empty:
            return
        common = self._build_common_labels(model_name, selection)
        for _, row in forecast.iterrows():
            ts = row.get("ds")
            if ts is None:
                continue
            try:
                # ``Timestamp.timestamp()`` returns seconds; we want ms.
                ts_ms = int(float(ts.timestamp()) * 1000)
            except (AttributeError, TypeError, ValueError):
                continue
            yhat_raw = row.get("yhat")
            if yhat_raw is not None and math.isfinite(float(yhat_raw)):
                yield ForecastCurvePoint(
                    metric=f"{self._metric_id}_forecast",
                    labels=common,
                    value=float(yhat_raw),
                    timestamp_ms=ts_ms,
                )
            for level in self._levels:
                lo_col = f"yhat_lower_{level}"
                hi_col = f"yhat_upper_{level}"
                if lo_col in forecast.columns:
                    lo = row[lo_col]
                    if math.isfinite(float(lo)):
                        yield ForecastCurvePoint(
                            metric=f"{self._metric_id}_forecast_lower",
                            labels={**common, "level": str(level)},
                            value=float(lo),
                            timestamp_ms=ts_ms,
                        )
                if hi_col in forecast.columns:
                    hi = row[hi_col]
                    if math.isfinite(float(hi)):
                        yield ForecastCurvePoint(
                            metric=f"{self._metric_id}_forecast_upper",
                            labels={**common, "level": str(level)},
                            value=float(hi),
                            timestamp_ms=ts_ms,
                        )

    def ensemble_weight_points(
        self,
        weights: dict[str, float],
    ) -> Iterable[EnsembleWeightPoint]:
        """One ``forecast_ensemble_weight`` sample per component model."""
        common = self._common_with_id()
        for model_name, weight in weights.items():
            yield EnsembleWeightPoint(
                labels={**common, "model": model_name},
                weight=float(weight),
            )

    def contribution_points(
        self,
        contributions: list[ContributionResult],
    ) -> Iterable[ContributionPoint]:
        """Yield one ``<id>_forecast_contribution_ratio`` sample per regressor.

        The label set mirrors the forecast point family (base series labels
        + ``group``) plus a ``regressor`` discriminator. We deliberately
        omit ``model`` here: the contribution is a property of the data, not
        the model. Plug-in models that derive a per-model coefficient can
        emit a parallel metric in a future revision without colliding.
        """
        metric_name = f"{self._metric_id}_forecast_contribution_ratio"
        for contribution in contributions:
            labels = {
                **self._base_common,
                "group": self._group,
                "regressor": contribution.regressor_id,
            }
            yield ContributionPoint(
                metric=metric_name,
                labels=labels,
                value=float(contribution.ratio),
            )

    def threshold_eta_points(
        self,
        model_name: str,
        etas: list[ThresholdEta],
        selection: SelectionResult | None,
    ) -> Iterable[ThresholdEtaPoint]:
        """Yield one ``<id>_forecast_time_to_threshold_seconds`` per ETA.

        ``etas`` carries the per-(threshold, level) ETA computed by
        :func:`promforecast.capacity.compute_threshold_etas`. The metric
        name follows the forecast-point convention so dashboards can join
        on ``id`` between the underlying forecast and its derived ETA.
        Labels mirror the forecast point family (``model``, ``horizon``,
        plus optional ``best_model``/``model_fallback``) and add
        ``name`` + ``comparator`` for the threshold, and ``level`` when
        the ETA was computed against a band.
        """
        if not etas:
            return
        common = self._build_common_labels(model_name, selection)
        metric_name = f"{self._metric_id}_forecast_time_to_threshold_seconds"
        for eta in etas:
            labels = {
                **common,
                "name": eta.name,
                "comparator": eta.comparator,
            }
            if eta.level is not None:
                labels["level"] = str(eta.level)
            yield ThresholdEtaPoint(
                metric=metric_name,
                labels=labels,
                seconds=eta.seconds,
            )

    def growth_rate_points(
        self,
        model_name: str,
        rates: list[GrowthRate],
        selection: SelectionResult | None,
    ) -> Iterable[GrowthRatePoint]:
        """Yield one ``<id>_forecast_growth_rate`` per configured window.

        Labels carry ``window`` (the configured duration rendered as a
        Prometheus-style ``<n><unit>`` string so dashboards can group
        cleanly) and ``units`` (``units_per_second`` / ``percent``).
        """
        if not rates:
            return
        common = self._build_common_labels(model_name, selection)
        metric_name = f"{self._metric_id}_forecast_growth_rate"
        for rate in rates:
            labels = {
                **common,
                "window": _format_window(rate.window_seconds),
                "units": rate.units,
            }
            yield GrowthRatePoint(
                metric=metric_name,
                labels=labels,
                value=rate.rate,
            )

    def _build_common_labels(
        self,
        model_name: str,
        selection: SelectionResult | None,
    ) -> dict[str, str]:
        common: dict[str, str] = {
            **self._base_common,
            "group": self._group,
            "model": model_name,
            "horizon": "next",
        }
        if selection is not None:
            common["best_model"] = selection.best_model
            common["model_fallback"] = "true" if selection.fallback_applied else "false"
        return common

    def accuracy_points(
        self,
        backtest_results: dict[str, list[AccuracyResult]],
    ) -> Iterable[AccuracyPoint]:
        """Flatten per-series backtest results into ``AccuracyPoint``s."""
        common = self._common_with_id()
        for model_name, results in backtest_results.items():
            for res in results:
                yield AccuracyPoint(
                    labels={**common, "model": model_name, "horizon": res.horizon},
                    mape=res.mape,
                    mase=res.mase,
                )

    def deviation_points(
        self,
        backtests: dict[str, BacktestOutcome],
        emitted_models: list[str],
    ) -> Iterable[DeviationPoint]:
        """Compute deviation from the trailing backtest (actual, band) pair.

        The forecaster is stateless, so we use the most recent in-sample
        point where we have both an actual and a predicted band — i.e. the
        last row of the hold-out window — as the canonical "is the world
        still behaving like the model expects?" signal. One sample per
        (series, model, level) is emitted.
        """
        common = self._common_with_id()
        for model_name in emitted_models:
            outcome = backtests.get(model_name)
            if outcome is None:
                continue
            for level in self._levels:
                band = outcome.last_bands.get(level)
                if band is None:
                    continue
                dev = deviation_mod.compute(
                    actual=outcome.last_actual,
                    yhat=outcome.last_yhat,
                    yhat_lower=band.lower,
                    yhat_upper=band.upper,
                )
                yield DeviationPoint(
                    labels={**common, "model": model_name, "level": str(level)},
                    ratio=dev.ratio,
                    outside_band=dev.outside_band,
                )

    def quality_points(
        self,
        backtests: dict[str, BacktestOutcome],
        emitted_models: list[str],
        train_y: list[float],
        coverage: float,
        imputed_fraction: float = 0.0,
        score_kind: quality_mod.ScoreKind = "mape",
    ) -> Iterable[QualityPoint]:
        """Compute a composite quality score per emitted (series, model).

        Combines (1) overall accuracy across the configured accuracy
        horizons — MAPE for continuous series, MASE for intermittent
        series, (2) the mean backtest band width relative to
        ``mean(|train_y|)`` (paired with actuals during backtest, which
        is more meaningful than scoring the unvalidated forward bands),
        and (3) the lookback data coverage. When ``imputed_fraction > 0``
        the coverage sub-score is scaled by ``(1 - imputed_fraction)`` so
        a series whose lookback is mostly synthesised by gap imputation
        does not silently score well on the coverage dimension.

        ``score_kind`` chooses the scoring unit. The default ``"mape"``
        preserves the v1.0 behaviour; pass ``"mase"`` when the caller's
        ``data_profile`` is ``intermittent`` so a series whose actuals
        are mostly zero (and therefore have NaN MAPE everywhere) still
        gets a meaningful quality signal.

        Models with no backtest data — typically the auto-select fallback
        model when it isn't in the configured ``models`` list — are skipped
        rather than scored as zero. Emitting zero quality for fallback'd
        series would fire the ``ForecastQualityZero`` alert without signal;
        the ``ForecastFallbackPersistent`` alert covers that case better.

        See :mod:`promforecast.quality` for the exact formula.
        """
        abs_train = [abs(v) for v in train_y]
        scale = (sum(abs_train) / len(abs_train)) if abs_train else 0.0
        common = self._common_with_id()
        score_summary = overall_mase if score_kind == "mase" else overall_mape
        for model_name in emitted_models:
            outcome = backtests.get(model_name)
            if outcome is None:
                continue
            accuracy = score_summary(outcome.results)
            widths = [b.width for b in outcome.last_bands.values() if math.isfinite(b.width)]
            avg_width = sum(widths) / len(widths) if widths else float("nan")
            band_width_relative = (
                (avg_width / scale) if scale > 0 and math.isfinite(avg_width) else None
            )
            score = quality_mod.compute(
                backtest_mape=accuracy,
                band_width_relative=band_width_relative,
                coverage=coverage,
                imputed_fraction=imputed_fraction,
                score_kind=score_kind,
            )
            yield QualityPoint(
                labels={**common, "model": model_name},
                score=score,
            )

    def _common_with_id(self) -> dict[str, str]:
        """Common labels for the operational families (carry ``id`` + ``group``).

        Forecast points carry ``__name__``-stripped base labels but use the
        metric *name* itself as the identifier; the operational families need
        ``id`` as a distinct label so they can be joined back to the forecast
        family in dashboards.
        """
        return {**self._base_common, "id": self._metric_id, "group": self._group}
