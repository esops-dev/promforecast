"""Prophet wrapper that conforms to the ``promforecast.models.ForecastModel``
protocol.

Prophet's strengths — automatic changepoint detection, native holiday
modelling, smooth additive seasonalities — make it a solid alternative to
``statsforecast`` for series with strong calendar effects. The wrapper here
keeps the public surface identical to the built-in models so a config can
swap ``AutoARIMA`` for ``Prophet`` without any other change.

Two design choices worth flagging:

* ``prophet`` is imported lazily inside :meth:`fit_predict`. Loading it
  pulls in cmdstanpy and its Stan toolchain, which on cold containers can
  take seconds; deferring the import keeps the model-registry scan
  millisecond-fast.
* Prophet's ``interval_width`` is consulted at *predict* time, not fit
  time — the underlying MAP fit is identical regardless of the requested
  band width. The wrapper exploits this: one fit per series, one cheap
  ``predict()`` per configured confidence level. For two levels (e.g.
  ``[80, 95]``) this is ~half the CPU of a naive per-level-fit
  implementation; for three or more, the speedup is proportionally
  larger.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


class ProphetModel:
    """Single-series Prophet wrapper.

    Args mirror the rest of the registry:

    * ``season_length`` is accepted for protocol parity. Prophet detects
      seasonality on its own from the time index, so the value is not
      forwarded; we still capture it so the registry's
      ``Cls(season_length=..., **params)`` construction call works without
      a custom shim.
    * ``params`` is forwarded as keyword arguments to Prophet's
      constructor. Anything Prophet accepts (``growth``, ``changepoints``,
      ``yearly_seasonality``, ``holidays``, ...) lands here. Unknown keys
      surface as a ``TypeError`` at fit time. ``interval_width``, when
      supplied, pins the band width for *every* configured level; otherwise
      each level is mapped to ``level / 100`` as Prophet's native scale.
    """

    name = "Prophet"

    def __init__(self, season_length: int = 288, **params: Any) -> None:
        self._season_length = season_length
        self._params = params

    def fit_predict(
        self,
        df: pd.DataFrame,
        horizon: int,
        freq: str,
        levels: list[int],
        regressors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Fit Prophet once per series and emit a forecast row per level.

        See the module docstring for why this is one fit per series rather
        than one per (series, level): ``interval_width`` is a predict-time
        knob, so per-level bands cost only an extra cheap ``predict()``
        call after the model is fit.
        """
        import pandas as pd  # noqa: PLC0415
        from prophet import Prophet  # noqa: PLC0415

        # Read the override without mutating ``self._params`` — the
        # instance may be reused across runner invocations and a destructive
        # ``pop`` would silently disable the override on the second call.
        param_interval_width = self._params.get("interval_width")
        constructor_kwargs = {k: v for k, v in self._params.items() if k != "interval_width"}

        results: list[pd.DataFrame] = []
        for uid, group in df.groupby("unique_id", sort=False):
            series = group[["ds", "y"]].copy()
            # Attach exogenous regressors aligned to the series ``ds``
            # column. Prophet expects the same columns to be present in
            # the prediction frame, so we keep a per-series copy of the
            # future regressor values where possible.
            extra_cols: list[str] = []
            if regressors is not None and not regressors.empty:
                merged = series.merge(regressors, on="ds", how="left")
                extra_cols = [c for c in regressors.columns if c != "ds"]
                series = merged

            # Single fit per series. The constructor's ``interval_width``
            # is irrelevant — it gets overwritten per-level on the fitted
            # instance below — but we set a sensible default so a debugger
            # snapshot of the unmodified model isn't misleading.
            initial_width = (
                param_interval_width if param_interval_width is not None else levels[0] / 100
            )
            model = Prophet(interval_width=initial_width, **constructor_kwargs)
            for col in extra_cols:
                model.add_regressor(col)
            model.fit(series)

            future = model.make_future_dataframe(
                periods=horizon,
                freq=freq,
                include_history=False,
            )
            # Carry forward the most recent regressor value for each future
            # row. A more sophisticated implementation could surface a
            # per-regressor forecast; this conservative default avoids
            # inventing values out of thin air.
            for col in extra_cols:
                future[col] = series[col].iloc[-1]

            level_frames: list[pd.DataFrame] = []
            for level in levels:
                # Mutate only the band width on the *fitted* model. Prophet
                # reads ``self.interval_width`` inside ``predict()`` to
                # compute the percentile bounds from the cached forecast
                # samples; it is *not* a property of the fit itself.
                model.interval_width = (
                    param_interval_width if param_interval_width is not None else level / 100
                )
                pred = model.predict(future)
                level_frames.append(
                    pred[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
                        columns={
                            "yhat_lower": f"yhat_lower_{level}",
                            "yhat_upper": f"yhat_upper_{level}",
                        }
                    )
                )

            # The mean (``yhat``) is identical across levels — collapse on
            # ``ds`` so the per-level lower/upper columns share one row.
            merged = level_frames[0][["ds", "yhat"]]
            for level, frame in zip(levels, level_frames, strict=True):
                merged = merged.merge(
                    frame[["ds", f"yhat_lower_{level}", f"yhat_upper_{level}"]],
                    on="ds",
                    how="left",
                )
            merged.insert(0, "unique_id", uid)
            results.append(merged)

        return pd.concat(results, ignore_index=True)
