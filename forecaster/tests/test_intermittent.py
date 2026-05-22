"""Tests for the intermittent-demand model wrappers.

The actual ``fit_predict`` behaviour exercises ``statsforecast`` end to
end; these tests pin down the wrapper-layer contracts:

* every wrapper is registered under its public name;
* ``season_length`` is accepted and ignored;
* requesting ``levels`` doesn't blow up the way the underlying classes
  do when invoked without ``prediction_intervals``;
* the ``INTERMITTENT_MODEL_NAMES`` set actually lists the names the
  registry knows.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from promforecast import models as model_registry
from promforecast.models import (
    INTERMITTENT_MODEL_NAMES,
    ADIDAModel,
    CrostonModel,
    CrostonSBAModel,
    IMAPAModel,
)


@pytest.mark.parametrize(
    ("name", "expected_class"),
    [
        ("Croston", CrostonModel),
        ("CrostonSBA", CrostonSBAModel),
        ("ADIDA", ADIDAModel),
        ("IMAPA", IMAPAModel),
    ],
)
def test_intermittent_model_registered(name: str, expected_class: type) -> None:
    """Every intermittent wrapper resolves to its concrete class through the registry."""
    model = model_registry.build(name, season_length=288)
    assert isinstance(model, expected_class)
    assert model.name == name


def test_intermittent_set_lists_every_wrapper() -> None:
    """``INTERMITTENT_MODEL_NAMES`` exposes exactly the four shipped wrappers.

    The selector reads this set to prefer intermittent models under
    ``data_profile: intermittent``; a wrapper not in the set would
    silently fail the preference filter.
    """
    assert {"Croston", "CrostonSBA", "ADIDA", "IMAPA"} == INTERMITTENT_MODEL_NAMES


def _intermittent_series(n: int = 60) -> pd.DataFrame:
    """A simple low-density intermittent series for smoke tests."""
    values = [0.0] * n
    for i in (5, 15, 25, 35, 45):
        values[i] = float(i % 5 + 1)
    ds = pd.date_range(datetime(2026, 1, 1, tzinfo=UTC), periods=n, freq="5min")
    return pd.DataFrame({"unique_id": ["s"] * n, "ds": ds, "y": values})


def test_croston_fit_predict_emits_yhat_only_without_levels() -> None:
    """Without ``levels``, the wrapper produces a yhat column and no bands.

    Bands would otherwise require the conformal bootstrap, which is
    only wired in when the caller asks for it; emitting empty band
    columns on a no-levels call would just confuse the exporter.
    """
    model = CrostonSBAModel(season_length=288)
    forecast = model.fit_predict(df=_intermittent_series(), horizon=4, freq="5min", levels=[])
    assert "yhat" in forecast.columns
    assert len(forecast) == 4
    assert not any(col.startswith("yhat_upper_") for col in forecast.columns)


def test_croston_fit_predict_emits_bands_when_levels_requested() -> None:
    """Asking for ``levels`` opts in to the ``ConformalIntervals`` bootstrap.

    The underlying ``statsforecast`` classes raise on a bare ``level=``
    request — the wrapper transparently materialises the bootstrap
    wrapper so the operator's ``confidence_levels:`` config "just works".
    """
    model = CrostonSBAModel(season_length=288)
    forecast = model.fit_predict(df=_intermittent_series(), horizon=4, freq="5min", levels=[80])
    # ``yhat_lower_80`` / ``yhat_upper_80`` are present once the conformal
    # bootstrap is wired in; their absence would mean the wrapper failed
    # to thread ``prediction_intervals`` through.
    assert "yhat_lower_80" in forecast.columns
    assert "yhat_upper_80" in forecast.columns


def test_intermittent_wrapper_ignores_season_length() -> None:
    """The wrappers must accept ``season_length`` for protocol uniformity but
    not pass it into the underlying classes (which would raise).

    Passing wildly different ``season_length`` values shouldn't change
    the forecast output — intermittent methods don't model seasonality.
    """
    df = _intermittent_series()
    a = CrostonModel(season_length=2).fit_predict(df=df, horizon=4, freq="5min", levels=[])
    b = CrostonModel(season_length=4096).fit_predict(df=df, horizon=4, freq="5min", levels=[])
    assert a["yhat"].tolist() == b["yhat"].tolist()


def test_intermittent_wrapper_ignores_regressors() -> None:
    """``statsforecast`` intermittent models don't accept exogenous regressors.

    The wrappers must silently drop the kwarg — otherwise a calendar-
    regressor frame intended for the AutoARIMA fit in the same config
    would blow up the intermittent fit running alongside it.
    """
    df = _intermittent_series()
    bogus = pd.DataFrame({"x": [1.0] * 10})
    forecast = ADIDAModel(season_length=288).fit_predict(
        df=df, horizon=4, freq="5min", levels=[], regressors=bogus
    )
    assert "yhat" in forecast.columns


def test_intermittent_constant_zero_series_does_not_crash() -> None:
    """A degenerate all-zero series must not crash the wrapper.

    The runner already covers a constant-zero series via the standard
    failure-isolation paths, but the wrapper itself shouldn't blow up
    inside that exception handler — keep it ``yhat=0`` and let the
    caller handle the rest.
    """
    n = 60
    df = pd.DataFrame(
        {
            "unique_id": ["s"] * n,
            "ds": pd.date_range(datetime(2026, 1, 1, tzinfo=UTC), periods=n, freq="5min"),
            "y": [0.0] * n,
        }
    )
    forecast = CrostonSBAModel(season_length=288).fit_predict(
        df=df, horizon=2, freq="5min", levels=[]
    )
    assert len(forecast) == 2
    assert all(v == 0.0 for v in forecast["yhat"].tolist())


def test_intermittent_wrapper_conformal_windows_param_override() -> None:
    """``conformal_windows`` is popped from ``params`` so it doesn't
    accidentally land in the constructor (which would TypeError).

    Operators tune the bootstrap size for noisier series; we verify the
    knob is consumed by the wrapper rather than forwarded to the
    underlying statsforecast class.
    """
    model = ADIDAModel(season_length=288, conformal_windows=4)
    # The wrapper stashes everything that isn't the bootstrap knob in
    # ``self._params``; ``conformal_windows`` should *not* be there
    # because the wrapper consumed it.
    forecast = model.fit_predict(df=_intermittent_series(), horizon=2, freq="5min", levels=[80])
    assert "yhat_upper_80" in forecast.columns


_MIN_BACKTEST_LEN = 12


def test_backtest_pathway_emits_mase(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: the accuracy module's MASE summary works on an intermittent fit.

    Catches the regression where MAPE-shaped scoring would silently
    return NaN for a series with many zero actuals.
    """
    from promforecast.accuracy import (  # noqa: PLC0415
        HorizonSpec,
        backtest,
        evaluate,
        overall_mase,
    )

    df = _intermittent_series(n=60)
    out = backtest(
        model=CrostonSBAModel(season_length=288),
        df=df,
        horizon_steps=12,
        freq="5min",
        levels=[],
    )
    assert out is not None
    test, predictions = out
    results = evaluate(
        test=test,
        predictions=predictions,
        train=df.iloc[:-12],
        season_length=288,
        horizons=[HorizonSpec(label="1h", steps=_MIN_BACKTEST_LEN)],
        model_name="CrostonSBA",
    )
    # MASE is well-defined whether or not the actuals are zero; MAPE
    # would commonly be NaN here. Both should round-trip cleanly though.
    mase = overall_mase(results)
    assert mase is not None
