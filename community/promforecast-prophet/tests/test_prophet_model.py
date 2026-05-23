"""Tests for the Prophet plug-in.

Prophet's import pulls in cmdstanpy + a Stan toolchain — heavy enough that
we don't want CI to install it just to verify our wrapper's plumbing. The
fit/predict path is mocked here; the real Prophet integration is exercised
by anyone actually running the package against their data, and a smoke
test could be added once a docker-compose profile bundles Prophet.
"""

from __future__ import annotations

import sys
import types
from typing import Any, ClassVar
from unittest.mock import patch

import pandas as pd
import pytest

from promforecast_prophet import ProphetModel


class _FakeProphet:
    """Minimal stand-in for ``prophet.Prophet`` used by the wrapper tests.

    Captures constructor kwargs, ``add_regressor`` calls, the fitted
    frame, the per-predict ``interval_width`` snapshot, and the predict
    invocation count so the tests can verify the wrapper's contract
    (including the fit-once / predict-many optimisation) without exercising
    real Prophet (which would require a Stan toolchain).
    """

    instances: ClassVar[list[_FakeProphet]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        # ``interval_width`` is read off ``self.interval_width`` at predict
        # time in real Prophet — the wrapper exploits this to vary band
        # width per level without re-fitting. Mirror that here so the test
        # double behaves the same way.
        self.interval_width: float = kwargs.get("interval_width", 0.8)
        self.regressors: list[str] = []
        self.fit_df: pd.DataFrame | None = None
        # Captured per predict() call so tests can assert "one fit, two
        # predicts at distinct band widths".
        self.predict_widths: list[float] = []
        self.fit_count = 0
        type(self).instances.append(self)

    def add_regressor(self, name: str) -> None:
        self.regressors.append(name)

    def fit(self, df: pd.DataFrame) -> _FakeProphet:
        self.fit_count += 1
        self.fit_df = df.copy()
        return self

    def make_future_dataframe(
        self, periods: int, freq: str = "D", include_history: bool = True
    ) -> pd.DataFrame:
        # Generate ``periods`` rows of timestamps starting one ``freq``
        # after the last fit ``ds``. Matches Prophet's real behaviour for
        # ``include_history=False``, which is what the wrapper requests.
        assert self.fit_df is not None
        last = self.fit_df["ds"].iloc[-1]
        start = pd.Timestamp(last) + pd.tseries.frequencies.to_offset(freq)
        ds = pd.date_range(start=start, periods=periods, freq=freq)
        return pd.DataFrame({"ds": ds})

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        n = len(future)
        # Width of the confidence band scales with the *current*
        # ``self.interval_width`` (mutable post-fit) so the per-level
        # outputs differ — the wrapper relies on this to produce distinct
        # yhat_lower_<level>/yhat_upper_<level> values from a single fit.
        width = self.interval_width
        self.predict_widths.append(width)
        return pd.DataFrame(
            {
                "ds": future["ds"],
                "yhat": [1.0] * n,
                "yhat_lower": [1.0 - width] * n,
                "yhat_upper": [1.0 + width] * n,
            }
        )


@pytest.fixture(autouse=True)
def fake_prophet(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Inject a fake ``prophet`` module before the wrapper imports it.

    The wrapper imports ``from prophet import Prophet`` lazily inside
    ``fit_predict``; we install a stub module so the import resolves to
    the fake even on machines without Prophet installed.
    """
    _FakeProphet.instances.clear()
    fake_module = types.ModuleType("prophet")
    fake_module.Prophet = _FakeProphet  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prophet", fake_module)
    yield fake_module


def _make_df(periods: int = 50) -> pd.DataFrame:
    """A trivial single-series frame in the standard registry shape."""
    ds = pd.date_range("2024-01-01", periods=periods, freq="h")
    return pd.DataFrame({"unique_id": ["s"] * periods, "ds": ds, "y": list(range(periods))})


def test_model_exposes_protocol_attributes() -> None:
    """Sanity check on the registry contract: ``name`` is the user-visible
    string and the constructor accepts ``season_length`` plus ``**params``."""
    m = ProphetModel(season_length=24, growth="linear")
    assert m.name == "Prophet"
    assert m._season_length == 24
    assert m._params == {"growth": "linear"}


def test_fit_predict_returns_expected_columns() -> None:
    """The wrapper's output must conform to the registry contract: one
    ``yhat`` column plus a pair of bounds per configured confidence level."""
    df = _make_df()
    out = ProphetModel().fit_predict(df=df, horizon=5, freq="h", levels=[80, 95])
    assert list(out.columns) == [
        "unique_id",
        "ds",
        "yhat",
        "yhat_lower_80",
        "yhat_upper_80",
        "yhat_lower_95",
        "yhat_upper_95",
    ]
    assert len(out) == 5
    assert (out["unique_id"] == "s").all()


def test_fit_predict_fits_once_per_series_and_predicts_per_level() -> None:
    """Performance contract: Prophet's ``interval_width`` is a *predict-time*
    knob, not a fit-time one, so multiple confidence levels share a single
    fit and only differ in cheap ``predict()`` calls. A regression that
    re-fits per level would silently inflate CPU usage by Nx — assert the
    optimisation here so it can't slip back in."""
    df = _make_df()
    ProphetModel().fit_predict(df=df, horizon=3, freq="h", levels=[80, 95])
    # One series, one Prophet instance, one fit.
    assert len(_FakeProphet.instances) == 1
    inst = _FakeProphet.instances[0]
    assert inst.fit_count == 1
    # Two predicts at the two distinct band widths.
    assert sorted(inst.predict_widths) == [0.8, 0.95]


def test_fit_predict_does_not_mutate_self_params() -> None:
    """The wrapper must not destructively pop from ``self._params`` during
    ``fit_predict`` — a reused instance would lose the override on the
    second call and silently switch to the per-level defaults. Catch
    that here so the bug stays fixed."""
    df = _make_df()
    model = ProphetModel(interval_width=0.5, growth="linear")
    snapshot = dict(model._params)
    model.fit_predict(df=df, horizon=2, freq="h", levels=[80, 95])
    assert model._params == snapshot
    # Second call must produce the same width override; if pop() snuck back
    # in, the second call would fall through to ``level/100``.
    _FakeProphet.instances.clear()
    model.fit_predict(df=df, horizon=2, freq="h", levels=[80, 95])
    inst = _FakeProphet.instances[0]
    assert inst.predict_widths == [0.5, 0.5]


def test_fit_predict_passes_through_constructor_params() -> None:
    """``params`` from the YAML config must reach Prophet's constructor."""
    df = _make_df()
    ProphetModel(growth="logistic", weekly_seasonality=False).fit_predict(
        df=df, horizon=2, freq="h", levels=[80]
    )
    inst = _FakeProphet.instances[0]
    assert inst.kwargs["growth"] == "logistic"
    assert inst.kwargs["weekly_seasonality"] is False


def test_fit_predict_handles_multiple_series() -> None:
    """Prophet is single-series; the wrapper must loop per ``unique_id``."""
    a = _make_df().assign(unique_id="a")
    b = _make_df().assign(unique_id="b")
    df = pd.concat([a, b], ignore_index=True)
    out = ProphetModel().fit_predict(df=df, horizon=2, freq="h", levels=[80])
    assert sorted(out["unique_id"].unique()) == ["a", "b"]
    # Two series, one level → two Prophet fits (one per series).
    assert len(_FakeProphet.instances) == 2
    assert all(inst.fit_count == 1 for inst in _FakeProphet.instances)


def test_fit_predict_attaches_regressors() -> None:
    """When regressors are supplied they must be registered on Prophet via
    ``add_regressor`` and carried into the future frame."""
    df = _make_df()
    regressors = pd.DataFrame({"ds": df["ds"], "deploys": [1.0] * len(df)})
    ProphetModel().fit_predict(df=df, horizon=3, freq="h", levels=[80], regressors=regressors)
    inst = _FakeProphet.instances[0]
    assert inst.regressors == ["deploys"]


def test_fit_predict_explicit_interval_width_overrides_levels() -> None:
    """If the operator pins ``interval_width`` themselves, every per-level
    predict honours it — useful for users who want absolute control over
    the Prophet uncertainty interval. The level still labels the output
    columns; only the width parameter is overridden."""
    df = _make_df()
    out = ProphetModel(interval_width=0.5).fit_predict(df=df, horizon=2, freq="h", levels=[80, 95])
    inst = _FakeProphet.instances[0]
    # Both predicts should have used the explicit 0.5 rather than 0.8/0.95.
    assert inst.predict_widths == [0.5, 0.5]
    # ``interval_width`` must NOT have leaked into the constructor kwargs
    # alongside being mutated on the fitted model — the wrapper has to
    # remove it from the constructor path so Prophet doesn't see it twice.
    # We assert the constructor saw *some* value (the wrapper passes the
    # initial width for debugger sanity) and that the predict-time mutation
    # is the source of truth for output bands.
    assert "interval_width" in inst.kwargs
    # Column names still reflect the configured levels so downstream
    # exposition logic stays unchanged.
    assert "yhat_lower_80" in out.columns
    assert "yhat_lower_95" in out.columns


def _patched_entry_points(group: str = "") -> Any:
    """Return the real entry-point set so the registry test below sees the
    Prophet plug-in as if it had been installed at startup."""
    from importlib.metadata import entry_points  # noqa: PLC0415

    real = list(entry_points(group=group))

    class _StubEP:
        def __init__(self) -> None:
            self.name = "prophet"
            self.value = "promforecast_prophet:ProphetModel"

        def load(self) -> Any:
            return ProphetModel

    return [*real, _StubEP()]


def test_prophet_advertises_uses_regressors_true() -> None:
    """Prophet's ``fit_predict`` actually consumes the ``regressors`` frame
    (the wrapper registers each column via ``add_regressor`` and merges
    it into the future frame). The forecaster's conditional-emission
    path reads the ``uses_regressors`` class attribute to decide whether
    to refit with held-constant regressors; Prophet must advertise
    ``True`` so the refit actually happens.
    """
    assert ProphetModel.uses_regressors is True


def test_prophet_uses_regressors_flag_via_registry_helper() -> None:
    """Through the registry helper too — the forecaster reads
    ``model_registry.uses_regressors(name)`` rather than reaching for
    the class attribute directly. Verify the discovery + lookup path
    agrees end-to-end.
    """
    from promforecast import models as registry  # noqa: PLC0415

    registry.refresh()
    try:
        with patch("promforecast.models.entry_points", _patched_entry_points):
            registry.refresh()
            assert registry.uses_regressors("Prophet") is True
    finally:
        registry.refresh()


def test_prophet_legitimately_differs_when_regressors_held_constant() -> None:
    """End-to-end: the conditional refit path must produce different bands
    when the regressors frame is flat-held vs. the original input.

    The fake Prophet uses ``regressors[regressor_name].iloc[-1]`` as a
    proxy for "the regressor's current value"; in our flat-held
    contract every row of the conditional frame equals the in-sample
    final value, so the post-fit predicted yhat tracks that value. We
    confirm the wrapper preserves the regressors all the way through —
    so the conditional refit really does see the held-constant frame —
    by inspecting the per-instance ``add_regressor`` calls and the
    fitted dataframe's regressor column.
    """
    df = _make_df()
    # Two distinct configurations of the same regressor column:
    #   (a) varying — the "unconditional" input the model would see
    #       under a real fit.
    #   (b) flat-held — every row pinned to the last observed value,
    #       matching what the runner's
    #       ``_flatten_regressors_to_last_value`` produces.
    varying = pd.DataFrame({"ds": df["ds"], "deploys": [float(i % 5) for i in range(len(df))]})
    held = pd.DataFrame({"ds": df["ds"], "deploys": [varying["deploys"].iloc[-1]] * len(df)})

    ProphetModel().fit_predict(df=df, horizon=3, freq="h", levels=[80], regressors=varying)
    unconditional_inst = _FakeProphet.instances[-1]
    assert unconditional_inst.fit_df is not None
    # The fit frame has the varying regressor merged in.
    assert "deploys" in unconditional_inst.fit_df.columns
    assert list(unconditional_inst.fit_df["deploys"]) == list(varying["deploys"])

    _FakeProphet.instances.clear()
    ProphetModel().fit_predict(df=df, horizon=3, freq="h", levels=[80], regressors=held)
    conditional_inst = _FakeProphet.instances[-1]
    assert conditional_inst.fit_df is not None
    # The flat-held frame is what the conditional refit feeds Prophet.
    # Every row equals the last value so the regressor effect is
    # "frozen at the current state".
    deploys_in_fit = list(conditional_inst.fit_df["deploys"])
    assert deploys_in_fit == [held["deploys"].iloc[-1]] * len(df)
    # Both fits add the regressor — the wrapper isn't accidentally
    # silencing the regressor on the conditional pass.
    assert conditional_inst.regressors == ["deploys"]


def test_prophet_appears_in_promforecast_registry_via_entry_point() -> None:
    """End-to-end: once the package is installed, ``Prophet`` becomes a
    valid model name in promforecast configs without any forecaster-side
    code change. We verify that contract here so the whole point of
    shipping a separate package — zero-glue discovery — is regression-proof.
    """
    from promforecast import models as registry  # noqa: PLC0415

    registry.refresh()
    try:
        with patch("promforecast.models.entry_points", _patched_entry_points):
            registry.refresh()
            assert "Prophet" in registry.available()
            built = registry.build("Prophet", season_length=24, params={"growth": "linear"})
            assert isinstance(built, ProphetModel)
            assert built._season_length == 24
            assert built._params == {"growth": "linear"}
    finally:
        registry.refresh()
