"""Tests for the model registry.

The actual ``fit_predict`` behaviour of statsforecast wrappers is exercised
end-to-end via docker-compose; here we just verify each name is registered,
constructs without error, and exposes the right ``name`` attribute.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from promforecast import models as model_registry
from promforecast.models import (
    AutoARIMAModel,
    AutoETSModel,
    ModelRegistrationError,
    MSTLModel,
    SeasonalNaiveModel,
    UnknownModelError,
)


@pytest.mark.parametrize(
    ("name", "expected_class"),
    [
        ("AutoARIMA", AutoARIMAModel),
        ("AutoETS", AutoETSModel),
        ("MSTL", MSTLModel),
        ("SeasonalNaive", SeasonalNaiveModel),
    ],
)
def test_build_returns_correct_class(name: str, expected_class: type) -> None:
    model = model_registry.build(name, season_length=288)
    assert isinstance(model, expected_class)
    assert model.name == name


def test_build_unknown_model_raises() -> None:
    with pytest.raises(UnknownModelError):
        model_registry.build("Prophet", season_length=288)


def test_each_model_satisfies_protocol() -> None:
    for name in ("AutoARIMA", "AutoETS", "MSTL", "SeasonalNaive"):
        model = model_registry.build(name, season_length=288)
        assert callable(model.fit_predict)


def test_build_accepts_per_model_params() -> None:
    """``params`` must reach the model's constructor (stored for later
    forwarding to ``statsforecast``)."""
    model = model_registry.build("AutoARIMA", season_length=288, params={"max_p": 3})
    assert isinstance(model, AutoARIMAModel)
    # Internal storage is a private detail but easy enough to assert via the
    # captured kwargs — the runner relies on this being kept verbatim.
    assert model._params == {"max_p": 3}  # type: ignore[attr-defined]


def test_build_stores_params_for_pass_through_at_fit_time() -> None:
    """Wrappers accept ``**params`` and don't validate at build time — typos
    surface as a ``TypeError`` at the first ``fit_predict`` call, where the
    runner wraps them into the standard fit-failure path. We assert here that
    arbitrary keys are accepted at build so wrappers stay shimless."""
    model = model_registry.build("SeasonalNaive", season_length=288, params={"garbage": 1})
    assert model._params == {"garbage": 1}  # type: ignore[attr-defined]


def test_available_lists_builtins() -> None:
    """The discovery surface returns every built-in entry point. Sorted so
    docs and CLI output stay deterministic."""
    names = model_registry.available()
    assert names == ["AutoARIMA", "AutoETS", "MSTL", "SeasonalNaive"]


def test_unknown_model_error_lists_alternatives() -> None:
    """The error message must list available model names so a typo is
    self-correcting without grepping the source."""
    with pytest.raises(UnknownModelError, match=r"available:.*AutoARIMA"):
        model_registry.build("AutoArimaa", season_length=288)


class _FakeProphet:
    """Stand-in for an external plug-in registered via entry points.

    Lives next to the tests so the registry's discovery path is exercised
    without depending on the real ``promforecast-prophet`` package being
    importable in this venv.
    """

    name = "Prophet"

    def __init__(self, season_length: int = 288, **params: Any) -> None:
        self._season_length = season_length
        self._params = params

    def fit_predict(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class _FakeEntryPoint:
    """Minimal duck-typed ``EntryPoint`` for the ``entry_points()`` patch.

    importlib.metadata.EntryPoint is a NamedTuple with strict construction
    rules; rolling our own stand-in keeps the test honest about what the
    registry actually reads from each EP (``name``, ``value``, ``load``).
    """

    def __init__(self, name: str, value: str, target: object) -> None:
        self.name = name
        self.value = value
        self._target = target

    def load(self) -> object:
        return self._target


def test_third_party_model_is_discovered_via_entry_point() -> None:
    """A package that registers ``promforecast.models`` ships its model into
    the registry without any forecaster-side code change. This is exactly
    how ``promforecast-prophet`` is wired — verifying the contract here
    means the real package can rely on it.
    """
    fake_ep = _FakeEntryPoint("prophet", "tests.test_models_registry:_FakeProphet", _FakeProphet)
    real_eps = list(
        __import__("importlib.metadata", fromlist=["entry_points"]).entry_points(
            group=model_registry.ENTRY_POINT_GROUP
        )
    )

    def _patched(group: str = "") -> list[_FakeEntryPoint]:
        # The registry calls entry_points(group=...) — return the real
        # built-ins plus our injected one so discovery stays realistic.
        return [*real_eps, fake_ep]  # type: ignore[list-item]

    model_registry.refresh()
    try:
        with patch("promforecast.models.entry_points", _patched):
            model_registry.refresh()
            assert "Prophet" in model_registry.available()
            built = model_registry.build(
                "Prophet", season_length=24, params={"seasonality_mode": "multiplicative"}
            )
            assert isinstance(built, _FakeProphet)
            assert built._season_length == 24
            assert built._params == {"seasonality_mode": "multiplicative"}
    finally:
        # Restore the real registry so subsequent tests see the built-ins
        # exactly as they ship.
        model_registry.refresh()


def test_duplicate_model_name_raises() -> None:
    """Two entry points claiming the same ``name`` must fail loudly — the
    forecaster can't pick a winner without surprising someone, so it
    refuses to start. Operators see the offending pair in the error.
    """

    class _DupArima:
        name = "AutoARIMA"  # collides with the built-in

        def __init__(self, season_length: int = 288, **params: Any) -> None: ...

        def fit_predict(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise NotImplementedError

    fake_ep = _FakeEntryPoint("dup_arima", "tests.test_models_registry:_DupArima", _DupArima)
    real_eps = list(
        __import__("importlib.metadata", fromlist=["entry_points"]).entry_points(
            group=model_registry.ENTRY_POINT_GROUP
        )
    )

    def _patched(group: str = "") -> list[_FakeEntryPoint]:
        return [*real_eps, fake_ep]  # type: ignore[list-item]

    model_registry.refresh()
    try:
        with patch("promforecast.models.entry_points", _patched):
            model_registry.refresh()
            with pytest.raises(ModelRegistrationError, match="duplicate model name 'AutoARIMA'"):
                model_registry.available()
    finally:
        model_registry.refresh()
