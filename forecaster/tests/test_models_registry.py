"""Tests for the model registry.

The actual ``fit_predict`` behaviour of statsforecast wrappers is exercised
end-to-end via docker-compose; here we just verify each name is registered,
constructs without error, and exposes the right ``name`` attribute.
"""

from __future__ import annotations

import pytest

from promforecast import models as model_registry
from promforecast.models import (
    AutoARIMAModel,
    AutoETSModel,
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
