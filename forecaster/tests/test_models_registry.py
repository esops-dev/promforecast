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
