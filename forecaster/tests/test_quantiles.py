"""Unit tests for arbitrary-quantile derivation, conformal, config + cardinality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from promforecast import conformal, quantiles
from promforecast.config import (
    Config,
    DatasourceConfig,
    DefaultsConfig,
    GroupConfig,
    QueryConfig,
    QueryEmissionConfig,
    SafetyConfig,
    ServerConfig,
)
from promforecast.validate import _estimate_cardinality


def _banded_frame() -> pd.DataFrame:
    ds = pd.date_range("2026-01-01", periods=5, freq="1h")
    return pd.DataFrame(
        {
            "unique_id": ["s"] * 5,
            "ds": ds,
            "yhat": [10.0] * 5,
            "yhat_lower_80": [8.0] * 5,
            "yhat_upper_80": [12.0] * 5,
            "yhat_lower_90": [7.0] * 5,
            "yhat_upper_90": [13.0] * 5,
            "yhat_lower_98": [5.0] * 5,
            "yhat_upper_98": [15.0] * 5,
        }
    )


def test_level_and_side_mapping() -> None:
    assert quantiles.level_and_side_for_quantile(0.5) is None
    assert quantiles.level_and_side_for_quantile(0.9) == (80, "upper")
    assert quantiles.level_and_side_for_quantile(0.95) == (90, "upper")
    assert quantiles.level_and_side_for_quantile(0.99) == (98, "upper")
    assert quantiles.level_and_side_for_quantile(0.1) == (80, "lower")
    assert quantiles.level_and_side_for_quantile(0.25) == (50, "lower")
    assert quantiles.level_and_side_for_quantile(0.05) == (90, "lower")


def test_extreme_quantile_clamps_to_99() -> None:
    assert quantiles.level_and_side_for_quantile(0.999) == (99, "upper")
    assert quantiles.level_and_side_for_quantile(0.001) == (99, "lower")


def test_quantile_levels_dedupes_and_drops_median() -> None:
    # 0.9 -> 80, 0.1 -> 80 (dup), 0.95 -> 90, 0.5 -> median (no level)
    assert quantiles.quantile_levels([0.1, 0.5, 0.9, 0.95]) == [80, 90]


def test_format_quantile() -> None:
    assert quantiles.format_quantile(0.95) == "0.95"
    assert quantiles.format_quantile(0.5) == "0.5"
    assert quantiles.format_quantile(0.1) == "0.1"


def test_add_quantile_columns_reads_correct_band_edges() -> None:
    f = _banded_frame()
    qs = [0.1, 0.5, 0.9, 0.95, 0.99]
    resolved = quantiles.add_quantile_columns(f, qs)
    assert resolved == qs
    assert f["yhat_quantile_0.1"].iloc[0] == 8.0  # lower_80
    assert f["yhat_quantile_0.5"].iloc[0] == 10.0  # yhat
    assert f["yhat_quantile_0.9"].iloc[0] == 12.0  # upper_80
    assert f["yhat_quantile_0.95"].iloc[0] == 13.0  # upper_90
    assert f["yhat_quantile_0.99"].iloc[0] == 15.0  # upper_98


def test_add_quantile_columns_skips_missing_band() -> None:
    """A quantile whose source level isn't on the frame is skipped, not faked."""
    f = _banded_frame().drop(columns=["yhat_upper_98", "yhat_lower_98"])
    resolved = quantiles.add_quantile_columns(f, [0.9, 0.99])
    assert resolved == [0.9]  # 0.99 (needs level 98) dropped
    assert "yhat_quantile_0.99" not in f.columns


def test_add_quantile_columns_applies_signed_offset() -> None:
    f = _banded_frame()
    quantiles.add_quantile_columns(f, [0.9], offsets={0.9: 1.5})
    assert f["yhat_quantile_0.9"].iloc[0] == 12.0 + 1.5


def test_compute_quantile_offsets_matches_empirical_coverage() -> None:
    """The signed offset is the q-quantile of (actual - predicted_q)."""
    f = _banded_frame()
    actuals = np.array([10.5, 9.5, 11.0, 10.0, 12.0])
    res = conformal.compute_quantile_offsets(
        calib_actuals=actuals, calib_predictions=f, quantiles=[0.5, 0.9]
    )
    # median predicted 10.0 -> residuals [0.5,-0.5,1,0,2], 0.5-quantile = 0.5
    assert res.offsets[0.5] == pytest.approx(0.5)
    # upper_80 predicted 12.0 -> residuals [-1.5,-2.5,-1,-2,0], 0.9-quantile is negative
    assert res.offsets[0.9] < 0
    assert res.samples == 5


def test_compute_quantile_offsets_empty_on_degenerate() -> None:
    res = conformal.compute_quantile_offsets(
        calib_actuals=np.array([1.0]),
        calib_predictions=_banded_frame(),
        quantiles=[0.9],
    )
    assert res.offsets == {}


# ---- config validation ----


def test_valid_quantiles_accepted() -> None:
    q = QueryConfig(id="m", promql="up", emission=QueryEmissionConfig(quantiles=[0.1, 0.5, 0.99]))
    assert q.emission.quantiles == [0.1, 0.5, 0.99]


def _cfg(query: QueryConfig) -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(),
        groups=[GroupConfig(name="g", queries=[query])],
    )


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_quantile_out_of_range_rejected(bad: float) -> None:
    from promforecast.config import _validate_query_overrides  # noqa: PLC0415

    q = QueryConfig(id="m", promql="up", emission=QueryEmissionConfig(quantiles=[bad]))
    with pytest.raises(ValueError, match="strictly"):
        _validate_query_overrides(group_name="g", query=q)


def test_duplicate_quantile_rejected() -> None:
    from promforecast.config import _validate_query_overrides  # noqa: PLC0415

    q = QueryConfig(id="m", promql="up", emission=QueryEmissionConfig(quantiles=[0.9, 0.9]))
    with pytest.raises(ValueError, match="duplicate"):
        _validate_query_overrides(group_name="g", query=q)


def test_quantiles_rejected_on_events_profile() -> None:
    from promforecast.config import _validate_query_overrides  # noqa: PLC0415

    q = QueryConfig(
        id="m",
        promql="up",
        data_profile="events",
        emission=QueryEmissionConfig(quantiles=[0.9]),
    )
    with pytest.raises(ValueError, match=r"emission\.quantiles"):
        _validate_query_overrides(group_name="g", query=q)


# ---- cardinality ----


def test_cardinality_counts_quantile_family() -> None:
    base = QueryConfig(id="m", promql="up")
    with_q = QueryConfig(
        id="m", promql="up", emission=QueryEmissionConfig(quantiles=[0.1, 0.9, 0.99])
    )
    base_est = _estimate_cardinality(_cfg(base))
    q_est = _estimate_cardinality(_cfg(with_q))
    assert base_est.quantile_lines == 0
    # series_cap * n_models * n_quantiles (conformal off by default).
    cap = _cfg(with_q).safety.max_series_per_query
    n_models = len(_cfg(with_q).defaults.models)
    assert q_est.quantile_lines == cap * n_models * 3
    assert q_est.total > base_est.total
