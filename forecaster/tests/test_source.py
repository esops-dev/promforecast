"""Tests for the PromQL response parser."""

from __future__ import annotations

import pytest

from promforecast.source import QueryError, _parse_matrix


def test_parse_matrix_basic() -> None:
    payload = {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {"__name__": "up", "job": "node"},
                    "values": [[1700000000, "1"], [1700000060, "0"]],
                }
            ],
        },
    }
    frames = _parse_matrix(payload)
    assert len(frames) == 1
    assert frames[0].labels == {"__name__": "up", "job": "node"}
    assert frames[0].values == [1.0, 0.0]
    assert len(frames[0].timestamps) == 2


def test_parse_matrix_drops_nan() -> None:
    payload = {
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": {"job": "n"},
                    "values": [[1700000000, "1"], [1700000060, "NaN"], [1700000120, "2"]],
                }
            ],
        },
    }
    frames = _parse_matrix(payload)
    assert frames[0].values == [1.0, 2.0]


def test_parse_matrix_skips_empty_series() -> None:
    payload = {
        "data": {
            "resultType": "matrix",
            "result": [
                {"metric": {"job": "empty"}, "values": []},
                {"metric": {"job": "ok"}, "values": [[1700000000, "5"]]},
            ],
        },
    }
    frames = _parse_matrix(payload)
    assert len(frames) == 1
    assert frames[0].labels == {"job": "ok"}


def test_parse_matrix_rejects_wrong_result_type() -> None:
    payload = {"data": {"resultType": "vector", "result": []}}
    with pytest.raises(QueryError):
        _parse_matrix(payload)
