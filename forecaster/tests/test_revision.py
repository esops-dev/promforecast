"""Unit tests for the forecast-to-forecast revision module."""

from __future__ import annotations

import math

import pytest

from promforecast.revision import RevisionCache, compute_revision_magnitude


def test_compute_revision_magnitude_zero_when_unchanged() -> None:
    assert compute_revision_magnitude(previous=10.0, current=10.0) == 0.0


def test_compute_revision_magnitude_relative_shift() -> None:
    # 30% increase
    assert compute_revision_magnitude(previous=100.0, current=130.0) == pytest.approx(0.3)
    # Sign-insensitive
    assert compute_revision_magnitude(previous=100.0, current=70.0) == pytest.approx(0.3)


def test_compute_revision_magnitude_handles_zero_previous() -> None:
    # Should not blow up; uses floor 1e-9
    magnitude = compute_revision_magnitude(previous=0.0, current=0.5)
    assert math.isfinite(magnitude)
    # 0.5 / 1e-9 = 5e8 — very large, indicating "huge swing from zero"
    assert magnitude > 1e6


def test_compute_revision_magnitude_nan_on_non_finite() -> None:
    assert math.isnan(compute_revision_magnitude(previous=float("nan"), current=1.0))
    assert math.isnan(compute_revision_magnitude(previous=1.0, current=float("inf")))


def test_revision_cache_round_trip() -> None:
    cache = RevisionCache()
    cache.put(group="g", query="q", series_key="s", model="m", horizon="1h", value=12.5)
    assert cache.get(group="g", query="q", series_key="s", model="m", horizon="1h") == 12.5
    # Missing entry
    assert cache.get(group="g", query="q", series_key="s", model="m", horizon="24h") is None


def test_revision_cache_skips_nan_writes() -> None:
    cache = RevisionCache()
    cache.put(
        group="g",
        query="q",
        series_key="s",
        model="m",
        horizon="1h",
        value=float("nan"),
    )
    # NaN entries are dropped so the next fit sees "no prior data"
    # rather than poisoning the diff.
    assert cache.get(group="g", query="q", series_key="s", model="m", horizon="1h") is None


def test_revision_cache_prune_to_drops_dead_pairs() -> None:
    cache = RevisionCache()
    cache.put(group="g", query="q1", series_key="s", model="m", horizon="1h", value=1.0)
    cache.put(group="g", query="q2", series_key="s", model="m", horizon="1h", value=2.0)
    cache.prune_to({("g", "q1")})
    assert cache.get(group="g", query="q1", series_key="s", model="m", horizon="1h") == 1.0
    assert cache.get(group="g", query="q2", series_key="s", model="m", horizon="1h") is None


def test_revision_cache_evicts_oldest_when_over_capacity() -> None:
    # Guard against series-key churn (ephemeral pod IPs, container IDs):
    # without the LRU cap the cache would grow forever between reloads.
    cache = RevisionCache(max_entries=3)
    for i in range(5):
        cache.put(group="g", query="q", series_key=f"s{i}", model="m", horizon="1h", value=float(i))
    assert len(cache) == 3
    # The three most recent writes survive; the older two are evicted.
    assert cache.get(group="g", query="q", series_key="s0", model="m", horizon="1h") is None
    assert cache.get(group="g", query="q", series_key="s1", model="m", horizon="1h") is None
    assert cache.get(group="g", query="q", series_key="s4", model="m", horizon="1h") == 4.0


def test_revision_cache_touch_on_read_keeps_hot_entries() -> None:
    # A reader-touch on the oldest key promotes it to most-recent, so
    # subsequent eviction takes a now-older entry instead.
    cache = RevisionCache(max_entries=3)
    cache.put(group="g", query="q", series_key="s0", model="m", horizon="1h", value=0.0)
    cache.put(group="g", query="q", series_key="s1", model="m", horizon="1h", value=1.0)
    cache.put(group="g", query="q", series_key="s2", model="m", horizon="1h", value=2.0)
    # Touch s0 — it's now the most-recent entry.
    assert cache.get(group="g", query="q", series_key="s0", model="m", horizon="1h") == 0.0
    # Adding s3 should evict s1 (the new oldest after the touch), not s0.
    cache.put(group="g", query="q", series_key="s3", model="m", horizon="1h", value=3.0)
    assert cache.get(group="g", query="q", series_key="s0", model="m", horizon="1h") == 0.0
    assert cache.get(group="g", query="q", series_key="s1", model="m", horizon="1h") is None
    assert cache.get(group="g", query="q", series_key="s3", model="m", horizon="1h") == 3.0
