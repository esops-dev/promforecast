"""Unit tests for :class:`promforecast.band_coverage.BandCoverageTracker`."""

from __future__ import annotations

from datetime import timedelta

from promforecast.band_coverage import BandCoverageTracker


def _observe(
    tracker: BandCoverageTracker,
    *,
    inside: bool,
    window: timedelta = timedelta(hours=4),
    step: timedelta = timedelta(hours=1),
) -> None:
    tracker.observe(
        group="g",
        query="q",
        series_key="s",
        model="m",
        level=95,
        window_label="4h",
        window=window,
        step=step,
        inside=inside,
    )


def test_tracker_reports_fraction_from_actual_bucket_size() -> None:
    """Fraction must be computed against the bucket fill, not its maxlen.

    Regression guard: an early draft sized the buckets by ``window /
    data_step`` but recorded one observation per fit, so the fraction
    was undercounted by the refresh ratio. The fix is to use the
    refresh interval as ``step`` so the bucket fills at the rate
    observations arrive.
    """
    tracker = BandCoverageTracker()
    _observe(tracker, inside=True)
    _observe(tracker, inside=False)
    _observe(tracker, inside=True)

    samples = tracker.samples()
    assert len(samples) == 1
    sample = samples[0]
    assert sample.samples == 3
    assert abs(sample.fraction - (2 / 3)) < 1e-9


def test_tracker_rolls_off_oldest_observation_when_maxlen_reached() -> None:
    """A 4h/1h bucket holds 4 observations; the fifth drops the first."""
    tracker = BandCoverageTracker()
    # First four: 1 inside, 3 outside → fraction 0.25.
    _observe(tracker, inside=True)
    for _ in range(3):
        _observe(tracker, inside=False)
    snap = tracker.samples()[0]
    assert snap.samples == 4
    assert snap.fraction == 0.25
    # Fifth observation inside → first (also inside) rolls off; bucket
    # now [False, False, False, True] → fraction 0.25 again.
    _observe(tracker, inside=True)
    snap = tracker.samples()[0]
    assert snap.samples == 4
    assert snap.fraction == 0.25


def test_tracker_emits_labels_for_dashboard_consumption() -> None:
    """Sample labels must carry the bare minimum a dashboard joins on."""
    tracker = BandCoverageTracker()
    _observe(tracker, inside=True)
    sample = tracker.samples()[0]
    assert sample.labels["id"] == "q"
    assert sample.labels["group"] == "g"
    assert sample.labels["model"] == "m"
    assert sample.labels["level"] == "95"
    assert sample.labels["window"] == "4h"


def test_tracker_forget_drops_dead_groups_and_queries() -> None:
    """``forget`` removes buckets whose (group, query) is no longer live."""
    tracker = BandCoverageTracker()
    tracker.observe(
        group="g1",
        query="q1",
        series_key="s",
        model="m",
        level=95,
        window_label="4h",
        window=timedelta(hours=4),
        step=timedelta(hours=1),
        inside=True,
    )
    tracker.observe(
        group="g2",
        query="q2",
        series_key="s",
        model="m",
        level=95,
        window_label="4h",
        window=timedelta(hours=4),
        step=timedelta(hours=1),
        inside=True,
    )
    tracker.forget({("g1", "q1")})
    surviving = {(s.labels["group"], s.labels["id"]) for s in tracker.samples()}
    assert surviving == {("g1", "q1")}


def test_tracker_resizes_bucket_on_window_change() -> None:
    """Changing the window must rebuild the bucket so stale state can't leak.

    Without the reset, an operator who enlarges the window would see the
    old observations persist alongside new ones with a confusing
    boundary at the change point.
    """
    tracker = BandCoverageTracker()
    # Start with a 2h/1h bucket; record 2 observations to fill it.
    _observe(tracker, inside=True, window=timedelta(hours=2), step=timedelta(hours=1))
    _observe(tracker, inside=False, window=timedelta(hours=2), step=timedelta(hours=1))
    assert tracker.samples()[0].samples == 2

    # Now enlarge the window: the old bucket is dropped (maxlen mismatch)
    # so the new observation starts a fresh series.
    _observe(tracker, inside=True, window=timedelta(hours=4), step=timedelta(hours=1))
    sample = tracker.samples()[0]
    assert sample.samples == 1
    assert sample.fraction == 1.0
