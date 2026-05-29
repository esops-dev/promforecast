"""Tests for the static calendar-file regressor source (YAML + iCal)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from promforecast import calendar_file


def _write(tmp_path: Path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text(body)
    return str(path)


def test_parse_yaml_entries(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "freezes.yaml",
        """
entries:
  - start: 2026-01-01
    end: 2026-01-05
    name: new-year-freeze
  - start: 2026-07-01
    end: 2026-07-01
    name: q3-close
""",
    )
    entries = calendar_file.load_calendar(path)
    assert len(entries) == 2
    assert entries[0].name == "new-year-freeze"
    assert entries[0].contains(dt.date(2026, 1, 3))
    assert not entries[0].contains(dt.date(2026, 1, 6))
    assert entries[1].start == entries[1].end == dt.date(2026, 7, 1)


def test_parse_yaml_bare_list(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "list.yaml",
        """
- start: 2026-02-01
  end: 2026-02-02
  name: campaign
""",
    )
    entries = calendar_file.load_calendar(path)
    assert entries[0].name == "campaign"


def test_parse_yaml_rejects_end_before_start(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "bad.yaml",
        """
entries:
  - start: 2026-02-05
    end: 2026-02-01
    name: backwards
""",
    )
    with pytest.raises(calendar_file.CalendarFileError):
        calendar_file.load_calendar(path)


def test_parse_yaml_rejects_missing_name(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "noname.yaml",
        """
entries:
  - start: 2026-02-05
    end: 2026-02-06
""",
    )
    with pytest.raises(calendar_file.CalendarFileError):
        calendar_file.load_calendar(path)


def test_parse_ical(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "cal.ics",
        """BEGIN:VCALENDAR
BEGIN:VEVENT
DTSTART;VALUE=DATE:20260101
DTEND;VALUE=DATE:20260105
SUMMARY:New Year Freeze
END:VEVENT
BEGIN:VEVENT
DTSTART:20260701T090000Z
SUMMARY:Q3 Close
END:VEVENT
END:VCALENDAR
""",
    )
    entries = calendar_file.load_calendar(path)
    assert len(entries) == 2
    assert entries[0].name == "New Year Freeze"
    assert entries[0].contains(dt.date(2026, 1, 3))
    # All-day DTEND is exclusive (RFC 5545): 20260105 means "through Jan 4".
    assert entries[0].end == dt.date(2026, 1, 4)
    assert not entries[0].contains(dt.date(2026, 1, 5))
    # No DTEND -> single-day window.
    assert entries[1].start == entries[1].end == dt.date(2026, 7, 1)


def test_missing_file_raises() -> None:
    with pytest.raises(calendar_file.CalendarFileError):
        calendar_file.get_calendar("/nonexistent/path/to/calendar.yaml")


def test_get_calendar_caches_and_reparses_on_mtime_change(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("entries:\n  - start: 2026-01-01\n    end: 2026-01-02\n    name: a\n")
    first = calendar_file.get_calendar(str(path))
    assert [e.name for e in first] == ["a"]
    # Rewrite with a different mtime so the cache is invalidated.
    import os  # noqa: PLC0415

    path.write_text("entries:\n  - start: 2026-01-01\n    end: 2026-01-02\n    name: b\n")
    os.utime(path, (1_000_000_000, 1_000_000_000))
    second = calendar_file.get_calendar(str(path))
    assert [e.name for e in second] == ["b"]
