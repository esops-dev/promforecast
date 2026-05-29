"""Static internal-calendar regressor source.

Reads a YAML or iCal file mounted into the pod and turns it into a set of
named date windows (release freezes, fiscal-quarter ends, marketing-campaign
launches). The :mod:`promforecast.regressors` module then materialises those
windows into a regressor column with the same binary / one-hot shape as the
``holidays`` regressor:

* ``expand: false`` (default) — a single binary column that is ``1`` on any
  day inside any window and ``0`` otherwise.
* ``expand: true`` — one column per distinct entry ``name``.

Two file formats are supported, chosen by extension:

* ``.yaml`` / ``.yml`` (preferred, git-friendly) — a mapping with an
  ``entries:`` list, each entry ``{start, end, name}``. A bare top-level
  list is also accepted.
* ``.ics`` (for tooling compatibility) — minimal iCalendar: every ``VEVENT``
  contributes one window from ``DTSTART`` / ``DTEND`` / ``SUMMARY``.

Parsed calendars are cached per path and re-read when the file's mtime
changes, so a ConfigMap update to the mounted calendar is picked up on the
next fit without a process restart — the same effect as the main config's
configmap-watch reload path.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path

import yaml


class CalendarFileError(ValueError):
    """Raised when a calendar file is missing, unreadable, or malformed."""


@dataclass(frozen=True)
class CalendarEntry:
    """A single named, inclusive ``[start, end]`` date window."""

    start: dt.date
    end: dt.date
    name: str

    def contains(self, day: dt.date) -> bool:
        return self.start <= day <= self.end


# Per-path cache: path -> (mtime, parsed entries). Re-read on mtime change so
# a ConfigMap update to the mounted file is picked up without a restart.
_CACHE: dict[str, tuple[float, list[CalendarEntry]]] = {}


def get_calendar(path: str) -> list[CalendarEntry]:
    """Return the parsed calendar for ``path``, re-reading on mtime change."""
    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        raise CalendarFileError(f"calendar file not readable: {path} ({exc})") from exc
    cached = _CACHE.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    entries = load_calendar(path)
    _CACHE[path] = (mtime, entries)
    return entries


def load_calendar(path: str) -> list[CalendarEntry]:
    """Parse a calendar file (YAML or iCal) into :class:`CalendarEntry` list.

    Dispatches on the file extension. Always reads from disk — callers that
    want mtime-based caching should use :func:`get_calendar`.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    try:
        text = p.read_text()
    except OSError as exc:
        raise CalendarFileError(f"calendar file not readable: {path} ({exc})") from exc
    if suffix == ".ics":
        return _parse_ical(text, path=path)
    return _parse_yaml(text, path=path)


def _parse_yaml(text: str, *, path: str) -> list[CalendarEntry]:
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CalendarFileError(f"calendar file {path} is not valid YAML: {exc}") from exc
    if isinstance(doc, dict):
        raw_entries = doc.get("entries")
    elif isinstance(doc, list):
        raw_entries = doc
    else:
        raise CalendarFileError(
            f"calendar file {path} must be a mapping with 'entries:' or a top-level list"
        )
    if not isinstance(raw_entries, list):
        raise CalendarFileError(f"calendar file {path}: 'entries' must be a list")
    entries: list[CalendarEntry] = []
    for index, item in enumerate(raw_entries):
        if not isinstance(item, dict):
            raise CalendarFileError(f"calendar file {path}: entry {index} must be a mapping")
        entries.append(_entry_from_mapping(item, path=path, index=index))
    return entries


def _entry_from_mapping(item: dict[object, object], *, path: str, index: int) -> CalendarEntry:
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        raise CalendarFileError(f"calendar file {path}: entry {index} requires a non-empty 'name'")
    start = _coerce_date(item.get("start"), path=path, index=index, field="start")
    end = _coerce_date(item.get("end"), path=path, index=index, field="end")
    if end < start:
        raise CalendarFileError(
            f"calendar file {path}: entry {index} ({name!r}) has end before start"
        )
    return CalendarEntry(start=start, end=end, name=name.strip())


def _coerce_date(value: object, *, path: str, index: int, field: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value.strip()[:10])
        except ValueError as exc:
            raise CalendarFileError(
                f"calendar file {path}: entry {index} field {field!r} is not an ISO date: {value!r}"
            ) from exc
    raise CalendarFileError(
        f"calendar file {path}: entry {index} requires a date in field {field!r}"
    )


def _parse_ical(text: str, *, path: str) -> list[CalendarEntry]:
    """Minimal iCalendar parser: one window per VEVENT.

    Handles the common shape mainstream calendar tools export: line-folded
    ``BEGIN:VEVENT`` / ``END:VEVENT`` blocks with ``DTSTART``, ``DTEND``,
    and ``SUMMARY`` properties. Property parameters (``;VALUE=DATE``,
    ``;TZID=...``) are tolerated and ignored — only the date portion of the
    value is used. An ``.ics`` without an explicit ``DTEND`` treats the
    event as a single-day window.

    Per RFC 5545, ``DTEND`` for an all-day (``VALUE=DATE``) event is
    *exclusive* — it names the day after the last day of the event — so a
    date-only ``DTEND`` is rolled back one day to recover the inclusive
    ``[start, end]`` window this module works in. A date-time ``DTEND``
    (with a ``T`` component) is treated as inclusive of its date.
    """
    entries: list[CalendarEntry] = []
    in_event = False
    start: dt.date | None = None
    end: dt.date | None = None
    end_is_date_only = False
    summary: str | None = None
    for raw_line in _unfold_ical(text):
        line = raw_line.strip()
        upper = line.upper()
        if upper == "BEGIN:VEVENT":
            in_event, start, end, end_is_date_only, summary = True, None, None, False, None
            continue
        if upper == "END:VEVENT":
            if start is not None:
                name = summary.strip() if summary and summary.strip() else "event"
                resolved_end = start if end is None else end
                # All-day DTEND is exclusive; pull it back to the last
                # included day (never below ``start``).
                if end is not None and end_is_date_only and resolved_end > start:
                    resolved_end = resolved_end - dt.timedelta(days=1)
                entries.append(CalendarEntry(start=start, end=resolved_end, name=name))
            in_event = False
            continue
        if not in_event or ":" not in line:
            continue
        prop, _, value = line.partition(":")
        key = prop.split(";", 1)[0].upper()
        if key == "DTSTART":
            start = _ical_date(value, path=path)
        elif key == "DTEND":
            end = _ical_date(value, path=path)
            end_is_date_only = "T" not in value.strip()
        elif key == "SUMMARY":
            summary = value
    return entries


def _unfold_ical(text: str) -> list[str]:
    """Reverse RFC 5545 line folding (continuation lines start with space/tab)."""
    lines: list[str] = []
    for line in text.splitlines():
        if line[:1] in (" ", "\t") and lines:
            lines[-1] += line[1:]
        else:
            lines.append(line)
    return lines


def _ical_date(value: str, *, path: str) -> dt.date:
    token = value.strip()
    # Date-time values look like 20260101T090000Z; date-only like 20260101.
    digits = token.split("T", 1)[0]
    try:
        return dt.datetime.strptime(digits, "%Y%m%d").date()
    except ValueError:
        # Some producers emit ISO-with-dashes; fall back to fromisoformat.
        try:
            return dt.date.fromisoformat(token[:10])
        except ValueError as exc:
            raise CalendarFileError(
                f"calendar file {path}: unparseable iCal date {value!r}"
            ) from exc
