"""Meta-tests for the per-family snapshot inheritance registry.

When a partial tick runs only a subset of a group's queries (per-query
refresh cadences), :func:`promforecast.runner._inherit_prior_for_skipped_queries`
walks :data:`promforecast.runner._INHERITANCE_REGISTRY` and copies the
prior snapshot's rows for any query that's not scheduled this tick. The
registry is the single source of truth — adding a new emission family
means appending one ``_InheritanceSpec`` here, and *forgetting* to add
that one line silently strips the new family from partial-tick
snapshots.

These meta-tests fail loudly when:

* a new ``list[...Point]`` field is added to :class:`GroupSnapshot`
  without a corresponding registry entry, or
* a registered entry points at a field that no longer exists on either
  :class:`GroupSnapshot` or :class:`_RunContext`.

Without this guard, the registry-driven dispatch trades the older
hand-coded copy block for a different silent-regression vector: a
missing registry line drops the family from partial-tick exposition
until the next full run. Catching it at test time (and naming the
offending field) is the lowest-cost defence.
"""

from __future__ import annotations

import dataclasses
import typing

from promforecast import exporter as exporter_mod
from promforecast import runner as runner_mod


def _is_point_list_field(field: dataclasses.Field[object]) -> bool:
    """Return True if ``field`` looks like a snapshot point list.

    We accept any ``list[...]`` annotation; the registry is keyed off the
    attribute name, not the contained type, so a future point type that
    doesn't sit in ``promforecast.exporter`` would still need to register.
    """
    type_str = str(field.type)
    # Defensive: ``from __future__ import annotations`` keeps annotations
    # as strings, so we match the textual form rather than calling
    # ``typing.get_origin`` which would require resolving the forward
    # references first.
    return type_str.startswith("list[")


def _snapshot_list_field_names() -> set[str]:
    fields = dataclasses.fields(exporter_mod.GroupSnapshot)
    return {f.name for f in fields if _is_point_list_field(f)}


def _registry_prior_attrs() -> set[str]:
    return {spec.prior_attr for spec in runner_mod._INHERITANCE_REGISTRY}


def _registry_ctx_attrs() -> set[str]:
    return {spec.ctx_attr for spec in runner_mod._INHERITANCE_REGISTRY}


# Fields on ``GroupSnapshot`` that are intentionally NOT inherited via the
# registry. Each entry needs a one-line justification so future
# maintainers don't expand the whitelist by reflex.
_EXCLUDED_FROM_INHERITANCE: dict[str, str] = {
    # Rebuilt fresh on every tick from ``Runner._band_coverage`` (a
    # rolling tracker that survives across runs); inheriting the prior
    # snapshot's frozen samples would double-count observations.
    "band_coverage_points": (
        "BandCoveragePoint rows are rebuilt from the persistent "
        "BandCoverageTracker on every tick, not inherited."
    ),
    # Config-derived: rebuilt every run from the current group config
    # (via ``_build_threshold_action_cost_points``). Inheriting the prior
    # snapshot would mean a threshold removed from YAML keeps emitting
    # its cost row forever — the opposite of "tied to the threshold's
    # lifecycle".
    "threshold_action_cost_points": (
        "ThresholdActionCostPoint rows are derived from live config "
        "on every run; inheritance would defeat the lifecycle semantics."
    ),
}


def test_inheritance_registry_covers_every_snapshot_list_field() -> None:
    """Every ``list[...]`` field on ``GroupSnapshot`` must either be
    registered for inheritance or explicitly whitelisted with a reason.

    If this fails, the named field needs one of:
    * a new ``_InheritanceSpec`` entry in ``_INHERITANCE_REGISTRY``, or
    * an entry in ``_EXCLUDED_FROM_INHERITANCE`` with a one-line
      justification (e.g. "rebuilt from a tracker every tick").
    """
    snapshot_fields = _snapshot_list_field_names()
    registered = _registry_prior_attrs()
    expected_handled = registered | set(_EXCLUDED_FROM_INHERITANCE)
    missing = snapshot_fields - expected_handled
    assert not missing, (
        "GroupSnapshot list fields without an inheritance registry entry: "
        f"{sorted(missing)}. Add a `_InheritanceSpec` in "
        "`promforecast.runner._INHERITANCE_REGISTRY` so partial-tick "
        "snapshots don't drop the family, or add an entry in "
        "`_EXCLUDED_FROM_INHERITANCE` with the reason."
    )


def test_inheritance_registry_targets_real_run_context_attrs() -> None:
    """Each registry entry's ``ctx_attr`` must exist on ``_RunContext``.

    Catches a typo or rename where the registry's target field no longer
    exists — the inheritance loop would otherwise raise ``AttributeError``
    at runtime on the first partial tick instead of at test time.
    """
    ctx_fields = {f.name for f in dataclasses.fields(runner_mod._RunContext)}
    targeted = _registry_ctx_attrs()
    missing = targeted - ctx_fields
    assert not missing, (
        f"_INHERITANCE_REGISTRY ctx_attr values not present on _RunContext: {sorted(missing)}."
    )


def test_inheritance_registry_sources_from_real_snapshot_attrs() -> None:
    """Each registry entry's ``prior_attr`` must exist on ``GroupSnapshot``."""
    snapshot_fields = {f.name for f in dataclasses.fields(exporter_mod.GroupSnapshot)}
    sourced = _registry_prior_attrs()
    missing = sourced - snapshot_fields
    assert not missing, (
        f"_INHERITANCE_REGISTRY prior_attr values not present on GroupSnapshot: {sorted(missing)}."
    )


def test_inheritance_registry_keyby_modes_are_known() -> None:
    """Defensive: ``_query_id_of`` dispatches on ``by`` — only ``label`` and
    ``metric`` are wired up today. A new mode means a corresponding branch
    in :func:`promforecast.runner._state._query_id_of` must land too.
    """
    valid = {"label", "metric"}
    actual = {spec.by for spec in runner_mod._INHERITANCE_REGISTRY}
    invalid = actual - valid
    assert not invalid, (
        f"unknown ``by`` modes in _INHERITANCE_REGISTRY: {sorted(invalid)}. "
        f"If a new dispatch mode is needed, also extend _query_id_of."
    )


def test_typing_get_origin_path_for_list_detection_stays_consistent() -> None:
    """Sanity check: dataclasses surface ``list[...]`` annotations as the
    same string form across Python 3.12 builds. If the textual form
    changes (e.g. ``typing.List`` instead of ``list``) the detection
    helper above silently misses fields.

    The check is local to the helper; we just verify it correctly
    classifies a known list field and a known non-list field.
    """
    name_to_field = {f.name: f for f in dataclasses.fields(exporter_mod.GroupSnapshot)}
    # ``points`` is a list field.
    assert _is_point_list_field(name_to_field["points"]), (
        "GroupSnapshot.points should be detected as a list field; "
        "if this fails the field-detection helper is broken."
    )
    # ``group`` is a scalar str field.
    assert not _is_point_list_field(name_to_field["group"]), (
        "GroupSnapshot.group should NOT be detected as a list field."
    )
    # And typing.get_origin still resolves list properly when given a
    # concrete annotation — useful as documentation of the expected
    # invariant if Python's annotation handling evolves.
    assert typing.get_origin(list[int]) is list
