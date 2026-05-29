"""Hot-reload coverage for the v1.9 outbound/inbound integrations.

A config reload that adds, changes, or removes ``triggers:`` /
``annotations:`` / webhook regressors must reconcile the live trigger
engine, annotation sink, and webhook client without a process restart.
The fit path is irrelevant here, so ``run_group`` is stubbed to a no-op to
keep the respawned group loops inert.
"""

from __future__ import annotations

import pytest

from promforecast.annotations import GrafanaAnnotationSink
from promforecast.config import (
    AnnotationsConfig,
    Config,
    DatasourceConfig,
    DefaultsConfig,
    GroupConfig,
    QueryConfig,
    RegressorConfig,
    SafetyConfig,
    ServerConfig,
    TriggerConfig,
)
from promforecast.exporter import Exporter
from promforecast.runner import Runner
from promforecast.source import PromSource


class _FakeSource(PromSource):
    def __init__(self) -> None:
        pass

    async def aclose(self) -> None:
        return None


def _config(
    *,
    triggers: list[TriggerConfig] | None = None,
    annotations: AnnotationsConfig | None = None,
    webhook: bool = False,
) -> Config:
    regressors = (
        [RegressorConfig(id="wh", type="webhook", url="https://bridge/x")] if webhook else []
    )
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        server=ServerConfig(),
        safety=SafetyConfig(),
        defaults=DefaultsConfig(models=["AutoARIMA"]),
        groups=[
            GroupConfig(
                name="g",
                queries=[QueryConfig(id="m", promql="up", regressors=regressors)],
            )
        ],
        triggers=triggers or [],
        annotations=annotations or AnnotationsConfig(),
    )


@pytest.mark.asyncio
async def test_reload_wires_and_tears_down_integrations(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop_run_group(self: Runner, group: object, queries: object = None) -> None:
        return None

    monkeypatch.setattr(Runner, "run_group", _noop_run_group)

    runner = Runner(config=_config(), source=_FakeSource(), exporter=Exporter())
    # Nothing configured at startup.
    assert runner._trigger_engine is None
    assert runner._annotation_sink is None
    assert runner._webhook_client is None

    try:
        # Reload adds all three.
        await runner.apply_config(
            _config(
                triggers=[
                    TriggerConfig(name="t", condition="up == 1", webhook_url="https://hook/x")
                ],
                annotations=AnnotationsConfig(
                    enabled=True, grafana_url="https://grafana", events=["drift"]
                ),
                webhook=True,
            )
        )
        assert runner._trigger_engine is not None
        assert runner._annotation_sink is not None
        assert runner._webhook_client is not None

        # Reload removes all three again.
        await runner.apply_config(_config())
        assert runner._trigger_engine is None
        assert runner._annotation_sink is None
        assert runner._webhook_client is None
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_reload_preserves_annotation_dedup_when_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An unrelated reload (a group changes, annotations block identical) must
    # NOT rebuild the annotation sink, or the rising-edge dedup state resets
    # and every active annotation re-posts.
    async def _noop_run_group(self: Runner, group: object, queries: object = None) -> None:
        return None

    monkeypatch.setattr(Runner, "run_group", _noop_run_group)

    ann = AnnotationsConfig(enabled=True, grafana_url="https://grafana", events=["drift"])
    exporter = Exporter()
    # The sink is built and injected by main.py at startup; mirror that here.
    runner = Runner(
        config=_config(annotations=ann),
        source=_FakeSource(),
        exporter=exporter,
        annotation_sink=GrafanaAnnotationSink(ann, exporter),
    )
    sink_before = runner._annotation_sink
    assert sink_before is not None
    try:
        # Change only the group's query id; annotations block is identical.
        changed = _config(annotations=ann)
        changed.groups[0].queries[0] = QueryConfig(id="m2", promql="up")
        await runner.apply_config(changed)
        assert runner._annotation_sink is sink_before  # same instance, state intact
    finally:
        await runner.stop()
