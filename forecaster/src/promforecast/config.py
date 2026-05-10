"""Configuration schema and validation.

The YAML config is the single source of truth. The pydantic models below and
``charts/promforecast/values.schema.json`` must agree — the CI helm-schema
check enforces this.
"""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from ._naming import validate_metric_name

CALENDAR_FEATURE_TYPES = ("hour_of_day", "day_of_week", "is_weekend")

_PROM_DURATION_RE = re.compile(
    r"^\s*(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m(?!s))?(?:(\d+)s)?(?:(\d+)ms)?\s*$"
)


def _parse_duration(value: object) -> object:
    """Accept Prometheus-style durations (``5m``, ``1h``, ``14d``).

    Pydantic's native timedelta parser only accepts ISO-8601 ("PT5M") and
    seconds. The Prometheus ecosystem uses the compact form everywhere, so
    config files become unfriendly without this coercion.
    """
    if not isinstance(value, str):
        return value
    match = _PROM_DURATION_RE.match(value)
    if not match or not any(match.groups()):
        return value  # let pydantic emit its native error
    days, hours, minutes, seconds, millis = (int(g) if g else 0 for g in match.groups())
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds, milliseconds=millis)


Duration = Annotated[timedelta, BeforeValidator(_parse_duration)]


class AuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["none", "bearer", "basic", "mtls"] = "none"


class DatasourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    timeout: Duration = timedelta(seconds=30)
    auth: AuthConfig = AuthConfig()


class ReloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    watch_configmap: bool = True


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    listen: str = ":9091"
    refresh_interval: Duration = timedelta(hours=1)
    reload: ReloadConfig = ReloadConfig()


class SafetyConfig(BaseModel):
    """Cardinality and resource guards.

    ``max_series_per_query`` is enforced **client-side**, after the TSDB
    response is parsed. A query that returns many more series than the cap
    still costs bandwidth and a brief memory spike; if you expect runaway
    cardinality, use a ``topk(N, ...)`` wrapper in your PromQL. This is
    deliberate: server-side ``?limit=`` parameters are non-standard and
    differ between Prometheus / VictoriaMetrics / Mimir.
    """

    model_config = ConfigDict(extra="forbid")

    max_series_per_query: int = 500
    max_total_series: int = 5000
    series_overflow: Literal["drop_lowest_priority", "reject", "sample"] = "drop_lowest_priority"
    fit_timeout: Duration = timedelta(seconds=60)
    query_timeout: Duration = timedelta(seconds=30)


class AccuracyConfig(BaseModel):
    """Backtesting + model-selection knobs.

    Grouped together because they share one underlying cost: the per-fit
    hold-out backtest. ``evaluate``, ``auto_select`` (any non-false value),
    and the deviation/quality emitters in :class:`EmissionConfig` all rely
    on it. Set ``evaluate: false`` and disable both emission flags to skip
    backtesting entirely when CPU is tight.
    """

    model_config = ConfigDict(extra="forbid")

    evaluate: bool = True
    horizons: list[Duration] = Field(
        default_factory=lambda: [
            timedelta(hours=1),
            timedelta(hours=6),
            timedelta(hours=24),
        ]
    )
    auto_select: bool | Literal["ensemble"] = False
    fallback_model: str = "SeasonalNaive"
    fallback_mape_threshold: float | None = 50.0


class EmissionConfig(BaseModel):
    """Per-metric-family on/off toggles.

    Both rely on backtest data (see :class:`AccuracyConfig`); turning them
    on without ``accuracy.evaluate`` is allowed because the backtest will be
    run on demand for the deviation/quality summaries even if the per-horizon
    MAPE/MASE samples are not exposed.
    """

    model_config = ConfigDict(extra="forbid")

    deviation: bool = True
    quality: bool = True


class ModelSpec(BaseModel):
    """A single configured model, with optional per-model parameter overrides.

    A bare string in YAML (``- AutoARIMA``) is coerced to ``ModelSpec(name=...)``
    by :func:`_coerce_model_spec` so existing configs keep working unchanged.
    Power users can override individual hyperparameters via ``params:``;
    those are forwarded to the model's ``fit_predict`` as keyword arguments.

    Unknown ``params`` keys are accepted by the schema (each model decides
    which ones it actually uses) so the registry can evolve without forcing
    a schema bump for every new tunable.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


def _coerce_model_spec(value: object) -> object:
    """Accept ``- AutoARIMA`` shorthand alongside the full ``{name: ...}`` dict."""
    if isinstance(value, str):
        return {"name": value}
    return value


ModelSpecField = Annotated[ModelSpec, BeforeValidator(_coerce_model_spec)]


class DefaultsConfig(BaseModel):
    """Cascaded defaults applied to every group."""

    model_config = ConfigDict(extra="forbid")

    lookback: Duration = timedelta(days=14)
    step: Duration = timedelta(minutes=5)
    horizon: Duration = timedelta(hours=24)
    models: list[ModelSpecField] = Field(
        default_factory=lambda: [ModelSpec(name="AutoARIMA"), ModelSpec(name="SeasonalNaive")]
    )
    season_length: int = 288
    confidence_levels: list[int] = Field(default_factory=lambda: [80, 95])
    min_points: int = 200
    accuracy: AccuracyConfig = AccuracyConfig()
    emission: EmissionConfig = EmissionConfig()


class RegressorConfig(BaseModel):
    """Exogenous regressor wired into a query.

    Two flavours:

    * ``type: calendar`` — built-in deterministic features
      (``hour_of_day``, ``day_of_week``, ``is_weekend``). ``id`` doubles as
      the feature name and must match one of those built-ins.
    * ``type: custom`` (default) — a PromQL query whose returned series is
      aligned to the primary query's time index. ``promql`` is required.

    ``required`` flips the failure mode from graceful (skip the regressor,
    keep forecasting) to fatal-per-series (the whole series is skipped if
    the regressor can't be resolved). Failures are counted regardless via
    ``forecast_regressor_failures_total``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    promql: str | None = None
    type: Literal["custom", "calendar"] = "custom"
    required: bool = False


class QueryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    promql: str
    regressors: list[RegressorConfig] = Field(default_factory=list)


class GroupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    priority: int = 1
    queries: list[QueryConfig]


class RemoteWriteSinkConfig(BaseModel):
    """Optional push sink that writes the *full* forecast curve to a TSDB.

    Posts each forecast row as a Prometheus-format line with an explicit
    future millisecond timestamp to ``url`` after every successful run.
    Combined with VictoriaMetrics' ``-futureRetention`` flag (set to ``2d``
    by default in the ``promforecast-stack`` umbrella chart), this makes
    forecast curves first-class historical data: graphable, queryable, and
    feedable into recording rules. The ``/metrics`` exporter remains the
    pull-mode default and continues to expose snapshot gauges in parallel.

    The default endpoint shape (``/api/v1/import/prometheus``) is the
    VictoriaMetrics text-import endpoint — same data path as a Prometheus
    federation push, no protobuf/snappy dependency required. Mimir/Thanos
    users can point ``url`` at their equivalent ingest endpoints.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    url: str = ""
    timeout: Duration = timedelta(seconds=30)
    headers: dict[str, str] = Field(default_factory=dict)
    # Rows per HTTP body. The default keeps each request well under the
    # typical TSDB ingest cap; tighten it for shops behind aggressive
    # ingress proxies.
    chunk_size: int = Field(default=10_000, ge=1)


class SinkConfig(BaseModel):
    """Forecast output sinks. The ``/metrics`` exporter is always on."""

    model_config = ConfigDict(extra="forbid")

    remote_write: RemoteWriteSinkConfig = RemoteWriteSinkConfig()


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasource: DatasourceConfig
    server: ServerConfig = ServerConfig()
    safety: SafetyConfig = SafetyConfig()
    defaults: DefaultsConfig = DefaultsConfig()
    sink: SinkConfig = SinkConfig()
    groups: list[GroupConfig]


def load(path: str | Path) -> Config:
    """Load and validate a YAML config file.

    Validates that all query IDs are usable as Prometheus metric names —
    output names are derived as ``<id>_forecast`` and rejecting bad IDs at
    boot is much friendlier than failing on first scrape. Auth modes other
    than ``none`` are accepted by the schema (so configs stay forward-
    compatible) but rejected here so a misconfigured ``bearer`` doesn't
    silently send unauthenticated requests.
    """
    raw = yaml.safe_load(Path(path).read_text())
    config = Config.model_validate(raw)
    if config.datasource.auth.type != "none":
        raise NotImplementedError(
            f"datasource.auth.type={config.datasource.auth.type!r} is not implemented yet; "
            "only 'none' is supported"
        )
    if config.sink.remote_write.enabled and not config.sink.remote_write.url:
        # Catch the common misconfiguration ("enabled but forgot the url")
        # at load time rather than letting the first push silently 404.
        raise ValueError("sink.remote_write.enabled is true but sink.remote_write.url is empty")
    for group in config.groups:
        for query in group.queries:
            validate_metric_name(query.id)
            for reg in query.regressors:
                _validate_regressor(group_name=group.name, query_id=query.id, reg=reg)
    return config


def _validate_regressor(*, group_name: str, query_id: str, reg: RegressorConfig) -> None:
    """Per-regressor sanity checks.

    Calendar regressors must use one of the supported feature names and
    must not carry a ``promql:`` (it would be silently ignored). Custom
    regressors must carry a non-empty ``promql:`` (otherwise the runner
    has nothing to query). Both checks fail the load with a message that
    points at the offending group/query so operators can find it quickly.
    """
    label = f"groups[{group_name}].queries[{query_id}].regressors[{reg.id}]"
    if reg.type == "calendar":
        if reg.id not in CALENDAR_FEATURE_TYPES:
            raise ValueError(
                f"{label}: calendar regressor id must be one of "
                f"{CALENDAR_FEATURE_TYPES}; got {reg.id!r}"
            )
        if reg.promql:
            raise ValueError(f"{label}: calendar regressors must not set promql")
    elif not reg.promql:
        raise ValueError(f"{label}: custom regressors require a non-empty promql")
