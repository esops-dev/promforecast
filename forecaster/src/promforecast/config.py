"""Configuration schema and validation.

The YAML config is the single source of truth. The pydantic models below and
``charts/promforecast/values.schema.json`` must agree — the CI helm-schema
check enforces this.
"""

from __future__ import annotations

import re
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

import structlog
import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from ._naming import validate_metric_name

logger = structlog.get_logger(__name__)

CALENDAR_FEATURE_TYPES = ("hour_of_day", "day_of_week", "is_weekend")

# Versioned schema. ``apiVersion`` mirrors the Kubernetes convention so a
# config file declares which schema it conforms to and the loader can route
# between versions transparently. ``promforecast.io/v1`` is the stable
# schema; ``promforecast.io/v1alpha1`` is kept as a backward-compatible
# alias of v1 (every v1alpha1 config validates unchanged as v1) and loads
# with a one-shot ``DeprecationWarning`` nudging operators to retag. Any
# future breaking change will ship under a fresh ``apiVersion`` with both
# the old and new schemas accepted during a documented deprecation window
# — see ``docs/config-schema.md`` for the migration policy.
CURRENT_API_VERSION = "promforecast.io/v1"
DEPRECATED_API_VERSIONS: frozenset[str] = frozenset({"promforecast.io/v1alpha1"})
SUPPORTED_API_VERSIONS: frozenset[str] = frozenset({CURRENT_API_VERSION}) | DEPRECATED_API_VERSIONS


class UnsupportedApiVersionError(ValueError):
    """Raised when the config declares an ``apiVersion`` we don't recognise."""


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
    """TSDB read endpoint.

    ``max_concurrent_requests`` caps in-flight PromQL fetches across every
    group so a fan-out (many groups times many regressors) can't accidentally
    open hundreds of concurrent TCP connections to the TSDB. The default
    of 8 is comfortable for a single-replica forecaster talking to a
    single-node VictoriaMetrics; tune up if profiling shows query queueing,
    tune down if your TSDB is shared and rate-sensitive.

    ``max_keepalive_connections`` shapes the underlying httpx pool so the
    semaphore limit and the pool limit agree: with a smaller pool than the
    semaphore, the pool itself becomes the implicit bottleneck and the
    intent of the semaphore knob is obscured.
    """

    model_config = ConfigDict(extra="forbid")

    url: str
    timeout: Duration = timedelta(seconds=30)
    auth: AuthConfig = AuthConfig()
    max_concurrent_requests: int = Field(default=8, ge=1)
    max_keepalive_connections: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def _check_keepalive_le_concurrent(self) -> DatasourceConfig:
        # httpx allows keepalive > total but the keepalive cap is
        # bounded by the total cap at runtime, so the larger value is
        # just misleading config. Reject at load time so the operator
        # sees the mistake instead of silently watching one knob have
        # no effect.
        if self.max_keepalive_connections > self.max_concurrent_requests:
            raise ValueError(
                "datasource.max_keepalive_connections "
                f"({self.max_keepalive_connections}) must be <= "
                f"max_concurrent_requests ({self.max_concurrent_requests}); "
                "the keepalive cap can't exceed the total connection cap."
            )
        return self


class ReloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    watch_configmap: bool = True


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    listen: str = ":9091"
    refresh_interval: Duration = timedelta(hours=1)
    reload: ReloadConfig = ReloadConfig()
    # Off by default — the endpoint exposes the parsed (sanitised) config
    # over HTTP without authentication, which is appropriate inside a trusted
    # cluster but not on a public address. Operators flip this on when they
    # need to confirm what the forecaster is *actually* running.
    expose_config_endpoint: bool = False


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

    The deviation and quality toggles rely on backtest data (see
    :class:`AccuracyConfig`); turning them on without ``accuracy.evaluate``
    is allowed because the backtest will be run on demand for the
    deviation/quality summaries even if the per-horizon MAPE/MASE samples
    are not exposed.

    ``contribution`` controls the per-regressor explainability metric. When
    a query has no regressors configured, the toggle is a no-op.
    """

    model_config = ConfigDict(extra="forbid")

    deviation: bool = True
    quality: bool = True
    contribution: bool = True


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


class DiscoveryVariable(BaseModel):
    """One discovery variable for dynamic series fan-out.

    The variable's ``promql`` is run on every refresh tick and the distinct
    values of its ``label`` become the variable's value set. The primary
    query's PromQL is then rendered once per value (Cartesian product when
    multiple variables are configured) and fed into the normal fetch path.

    Keeping ``promql`` and ``label`` decoupled is deliberate: the discovery
    query is typically a cheap ``up{job="..."}`` or
    ``count by (instance)(node_uname_info)`` while the primary query is
    expensive; we want the cheap probe to drive the expensive fan-out.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    label: str
    promql: str


class QueryConfig(BaseModel):
    """A single PromQL query plus optional per-query overrides.

    ``step`` and ``season_length`` shadow the matching keys on
    :class:`DefaultsConfig` so a single forecaster can mix metrics with very
    different natural cadences (e.g. 5-minute node CPU vs hourly business
    KPIs) without forking the config or running multiple instances.

    * ``step``: query/sample cadence as a Prometheus duration. ``None``
      (default) inherits ``defaults.step``.
    * ``season_length``: number of steps in one season. ``None`` inherits
      ``defaults.season_length``; the literal string ``"auto"`` asks the
      forecaster to derive a daily seasonality from the effective step
      (e.g. step=5m → 288, step=1h → 24, step=1d → 7).
    * ``discover``: optional discovery variables. When present, the primary
      ``promql`` is treated as a template; ``$var`` or ``${var}`` markers
      are substituted with the values produced by each variable's discovery
      query at runtime. See :mod:`promforecast.discovery`.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    promql: str
    regressors: list[RegressorConfig] = Field(default_factory=list)
    step: Duration | None = None
    season_length: int | Literal["auto"] | None = None
    discover: list[DiscoveryVariable] = Field(default_factory=list)


class CanaryConfig(BaseModel):
    """Canary / weighted sampling for a group.

    When ``enabled: true`` the runner samples ``weight`` of every query's
    series before passing them through the per-query / global overflow
    caps. Useful for testing a new high-cardinality group against a
    production TSDB without paying the full fit cost up front, or for
    running an expensive model across a representative slice of a fleet
    instead of every host.

    **Precedence with ``safety.series_overflow``**: canary sampling runs
    *first* and reduces the population; ``max_series_per_query`` and
    ``max_total_series`` then apply to the sampled set. This ordering
    matches the intent of canary mode — the operator opts into a smaller
    population, and the global guards still bound it from above. The
    sampled-out series surface as
    ``forecast_series_dropped_total{group, reason="canary"}``.

    Sampling is deterministic per run: we use ``random.sample`` with the
    default RNG so a single run is reproducible during debugging, but
    successive runs rotate through the population. A fixed seed would
    bias the canary to the same N series forever; rotation gives broader
    coverage over time.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    weight: float = Field(default=0.1, gt=0.0, le=1.0)


class GroupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    priority: int = 1
    canary: CanaryConfig = Field(default_factory=CanaryConfig)
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
    # HTTP pool sizing for the push client. The sink instance is shared
    # across groups, and groups push concurrently when their refresh
    # intervals overlap, so the pool needs headroom above the per-group
    # sequential pattern. Defaults to 4 / 2 — comfortable for a handful
    # of groups; bump for deployments with many groups whose schedules
    # align. Exposed as discrete fields (rather than a single derived
    # value) so deployments can tune connections vs. keepalive
    # independently when fronted by an HTTP proxy that limits idle
    # connections more aggressively than total connections.
    max_connections: int = Field(default=4, ge=1)
    max_keepalive_connections: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _check_keepalive_le_connections(self) -> RemoteWriteSinkConfig:
        # Same intent as the datasource validator — keepalive > total
        # is a no-op at runtime; reject so the misconfig is visible.
        if self.max_keepalive_connections > self.max_connections:
            raise ValueError(
                "sink.remote_write.max_keepalive_connections "
                f"({self.max_keepalive_connections}) must be <= "
                f"max_connections ({self.max_connections})."
            )
        return self


class SinkConfig(BaseModel):
    """Forecast output sinks. The ``/metrics`` exporter is always on."""

    model_config = ConfigDict(extra="forbid")

    remote_write: RemoteWriteSinkConfig = RemoteWriteSinkConfig()


class HighAvailabilityConfig(BaseModel):
    """Leader election + shared-snapshot wiring for multi-replica deployments.

    Off by default. When ``enabled: true`` the forecaster acquires a
    coordination.k8s.io/v1 Lease before scheduling any work; followers serve
    ``/metrics`` from the Redis snapshot cache the leader writes after each
    run. Setting ``enabled: true`` without ``redis.address`` is rejected at
    load time because followers would have nothing to read from.

    The Lease parameters mirror the defaults the Kubernetes control plane
    uses (``lease_duration=30s``, ``renew_deadline=20s``, ``retry_period=4s``);
    they're exposed mostly for tests and very-fast-failover environments.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    lease_name: str = "promforecast-leader"
    namespace: str = ""
    identity: str = ""
    lease_duration_seconds: int = Field(default=30, ge=2)
    renew_deadline_seconds: int = Field(default=20, ge=1)
    retry_period_seconds: int = Field(default=4, ge=1)

    class RedisConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        address: str = ""
        password: str = ""
        snapshot_ttl: Duration = timedelta(minutes=5)

    redis: RedisConfig = Field(default_factory=RedisConfig)

    @model_validator(mode="after")
    def _enabled_requires_redis(self) -> HighAvailabilityConfig:
        # client-go ordering invariant: renew_deadline must be strictly less
        # than lease_duration. We surface it at config load rather than at
        # election start so a typo fails fast.
        if self.renew_deadline_seconds >= self.lease_duration_seconds:
            raise ValueError(
                "highAvailability.renew_deadline_seconds must be < lease_duration_seconds"
            )
        if self.enabled and not self.redis.address:
            raise ValueError(
                "highAvailability.enabled is true but highAvailability.redis.address is empty; "
                "followers need a Redis backend to serve the leader's snapshot"
            )
        return self


class OTLPTelemetryConfig(BaseModel):
    """OTLP exporter target for the optional OpenTelemetry tracer.

    Only the gRPC OTLP transport is exposed today — it's the default for
    the OTel Collector and matches what every managed APM accepts. The
    HTTP/protobuf variant is a future addition if a deployment ever needs
    it.

    ``insecure`` mirrors the OTLP exporter's flag of the same name. The
    default of ``True`` matches how most in-cluster collectors are
    deployed (cleartext on the cluster network); flip it for collectors
    fronted by TLS.
    """

    model_config = ConfigDict(extra="forbid")

    endpoint: str = ""
    insecure: bool = True
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: Duration = timedelta(seconds=10)


class TelemetryConfig(BaseModel):
    """Optional OpenTelemetry tracing wiring.

    Off by default. When ``enabled: true`` (which is implied by a non-empty
    ``otlp.endpoint``) the forecaster initialises an OTLP-gRPC exporter at
    boot and instruments the query → fit → export pipeline with spans
    tagged ``group``, ``query``, ``model``, ``series_count``, and span
    durations. When disabled the tracer is a no-op (``nullcontext()``) and
    the OpenTelemetry package is *not* imported, so a deployment that
    doesn't want tracing pays zero runtime cost.

    The ``service_name`` becomes the OTel resource attribute
    ``service.name``; the default matches the binary so an APM out of the
    box groups every replica under a single service.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    service_name: str = "promforecast"
    sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    otlp: OTLPTelemetryConfig = Field(default_factory=OTLPTelemetryConfig)

    @model_validator(mode="after")
    def _enabled_requires_endpoint(self) -> TelemetryConfig:
        # Treat ``otlp.endpoint`` as the canonical on switch — flipping
        # ``enabled: true`` without an endpoint would otherwise start the
        # SDK against ``localhost:4317`` and silently drop every span.
        if self.enabled and not self.otlp.endpoint:
            raise ValueError(
                "telemetry.enabled is true but telemetry.otlp.endpoint is empty; "
                "configure an OTLP-gRPC endpoint (e.g. otel-collector:4317)"
            )
        return self


class QueryCacheConfig(BaseModel):
    """PromQL query-result cache.

    Off by default — production deployments with one group and one refresh
    interval gain nothing. The win shows up when many groups (or many
    queries inside a group) share PromQL prefixes against the same TSDB and
    the same lookback window: the cache turns the duplicate fetches into a
    single one.

    ``backend: auto`` picks Redis when HA is enabled (so the cache survives
    replica restarts and is shared across followers), otherwise the
    in-process LRU. ``backend: memory`` forces the LRU even with HA on;
    ``backend: redis`` forces Redis (and requires ``highAvailability``
    to provide a Redis address, or this section's own ``redis_address``).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    backend: Literal["auto", "memory", "redis"] = "auto"
    ttl: Duration = timedelta(minutes=5)
    max_entries: int = Field(default=1024, ge=1)
    # When ``backend: redis`` is forced and HA is *not* enabled, the cache
    # needs its own Redis address. When HA is enabled, this is optional and
    # the HA Redis is reused.
    redis_address: str = ""
    redis_password: str = ""


class Config(BaseModel):
    # ``populate_by_name=True`` lets in-Python construction use ``api_version``
    # while YAML keeps the Kubernetes-idiomatic ``apiVersion`` spelling.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # Default keeps existing in-Python ``Config(...)`` usage (tests, dry-run
    # scaffolding) compatible — the YAML loader peeks at ``apiVersion``
    # before pydantic validation, so the field's default is only ever
    # observed when no YAML is involved. The Literal accepts both the
    # stable ``v1`` and the deprecated ``v1alpha1`` so a config that pinned
    # the alpha tag still parses; the loader emits a DeprecationWarning
    # when the alpha tag is declared explicitly.
    api_version: Literal[
        "promforecast.io/v1",
        "promforecast.io/v1alpha1",
    ] = Field(
        # Inlined as a literal so mypy can narrow the default to the
        # Literal type — using ``CURRENT_API_VERSION`` (typed ``str``)
        # would force a cast. Keep the literal in lockstep with
        # ``CURRENT_API_VERSION`` above.
        default="promforecast.io/v1",
        alias="apiVersion",
    )
    datasource: DatasourceConfig
    server: ServerConfig = ServerConfig()
    safety: SafetyConfig = SafetyConfig()
    defaults: DefaultsConfig = DefaultsConfig()
    sink: SinkConfig = SinkConfig()
    high_availability: HighAvailabilityConfig = Field(
        default_factory=HighAvailabilityConfig, alias="highAvailability"
    )
    query_cache: QueryCacheConfig = QueryCacheConfig()
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    groups: list[GroupConfig]


def load(path: str | Path) -> Config:
    """Load and validate a YAML config file.

    Validates that all query IDs are usable as Prometheus metric names —
    output names are derived as ``<id>_forecast`` and rejecting bad IDs at
    boot is much friendlier than failing on first scrape. Auth modes other
    than ``none`` are accepted by the schema (so configs stay forward-
    compatible) but rejected here so a misconfigured ``bearer`` doesn't
    silently send unauthenticated requests.

    The ``apiVersion`` field is enforced before pydantic validation so an
    unsupported version is rejected with a precise error message rather
    than the noisier ``Literal[...]`` mismatch pydantic would produce. A
    missing ``apiVersion`` is accepted (and back-filled with the current
    version) but emits a one-shot warning so operators add it before the
    schema is frozen.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config file must contain a YAML mapping at top level: {path}")
    raw = _check_api_version(raw, source=str(path))
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
            _validate_query_overrides(group_name=group.name, query=query)
            for reg in query.regressors:
                _validate_regressor(group_name=group.name, query_id=query.id, reg=reg)
    return config


def _check_api_version(raw: dict[str, Any], *, source: str) -> dict[str, Any]:
    """Validate (and inject) the top-level ``apiVersion`` declaration.

    * Missing ``apiVersion`` -> warn once, default to the current version.
      Pre-versioning configs keep working without an edit, but the warning
      nudges operators to pin the schema before the next breaking bump.
    * ``promforecast.io/v1`` -> the stable schema; loads silently.
    * ``promforecast.io/v1alpha1`` -> deprecated alias of v1 (the schemas
      are byte-identical). Loads with a one-shot ``DeprecationWarning``
      pointing at the migration recipe in ``docs/config-schema.md``.
    * Unknown ``apiVersion`` -> raise :class:`UnsupportedApiVersionError`
      with the offending value and the supported set so a typo fails
      clearly rather than bottoming out in a generic pydantic ``Literal``
      error.

    Returns the (possibly mutated) raw dict so the caller can hand it to
    pydantic without an extra copy.
    """
    declared = raw.get("apiVersion")
    if declared is None:
        warnings.warn(
            f"config {source}: missing 'apiVersion' field; "
            f"defaulting to {CURRENT_API_VERSION!r}. "
            f"Add 'apiVersion: {CURRENT_API_VERSION}' to silence this warning "
            "and pin the schema for future upgrades.",
            DeprecationWarning,
            stacklevel=3,
        )
        logger.warning(
            "config_api_version_missing",
            source=source,
            assumed=CURRENT_API_VERSION,
        )
        raw["apiVersion"] = CURRENT_API_VERSION
        return raw
    if declared in DEPRECATED_API_VERSIONS:
        warnings.warn(
            f"config {source}: apiVersion {declared!r} is deprecated and will be "
            f"removed in a future major release; the v1alpha1 schema is identical "
            f"to {CURRENT_API_VERSION!r}, so changing the apiVersion field is the "
            "only required edit. See docs/config-schema.md for the migration recipe.",
            DeprecationWarning,
            stacklevel=3,
        )
        logger.warning(
            "config_api_version_deprecated",
            source=source,
            declared=declared,
            current=CURRENT_API_VERSION,
        )
        return raw
    if declared not in SUPPORTED_API_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_API_VERSIONS))
        raise UnsupportedApiVersionError(
            f"config {source}: unsupported apiVersion {declared!r}; supported versions: {supported}"
        )
    return raw


def _validate_query_overrides(*, group_name: str, query: QueryConfig) -> None:
    """Reject obviously broken per-query ``step`` / ``season_length``.

    The pydantic schema accepts both fields as optional with broad types so
    forward-compat configs keep parsing; the semantic checks (positive step,
    positive season_length) live here so they fail at boot with a precise
    label rather than blowing up during the first fit.
    """
    label = f"groups[{group_name}].queries[{query.id}]"
    if query.step is not None and query.step.total_seconds() <= 0:
        raise ValueError(f"{label}: step must be a positive duration")
    if isinstance(query.season_length, int) and query.season_length <= 0:
        raise ValueError(f"{label}: season_length must be a positive integer")


def effective_step(query: QueryConfig, defaults: DefaultsConfig) -> timedelta:
    """Resolve the actual step for ``query`` from its override + defaults."""
    return query.step if query.step is not None else defaults.step


def effective_season_length(
    query: QueryConfig,
    defaults: DefaultsConfig,
    *,
    step: timedelta | None = None,
) -> int:
    """Resolve the actual season_length for ``query``.

    Handles three cases in order of precedence:

    1. Query-level integer override -> use as-is.
    2. Query-level ``"auto"`` -> derive a daily seasonality from the
       effective step. Sub-daily steps map to ``86400 / step_seconds``;
       step >= 1 day collapses to 7 (weekly seasonality).
    3. No query-level override -> inherit ``defaults.season_length``.

    A bottom floor of ``1`` is enforced because statsforecast's models can't
    handle ``season_length=0``.
    """
    sl = query.season_length
    if isinstance(sl, int):
        return max(1, sl)
    if sl == "auto":
        s = step if step is not None else effective_step(query, defaults)
        return _derive_season_length_from_step(s)
    return defaults.season_length


_SECRET_REDACTION = "<redacted>"


def sanitised_dump(config: Config) -> dict[str, Any]:
    """Return the parsed config as a JSON-friendly dict with secrets redacted.

    Used by the optional ``/config`` debug endpoint. The redaction list is
    deliberately small but precise:

    * ``datasource.auth.*`` (credential type + any future credential fields)
    * ``sink.remote_write.headers`` (auth tokens commonly travel in headers)

    Operators see the metric IDs, PromQL, model/regressor configuration —
    everything they need to confirm the running config matches their git
    revision — without /config leaking a bearer token if one is later added
    to ``datasource.auth`` or to a custom remote_write header.

    ``by_alias=True`` so the JSON keys match the YAML the operator wrote
    (e.g. ``apiVersion``, not the Python field name ``api_version``); the
    point of /config is to confirm the deployed config, and a key-name
    drift between input and output would defeat that.
    """
    payload = config.model_dump(mode="json", by_alias=True)
    auth = payload.get("datasource", {}).get("auth")
    if isinstance(auth, dict) and auth.get("type") not in (None, "none"):
        payload["datasource"]["auth"] = {"type": auth.get("type"), "credentials": _SECRET_REDACTION}
    sink = payload.get("sink", {}).get("remote_write")
    if isinstance(sink, dict):
        headers = sink.get("headers") or {}
        if headers:
            sink["headers"] = {key: _SECRET_REDACTION for key in headers}
    return payload


def _derive_season_length_from_step(step: timedelta) -> int:
    """Daily-seasonality heuristic for the ``season_length: auto`` shortcut.

    For sub-daily steps return ``round(86400 / step_seconds)``; for steps of
    a day or longer fall back to a weekly seasonality of 7. Both branches
    floor at 1 so the result is always a valid season_length.
    """
    step_seconds = max(1, int(step.total_seconds()))
    day_seconds = 86_400
    if step_seconds < day_seconds:
        return max(1, round(day_seconds / step_seconds))
    return 7


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
