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
# between versions when a stable schema lands. Today only ``v1alpha1``
# exists; configs without an ``apiVersion`` are accepted but emit a
# one-shot warning so operators add it before the schema is frozen.
CURRENT_API_VERSION = "promforecast.io/v1alpha1"
SUPPORTED_API_VERSIONS: frozenset[str] = frozenset({CURRENT_API_VERSION})


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
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    promql: str
    regressors: list[RegressorConfig] = Field(default_factory=list)
    step: Duration | None = None
    season_length: int | Literal["auto"] | None = None


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


class Config(BaseModel):
    # ``populate_by_name=True`` lets in-Python construction use ``api_version``
    # while YAML keeps the Kubernetes-idiomatic ``apiVersion`` spelling.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # Default keeps existing in-Python ``Config(...)`` usage (tests, dry-run
    # scaffolding) compatible — the YAML loader peeks at ``apiVersion``
    # before pydantic validation, so the field's default is only ever
    # observed when no YAML is involved.
    api_version: Literal["promforecast.io/v1alpha1"] = Field(
        default="promforecast.io/v1alpha1",
        alias="apiVersion",
    )
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
      This keeps pre-versioning configs working through the v0.x line so an
      upgrade isn't gated on a config edit, but nudges operators to add it
      before the schema freezes.
    * Unknown ``apiVersion`` -> raise :class:`UnsupportedApiVersionError`
      with the offending value and the supported set, so a typo
      (``promforecast.io/v1`` before v1 ships) fails clearly rather than
      bottoming out in a generic pydantic ``Literal`` error.

    Returns the (possibly mutated) raw dict so the caller can hand it to
    pydantic without an extra copy.
    """
    declared = raw.get("apiVersion")
    if declared is None:
        warnings.warn(
            f"config {source}: missing 'apiVersion' field; "
            f"defaulting to {CURRENT_API_VERSION!r}. "
            "Add 'apiVersion: promforecast.io/v1alpha1' to silence this warning "
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
