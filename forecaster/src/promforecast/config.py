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


class ConformalConfig(BaseModel):
    """Split-conformal band calibration.

    Opt-in calibration that wraps any registered model. When enabled, the
    forecaster reserves a calibration hold-out at the tail of the lookback
    window, fits the model on the prefix, and uses the absolute residuals on
    the hold-out to compute the smallest additive band correction that
    achieves the requested empirical coverage. The correction is applied
    symmetrically (``±offset`` to upper / lower) to each emitted forecast
    confidence band, and the offset per (series, model, level) is exposed
    as ``forecast_band_calibration_offset`` so operators can see how much
    the bands moved.

    The default ``holdout_fraction`` of 0.2 matches the conformal-prediction
    literature's split-sample default and gives a meaningful sample for the
    quantile estimate at the 95th percentile (it leaves ``0.2 * lookback``
    points to estimate the upper tail). Smaller fractions risk a single
    outlier dominating the quantile; larger fractions starve the in-fit
    history.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: Literal["split"] = "split"
    holdout_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)


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
    conformal: ConformalConfig = ConformalConfig()


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


class ThresholdConfig(BaseModel):
    """One named threshold for the time-to-threshold ETA metric.

    Each threshold becomes a sample of ``<id>_forecast_time_to_threshold_seconds``
    labelled with ``name`` (e.g. ``warn``/``critical``). ``comparator``
    chooses whether the metric fires when the forecast first crosses
    *above* (``gt``) or *below* (``lt``) ``value``; the ETA is the number
    of seconds from the forecast's start until that crossing, or NaN if
    no crossing happens within the horizon.

    When confidence levels are emitted, the same family is emitted per
    level using ``yhat_upper`` (for ``gt``) or ``yhat_lower`` (for ``lt``)
    so capacity ETAs are conservative at the upper band — "disk fills
    no later than 24h from now" rather than "the central forecast hits
    full in 24h".
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    comparator: Literal["gt", "lt"] = "gt"


class GrowthRateEmissionConfig(BaseModel):
    """Per-query growth-rate emission.

    When ``enabled: true`` the forecaster emits ``<id>_forecast_growth_rate``
    once per configured ``windows`` entry. The value is the slope of the
    forecast curve over that window expressed as either
    ``units_per_second`` (default — directly interpretable in Grafana
    with ``unit: short/s`` axes) or as a relative ``percent`` change
    (``window_end / window_start - 1``).

    Cardinality multiplier is ``N_series * N_windows`` per query; the
    ``promforecast validate`` cardinality estimator counts the family.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    windows: list[Duration] = Field(default_factory=lambda: [timedelta(days=1), timedelta(days=7)])
    units: Literal["units_per_second", "percent"] = "units_per_second"


class QueryEmissionConfig(BaseModel):
    """Per-query emission toggles.

    Today three families of settings live here:

    * :class:`GrowthRateEmissionConfig` (the v1.2 forecast growth-rate
      metric)
    * ``clip_non_negative`` and ``round_to_integer`` (v1.3 count-aware
      emission safeguards for non-negative / integer-valued metrics)

    All three are kept on the per-query :class:`EmissionConfig` block so
    operators opt in *per metric*: a config that mixes a count-valued
    error counter with a continuous CPU-busy-percent series isn't
    forced to choose a single rounding policy for both.

    ``clip_non_negative`` / ``round_to_integer`` default to ``None`` so
    the runner can distinguish "operator left it unset" from "operator
    explicitly said off". When a query carries ``data_profile: count``,
    the runner treats unset toggles as opt-in (both default to ``True``)
    — that's the documented v1.3 behaviour. An operator who wants the
    count profile but explicit continuous emission still gets the
    override by setting the toggle to ``False`` directly.
    """

    model_config = ConfigDict(extra="forbid")

    growth_rate: GrowthRateEmissionConfig = Field(default_factory=GrowthRateEmissionConfig)
    clip_non_negative: bool | None = None
    round_to_integer: bool | None = None


class ColdStartBorrowConfig(BaseModel):
    """How to borrow a seasonal shape from sibling series for a cold-start fit.

    Activated by the parent :class:`ColdStartConfig` when the target
    series has fewer than ``min_points`` samples in its own lookback.
    The borrow query is expected to return *peer* series — the same
    metric family for comparable workloads (e.g. the same metric on
    other instances of the same service) — that the forecaster
    normalises to its own mean and then medians per step to produce a
    seasonal prior. The prior is scaled back to the target's recent
    average before being fed into :class:`SeasonalNaiveModel` as the
    cold-start forecast.

    ``method`` is fixed at ``median_normalized`` in v1.3 — the only
    method shipped today — but the field exists so future variants
    (``mean_normalized``, ``trimmed_median``) can ship without a schema
    bump.
    """

    model_config = ConfigDict(extra="forbid")

    promql: str
    method: Literal["median_normalized"] = "median_normalized"
    min_siblings: int = Field(default=3, ge=1)


class ColdStartConfig(BaseModel):
    """Cold-start seasonality borrowing for a query.

    Off by default. When ``borrow_from`` is configured and the target
    series has fewer than the runner's ``min_points``, the forecaster
    queries the sibling set named by :class:`ColdStartBorrowConfig` and
    synthesises a seasonal prior rather than dropping the series with
    ``reason="min_points"``.

    ``max_quality_score`` caps the ``forecast_quality_score`` for cold-
    start fits so they cannot pass strict alert gates like
    ``forecast_quality_score > 0.6``. The default of ``0.5`` is the
    documented behaviour; operators who want a different cap can tune
    it per query.

    Cold-start fits emit a ``cold_start="true"`` label on every
    forecast row so dashboards and alerts can route around them.

    **Explicit non-goal**: this is not full hierarchical reconciliation
    — the borrowed shape is a prior, not a constrained joint forecast.
    True bottom-up/top-down reconciliation is a v2 model-pack concern.
    """

    model_config = ConfigDict(extra="forbid")

    borrow_from: ColdStartBorrowConfig | None = None
    # 0.5 = halfway. Anything above 0.6 would let cold-start forecasts
    # pass the standard ``forecast_quality_score > 0.6`` alert gate
    # documented in the example alerts; we cap below that explicitly so
    # the default behaves the way the alerts assume.
    max_quality_score: float = Field(default=0.5, ge=0.0, le=1.0)


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
    * ``thresholds``: named ETA thresholds (see :class:`ThresholdConfig`).
    * ``emission``: per-query emission toggles (see :class:`QueryEmissionConfig`).
    * ``data_profile``: optional hint about the metric's shape
      (``intermittent`` | ``count`` | ``continuous``). The selector
      uses MASE-based ranking for ``intermittent`` series so auto-select
      can prefer the dedicated intermittent-demand methods; ``count``
      flips the emission safeguards (``clip_non_negative``,
      ``round_to_integer``) to default-on when the operator hasn't
      explicitly chosen.
    * ``cold_start``: opt-in cold-start sibling borrowing (see
      :class:`ColdStartConfig`).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    promql: str
    regressors: list[RegressorConfig] = Field(default_factory=list)
    step: Duration | None = None
    season_length: int | Literal["auto"] | None = None
    discover: list[DiscoveryVariable] = Field(default_factory=list)
    thresholds: list[ThresholdConfig] = Field(default_factory=list)
    emission: QueryEmissionConfig = Field(default_factory=QueryEmissionConfig)
    data_profile: Literal["intermittent", "count", "continuous"] | None = None
    cold_start: ColdStartConfig = Field(default_factory=ColdStartConfig)


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


class GapImputationConfig(BaseModel):
    """How to fill gaps in the lookback series before fitting.

    The default is intentionally conservative: a forward-fill for gaps up to
    ``max_gap`` (30 minutes), and ``drop_series`` once the gap is larger so a
    multi-hour outage doesn't smuggle fabricated values into the forecast.
    Operators who want a different behaviour flip ``strategy`` per group; the
    ``forecast_input_gaps_filled_total`` counter records what the runner did
    so the choice is observable from /metrics.

    ``max_gap`` is compared against the largest gap *between adjacent
    samples* — not the cumulative number of missing points. A single
    35-minute outage trips the fallback even if the series is otherwise
    fully populated.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["forward_fill", "linear_interp", "nan", "drop_series"] = "forward_fill"
    max_gap: Duration = timedelta(minutes=30)


class OutlierConfig(BaseModel):
    """Spike-resistant lookback preprocessing.

    Off by default — silent data correction surprises operators, so the
    feature is opt-in even though it is usually beneficial on noisy
    production metrics. ``method`` picks the detector; ``sensitivity``
    rolls up the per-method tuning (Hampel threshold, IQR multiplier, STL
    residual cutoff) into three named bands so callers don't have to
    reason about per-method magic numbers.

    ``replace_with`` controls what replaces a flagged point.
    ``median_neighbours`` is the default because it preserves the local
    level without inheriting the spike; ``linear_interp`` works better for
    monotone series; ``nan`` leaves it to downstream gap handling, which
    is appropriate when the operator wants outliers visible in the
    coverage signal.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: Literal["hampel", "iqr", "stl_residual"] = "hampel"
    sensitivity: Literal["low", "medium", "high"] = "medium"
    replace_with: Literal["median_neighbours", "linear_interp", "nan"] = "median_neighbours"


class ChangePointConfig(BaseModel):
    """Detect structural breaks in the lookback and adapt the fit window.

    Off by default. When enabled, the detector runs once per series before
    each fit; on a detected break the runner either truncates the lookback
    to the post-break tail (``truncate_lookback``) or down-weights pre-break
    samples (``reweight_recent``).

    ``min_post_break_points`` guards against an aggressive detector
    truncating a series to too few samples: if truncation would leave
    fewer than this many points, the runner keeps the full lookback. This
    is the most common knob to tune — set it to roughly one full seasonal
    cycle.

    **Not** the same algorithm as ``promanomaly`` even though both
    projects expose ``bocpd``/``ruptures``. Here the detector runs
    *input-side* (adjusts the lookback fed into the forecast model). In
    ``promanomaly`` the same names run *output-side* and emit a change
    point as an alert signal.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: Literal["bocpd", "ruptures"] = "ruptures"
    action: Literal["truncate_lookback", "reweight_recent"] = "truncate_lookback"
    min_post_break_points: int = Field(default=50, ge=2)
    # Used only when ``action == "reweight_recent"``. The exponential decay
    # applied to pre-break samples — 1.0 = no down-weighting, 0.0 = drop them
    # entirely. The default 0.5 halves each pre-break point's weight.
    reweight_decay: float = Field(default=0.5, ge=0.0, le=1.0)


class PreprocessConfig(BaseModel):
    """All pre-fit cleaning steps for a group.

    Steps run in the order ``gaps -> outliers -> change_point``: gaps are
    filled first so outlier detection sees a regularly-sampled signal, and
    change-point detection runs last on the cleaned series so a single
    spike doesn't masquerade as a regime shift.

    All three sub-sections default to inert behaviour (``forward_fill`` with
    a 30-minute cap, outliers off, change-point off), so adding the block
    to a config without further edits does nothing observable. Each
    individual feature flips on independently.
    """

    model_config = ConfigDict(extra="forbid")

    gaps: GapImputationConfig = GapImputationConfig()
    outliers: OutlierConfig = OutlierConfig()
    change_point: ChangePointConfig = ChangePointConfig()


class GroupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    priority: int = 1
    canary: CanaryConfig = Field(default_factory=CanaryConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
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
    threshold_names: set[str] = set()
    for threshold in query.thresholds:
        if not threshold.name:
            raise ValueError(f"{label}.thresholds: name must be non-empty")
        if threshold.name in threshold_names:
            raise ValueError(
                f"{label}.thresholds: duplicate threshold name {threshold.name!r}; "
                "each threshold name must be unique within a query"
            )
        threshold_names.add(threshold.name)
    _validate_cold_start(label=label, query=query)
    growth = query.emission.growth_rate
    if growth.enabled:
        if not growth.windows:
            raise ValueError(
                f"{label}.emission.growth_rate: at least one window is required when enabled"
            )
        for window in growth.windows:
            if window.total_seconds() <= 0:
                raise ValueError(
                    f"{label}.emission.growth_rate.windows: must be positive durations"
                )


def _validate_cold_start(*, label: str, query: QueryConfig) -> None:
    """Semantic checks for the ``cold_start`` block of a query.

    Pulled out of :func:`_validate_query_overrides` so neither helper
    grows past ruff's branch limit. The runtime path is already
    defensive about these conditions, but raising at load time turns a
    silent skip into a precise error message.
    """
    borrow = query.cold_start.borrow_from
    if borrow is None:
        return
    # Pydantic accepts the empty string as a valid ``str``; an empty
    # borrow PromQL would 4xx the TSDB silently every refresh.
    if not borrow.promql.strip():
        raise ValueError(f"{label}.cold_start.borrow_from: promql must be a non-empty PromQL query")
    # ``max_quality_score <= 0`` would emit cold-start quality samples at
    # zero and trip ``ForecastQualityZero`` every cycle. Operators who
    # want cold-start quality disabled should leave ``borrow_from`` out.
    if query.cold_start.max_quality_score <= 0:
        raise ValueError(
            f"{label}.cold_start.max_quality_score must be > 0 when "
            "borrow_from is configured; remove cold_start.borrow_from to "
            "disable cold-start entirely."
        )
    # Median-normalising a sparse zero-heavy series produces a degenerate
    # prior; the runner already skips this combination, but failing at
    # load time turns the silent skip into actionable feedback.
    if query.data_profile == "intermittent":
        raise ValueError(
            f"{label}.cold_start.borrow_from is set together with "
            "data_profile: intermittent; cold-start cannot synthesise a "
            "meaningful prior for intermittent series. Remove one of the "
            "two — the runner would otherwise silently fall through to "
            "the standard min_points drop on every refresh."
        )


def effective_step(query: QueryConfig, defaults: DefaultsConfig) -> timedelta:
    """Resolve the actual step for ``query`` from its override + defaults."""
    return query.step if query.step is not None else defaults.step


def effective_emission_safeguards(query: QueryConfig) -> tuple[bool, bool]:
    """Resolve ``(clip_non_negative, round_to_integer)`` for a query.

    A query with ``data_profile: count`` opts both toggles on by
    default so an operator who declared the profile gets count-safe
    emission without a second knob. An explicit boolean on either
    toggle wins over the profile default — that's the override
    pathway documented in v1.3.

    Returns the resolved booleans the runner uses; the values are not
    persisted back on the query (the original config is read-only by
    convention and ``data_profile`` already encodes the intent).
    """
    profile_default = query.data_profile == "count"
    clip = query.emission.clip_non_negative
    rounder = query.emission.round_to_integer
    return (
        profile_default if clip is None else clip,
        profile_default if rounder is None else rounder,
    )


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
