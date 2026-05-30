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

# Bounded set of regressor "types" the schema accepts. Stays small on purpose:
# every new type ships its own validator and runtime path, so a Literal here
# prevents typos from silently degrading to ``custom``.
REGRESSOR_TYPES = ("custom", "calendar", "holidays", "webhook", "calendar_file")

# Versioned schema. ``apiVersion`` mirrors the Kubernetes convention so a
# config file declares which schema it conforms to and the loader can route
# between versions transparently. ``promforecast.io/v1`` is the stable
# schema; ``promforecast.io/v1alpha1`` is kept as a backward-compatible
# alias of v1 (every v1alpha1 config validates unchanged as v1) and loads
# with a one-shot ``DeprecationWarning`` nudging operators to retag. Any
# future breaking change will ship under a fresh ``apiVersion`` with both
# the old and new schemas accepted during a documented deprecation window
# — see ``docs/reference/config-schema.md`` for the migration policy.
CURRENT_API_VERSION = "promforecast.io/v1"
DEPRECATED_API_VERSIONS: frozenset[str] = frozenset({"promforecast.io/v1alpha1"})
SUPPORTED_API_VERSIONS: frozenset[str] = frozenset({CURRENT_API_VERSION}) | DEPRECATED_API_VERSIONS


class UnsupportedApiVersionError(ValueError):
    """Raised when the config declares an ``apiVersion`` we don't recognise."""


_PROM_DURATION_RE = re.compile(
    r"^\s*(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m(?!s))?(?:(\d+)s)?(?:(\d+)ms)?\s*$"
)


def parse_duration(value: object) -> object:
    """Accept Prometheus-style durations (``5m``, ``1h``, ``14d``).

    Pydantic's native timedelta parser only accepts ISO-8601 ("PT5M") and
    seconds. The Prometheus ecosystem uses the compact form everywhere, so
    config files become unfriendly without this coercion.

    Public helper: re-used by ``preview.py`` (for ``horizon`` / ``lookback``
    overrides) and any future caller that wants the same syntax.
    """
    if not isinstance(value, str):
        return value
    match = _PROM_DURATION_RE.match(value)
    if not match or not any(match.groups()):
        return value  # let pydantic emit its native error
    days, hours, minutes, seconds, millis = (int(g) if g else 0 for g in match.groups())
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds, milliseconds=millis)


Duration = Annotated[timedelta, BeforeValidator(parse_duration)]


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
    # ``POST /forecast/preview`` returns a JSON forecast for ad-hoc inputs
    # (override regressors, override horizon, override lookback) without
    # touching the YAML config. Same network-level access-control model as
    # ``/-/reload``: rely on NetworkPolicy and service exposure. Off by
    # default; flip on to support what-if dashboards.
    expose_preview_endpoint: bool = False
    # ``POST /forecast/what-if/fleet`` runs the preview machinery across
    # every configured query whose ``thresholds:`` define a capacity
    # ceiling and ranks the predicted-soonest breaches. CPU-heavy
    # (O(minutes) rather than O(seconds) like the single-query preview)
    # so it's separately opt-in and rate-limited by
    # ``safety.fleet_what_if_min_interval``. Same network-level
    # access-control model as ``/-/reload`` / ``/forecast/preview``.
    expose_what_if_endpoint: bool = False
    # ``GET /forecast/warmup`` reports per-(group, query) "what is still
    # loading?" status for a fresh install: how many points each query
    # has versus how many it needs, the blocking reason (min_points,
    # missing regressor, ...), and a rough ETA to first usable forecast.
    # Off by default — the endpoint costs one cheap probe query per
    # configured query when called, which is fine for ad-hoc debugging
    # but would be wasteful for any monitoring system polling it on
    # cadence. Same network-level access-control model as
    # ``/-/reload``. See ``docs/operations/warmup.md``.
    expose_warmup_endpoint: bool = False


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
    # Cap on simultaneous per-query refresh jobs across the whole forecaster.
    # Per-query refresh intervals let groups schedule jobs independently;
    # without a global cap, a burst of cheap queries can
    # all line up on the same tick and saturate the executor pool. The
    # default of 8 mirrors the datasource concurrency cap so a single
    # forecaster doesn't open more parallel fits than it has TSDB read
    # slots.
    max_concurrent_fits: int = Field(default=8, ge=1)
    # Minimum interval between successive fleet what-if runs. The
    # multi-service stress test refits every configured query whose
    # ``thresholds:`` define a capacity ceiling against a hypothetical
    # traffic multiplier — a single invocation is O(minutes) on a
    # medium-sized config, so unthrottled access could starve scheduled
    # fits. The default of 5 minutes matches the typical scheduled
    # ``refresh_interval`` so a what-if call can't push the snapshot off
    # cadence. Setting this lower trades scheduling headroom for
    # interactive dashboards; setting it higher is appropriate for
    # forecasters that share an executor with high-cardinality groups.
    fleet_what_if_min_interval: Duration = timedelta(minutes=5)
    # Hard upper bound on the backfill window passed to
    # ``promforecast backfill --from <duration>``. Backfill walks one
    # fit per ``server.refresh_interval`` across the requested window —
    # at the default 1h refresh a 90d window is 2160 fits per group,
    # comfortably in the hours of CPU range. The cap rejects accidental
    # year-long requests at the CLI boundary rather than letting them
    # silently kick off an overnight job; raise it explicitly only when
    # the operator has sized the cluster for the work. See
    # ``docs/operations/backfill.md`` for the trade-off.
    backfill_max_window: Duration = timedelta(days=90)


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


class SigningConfig(BaseModel):
    """HMAC payload signing for an outbound or inbound webhook.

    The de-facto standard for webhook authentication: the sender computes
    an HMAC over the raw request body and ships it in a header; the
    receiver recomputes and compares. Both the webhook *regressor* (which
    verifies the response it pulls in) and the webhook *trigger* (which
    signs the request it pushes out) accept this block.

    ``secret_ref`` is resolved through the same ``${ENV_VAR}`` indirection
    the chart uses for datasource credentials — set it to ``${NAME}`` to
    read environment variable ``NAME`` (injected from a Kubernetes Secret
    via ``secretKeyRef``), or to a literal value for local testing. An
    empty resolved secret is rejected at load time so a missing Secret
    can't silently disable signing.
    """

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["hmac-sha256"] = "hmac-sha256"
    secret_ref: str


class RegressorConfig(BaseModel):
    """Exogenous regressor wired into a query.

    Flavours:

    * ``type: calendar`` — built-in deterministic features
      (``hour_of_day``, ``day_of_week``, ``is_weekend``). ``id`` doubles as
      the feature name and must match one of those built-ins.
    * ``type: custom`` (default) — a PromQL query whose returned series is
      aligned to the primary query's time index. ``promql`` is required.
    * ``type: holidays`` — wraps the ``holidays`` PyPI package. ``country``
      is required; optional ``regions`` narrows the per-region list
      (e.g. US states); ``expand: true`` emits one regressor column per
      holiday name (so the model can learn that Black Friday differs from
      Thanksgiving) instead of a single ``is_holiday`` binary column.
    * ``type: webhook`` — a signal that lives outside Prometheus (active
      incidents from PagerDuty, deploys-in-flight from a CI/CD API). The
      forecaster POSTs to ``url`` at fit time with the query id and the
      timestamp range and expects a JSON time series back
      (``{"timestamps": [...], "values": [...]}``). Results are cached per
      URL+range for ``cache_ttl``. ``signing`` opts into HMAC verification
      of the response.
    * ``type: calendar_file`` — a static internal calendar (release
      freezes, fiscal-quarter ends) read from a YAML or iCal file mounted
      into the pod at ``path``. ``expand`` matches the ``holidays`` shape:
      ``false`` (default) emits a single binary column, ``true`` emits one
      column per calendar-entry name.

    ``required`` flips the failure mode from graceful (skip the regressor,
    keep forecasting) to fatal-per-series (the whole series is skipped if
    the regressor can't be resolved). Failures are counted regardless via
    ``forecast_regressor_failures_total``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    promql: str | None = None
    type: Literal["custom", "calendar", "holidays", "webhook", "calendar_file"] = "custom"
    required: bool = False
    # Holiday-regressor parameters. Only consulted when ``type == "holidays"``.
    # ``country`` is the ISO-3166-1 alpha-2 country code accepted by the
    # ``holidays`` package. ``regions`` narrows to per-region calendars (e.g.
    # US states, Canadian provinces); ``[]`` (default) means the country-wide
    # union. ``expand`` switches between a single binary column (default,
    # bounded cardinality) and one-hot per holiday name. ``timezone`` is the
    # IANA name used to convert UTC query timestamps to a local calendar
    # date before the holiday lookup; defaults to ``UTC`` which matches the
    # original behaviour but is wrong for localised workloads (a US holiday
    # otherwise flips at UTC-midnight rather than US-Eastern midnight).
    #
    # ``expand`` is shared with ``type: calendar_file`` (same binary vs
    # one-hot semantics).
    country: str | None = None
    regions: list[str] = Field(default_factory=list)
    expand: bool = False
    timezone: str = "UTC"
    # Webhook-regressor parameters. Only consulted when ``type == "webhook"``.
    # ``url`` is the endpoint POSTed at fit time; ``timeout`` bounds the
    # request; ``cache_ttl`` controls how long a successful (url, range)
    # response is reused to avoid hammering the external service; ``headers``
    # carry static auth (bearer tokens, shared secrets). ``signing`` opts
    # into HMAC verification of the response body.
    url: str | None = None
    timeout: Duration = timedelta(seconds=5)
    cache_ttl: Duration = timedelta(hours=1)
    headers: dict[str, str] = Field(default_factory=dict)
    signing: SigningConfig | None = None
    # Calendar-file parameters. Only consulted when ``type == "calendar_file"``.
    # ``path`` is the mounted YAML/iCal file; the runner reloads it on the
    # same configmap-watch path as the main config.
    path: str | None = None


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


class ActionCostConfig(BaseModel):
    """Optional cost-of-action annotation for a threshold.

    Surfaces as ``forecast_threshold_action_cost{id, group, name, type}``
    whose sample value is ``value``. Dashboards multiplex
    ``forecast_time_to_threshold_seconds`` by this gauge to rank
    "soon-and-expensive" first — an ETA of 7 days on a 50,000-currency
    storage tier outranks an ETA of 1 day on a 50-currency backup
    volume.

    ``type`` is a free-text discriminator label (``usd``, ``eur``,
    ``priority``, ``engineer_hours``) so non-monetary cost framings
    work alongside currency ones. ``value`` is the numeric amount;
    ``> 0`` is enforced because a zero or negative cost makes no
    sense and would silently rank below every meaningful threshold.
    """

    model_config = ConfigDict(extra="forbid")

    type: str
    value: float = Field(gt=0.0)


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

    ``alert_within`` + ``severity`` opt the threshold into the
    ``<id>_forecast_will_breach`` predictive boolean: a derived 0/1 gauge
    that fires when the time-to-threshold ETA is below ``alert_within``.
    Both fields are required together (one without the other is
    rejected at load time) — the predictive metric is a derived shape and
    cannot be partially configured.

    ``action_cost`` opts the threshold into the
    ``forecast_threshold_action_cost`` gauge — see
    :class:`ActionCostConfig`. The cost rides with the threshold's
    lifecycle so a Grafana panel can rank capacity ETAs by
    cost-to-ignore rather than just time-to-breach.

    The ``le_drain`` comparator targets the queue-burn-down case. It
    differs from ``lt`` along four axes:

    * ``v <= threshold`` predicate (non-strict) so a forecast at exactly
      the drain value (e.g. queue lands on zero) registers as drained.
    * No-crossing sentinel is ``+Inf`` (rather than NaN) so the
      ``forecast_time_to_drain_seconds == +Inf for 30m`` alert can match
      "queue is not actually draining".
    * Band side picks ``yhat_upper`` (the *latest* drain — the
      conservative answer when alerting on stuck queues).
    * Emitted under ``<id>_forecast_time_to_drain_seconds`` instead of
      ``..._time_to_threshold_seconds`` so the two families never
      collide on the same query.

    ``le_drain`` thresholds **must not** carry ``alert_within`` /
    ``severity``: the derived ``forecast_will_breach`` gauge fires when
    the ETA is small, which for a drain means "queue will drain soon"
    — the opposite of a "predictive breach". The combination is
    rejected at config-load time so the meaning of will_breach stays
    unambiguous.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    comparator: Literal["gt", "lt", "le_drain"] = "gt"
    alert_within: Duration | None = None
    severity: Literal["warning", "critical"] | None = None
    action_cost: ActionCostConfig | None = None


class SLOConfig(BaseModel):
    """Per-query SLO error-budget forecasting block.

    Off-by-default. When configured on a query whose forecast represents
    an error rate (fraction of failing requests, fraction of time in
    error), the forecaster emits three derived families per configured
    window:

    * ``<id>_forecast_error_budget_remaining_seconds{window}`` — the
      static allowed-error budget for the window
      (``(1 - objective/100) * window``).
    * ``<id>_forecast_error_budget_burn_rate{window}`` — the predicted
      consumption rate divided by the allowed rate. ``1.0`` means
      "consuming at exactly the budget-exhausting rate".
    * ``<id>_forecast_error_budget_exhausted_in_seconds{window[, level]}`` —
      projected seconds until cumulative consumption at the forecast's
      current rate exhausts the budget. Central row carries no
      ``level``; band rows use ``yhat_upper`` (the conservative side —
      a higher predicted rate exhausts sooner).

    Aligns with the SRE workbook's multi-window/multi-burn-rate
    pattern: pair short and long ``windows`` (e.g. ``[5m, 1h, 30d]``)
    and alert on the burn rate against each.

    Cardinality multiplier ``N_windows * (N_levels + 1)`` per opted-in
    query, counted by ``promforecast validate``.
    """

    model_config = ConfigDict(extra="forbid")

    # Objective is the SLO target as a percentage in (0, 100): 99.9
    # means "allow 0.1% errors". The bounds exclude both endpoints so a
    # config can't accidentally request "100% success" (zero budget,
    # divide-by-zero on burn rate) or "0% success" (every observation
    # is an error, budget meaningless).
    objective: float = Field(gt=0.0, lt=100.0)
    windows: list[Duration] = Field(default_factory=lambda: [timedelta(days=30)])


class RecommendedWindowsConfig(BaseModel):
    """Maintenance-window recommendation block.

    Off-by-default. When configured, the forecaster sweeps the next
    ``lookahead`` of the forecast horizon for the lowest-predicted-load
    windows of ``duration`` and emits a single ranked family —
    ``forecast_recommended_maintenance_window{id, group, rank,
    duration_seconds, start_timestamp_seconds}`` — whose value is the
    mean predicted load within that window. Rank 1 is the best
    (lowest-load) window; subsequent ranks are emitted up to ``count``.

    The forecast already has all the data; this turns the "stare at a
    weekly seasonality panel at 11 PM" decision into a config-knob
    emission family operators can pin to a Grafana panel or feed into
    a workflow.

    Cardinality multiplier ``count`` per opted-in query — bounded by
    construction. The ``start_timestamp_seconds`` label churns each
    refresh (different windows become available as time moves
    forward); operators on retention-sensitive TSDBs can drop the label
    via ``metric_relabel_configs`` if the churn outpaces their
    retention without losing the metric's value.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    duration: Duration = timedelta(minutes=30)
    lookahead: Duration = timedelta(hours=48)
    count: int = Field(default=3, ge=1)


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


class DriftEmissionConfig(BaseModel):
    """Per-query forecast-drift telemetry.

    When ``enabled: true`` the runner compares each new forecast vector
    against the previous successful fit for the same series and emits a
    drift score per (series, model). Two scoring methods are supported:

    * ``method: mape`` (default) — symmetric mean absolute percentage
      change between the two yhat curves. Cheap, dimension-free, and the
      most operator-friendly choice for "the prediction shifted by N%".
    * ``method: wasserstein`` — 1-D Wasserstein (earth-mover) distance
      between the two curves treated as empirical distributions, scaled
      by ``mean(|previous|)`` so it stays dimension-free. Useful when the
      operator cares about distributional shape changes rather than
      pointwise magnitude.

    ``alert_threshold`` bumps the ``forecast_drift_alerts_total`` counter
    whenever the score crosses the threshold. Counter-based rather than a
    gauge boolean so an alert on ``increase(...) > 0`` survives transient
    flips. ``None`` disables the alert counter even when scoring is on.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    method: Literal["mape", "wasserstein"] = "mape"
    alert_threshold: float | None = 0.25
    # Rolling window (in fits) used by the drift-explainer's Pearson
    # correlation between drift_score and each configured regressor.
    # Default 30 fits matches the v1.8 spec; lower values produce
    # noisier correlations, higher values are slower to react to a
    # genuine regime change in which regressor explains the drift.
    correlation_window: int = Field(default=30, ge=2)


class BandCoverageConfig(BaseModel):
    """Per-query empirical band-coverage diagnostics.

    When ``enabled: true`` the exporter tracks a rolling boolean
    (actual ∈ band) per (series, level, window) and emits
    ``forecast_band_empirical_coverage{id, group, model, level, window}``.
    The gauge reports the fraction of recent actuals that fell inside the
    nominal band, so an operator can detect "the 95% band is empirically
    covering 40%" before backtest MAPE picks the regression up.

    Memory cost is bounded: ``window / step`` booleans per (series, level,
    window). With the default windows of ``1d`` + ``7d`` and a 5m step
    that's 288 + 2016 = ~2.3 kbits per (series, level), a fraction of a
    megabyte for the canonical 5000-series cap.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    windows: list[Duration] = Field(default_factory=lambda: [timedelta(days=1), timedelta(days=7)])


class QueryEmissionConfig(BaseModel):
    """Per-query emission toggles.

    Several families of settings live here:

    * :class:`GrowthRateEmissionConfig` — forecast growth-rate metric.
    * ``clip_non_negative`` and ``round_to_integer`` — count-aware
      emission safeguards for non-negative / integer-valued metrics.
    * :class:`DriftEmissionConfig` — forecast-vs-previous-forecast
      drift telemetry.
    * :class:`BandCoverageConfig` — empirical band coverage rolling
      gauges.
    * ``conditional`` — parallel ``<id>_forecast_conditional*`` family
      that re-runs the fit with regressors held at their current value.
    * ``decompose`` — ``<id>_forecast_component`` trend/seasonal/level/
      residual split of the predicted curve.
    * ``quantiles`` — parallel ``<id>_forecast_quantile{q}`` family at
      arbitrary quantiles, derived from the same fit.

    Every family is kept on the per-query :class:`EmissionConfig` block so
    operators opt in *per metric*: a config that mixes a count-valued
    error counter with a continuous CPU-busy-percent series isn't
    forced to choose a single rounding policy for both.

    ``clip_non_negative`` / ``round_to_integer`` default to ``None`` so
    the runner can distinguish "operator left it unset" from "operator
    explicitly said off". When a query carries ``data_profile: count``,
    the runner treats unset toggles as opt-in (both default to ``True``).
    An operator who wants the count profile but explicit continuous
    emission still gets the override by setting the toggle to ``False``
    directly.

    ``conditional`` is off by default. When on, the runner re-fits every
    emitted model with a regressors frame whose values are flat-held at
    the most recent observation. Models that ignore the regressors
    parameter (the built-in ARIMA/ETS/SeasonalNaive wrappers) produce
    a conditional forecast equal to the unconditional one — the family
    still emits so downstream alerts have a stable schema; regressor-aware
    models register additional gain.
    """

    model_config = ConfigDict(extra="forbid")

    growth_rate: GrowthRateEmissionConfig = Field(default_factory=GrowthRateEmissionConfig)
    clip_non_negative: bool | None = None
    round_to_integer: bool | None = None
    drift: DriftEmissionConfig = Field(default_factory=DriftEmissionConfig)
    band_coverage: BandCoverageConfig = Field(default_factory=BandCoverageConfig)
    conditional: bool = False
    # Component decomposition emission (trend/seasonal/level/residual). Off by
    # default; opt-in per query because decomposition adds N_components rows
    # per (series, model) per horizon step. See `models.py` for which models
    # actually decompose; non-decomposable models bump a skip counter.
    decompose: bool = False
    # Arbitrary-quantile emission. Empty (default) emits nothing extra; a
    # list like ``[0.1, 0.5, 0.9, 0.99]`` emits a parallel
    # ``<id>_forecast_quantile{q="0.9"}`` family alongside the
    # ``_forecast_lower``/``_forecast_upper`` band family (which stays for
    # backward compatibility). Each quantile is derived from the same
    # single fit — no re-fit per quantile — by mapping it to the
    # statsforecast confidence level that brackets it; ``0.5`` is the
    # central forecast. Values must be strictly between 0 and 1. When
    # ``accuracy.conformal.enabled`` is on, a per-quantile coverage
    # calibration widens each quantile and the offset is exposed as
    # ``forecast_band_calibration_offset{q="..."}``. See
    # ``docs/quantile.md``.
    quantiles: list[float] = Field(default_factory=list)


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

    ``method`` is fixed at ``median_normalized`` — the only method
    shipped today — but the field exists so future variants
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
    True bottom-up/top-down reconciliation is a future model-pack concern.
    """

    model_config = ConfigDict(extra="forbid")

    borrow_from: ColdStartBorrowConfig | None = None
    # 0.5 = halfway. Anything above 0.6 would let cold-start forecasts
    # pass the standard ``forecast_quality_score > 0.6`` alert gate
    # documented in the example alerts; we cap below that explicitly so
    # the default behaves the way the alerts assume.
    max_quality_score: float = Field(default=0.5, ge=0.0, le=1.0)


class CanaryComparisonConfig(BaseModel):
    """Canary-vs-baseline forecast comparison for a query.

    Distinct from the group-level :class:`CanaryConfig` (which *samples* a
    fraction of series for cheaper fits). This block models a canary
    deploy: the query's series are split by label selector into a
    *baseline* population (the stable pods) and a *canary* population (the
    new version). The forecaster fits the baseline, projects what the
    canary *should* look like, and emits ``<id>_canary_forecast_deviation``
    — how far the canary actuals are from the baseline-predicted forecast.

    ``canary_selector`` / ``baseline_selector`` are Prometheus-style label
    matchers (``{version="canary"}``, ``{version!="canary"}``) applied to
    the labels already attached to each returned series. Only ``=`` and
    ``!=`` matchers are supported — the selectors partition an
    already-fetched result set, they are not pushed down to the TSDB.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    canary_selector: str = ""
    baseline_selector: str = ""


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
      (``intermittent`` | ``count`` | ``continuous`` | ``events``). The selector
      uses MASE-based ranking for ``intermittent`` series so auto-select
      can prefer the dedicated intermittent-demand methods; ``count``
      flips the emission safeguards (``clip_non_negative``,
      ``round_to_integer``) to default-on when the operator hasn't
      explicitly chosen. ``events`` routes the query to the
      incident-pattern detection path (see
      :mod:`promforecast.incident_pattern`) rather than the
      continuous-forecast path — the query's PromQL is expected to
      return a counter that ticks on each discrete event (e.g.
      ``pagerduty_incidents_total``) and the forecaster emits a
      ``forecast_incident_recurrence_score`` heatmap per
      (day_of_week, hour_of_day) bucket instead of a forecast curve.
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
    data_profile: Literal["intermittent", "count", "continuous", "events"] | None = None
    cold_start: ColdStartConfig = Field(default_factory=ColdStartConfig)
    # Per-query SLO error-budget forecasting. ``None`` (default) skips
    # the family entirely; set the block to opt in. See :class:`SLOConfig`.
    slo: SLOConfig | None = None
    # Per-query maintenance-window recommendation. Off by default.
    recommended_windows: RecommendedWindowsConfig = Field(default_factory=RecommendedWindowsConfig)
    # Per-query refresh override. When ``None`` (default) the query
    # inherits ``server.refresh_interval`` from its group. Setting it
    # explicitly schedules an independent timer for this query, so a
    # group can mix cheap-and-stable hourly metrics with expensive daily
    # capacity refits without forking the YAML into two groups.
    refresh_interval: Duration | None = None
    # Per-query canary-vs-baseline comparison. Off by default; see
    # :class:`CanaryComparisonConfig`.
    canary: CanaryComparisonConfig = Field(default_factory=CanaryComparisonConfig)


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


class BlacklistWindowConfig(BaseModel):
    """One operator-known anomalous window to drop from the lookback.

    Samples whose timestamp falls inside ``[start, end)`` are dropped
    *before* gap imputation, so the standard ``gaps`` strategy treats the
    blacklisted span as a gap and applies the configured fill / drop
    policy. The window is interpreted as a half-open interval so two
    adjacent windows touching at the boundary cover the boundary point
    exactly once.

    Two shapes are supported:

    * **Absolute** — ``start`` / ``end`` are ISO-8601 timestamps (e.g.
      ``2024-11-29T00:00:00Z``). Useful for "the November 2024 incident
      window" or "the Q1 migration freeze". The ``timezone`` field is
      ignored for absolute windows because the ISO-8601 string already
      carries an offset (or implies UTC when none is given).
    * **Recurring** — ``recurring`` carries a calendar pattern (``annual``
      today; ``weekly`` / ``monthly`` are reserved for future expansion).
      ``start`` / ``end`` are formatted as ``MM-DDTHH:MM`` (year omitted)
      so the window applies to every year in the lookback. ``timezone``
      controls the local calendar the window is anchored in — defaults
      to ``UTC``, but US Black-Friday-style operators typically want
      ``America/New_York`` or their site's local zone so the window
      flips at local midnight rather than UTC-midnight. Same convention
      as the v1.4 ``holidays`` regressor's ``timezone`` field.

    The optional ``reason`` is a free-text tag surfaced on the per-window
    counter (``forecast_input_blacklisted_samples_total{reason}``) so an
    operator can audit which windows are actually firing without log
    scraping. Reason cardinality is bounded by the operator's config —
    avoid templating per-incident values into it.
    """

    model_config = ConfigDict(extra="forbid")

    start: str
    end: str
    reason: str = ""
    # ``recurring: null`` (default) treats ``start``/``end`` as absolute
    # ISO-8601 timestamps. ``recurring: annual`` treats them as
    # ``MM-DDTHH:MM`` calendar patterns and applies the window across
    # every year in the lookback. Other periodicities are reserved.
    recurring: Literal["annual"] | None = None
    # IANA timezone name used to anchor a recurring window's calendar
    # boundary. Ignored when ``recurring`` is null (the absolute form
    # carries its own offset in the ISO-8601 string). Defaults to
    # ``UTC`` so a missing field never silently shifts the window.
    timezone: str = "UTC"

    @model_validator(mode="after")
    def _check_shape(self) -> BlacklistWindowConfig:
        if not self.start or not self.end:
            raise ValueError("blacklist window requires both start and end")
        if self.recurring == "annual":
            # ``MM-DDTHH:MM`` shape: 5 chars + ``T`` + 5 chars = 11.
            # Accept the trailing seconds for symmetry with ISO-8601.
            if "T" not in self.start or "T" not in self.end:
                raise ValueError(
                    "annual blacklist windows require 'MM-DDTHH:MM' shape "
                    f"(got start={self.start!r}, end={self.end!r})"
                )
        # Absolute window — the parser in preprocess.py reads via
        # ``datetime.fromisoformat``; do a cheap structural check
        # here so a typo fails at boot rather than mid-fit.
        elif "T" not in self.start or "T" not in self.end:
            raise ValueError(
                "absolute blacklist windows require ISO-8601 timestamps "
                f"(got start={self.start!r}, end={self.end!r})"
            )
        return self


class PreprocessConfig(BaseModel):
    """All pre-fit cleaning steps for a group.

    Steps run in the order ``blacklist -> gaps -> outliers -> change_point``:
    blacklisted samples are dropped first so gap imputation treats the window
    as a gap (and may fall back to ``drop_series`` if the window is longer
    than ``gaps.max_gap``); gaps are filled so outlier detection sees a
    regularly-sampled signal; outliers are scrubbed before change-point
    detection so a single spike doesn't masquerade as a regime shift.

    All four sub-sections default to inert behaviour (``blacklist_windows``
    empty, ``forward_fill`` with a 30-minute cap, outliers off, change-point
    off), so adding the block to a config without further edits does nothing
    observable. Each individual feature flips on independently.
    """

    model_config = ConfigDict(extra="forbid")

    gaps: GapImputationConfig = GapImputationConfig()
    outliers: OutlierConfig = OutlierConfig()
    change_point: ChangePointConfig = ChangePointConfig()
    blacklist_windows: list[BlacklistWindowConfig] = Field(default_factory=list)


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


class ShardingConfig(BaseModel):
    """Horizontal sharding: N *active* replicas, each fitting a disjoint
    subset of the configured groups.

    Off by default. This is a different scaling model from
    :class:`HighAvailabilityConfig` — HA is "one active leader plus hot
    followers serving the leader's cached snapshot", which tops out around
    one well-sized leader's fit budget. Sharding instead spreads the fit
    work across every replica, so horizontal scale becomes a knob (the
    chart's ``replicas`` field). The two are mutually exclusive: HA gates
    fitting on leadership, sharding wants every replica fitting.

    Two ownership modes:

    * ``consistent_hash`` — each replica computes which groups it owns from
      a stable hash of the group name modulo the replica count, compared
      against its own shard index. No cross-replica coordination is needed
      beyond agreeing on the replica count and each replica knowing its own
      index (the chart runs a StatefulSet so the pod ordinal is the index).
    * ``leader_assigned`` — a ``coordination.k8s.io/v1`` Lease elects one
      replica as the assigner; it computes a balanced group→shard mapping
      over the live membership and publishes it to the shared Redis, and
      every replica reads it to learn what it owns. Rebalances when a
      replica joins or leaves. Requires a Redis backend (the same one HA
      and the query cache use) for the membership registry and the
      published assignment.

    Output metrics for the groups a replica owns gain a ``shard_id`` label,
    and a ``forecast_shard_owns_group{shard_id, group}`` gauge exposes the
    assignment for debugging. The ``shard_id`` label is tool-private (it
    describes which replica fit the group, not the underlying signal) so it
    is not part of the cross-tool label contract; it is only emitted when
    sharding is enabled, leaving single-replica labelsets unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mode: Literal["consistent_hash", "leader_assigned"] = "consistent_hash"
    # Total number of shards. The chart sets this from ``replicas`` so the
    # config and the StatefulSet replica count can't drift. ``ge=1`` keeps a
    # zero from silently disabling all ownership.
    replica_count: int = Field(default=1, ge=1)
    # This replica's 0-based shard index for ``consistent_hash`` ownership.
    # ``-1`` (the default) means "derive at boot" — from the StatefulSet pod
    # ordinal (the trailing ``-N`` of ``PROMFORECAST_POD_NAME``) or the
    # explicit ``PROMFORECAST_SHARD_INDEX`` env var. Set a concrete value
    # only for local/test runs where neither env var is present.
    shard_index: int = Field(default=-1, ge=-1)
    # Label value identifying this shard on output metrics. Empty means
    # "derive at boot" — the pod name, else ``shard-<index>``. Operators
    # rarely set this; it exists so a non-Kubernetes deployment can pin a
    # stable id.
    shard_id: str = ""
    # ``leader_assigned`` membership: each replica writes a heartbeat key
    # with this TTL; the assigner lists the live members from those keys.
    # The TTL must comfortably exceed the heartbeat interval so a replica
    # mid-GC-pause isn't dropped from the membership.
    heartbeat_interval_seconds: int = Field(default=10, ge=1)
    member_ttl_seconds: int = Field(default=30, ge=2)
    # Lease parameters for ``leader_assigned`` (mirrors the HA defaults).
    lease_name: str = "promforecast-shard-assigner"
    lease_duration_seconds: int = Field(default=30, ge=2)
    renew_deadline_seconds: int = Field(default=20, ge=1)
    retry_period_seconds: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def _validate(self) -> ShardingConfig:
        if self.shard_index >= 0 and self.shard_index >= self.replica_count:
            raise ValueError(
                f"sharding.shard_index ({self.shard_index}) must be < "
                f"replica_count ({self.replica_count})"
            )
        if self.member_ttl_seconds <= self.heartbeat_interval_seconds:
            raise ValueError(
                "sharding.member_ttl_seconds must be > heartbeat_interval_seconds "
                f"(got {self.member_ttl_seconds} <= {self.heartbeat_interval_seconds})"
            )
        if self.renew_deadline_seconds >= self.lease_duration_seconds:
            raise ValueError("sharding.renew_deadline_seconds must be < lease_duration_seconds")
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


class DeployAnchorConfig(BaseModel):
    """Where to look for "a deploy just happened" events.

    Only ``regressor`` is accepted today: the runner inspects the
    named regressor on the named query and registers a deploy whenever
    the regressor's most recent value transitions from <= ``threshold``
    to > ``threshold``. Pair this with a counter regressor like
    ``changes(kube_deployment_status_observed_generation[1h])`` or a
    hand-emitted ``deploys_total`` from a CI pipeline.

    Change-point detection as an anchor source (using the existing
    detector against the metric itself rather than an explicit
    regressor) is tracked as a follow-up — until that lands the
    schema rejects it at parse time so a config can't pass
    ``helm install`` and then fail the forecaster on boot.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["regressor"] = "regressor"
    query: str
    regressor_id: str | None = None
    # The regressor value must be *strictly greater than* this to
    # trigger a deploy; the previous row must have been *<=*. The
    # default of 0 matches the typical "counter ticks above zero"
    # convention; non-zero thresholds are useful when the regressor
    # itself is noisy (e.g. ``rate(...)`` over a wide window).
    threshold: float = 0.0


class DeploySignalConfig(BaseModel):
    """One of the golden signals contributing to a deploy's risk score."""

    model_config = ConfigDict(extra="forbid")

    query: str
    # Weight applied to this signal's absolute deviation ratio before
    # the weighted-mean composite. The conventional shape is
    # latency >= traffic >= errors >= saturation; operators tune
    # these to the service's specific failure modes.
    weight: float = Field(default=1.0, gt=0.0)


class ResourceSizingWorkloadConfig(BaseModel):
    """One workload's right-sizing wiring.

    Off by default. Activated by configuring the parent
    :class:`ResourceSizingConfig` block. Each entry binds a Kubernetes
    workload to the forecast it should be sized against, plus
    optionally a PromQL query that reports the *current* limit/request
    so the runner can compute an expected savings percentage.

    ``forecast_query`` must match the ``id`` of a configured query;
    the runner reads its predicted peak (max ``yhat_upper`` at the
    highest configured confidence level, falling back to ``yhat``)
    and applies the headroom multiplier.

    ``current_promql`` is optional. When set, the runner fetches it
    via the existing :class:`PromSource` (a cheap instant-style
    range query) and emits the matching ``current`` and
    ``expected_savings_pct`` gauges alongside the recommendation. When
    unset only the recommendation is emitted; operators can derive
    the savings comparison via PromQL recording rules against their
    own infrastructure metrics (``kube_pod_container_resource_requests``,
    ...).
    """

    model_config = ConfigDict(extra="forbid")

    workload: str
    kind: Literal["cpu", "memory"]
    forecast_query: str
    current_promql: str | None = None


class ResourceSizingConfig(BaseModel):
    """Per-workload right-sizing recommendations from forecast peaks.

    Off by default. When configured, the runner walks each
    :class:`ResourceSizingWorkloadConfig` after every group run,
    reads the forecast snapshot for the named query, projects the
    headroom-adjusted peak, and emits
    ``forecast_recommended_resource_size{workload, kind, confidence}``
    plus optional ``current`` / ``savings_pct`` siblings.

    ``headroom_pct`` is added on top of the predicted peak — a 30%
    headroom means the recommendation is ``peak * 1.30``. Defaults to
    ``30`` because anything below 20% leaves no rope for unexpected
    load and anything above 50% defeats the purpose of right-sizing.

    ``quality_floor`` gates the emission against
    ``forecast_quality_score``. A forecast whose quality is below the
    floor produces no recommendation for that workload — better to
    omit the row than mislead operators with a recommendation built
    on an untrustworthy forecast.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    headroom_pct: float = Field(default=30.0, ge=0.0, le=200.0)
    quality_floor: float = Field(default=0.6, ge=0.0, le=1.0)
    workloads: list[ResourceSizingWorkloadConfig] = Field(default_factory=list)


class RecommendationsConfig(BaseModel):
    """Operator-facing recommendation outputs.

    Off by default. Today only :class:`ResourceSizingConfig` is
    surfaced under this block; future recommendation families land
    here so a config can enable them all under one umbrella.
    """

    model_config = ConfigDict(extra="forbid")

    resource_sizing: ResourceSizingConfig = Field(default_factory=ResourceSizingConfig)


class HealthScoreConfig(BaseModel):
    """Per-service forecast health composite.

    Off by default — when configured, the runner aggregates
    ``forecast_quality_score`` samples across all groups and emits
    ``forecast_service_health_score`` per distinct combination of the
    labels listed in ``aggregate_by``. The metric is intended for
    "how reliable is our prediction infrastructure for this service /
    environment?" reporting, distinct from "how reliable is the
    service itself?".

    ``aggregate_by`` lists the source-side labels (e.g. ``service``,
    ``environment``, ``cluster``) that become discriminator labels on
    the emitted metric; quality_points whose label set is missing any
    of these are dropped from the aggregation (so a single un-labelled
    forecast can't silently inflate or deflate every bucket).

    ``aggregator`` picks the summary statistic — currently only
    ``weighted_mean`` is supported (every quality_point is one series,
    so the weight is implicitly 1 per row regardless of ``weight_by``).
    ``weight_by`` is exposed for forward compatibility with future
    pre-aggregated quality emission shapes; today it does not change
    the math but the configured value rides through onto the emitted
    ``weight_by`` documentation label so downstream BI dashboards
    can record which interpretation was in effect.

    ``quality_floor`` (optional) drops quality scores below the floor
    before aggregating. Useful when cold-start or low-quality forecasts
    should not pull a service's health score down — set the floor at
    the same value as the alert gate (typically ``0.6``) so the
    aggregated number aligns with the alerting story.
    """

    model_config = ConfigDict(extra="forbid")

    aggregate_by: list[str] = Field(min_length=1)
    aggregator: Literal["weighted_mean"] = "weighted_mean"
    weight_by: Literal["uniform", "forecast_series_count"] = "forecast_series_count"
    quality_floor: float | None = Field(default=None, ge=0.0, le=1.0)


class GovernanceConfig(BaseModel):
    """Operator / platform-team aggregated views over forecast quality.

    Off by default. The only sub-block today is :class:`HealthScoreConfig`;
    new governance-style aggregations will land here so a config can
    enable them all under one block.
    """

    model_config = ConfigDict(extra="forbid")

    health_score: HealthScoreConfig | None = None


class DeploySignalsConfig(BaseModel):
    """One service's deploy-risk composite definition.

    Per service: an anchor (how deploys are detected), a list of
    golden-signal queries with weights, the active window (how long
    a deploy stays "fresh" for emission), and a bound on concurrent
    active deploys.
    """

    model_config = ConfigDict(extra="forbid")

    service: str
    anchor: DeployAnchorConfig
    signals: list[DeploySignalConfig]
    # How long after detection the deploy is considered "active" and
    # its risk score is emitted. After the window elapses the gauge
    # disappears; operators read this as "rollout window closed,
    # back to steady-state monitoring".
    active_window: Duration = timedelta(minutes=30)
    # Bound on simultaneously-active deploys per service. Prevents an
    # abnormal deploy pipeline (a flapping Argo Rollouts loop) from
    # blowing the per-(deployment_id, service) cardinality budget.
    max_active_deploys: int = Field(default=8, ge=1)


class TriggerConfig(BaseModel):
    """A forecast-driven outbound webhook.

    Distinct from a PrometheusRule alert: the payload carries forecast
    context (predicted value, confidence band, model identity, drift
    score). Distinct from a webhook *regressor*: the direction is inverse
    — the forecaster pushes out rather than pulls in.

    ``condition`` is a PromQL expression evaluated against the configured
    datasource after every fit (so it can reference the forecast metrics
    the sink wrote). The trigger POSTs to ``webhook_url`` when the
    condition transitions from false to true; for sustained conditions it
    re-fires no more often than ``repeat_interval``. ``payload_template``
    is an optional ``string.Template`` (``$var``-substitution) template
    rendered with the event fields (``$name``, ``$condition``, ``$value``,
    ``$fired_at``, ``$samples_json``) — ``$``-style rather than
    ``str.format`` so a JSON template's literal ``{``/``}`` need no
    escaping; an empty template ships the default JSON event. ``headers``
    carry static auth; ``signing`` opts into HMAC-signing the request
    body.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    condition: str
    webhook_url: str
    headers: dict[str, str] = Field(default_factory=dict)
    payload_template: str = ""
    repeat_interval: Duration = timedelta(hours=1)
    timeout: Duration = timedelta(seconds=5)
    signing: SigningConfig | None = None


# Bounded set of forecast events the Grafana-annotations sink can post.
# Each maps to a signal the runner already computes; keeping it a Literal
# stops a typo from silently disabling an annotation stream.
ANNOTATION_EVENT_TYPES = (
    "drift",
    "change_point",
    "deploy_risk",
    "threshold_breach_predicted",
    "deviation_outside_band",
)


class AnnotationsConfig(BaseModel):
    """Grafana-annotations sink for significant forecast events.

    When enabled, the forecaster POSTs to Grafana's annotations API every
    time an enabled event fires, so operators can correlate forecast
    events with the dashboards they already watch. Compatible with both
    Grafana OSS and Grafana Cloud (same API).

    ``api_token_ref`` is resolved through the same ``${ENV_VAR}``
    indirection as datasource credentials. ``events`` selects which event
    families post annotations; an empty list (default) means none even
    when ``enabled: true``.

    ``deploy_risk_min_score`` gates the ``deploy_risk`` event family: only
    a deploy whose composite risk score is at or above this threshold posts
    an annotation, so the stream reflects *elevated* risk rather than every
    active rollout. Ignored unless ``deploy_risk`` is in ``events``.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    grafana_url: str = ""
    api_token_ref: str = ""
    events: list[
        Literal[
            "drift",
            "change_point",
            "deploy_risk",
            "threshold_breach_predicted",
            "deviation_outside_band",
        ]
    ] = Field(default_factory=list)
    timeout: Duration = timedelta(seconds=5)
    deploy_risk_min_score: float = Field(default=0.5, ge=0.0, le=1.0)


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
    sharding: ShardingConfig = Field(default_factory=ShardingConfig)
    query_cache: QueryCacheConfig = QueryCacheConfig()
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    groups: list[GroupConfig]
    # Per-service deploy-anchored risk composites. Off by default
    # (empty list); each entry binds an anchor (regressor with a
    # "deploy happened" semantic) to a weighted set of golden-signal
    # queries. The runner emits ``forecast_deploy_risk_score`` per
    # active deploy. See :class:`DeploySignalsConfig`.
    deploy_signals: list[DeploySignalsConfig] = Field(default_factory=list)
    # Governance aggregations over forecast quality. Off by default;
    # see :class:`GovernanceConfig`.
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    # Operator-facing recommendations (right-sizing, ...). Off by
    # default; see :class:`RecommendationsConfig`.
    recommendations: RecommendationsConfig = Field(default_factory=RecommendationsConfig)
    # Forecast-driven outbound webhooks. Off by default (empty list);
    # see :class:`TriggerConfig`.
    triggers: list[TriggerConfig] = Field(default_factory=list)
    # Grafana-annotations sink for forecast events. Off by default;
    # see :class:`AnnotationsConfig`.
    annotations: AnnotationsConfig = Field(default_factory=AnnotationsConfig)

    @model_validator(mode="after")
    def _validate_scaling_modes(self) -> Config:
        # HA (single-active leader) and sharding (every replica active) are
        # opposite scaling models — enabling both is a misconfiguration, not
        # a richer setup. Reject it at load time so the contradiction can't
        # ship to a cluster.
        if self.sharding.enabled and self.high_availability.enabled:
            raise ValueError(
                "sharding.enabled and highAvailability.enabled are mutually exclusive: "
                "HA gates fitting on a single leader, sharding spreads fitting across "
                "every replica. Pick one."
            )
        # leader_assigned needs a Redis backend for the membership registry
        # and the published group→shard assignment. Accept the HA Redis or a
        # standalone query-cache Redis — main.py reuses whichever is set.
        if (
            self.sharding.enabled
            and self.sharding.mode == "leader_assigned"
            and not (self.high_availability.redis.address or self.query_cache.redis_address)
        ):
            raise ValueError(
                "sharding.mode=leader_assigned requires a Redis backend; set "
                "highAvailability.redis.address or query_cache.redis_address so "
                "replicas can share the membership registry and assignment map"
            )
        return self


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
            _validate_canary(group_name=group.name, query=query)
            for reg in query.regressors:
                _validate_regressor(group_name=group.name, query_id=query.id, reg=reg)
    _validate_deploy_signals(config)
    _validate_resource_sizing(config)
    _validate_triggers(config)
    _validate_annotations(config)
    return config


def _validate_canary(*, group_name: str, query: QueryConfig) -> None:
    """A canary comparison needs both selectors when enabled."""
    if not query.canary.enabled:
        return
    label = f"groups[{group_name}].queries[{query.id}].canary"
    if not query.canary.canary_selector.strip():
        raise ValueError(f"{label}: canary_selector is required when canary.enabled is true")
    if not query.canary.baseline_selector.strip():
        raise ValueError(f"{label}: baseline_selector is required when canary.enabled is true")


def _validate_triggers(config: Config) -> None:
    """Reject duplicate trigger names and empty condition/webhook_url.

    A duplicate ``name`` would collapse two triggers onto the same
    ``forecast_trigger_fires_total{name}`` series and make the
    repeat-interval bookkeeping ambiguous, so it fails fast at load.
    """
    seen: set[str] = set()
    for trigger in config.triggers:
        label = f"triggers[{trigger.name}]"
        if not trigger.name.strip():
            raise ValueError("triggers[]: name must be non-empty")
        if trigger.name in seen:
            raise ValueError(f"triggers: name {trigger.name!r} appears twice; names must be unique")
        seen.add(trigger.name)
        if not trigger.condition.strip():
            raise ValueError(f"{label}: condition must be a non-empty PromQL expression")
        if not trigger.webhook_url.strip():
            raise ValueError(f"{label}: webhook_url must be non-empty")


def _validate_annotations(config: Config) -> None:
    """A Grafana-annotations sink needs a URL when enabled."""
    ann = config.annotations
    if not ann.enabled:
        return
    if not ann.grafana_url.strip():
        raise ValueError(
            "annotations.enabled is true but annotations.grafana_url is empty; "
            "set the Grafana base URL (e.g. https://grafana.example.com)"
        )


def _validate_resource_sizing(config: Config) -> None:
    """Cross-reference every right-sizing workload against the live queries.

    Each workload entry's ``forecast_query`` must resolve to a
    configured query id; an unknown id is almost always a typo that
    would otherwise silently emit nothing. Duplicate (workload, kind)
    pairs are rejected so the emitted gauge stays single-valued —
    otherwise two entries would write to the same label set and the
    last one would silently win.
    """
    cfg = config.recommendations.resource_sizing
    if not cfg.enabled or not cfg.workloads:
        return
    queries_by_id: set[str] = {q.id for g in config.groups for q in g.queries}
    seen: set[tuple[str, str]] = set()
    for entry in cfg.workloads:
        label = f"recommendations.resource_sizing.workloads[{entry.workload}/{entry.kind}]"
        if entry.forecast_query not in queries_by_id:
            raise ValueError(
                f"{label}: forecast_query={entry.forecast_query!r} does not "
                "match any configured query id"
            )
        # ``current_promql`` is optional, but if the operator sets it
        # to the empty string the runtime fetcher would treat the
        # value as "configured" and issue a doomed PromQL request
        # every refresh. Reject at load time so the misconfig is
        # visible.
        if entry.current_promql is not None and not entry.current_promql.strip():
            raise ValueError(
                f"{label}: current_promql must be a non-empty PromQL query when "
                "set (omit the field to skip the current / savings_pct emission)"
            )
        if not entry.workload.strip():
            raise ValueError(f"{label}: workload must be non-empty")
        key = (entry.workload, entry.kind)
        if key in seen:
            raise ValueError(
                f"recommendations.resource_sizing.workloads: duplicate "
                f"(workload={entry.workload!r}, kind={entry.kind!r}); "
                "each pair must appear at most once"
            )
        seen.add(key)


def _validate_deploy_signals(config: Config) -> None:
    """Cross-reference every ``deploy_signals`` entry against the live queries.

    Each entry's ``anchor.query`` and every ``signals[].query`` must
    resolve to a configured query id; an unknown id is a typo that
    would otherwise silently emit a zero-row gauge forever. The
    anchor regressor must exist on the referenced query. Unsupported
    anchor ``type`` values fail at pydantic-parse time (see
    :class:`DeployAnchorConfig`).
    """
    if not config.deploy_signals:
        return
    queries_by_id: dict[str, QueryConfig] = {q.id: q for g in config.groups for q in g.queries}
    seen_services: set[str] = set()
    for entry in config.deploy_signals:
        label = f"deploy_signals[{entry.service}]"
        if not entry.service:
            raise ValueError(f"{label}: service must be non-empty")
        if entry.service in seen_services:
            raise ValueError(
                f"deploy_signals: service {entry.service!r} appears twice; "
                "each service must be defined exactly once"
            )
        seen_services.add(entry.service)
        if not entry.signals:
            raise ValueError(f"{label}.signals: must contain at least one signal")
        anchor_query = queries_by_id.get(entry.anchor.query)
        if anchor_query is None:
            raise ValueError(
                f"{label}.anchor.query={entry.anchor.query!r} does not "
                "match any configured query id"
            )
        if entry.anchor.regressor_id is None:
            raise ValueError(f"{label}.anchor.regressor_id is required when anchor.type=regressor")
        anchor_reg_ids = {r.id for r in anchor_query.regressors}
        if entry.anchor.regressor_id not in anchor_reg_ids:
            raise ValueError(
                f"{label}.anchor.regressor_id={entry.anchor.regressor_id!r} "
                f"is not configured on query {entry.anchor.query!r} "
                f"(configured regressors: {sorted(anchor_reg_ids) or 'none'})"
            )
        for signal in entry.signals:
            if signal.query not in queries_by_id:
                raise ValueError(
                    f"{label}.signals: query {signal.query!r} does not "
                    "match any configured query id"
                )


def _check_api_version(raw: dict[str, Any], *, source: str) -> dict[str, Any]:
    """Validate (and inject) the top-level ``apiVersion`` declaration.

    * Missing ``apiVersion`` -> warn once, default to the current version.
      Pre-versioning configs keep working without an edit, but the warning
      nudges operators to pin the schema before the next breaking bump.
    * ``promforecast.io/v1`` -> the stable schema; loads silently.
    * ``promforecast.io/v1alpha1`` -> deprecated alias of v1 (the schemas
      are byte-identical). Loads with a one-shot ``DeprecationWarning``
      pointing at the migration recipe in ``docs/reference/config-schema.md``.
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
            "only required edit. See docs/reference/config-schema.md for the migration recipe.",
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


def _validate_query_overrides(*, group_name: str, query: QueryConfig) -> None:  # noqa: PLR0912
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
        has_alert_within = threshold.alert_within is not None
        has_severity = threshold.severity is not None
        if has_alert_within != has_severity:
            raise ValueError(
                f"{label}.thresholds[{threshold.name}]: alert_within and severity "
                "must be set together (the will_breach metric is a derived shape "
                "and cannot be partially configured)"
            )
        if threshold.alert_within is not None and threshold.alert_within.total_seconds() <= 0:
            raise ValueError(
                f"{label}.thresholds[{threshold.name}]: alert_within must be a positive duration"
            )
        # The will_breach gauge derives ``1`` from "ETA is finite and
        # below alert_within". For a drain threshold the ETA shrinks as
        # the queue drains faster — so will_breach would fire on a
        # *healthy* fast-draining queue, the opposite of its
        # "predictive breach" meaning. Reject the combination at
        # load-time rather than emitting metrics whose meaning depends
        # on the comparator.
        if threshold.comparator == "le_drain" and has_alert_within:
            raise ValueError(
                f"{label}.thresholds[{threshold.name}]: alert_within + severity are not "
                "supported on le_drain thresholds; the will_breach gauge would invert "
                "(it would fire on a queue draining quickly). Use the +Inf-anchored "
                "QueueNotDraining-style alert on forecast_time_to_drain_seconds instead."
            )
        # ``type`` becomes a free-text label on /metrics — empty or
        # whitespace-only values would silently render as an empty
        # label which dashboards typically discard. Reject at boot.
        if threshold.action_cost is not None and not threshold.action_cost.type.strip():
            raise ValueError(
                f"{label}.thresholds[{threshold.name}].action_cost.type must be non-empty "
                "(e.g. 'usd', 'eur', 'engineer_hours')"
            )
    _validate_cold_start(label=label, query=query)
    _validate_events_profile(label=label, query=query)
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
    drift = query.emission.drift
    if drift.enabled and drift.alert_threshold is not None and drift.alert_threshold < 0:
        raise ValueError(
            f"{label}.emission.drift.alert_threshold must be >= 0 (or null to disable)"
        )
    band_cov = query.emission.band_coverage
    if band_cov.enabled:
        if not band_cov.windows:
            raise ValueError(
                f"{label}.emission.band_coverage: at least one window is required when enabled"
            )
        for window in band_cov.windows:
            if window.total_seconds() <= 0:
                raise ValueError(
                    f"{label}.emission.band_coverage.windows: must be positive durations"
                )
    if query.refresh_interval is not None and query.refresh_interval.total_seconds() <= 0:
        raise ValueError(f"{label}.refresh_interval must be a positive duration")
    _validate_quantiles(label=label, query=query)
    _validate_slo(label=label, query=query)
    _validate_recommended_windows(label=label, query=query)


def _validate_quantiles(*, label: str, query: QueryConfig) -> None:
    """Semantic checks for ``emission.quantiles``.

    Pydantic accepts any list of floats; the meaningful constraints
    (strictly inside ``(0, 1)``, no duplicates) live here so a bad
    quantile fails at load time with a precise label rather than emitting
    a degenerate ``q="0"`` / ``q="1"`` row (the 0th/100th percentile maps
    to an infinite band edge) every refresh.
    """
    quantiles = query.emission.quantiles
    if not quantiles:
        return
    seen: set[float] = set()
    for q in quantiles:
        if not (0.0 < q < 1.0):
            raise ValueError(
                f"{label}.emission.quantiles: every quantile must be strictly "
                f"between 0 and 1 (got {q}); the 0th/100th percentile maps to an "
                "infinite band edge"
            )
        if q in seen:
            raise ValueError(
                f"{label}.emission.quantiles: duplicate quantile {q}; each value must be unique"
            )
        seen.add(q)


def _validate_slo(*, label: str, query: QueryConfig) -> None:
    """Semantic checks for the SLO block of a query.

    The numeric bounds on ``objective`` are enforced by pydantic; this
    helper validates the per-window durations and the relationship
    between windows and the forecast horizon. Failing at load time
    means an over-budget config never reaches the runner.
    """
    if query.slo is None:
        return
    if not query.slo.windows:
        raise ValueError(f"{label}.slo: at least one window is required")
    for window in query.slo.windows:
        if window.total_seconds() <= 0:
            raise ValueError(f"{label}.slo.windows: must be positive durations")


def _validate_recommended_windows(*, label: str, query: QueryConfig) -> None:
    """Semantic checks for the recommended-windows block of a query.

    The duration must fit inside the lookahead and ``count`` must be
    positive (pydantic enforces ``count >= 1``; we still guard against
    a non-positive ``duration`` or ``lookahead`` slipping through).
    """
    rw = query.recommended_windows
    if not rw.enabled:
        return
    if rw.duration.total_seconds() <= 0:
        raise ValueError(f"{label}.recommended_windows.duration must be a positive duration")
    if rw.lookahead.total_seconds() <= 0:
        raise ValueError(f"{label}.recommended_windows.lookahead must be a positive duration")
    if rw.duration > rw.lookahead:
        raise ValueError(
            f"{label}.recommended_windows.duration ({rw.duration}) must be <= "
            f"lookahead ({rw.lookahead}); a window longer than the lookahead "
            "can never be recommended."
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


def _validate_events_profile(*, label: str, query: QueryConfig) -> None:
    """Reject query options that the ``data_profile: events`` path ignores.

    Event queries take a separate runtime path that emits only the
    incident-recurrence heatmap — no forecast curve, no model fit,
    no per-step bands. The following config options would be
    silently dropped on an events query, so we fail fast at load
    time rather than letting an operator wonder why the configured
    threshold never fires.

    The rejected options share the trait that they all only make
    sense for a continuous forecast: thresholds (need a forecast
    curve to project against), regressors (no fit consumes them),
    SLO blocks (need a rate forecast), recommended-windows (need a
    horizon-spanning forecast), cold-start (event series don't fit
    the sibling-borrowing shape), and the deviation / growth-rate /
    conditional emission toggles.
    """
    if query.data_profile != "events":
        return
    rejected: list[str] = []
    if query.thresholds:
        rejected.append("thresholds")
    if query.regressors:
        rejected.append("regressors")
    if query.slo is not None:
        rejected.append("slo")
    if query.recommended_windows.enabled:
        rejected.append("recommended_windows")
    if query.cold_start.borrow_from is not None:
        rejected.append("cold_start.borrow_from")
    if query.emission.conditional:
        rejected.append("emission.conditional")
    if query.emission.decompose:
        rejected.append("emission.decompose")
    if query.emission.growth_rate.enabled:
        rejected.append("emission.growth_rate")
    if query.emission.quantiles:
        rejected.append("emission.quantiles")
    if rejected:
        raise ValueError(
            f"{label}: data_profile: events does not support "
            f"{', '.join(rejected)}; the event-pattern path emits only the "
            "incident-recurrence heatmap (no forecast curve, no model fit). "
            "Remove the rejected options or change the data_profile."
        )


def effective_step(query: QueryConfig, defaults: DefaultsConfig) -> timedelta:
    """Resolve the actual step for ``query`` from its override + defaults."""
    return query.step if query.step is not None else defaults.step


def effective_refresh_interval(query: QueryConfig, server: ServerConfig) -> timedelta:
    """Resolve the actual refresh interval for ``query``.

    A query-level ``refresh_interval`` shadows the group/server-wide value
    so an operator can mix hourly business KPIs with daily capacity refits
    inside one group. ``None`` (the default) inherits ``server.refresh_interval``.
    """
    return query.refresh_interval if query.refresh_interval is not None else server.refresh_interval


def effective_emission_safeguards(query: QueryConfig) -> tuple[bool, bool]:
    """Resolve ``(clip_non_negative, round_to_integer)`` for a query.

    A query with ``data_profile: count`` opts both toggles on by
    default so an operator who declared the profile gets count-safe
    emission without a second knob. An explicit boolean on either
    toggle wins over the profile default — the explicit-override
    pathway is what an operator should reach for when the profile's
    default doesn't fit.

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


def _validate_holiday_calendar(*, label: str, reg: RegressorConfig) -> None:
    """Eagerly resolve the holiday calendar for ``reg`` to catch typos.

    The ``holidays`` package raises ``NotImplementedError`` (or a subclass)
    for unsupported countries and ``KeyError`` for unknown sub-divisions.
    We translate both into a precise ``ValueError`` carrying the
    offending YAML path so operators see "groups[g].queries[q].regressors[h]:
    holiday country 'XZ' is not supported" instead of waiting for the
    runner to bump ``forecast_regressor_failures_total{reason="invalid_type"}``
    on every refresh.

    Import is local so configs without holiday regressors never pay the
    ``holidays`` import cost at load time. The optional ``holidays``
    extras provides the dependency; the error message points operators
    at the install command.

    Calendar resolution is cached on ``(country, subdiv)`` so a config
    with dozens of holiday regressors against the same country pays the
    country-table materialisation cost exactly once per region.
    """
    try:
        from . import regressors as _regressors  # noqa: PLC0415
    except ImportError:  # pragma: no cover - the module is part of the package
        _regressors = None  # type: ignore[assignment]

    try:
        import holidays  # noqa: PLC0415
    except ImportError as exc:
        raise ValueError(
            f"{label}: holiday regressors require the optional 'holidays' extra. Install with "
            f"`pip install 'promforecast[holidays]'` (underlying error: {exc})"
        ) from exc

    # Timezone validation: zoneinfo.ZoneInfo raises on an unknown name.
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # noqa: PLC0415

    try:
        ZoneInfo(reg.timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"{label}: holiday regressor timezone is not a known IANA zone: {reg.timezone!r}"
        ) from exc

    country = str(reg.country)
    cache_get = _regressors._holiday_calendar_cache_get if _regressors is not None else None
    try:
        if not reg.regions:
            if cache_get is not None:
                cache_get(country, None)
            else:
                holidays.country_holidays(country)
        else:
            for subdiv in reg.regions:
                if cache_get is not None:
                    cache_get(country, subdiv)
                else:
                    holidays.country_holidays(country, subdiv=subdiv)
    except (NotImplementedError, KeyError) as exc:
        raise ValueError(
            f"{label}: holidays country/region not supported by the 'holidays' package: {exc}"
        ) from exc


def _validate_regressor(*, group_name: str, query_id: str, reg: RegressorConfig) -> None:  # noqa: PLR0912
    """Per-regressor sanity checks.

    * Calendar regressors must use one of the supported feature names and
      must not carry a ``promql:`` (it would be silently ignored).
    * Custom regressors must carry a non-empty ``promql:`` (otherwise the
      runner has nothing to query).
    * Holiday regressors must declare a ``country`` and must not carry a
      ``promql:``. They're deterministic per timestamp and never hit the
      TSDB.

    Each check fails the load with a message that points at the offending
    group/query so operators can find it quickly.
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
        if reg.country or reg.regions:
            raise ValueError(f"{label}: calendar regressors must not set country/regions")
    elif reg.type == "holidays":
        if reg.promql:
            raise ValueError(f"{label}: holiday regressors must not set promql")
        if not reg.country:
            raise ValueError(f"{label}: holiday regressors require a country code")
        # Resolve the calendar once at boot so a typo in ``country`` /
        # ``regions`` surfaces as a load-time error rather than as a
        # silent per-series ``invalid_type`` failure on every refresh.
        _validate_holiday_calendar(label=label, reg=reg)
    elif reg.type == "webhook":
        if not reg.url:
            raise ValueError(f"{label}: webhook regressors require a non-empty url")
        if reg.promql:
            raise ValueError(f"{label}: webhook regressors must not set promql")
        if reg.country or reg.regions:
            raise ValueError(f"{label}: webhook regressors must not set country/regions")
        if reg.path:
            raise ValueError(f"{label}: webhook regressors must not set path")
    elif reg.type == "calendar_file":
        if not reg.path:
            raise ValueError(f"{label}: calendar_file regressors require a non-empty path")
        if reg.promql:
            raise ValueError(f"{label}: calendar_file regressors must not set promql")
        if reg.country or reg.regions:
            raise ValueError(f"{label}: calendar_file regressors must not set country/regions")
        if reg.url:
            raise ValueError(f"{label}: calendar_file regressors must not set url")
    elif not reg.promql:
        raise ValueError(f"{label}: custom regressors require a non-empty promql")
