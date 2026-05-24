"""``promforecast diagnose`` — operator diagnostics against the long-term TSDB.

Two subcommands today:

* ``duplicates`` — scan for the dual-source forecast labelset state described
  in ``docs/reference/emission-paths.md``. Catches the case where the
  scraped and sink-pushed copies are *distinguishable* by labelset shape
  (different external_labels, ``honor_labels: false`` accidentally set,
  source signals missing a ``job`` label).
* ``sample-rate`` — covers the same-labelset case that ``duplicates``
  cannot see from labelset shape alone. For each forecast metric, probe
  the per-series sample rate via ``count_over_time``; any series whose
  count over the window exceeds the configured cap is flagged as
  likely-duplicate ingestion.

Exit codes (uniform across subcommands):

* ``0`` — nothing detected.
* ``1`` — duplicate / anomaly detected.
* ``2`` — datasource unreachable / malformed response.

Designed to drop into CI / pre-flight checks: the JSON output is stable
and the exit codes match the standard "tests passed / failed / errored"
convention so a ``promforecast diagnose ...`` step in a GitHub Action
gates the next stage cleanly.

The HTTP client is async and probes every forecast metric in parallel
under a small semaphore. Transient TSDB flakes (network blip, 5xx) are
retried with exponential backoff; 4xx (auth, bad query) fails fast so the
operator sees the real error instead of a retry storm masking it.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import IO, Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# HTTP status code for a successful TSDB response. Pulled out to a named
# constant so ruff's PLR2004 magic-number check isn't tripped on the call
# sites that compare against it.
_HTTP_OK = 200
_HTTP_5XX_FLOOR = 500
# Length of a Prometheus instant-query value tuple ``[timestamp, value]``.
_VALUE_PAIR_LEN = 2


# Metrics affected by the dual-source trap. The forecaster's operational
# families (``forecast_quality_score``, ``forecast_failures_total``, the
# ``*_time_to_threshold_seconds`` derived ETA metrics) are *not* in scope —
# they live on /metrics only and intentionally carry source-side labels
# like ``job="node-exporter"`` per design principle 8. See the operational-
# metrics section of ``docs/reference/emission-paths.md``.
FORECAST_METRIC_REGEX = r".+_forecast(_lower|_upper)?"

# Scrape-only marker labels. A labelset family carrying *any* of these is
# almost certainly the scraped copy: ``job`` is the canonical marker (set
# uniformly to ``promforecast`` by the standalone chart's ServiceMonitor),
# the others are common Kubernetes / Prometheus external_labels that the
# sink push doesn't introduce. The list is deliberately conservative — a
# false positive on a custom external_label is preferable to silently
# missing a real dual-source state.
_SCRAPE_MARKER_KEYS: frozenset[str] = frozenset(
    {"job", "service", "pod", "container", "namespace", "cluster", "replica"}
)


# ---------------------------------------------------------------------------
# Auth / client construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnoseAuth:
    """How to authenticate against the TSDB.

    All fields default to ``None`` / ``False`` so callers without auth (the
    dev stack, an in-cluster sidecar with NetworkPolicy gating) get a
    plain-HTTP client with no overhead.

    Precedence inside the dataclass: bearer overrides basic, both are
    independent of TLS knobs. The CLI layer enforces "at most one of
    bearer / basic" with argparse mutual-exclusion so the operator can't
    accidentally combine them.
    """

    bearer_token: str | None = None
    basic_user: str | None = None
    basic_password: str | None = None
    tls_cert: str | None = None
    tls_key: str | None = None
    tls_ca: str | None = None
    insecure_skip_verify: bool = False

    def build_client_kwargs(self) -> dict[str, Any]:
        """Translate the auth bundle into kwargs accepted by ``httpx.AsyncClient``."""
        kwargs: dict[str, Any] = {}
        if self.bearer_token:
            kwargs["headers"] = {"Authorization": f"Bearer {self.bearer_token}"}
        elif self.basic_user is not None:
            kwargs["auth"] = httpx.BasicAuth(self.basic_user, self.basic_password or "")
        if self.tls_cert and self.tls_key:
            kwargs["cert"] = (self.tls_cert, self.tls_key)
        if self.insecure_skip_verify:
            kwargs["verify"] = False
        elif self.tls_ca:
            kwargs["verify"] = self.tls_ca
        return kwargs


def _build_async_client(
    auth: DiagnoseAuth | None,
    *,
    timeout: float,
    max_concurrent: int,
) -> httpx.AsyncClient:
    """Construct an ``httpx.AsyncClient`` with auth + a sized connection pool.

    The pool size tracks the parallel-probe semaphore so the pool is never
    the implicit bottleneck — same approach as ``source.py``.
    """
    kwargs = (auth or DiagnoseAuth()).build_client_kwargs()
    kwargs.setdefault("timeout", timeout)
    kwargs.setdefault(
        "limits",
        httpx.Limits(max_connections=max_concurrent, max_keepalive_connections=max_concurrent),
    )
    return httpx.AsyncClient(**kwargs)


# ---------------------------------------------------------------------------
# Datatypes shared across subcommands
# ---------------------------------------------------------------------------


class DiagnoseError(RuntimeError):
    """Raised when the diagnose subcommand cannot reach the TSDB or the
    response is malformed. Surfaces as exit code 2 to the shell."""


@dataclass(frozen=True)
class LabelsetFamily:
    """One distinct labelset shape observed for a forecast metric.

    ``keys`` is the set of label names that appear on every series in
    this family (excluding ``__name__``). ``example_labels`` is one
    representative series so the operator can see a concrete labelset
    next to the abstract key list. ``count`` is how many series fell
    into this family — useful for sizing the cleanup work and for
    distinguishing the dominant family from a stragglers-only family.
    """

    keys: frozenset[str]
    example_labels: dict[str, str]
    count: int
    looks_scrape_shaped: bool


@dataclass
class MetricFinding:
    """One forecast metric and the families it appears under.

    Always has at least one family (a metric that returns zero series
    is excluded upstream). ``is_duplicate`` is true when the metric is
    in the dual-source state — that is, at least one scrape-shaped
    family *and* at least one sink-shaped family are both present.

    Intra-source variance (two families that are both scrape-shaped or
    both sink-shaped, differing only in some optional label like the
    conformal ``calibrated`` flag) is *not* a duplicate. The story this
    detector serves is specifically the scrape+sink dual-source trap;
    flagging every label-shape change inside one source would drown
    real findings under benign noise.
    """

    metric: str
    families: list[LabelsetFamily] = field(default_factory=list)

    @property
    def is_duplicate(self) -> bool:
        scrape_shaped = any(f.looks_scrape_shaped for f in self.families)
        sink_shaped = any(not f.looks_scrape_shaped for f in self.families)
        return scrape_shaped and sink_shaped


@dataclass
class DiagnoseReport:
    """All findings from one ``duplicates`` scan, plus metadata for the
    report header."""

    datasource_url: str
    metrics_scanned: int
    findings: list[MetricFinding]
    selector: str | None
    since_seconds: int | None

    @property
    def duplicates(self) -> list[MetricFinding]:
        return [f for f in self.findings if f.is_duplicate]

    @property
    def has_duplicates(self) -> bool:
        return bool(self.duplicates)


@dataclass(frozen=True)
class SampleRateAnomaly:
    """One forecast series whose sample count over the window exceeded the cap."""

    metric: str
    labels: dict[str, str]
    samples_in_window: int


@dataclass
class SampleRateReport:
    """All anomalies from one ``sample-rate`` scan plus the scan's parameters."""

    datasource_url: str
    window_seconds: int
    max_samples_per_window: int
    metrics_scanned: int
    series_examined: int
    anomalies: list[SampleRateAnomaly]
    selector: str | None

    @property
    def has_anomalies(self) -> bool:
        return bool(self.anomalies)


# ---------------------------------------------------------------------------
# Public entry points (sync wrappers for the CLI; tests can call the async
# functions directly with ``pytest.mark.asyncio``)
# ---------------------------------------------------------------------------


def run_diagnose_duplicates(
    *,
    datasource_url: str,
    selector: str | None = None,
    since_seconds: int | None = None,
    auth: DiagnoseAuth | None = None,
    out: IO[str] | None = None,
    json_output: bool = False,
    http_client: httpx.AsyncClient | None = None,
    max_concurrent: int = 8,
    timeout: float = 30.0,
) -> int:
    """Execute the diagnose-duplicates scan and write the report.

    Returns the process exit code (0 / 1 / 2). ``http_client`` is the
    seam tests use to inject a mock transport; production callers leave
    it ``None`` and a default client is built per invocation.
    """
    return asyncio.run(
        _run_diagnose_duplicates_async(
            datasource_url=datasource_url,
            selector=selector,
            since_seconds=since_seconds,
            auth=auth,
            out=out,
            json_output=json_output,
            http_client=http_client,
            max_concurrent=max_concurrent,
            timeout=timeout,
        )
    )


def run_diagnose_sample_rate(
    *,
    datasource_url: str,
    window_seconds: int,
    max_samples_per_window: int,
    selector: str | None = None,
    auth: DiagnoseAuth | None = None,
    out: IO[str] | None = None,
    json_output: bool = False,
    http_client: httpx.AsyncClient | None = None,
    max_concurrent: int = 8,
    timeout: float = 30.0,
) -> int:
    """Execute the sample-rate scan and write the report. Returns exit code."""
    return asyncio.run(
        _run_diagnose_sample_rate_async(
            datasource_url=datasource_url,
            window_seconds=window_seconds,
            max_samples_per_window=max_samples_per_window,
            selector=selector,
            auth=auth,
            out=out,
            json_output=json_output,
            http_client=http_client,
            max_concurrent=max_concurrent,
            timeout=timeout,
        )
    )


# ---------------------------------------------------------------------------
# Async cores
# ---------------------------------------------------------------------------


async def _run_diagnose_duplicates_async(
    *,
    datasource_url: str,
    selector: str | None,
    since_seconds: int | None,
    auth: DiagnoseAuth | None,
    out: IO[str] | None,
    json_output: bool,
    http_client: httpx.AsyncClient | None,
    max_concurrent: int,
    timeout: float,
) -> int:
    stream: IO[str] = out if out is not None else sys.stdout
    err = sys.stderr

    def write(line: str) -> None:
        stream.write(line + "\n")

    try:
        report = await scan_for_duplicates(
            datasource_url=datasource_url,
            selector=selector,
            since_seconds=since_seconds,
            auth=auth,
            http_client=http_client,
            max_concurrent=max_concurrent,
            timeout=timeout,
        )
    except DiagnoseError as exc:
        err.write(f"error: {exc}\n")
        return 2

    if json_output:
        write(json.dumps(_report_to_json(report), indent=2, sort_keys=True))
    else:
        _write_text_report(report, write)

    return 1 if report.has_duplicates else 0


async def _run_diagnose_sample_rate_async(
    *,
    datasource_url: str,
    window_seconds: int,
    max_samples_per_window: int,
    selector: str | None,
    auth: DiagnoseAuth | None,
    out: IO[str] | None,
    json_output: bool,
    http_client: httpx.AsyncClient | None,
    max_concurrent: int,
    timeout: float,
) -> int:
    stream: IO[str] = out if out is not None else sys.stdout
    err = sys.stderr

    def write(line: str) -> None:
        stream.write(line + "\n")

    try:
        report = await scan_for_sample_rate(
            datasource_url=datasource_url,
            window_seconds=window_seconds,
            max_samples_per_window=max_samples_per_window,
            selector=selector,
            auth=auth,
            http_client=http_client,
            max_concurrent=max_concurrent,
            timeout=timeout,
        )
    except DiagnoseError as exc:
        err.write(f"error: {exc}\n")
        return 2

    if json_output:
        write(json.dumps(_sample_rate_report_to_json(report), indent=2, sort_keys=True))
    else:
        _write_sample_rate_text_report(report, write)

    return 1 if report.has_anomalies else 0


async def scan_for_duplicates(
    *,
    datasource_url: str,
    selector: str | None,
    since_seconds: int | None,
    auth: DiagnoseAuth | None = None,
    http_client: httpx.AsyncClient | None = None,
    max_concurrent: int = 8,
    timeout: float = 30.0,
) -> DiagnoseReport:
    """Query the TSDB and return the structured finding set.

    Per-metric series probes fan out under a small semaphore so the scan
    completes in roughly one TSDB round-trip rather than N. The semaphore
    cap matches the TSDB read pool sizing used by ``source.py``.
    """
    base = datasource_url.rstrip("/")
    owns_client = http_client is None
    client = (
        http_client
        if http_client is not None
        else _build_async_client(auth, timeout=timeout, max_concurrent=max_concurrent)
    )
    try:
        metric_names = await _list_forecast_metric_names(client, base)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def probe(metric: str) -> tuple[str, list[dict[str, str]]]:
            async with semaphore:
                match = _build_selector(metric, selector)
                series = await _list_series(client, base, match, since_seconds=since_seconds)
                return metric, series

        results = await asyncio.gather(*[probe(m) for m in metric_names]) if metric_names else []
        findings: list[MetricFinding] = []
        for metric, series in results:
            if not series:
                continue
            families = _group_into_families(series)
            findings.append(MetricFinding(metric=metric, families=families))
        return DiagnoseReport(
            datasource_url=base,
            metrics_scanned=len(metric_names),
            findings=findings,
            selector=selector,
            since_seconds=since_seconds,
        )
    finally:
        if owns_client:
            await client.aclose()


async def scan_for_sample_rate(
    *,
    datasource_url: str,
    window_seconds: int,
    max_samples_per_window: int,
    selector: str | None,
    auth: DiagnoseAuth | None = None,
    http_client: httpx.AsyncClient | None = None,
    max_concurrent: int = 8,
    timeout: float = 30.0,
) -> SampleRateReport:
    """Probe per-series sample rate to find suspected duplicate ingest.

    Covers the same-labelset case ``scan_for_duplicates`` cannot detect:
    when the scrape and sink push samples into the *same* series (identical
    labelsets), the only remaining signal is that the series has more
    samples per window than one ingest path alone should produce.

    The threshold is operator-tuned because the "expected" rate depends on
    ``refresh_interval`` and the forecast step, neither of which is
    recoverable from the TSDB. A conservative default
    (``max_samples_per_window=5`` over ``window=10m``) catches scrape
    contamination (which typically adds 20-40 samples per 10min at a
    15-30s scrape_interval) while leaving slack for the sink's own
    multi-sample-per-fit emission pattern.
    """
    base = datasource_url.rstrip("/")
    owns_client = http_client is None
    client = (
        http_client
        if http_client is not None
        else _build_async_client(auth, timeout=timeout, max_concurrent=max_concurrent)
    )
    try:
        metric_names = await _list_forecast_metric_names(client, base)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def probe(metric: str) -> tuple[str, list[tuple[dict[str, str], int]]]:
            async with semaphore:
                counts = await _query_count_over_time(
                    client, base, metric, window_seconds=window_seconds, selector=selector
                )
                return metric, counts

        results = await asyncio.gather(*[probe(m) for m in metric_names]) if metric_names else []
        anomalies: list[SampleRateAnomaly] = []
        series_examined = 0
        for metric, counts in results:
            for labels, count in counts:
                series_examined += 1
                if count > max_samples_per_window:
                    anomalies.append(
                        SampleRateAnomaly(metric=metric, labels=labels, samples_in_window=count)
                    )
        # Stable ordering: by metric, then by descending count, then by labels.
        anomalies.sort(key=lambda a: (a.metric, -a.samples_in_window, sorted(a.labels.items())))
        return SampleRateReport(
            datasource_url=base,
            window_seconds=window_seconds,
            max_samples_per_window=max_samples_per_window,
            metrics_scanned=len(metric_names),
            series_examined=series_examined,
            anomalies=anomalies,
            selector=selector,
        )
    finally:
        if owns_client:
            await client.aclose()


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


def _is_retryable(exc: BaseException) -> bool:
    """Retry transport errors and 5xx; let 4xx (auth, bad query) fail fast.

    Mirrors the same policy ``source.py`` uses for the runtime PromQL
    client. Retrying a 401/403 just multiplies the operator-visible
    latency before the real error surfaces.
    """
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= _HTTP_5XX_FLOOR
    return False


async def _get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: list[tuple[str, str | int | float | bool | None]] | dict[str, str] | None = None,
) -> httpx.Response:
    """GET ``url`` with up to 3 attempts; transient flakes are smoothed over.

    Backoff cap matches ``source.py``: ~0.5/1/2/4s envelope. 4xx is raised
    immediately by the inner ``raise_for_status`` and *not* retried because
    ``_is_retryable`` returns False for it.
    """
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    ):
        with attempt:
            response = await client.get(url, params=params)
            if response.status_code >= _HTTP_5XX_FLOOR:
                # Raise an HTTPStatusError so tenacity sees a retryable
                # exception, not a successful return with a 5xx body.
                response.raise_for_status()
            return response
    raise DiagnoseError("retries exhausted")  # pragma: no cover


async def _list_forecast_metric_names(client: httpx.AsyncClient, base: str) -> list[str]:
    """Enumerate every metric name matching the forecast regex.

    Uses ``/api/v1/label/__name__/values`` with a server-side ``match[]``
    so the TSDB does the regex filter before responding. On a TSDB with
    tens of thousands of metric names, this turns a multi-megabyte body
    into a kilobyte-class one. Both Prometheus and VictoriaMetrics
    accept ``match[]`` on this endpoint; servers that ignore it fall
    back to returning everything and we client-side filter as before.
    """
    url = f"{base}/api/v1/label/__name__/values"
    # Send the server-side filter as a single ``match[]`` parameter. The
    # regex form ``{__name__=~"<regex>"}`` is preferred over an anchored
    # match because some TSDBs match the regex anchored already and adding
    # ``^/$`` would result in a double anchor.
    params: list[tuple[str, str | int | float | bool | None]] = [
        ("match[]", '{__name__=~"' + FORECAST_METRIC_REGEX + '"}')
    ]
    try:
        response = await _get_with_retry(client, url, params=params)
    except httpx.HTTPError as exc:
        raise DiagnoseError(
            f"datasource unreachable: {url}: {exc.__class__.__name__}: {exc}"
        ) from exc
    if response.status_code != _HTTP_OK:
        raise DiagnoseError(
            f"datasource returned HTTP {response.status_code} from {url}: {response.text[:200]}"
        )
    payload = _parse_json(response, url)
    data = payload.get("data")
    if not isinstance(data, list):
        raise DiagnoseError(f"unexpected response shape from {url}: missing 'data' list")
    pattern = re.compile(f"^{FORECAST_METRIC_REGEX}$")
    # Defensive client-side filter: some TSDBs (and proxies) silently
    # ignore unknown ``match[]`` params on the label-values endpoint and
    # return the full catalogue. The post-filter costs nothing on a
    # response the server already trimmed and keeps the result correct
    # when it didn't.
    return sorted(name for name in data if isinstance(name, str) and pattern.match(name))


async def _list_series(
    client: httpx.AsyncClient,
    base: str,
    match: str,
    *,
    since_seconds: int | None,
) -> list[dict[str, str]]:
    """Fetch all series matching ``match`` and return their label maps."""
    url = f"{base}/api/v1/series"
    params: list[tuple[str, str | int | float | bool | None]] = [("match[]", match)]
    if since_seconds is not None and since_seconds > 0:
        now = int(time.time())
        params.append(("start", str(now - since_seconds)))
        params.append(("end", str(now)))
    try:
        response = await _get_with_retry(client, url, params=params)
    except httpx.HTTPError as exc:
        raise DiagnoseError(
            f"datasource unreachable: {url}: {exc.__class__.__name__}: {exc}"
        ) from exc
    if response.status_code != _HTTP_OK:
        raise DiagnoseError(
            f"datasource returned HTTP {response.status_code} from {url}: {response.text[:200]}"
        )
    payload = _parse_json(response, url)
    data = payload.get("data")
    if not isinstance(data, list):
        raise DiagnoseError(f"unexpected response shape from {url}: missing 'data' list")
    series: list[dict[str, str]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        labels = {k: v for k, v in entry.items() if k != "__name__" and isinstance(v, str)}
        series.append(labels)
    return series


async def _query_count_over_time(
    client: httpx.AsyncClient,
    base: str,
    metric: str,
    *,
    window_seconds: int,
    selector: str | None,
) -> list[tuple[dict[str, str], int]]:
    """Run ``count_over_time(<metric>{<selector>}[<window>])`` and return per-series counts.

    The PromQL instant-query result is a vector — one entry per series
    matching the selector. Each entry's value is the sample count over
    the window for that series. Returns ``(labels_without___name__, count)``
    tuples preserving the TSDB's ordering.
    """
    base_selector = _build_selector(metric, selector)
    promql = f"count_over_time({base_selector}[{window_seconds}s])"
    url = f"{base}/api/v1/query"
    try:
        response = await _get_with_retry(client, url, params={"query": promql})
    except httpx.HTTPError as exc:
        raise DiagnoseError(
            f"datasource unreachable: {url}: {exc.__class__.__name__}: {exc}"
        ) from exc
    if response.status_code != _HTTP_OK:
        raise DiagnoseError(
            f"datasource returned HTTP {response.status_code} from {url}: {response.text[:200]}"
        )
    payload = _parse_json(response, url)
    data = payload.get("data") or {}
    if not isinstance(data, dict):
        raise DiagnoseError(f"unexpected response shape from {url}: 'data' is not an object")
    if data.get("resultType") != "vector":
        raise DiagnoseError(
            f"unexpected resultType from {url}: {data.get('resultType')!r}; expected 'vector'"
        )
    result = data.get("result") or []
    if not isinstance(result, list):
        return []
    counts: list[tuple[dict[str, str], int]] = []
    for entry in result:
        if not isinstance(entry, dict):
            continue
        metric_labels = entry.get("metric", {})
        if not isinstance(metric_labels, dict):
            continue
        labels = {k: v for k, v in metric_labels.items() if k != "__name__" and isinstance(v, str)}
        value = entry.get("value")
        if not isinstance(value, list) or len(value) != _VALUE_PAIR_LEN:
            continue
        try:
            count = int(float(value[1]))
        except (TypeError, ValueError):
            continue
        counts.append((labels, count))
    return counts


# ---------------------------------------------------------------------------
# Selector composition + family bucketing
# ---------------------------------------------------------------------------


def _build_selector(metric: str, user_selector: str | None) -> str:
    """Compose ``{__name__="<metric>", <user_selector_inner>}``.

    The ``--selector '{job="foo"}'`` flag is a fully-formed PromQL series
    selector; we splice its inner matchers into the metric-name selector
    so callers can pre-filter to one cluster / namespace / tenant. The
    helper is paranoid about the brace forms so an operator passing
    either ``{a="b"}`` or ``a="b"`` (no braces) lands on the same
    selector — the second form is what shells often produce when the
    user forgot the quoting.
    """
    if not user_selector:
        return "{" + f'__name__="{metric}"' + "}"
    inner = user_selector.strip()
    if inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1].strip()
    if not inner:
        return "{" + f'__name__="{metric}"' + "}"
    return "{" + f'__name__="{metric}",{inner}' + "}"


def _group_into_families(series: list[dict[str, str]]) -> list[LabelsetFamily]:
    """Bucket series by label-key fingerprint, then summarise each bucket.

    Two series belong to the same family iff they carry the same *set
    of label names*, regardless of the values. This catches the canonical
    dual-source case where the scraped copy carries ``job``/``instance``
    that the sink-pushed copy doesn't, without false-positiving on
    series that vary only in label *values* (e.g. one node having
    ``device=sda`` and another having ``device=nvme0n1``).
    """
    buckets: dict[frozenset[str], list[dict[str, str]]] = {}
    for labels in series:
        key = frozenset(labels.keys())
        buckets.setdefault(key, []).append(labels)
    families: list[LabelsetFamily] = []
    for keys, members in buckets.items():
        families.append(
            LabelsetFamily(
                keys=keys,
                example_labels=dict(sorted(members[0].items())),
                count=len(members),
                looks_scrape_shaped=bool(keys & _SCRAPE_MARKER_KEYS),
            )
        )
    families.sort(key=lambda f: (-f.count, sorted(f.keys)))
    return families


def _parse_json(response: httpx.Response, url: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise DiagnoseError(f"datasource returned non-JSON body from {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DiagnoseError(f"unexpected response shape from {url}: top-level not an object")
    if payload.get("status") != "success":
        raise DiagnoseError(
            f"datasource returned status={payload.get('status')!r} from {url}: "
            f"{payload.get('error') or payload.get('errorType') or '<no error message>'}"
        )
    return payload


# ---------------------------------------------------------------------------
# Report rendering — duplicates
# ---------------------------------------------------------------------------


def _report_to_json(report: DiagnoseReport) -> dict[str, Any]:
    """Stable JSON shape for CI consumption."""
    return {
        "datasource_url": report.datasource_url,
        "metrics_scanned": report.metrics_scanned,
        "selector": report.selector,
        "since_seconds": report.since_seconds,
        "has_duplicates": report.has_duplicates,
        "findings": [
            {
                "metric": f.metric,
                "is_duplicate": f.is_duplicate,
                "families": [
                    {
                        "keys": sorted(fam.keys),
                        "count": fam.count,
                        "looks_scrape_shaped": fam.looks_scrape_shaped,
                        "example_labels": fam.example_labels,
                    }
                    for fam in f.families
                ],
            }
            for f in report.findings
        ],
    }


def _write_text_report(report: DiagnoseReport, write: Callable[[str], None]) -> None:
    """Human-friendly report. JSON is preferred for CI; text for terminals."""
    write(f"Datasource: {report.datasource_url}")
    write(f"Metrics scanned: {report.metrics_scanned}")
    if report.selector:
        write(f"Selector filter: {report.selector}")
    if report.since_seconds:
        write(f"Lookback window: {report.since_seconds}s")
    write("")
    if not report.findings:
        write("No forecast metrics found in the datasource.")
        return
    if not report.has_duplicates:
        write(
            f"OK  no duplicate forecast labelsets detected across {len(report.findings)} metric(s)."
        )
        return
    write(
        f"FAIL  {len(report.duplicates)} of {len(report.findings)} forecast metric(s) "
        "appear under more than one labelset family."
    )
    write("")
    for finding in report.duplicates:
        write(f"  {finding.metric}:")
        for fam in finding.families:
            tag = "scrape-shaped" if fam.looks_scrape_shaped else "sink-shaped"
            write(f"    [{tag}] count={fam.count}  keys={sorted(fam.keys)}")
            write(f"        example: {fam.example_labels}")
        write("")
    _write_remediation(report, write)


def _write_remediation(report: DiagnoseReport, write: Callable[[str], None]) -> None:
    """Copy-pasteable next steps for an operator on the duplicate path."""
    write("Remediation:")
    write("")
    write("  1. Prometheus scrape — drop forecast metrics from the forecaster job.")
    write("     Add to the forecaster's scrape_config (externally-managed Prometheus)")
    write("     or to the ServiceMonitor (prometheus-operator):")
    write("")
    write("       metric_relabel_configs:")
    write("         - source_labels: [__name__]")
    write("           regex: '.+_forecast(_lower|_upper)?'")
    write("           action: drop")
    write("")
    write("  2. Helm chart values — flip the chart-managed drop on.")
    write("     Standalone chart:")
    write("")
    write("       sink:")
    write("         remote_write:")
    write("           enabled: true")
    write("       serviceMonitor:")
    write("         dropForecastSnapshots: true   # null = follow sink.enabled")
    write("")
    write("  3. Backfill cleanup — once the scrape-drop is in place, delete the")
    write("     scrape-shaped historical samples from VictoriaMetrics:")
    write("")
    for finding in report.duplicates:
        scrape_fam = next((f for f in finding.families if f.looks_scrape_shaped), None)
        if scrape_fam is None:
            continue
        selector = _build_delete_selector(finding.metric, scrape_fam)
        write(
            f"       curl -X POST '{report.datasource_url}/api/v1/admin/tsdb/delete_series' "
            f"--data-urlencode 'match[]={selector}'"
        )
    write("")
    write(
        "  4. Re-run ``promforecast diagnose duplicates`` to confirm the duplicate "
        "labelset family no longer appears."
    )
    write("")
    write(
        "  See docs/how-to/migrate-to-sink-first.md for the full migration "
        "walkthrough and docs/operations/diagnose.md for this subcommand."
    )


# Preferred-first ordering for the delete selector's pin label.
_DELETE_PIN_PREFERENCE: tuple[str, ...] = (
    "job",
    "service",
    "pod",
    "container",
    "namespace",
    "cluster",
    "replica",
)


def _build_delete_selector(metric: str, scrape_family: LabelsetFamily) -> str:
    """Pin the delete to the scrape-shaped family only.

    Picks the most-specific scrape-marker label observed in this family
    so the deletion targets the duplicate without touching the
    sink-pushed series or adjacent sink-pushed metrics that happen to
    share a wide external_label like ``cluster``.
    """
    for key in _DELETE_PIN_PREFERENCE:
        if key in scrape_family.keys:
            value = scrape_family.example_labels.get(key, "")
            return "{" + f'__name__="{metric}",{key}="{value}"' + "}"
    return "{" + f'__name__="{metric}"' + "}"


# ---------------------------------------------------------------------------
# Report rendering — sample-rate
# ---------------------------------------------------------------------------


def _sample_rate_report_to_json(report: SampleRateReport) -> dict[str, Any]:
    """Stable JSON shape for CI consumption."""
    return {
        "datasource_url": report.datasource_url,
        "window_seconds": report.window_seconds,
        "max_samples_per_window": report.max_samples_per_window,
        "metrics_scanned": report.metrics_scanned,
        "series_examined": report.series_examined,
        "selector": report.selector,
        "has_anomalies": report.has_anomalies,
        "anomalies": [
            {
                "metric": a.metric,
                "labels": a.labels,
                "samples_in_window": a.samples_in_window,
            }
            for a in report.anomalies
        ],
    }


def _write_sample_rate_text_report(report: SampleRateReport, write: Callable[[str], None]) -> None:
    """Human-friendly sample-rate report."""
    write(f"Datasource: {report.datasource_url}")
    write(f"Metrics scanned: {report.metrics_scanned}")
    write(f"Series examined: {report.series_examined}")
    write(f"Window: {report.window_seconds}s")
    write(f"Per-series cap: {report.max_samples_per_window} samples per window")
    if report.selector:
        write(f"Selector filter: {report.selector}")
    write("")
    if report.metrics_scanned == 0:
        write("No forecast metrics found in the datasource.")
        return
    if not report.has_anomalies:
        write(
            "OK  no series exceeded the per-window sample cap "
            f"across {report.series_examined} series."
        )
        return
    write(
        f"FAIL  {len(report.anomalies)} of {report.series_examined} series "
        "exceeded the configured sample-rate cap."
    )
    write("")
    last_metric: str | None = None
    for anomaly in report.anomalies:
        if anomaly.metric != last_metric:
            write(f"  {anomaly.metric}:")
            last_metric = anomaly.metric
        write(
            f"    count={anomaly.samples_in_window}  labels={dict(sorted(anomaly.labels.items()))}"
        )
    write("")
    write(
        "  Likely cause: two ingest paths (snapshot scrape + remote_write sink) "
        "writing to the same series. Apply the scrape-drop "
        "(``serviceMonitor.dropForecastSnapshots: true`` or the equivalent "
        "``metric_relabel_configs`` for externally-managed Prometheus); "
        "see docs/how-to/migrate-to-sink-first.md."
    )


# Re-exports kept for backwards-compatibility with existing test imports.
# ``_DELETE_PIN_PREFERENCE`` and the dataclasses above are the public-ish
# surface; keeping the underscored helpers importable is a deliberate
# choice that lets test_diagnose.py exercise the bucketing logic without
# spinning up an HTTP transport. None of these are user-facing.
__all__ = [
    "FORECAST_METRIC_REGEX",
    "DiagnoseAuth",
    "DiagnoseError",
    "DiagnoseReport",
    "LabelsetFamily",
    "MetricFinding",
    "SampleRateAnomaly",
    "SampleRateReport",
    "run_diagnose_duplicates",
    "run_diagnose_sample_rate",
    "scan_for_duplicates",
    "scan_for_sample_rate",
]
