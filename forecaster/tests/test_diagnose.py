"""Tests for ``promforecast diagnose``.

The detection logic is the load-bearing part: a clean install must return
exit code 0, a dual-source install must return exit code 1 and name every
duplicated metric, and an unreachable datasource must return exit code 2.

The TSDB is mocked at the HTTP layer with an ``httpx.MockTransport`` so
the tests run offline and against a deterministic series catalogue. The
transport works against both ``httpx.Client`` and ``httpx.AsyncClient``,
which lets us exercise the async diagnose internals from sync tests.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from promforecast.diagnose import (
    DiagnoseAuth,
    DiagnoseError,
    LabelsetFamily,
    _build_delete_selector,
    _build_selector,
    _group_into_families,
    run_diagnose_duplicates,
    run_diagnose_sample_rate,
    scan_for_duplicates,
    scan_for_sample_rate,
)


def _make_transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    """Build an async httpx client whose every request goes through ``handler``."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok(data: Any) -> httpx.Response:
    return httpx.Response(200, json={"status": "success", "data": data})


def _vector(entries: list[dict[str, Any]]) -> httpx.Response:
    """Build an instant-query vector response (``resultType: vector``)."""
    return httpx.Response(
        200, json={"status": "success", "data": {"resultType": "vector", "result": entries}}
    )


# ---------------------------------------------------------------------------
# Family bucketing — the heart of the detector
# ---------------------------------------------------------------------------


def test_group_into_families_distinguishes_sink_and_scrape_shapes() -> None:
    """Series with different label keys belong to different families."""
    series = [
        # Sink-pushed copies: lean labelset.
        {"group": "node_capacity", "model": "AutoARIMA", "instance": "node-a:9100"},
        {"group": "node_capacity", "model": "AutoARIMA", "instance": "node-b:9100"},
        # Scraped copies: carry the scrape-job marker.
        {
            "group": "node_capacity",
            "model": "AutoARIMA",
            "instance": "node-a:9100",
            "job": "promforecast",
        },
    ]
    families = _group_into_families(series)
    assert len(families) == 2
    # Sorted by descending count → the two-member sink family comes first.
    assert families[0].count == 2
    assert not families[0].looks_scrape_shaped
    assert families[1].count == 1
    assert families[1].looks_scrape_shaped


def test_group_into_families_treats_value_only_variation_as_one_family() -> None:
    """``device=sda`` vs ``device=nvme0n1`` is one family — same label keys."""
    series = [
        {"instance": "a", "device": "sda"},
        {"instance": "b", "device": "nvme0n1"},
    ]
    families = _group_into_families(series)
    assert len(families) == 1
    assert families[0].count == 2


# ---------------------------------------------------------------------------
# Selector composition
# ---------------------------------------------------------------------------


def test_build_selector_without_user_filter() -> None:
    assert _build_selector("foo_forecast", None) == '{__name__="foo_forecast"}'


def test_build_selector_splices_braced_user_selector() -> None:
    assert (
        _build_selector("foo_forecast", '{cluster="prod"}')
        == '{__name__="foo_forecast",cluster="prod"}'
    )


def test_build_selector_tolerates_unbraced_user_selector() -> None:
    """Shells often eat the braces; accept either form."""
    assert (
        _build_selector("foo_forecast", 'cluster="prod"')
        == '{__name__="foo_forecast",cluster="prod"}'
    )


def test_build_selector_ignores_empty_braces() -> None:
    assert _build_selector("foo_forecast", "{}") == '{__name__="foo_forecast"}'


# ---------------------------------------------------------------------------
# Delete-selector composition
# ---------------------------------------------------------------------------


def test_build_delete_selector_prefers_job_over_external_labels() -> None:
    """The delete selector must prefer ``job`` over broader external_labels."""
    family = LabelsetFamily(
        keys=frozenset({"job", "cluster", "replica", "instance", "group"}),
        example_labels={
            "cluster": "prod",
            "group": "node_capacity",
            "instance": "node-a:9100",
            "job": "promforecast",
            "replica": "prom-0",
        },
        count=10,
        looks_scrape_shaped=True,
    )
    selector = _build_delete_selector("node_cpu_busy_pct_forecast", family)
    assert "job=" in selector and "promforecast" in selector
    assert "cluster=" not in selector
    assert "replica=" not in selector


def test_build_delete_selector_pins_scrape_marker() -> None:
    family = LabelsetFamily(
        keys=frozenset({"job", "instance", "group", "model"}),
        example_labels={
            "group": "node_capacity",
            "instance": "forecaster:9091",
            "job": "promforecast",
            "model": "AutoARIMA",
        },
        count=12,
        looks_scrape_shaped=True,
    )
    selector = _build_delete_selector("node_cpu_busy_pct_forecast", family)
    assert selector == '{__name__="node_cpu_busy_pct_forecast",job="promforecast"}'


# ---------------------------------------------------------------------------
# End-to-end: clean install
# ---------------------------------------------------------------------------


def test_run_diagnose_returns_zero_on_clean_install() -> None:
    """Sink-only install: every metric has exactly one labelset family."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/label/__name__/values":
            return _ok(["node_cpu_busy_pct_forecast", "node_cpu_busy_pct_forecast_upper"])
        if request.url.path == "/api/v1/series":
            return _ok(
                [
                    {
                        "__name__": "node_cpu_busy_pct_forecast",
                        "group": "node_capacity",
                        "model": "AutoARIMA",
                        "instance": "node-a:9100",
                    },
                    {
                        "__name__": "node_cpu_busy_pct_forecast",
                        "group": "node_capacity",
                        "model": "AutoARIMA",
                        "instance": "node-b:9100",
                    },
                ]
            )
        return httpx.Response(404)

    client = _make_transport(handler)
    out = io.StringIO()
    rc = run_diagnose_duplicates(
        datasource_url="http://vm:8428/",
        out=out,
        http_client=client,
    )
    assert rc == 0
    rendered = out.getvalue()
    assert "no duplicate forecast labelsets detected" in rendered
    # The report header includes the canonicalised datasource URL (no
    # trailing slash) so a copy-pasted remediation snippet matches.
    assert "http://vm:8428" in rendered


# ---------------------------------------------------------------------------
# End-to-end: dual-source install
# ---------------------------------------------------------------------------


def test_run_diagnose_returns_one_when_dual_source() -> None:
    """The scrape + sink dual-source state must trip exit code 1."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/label/__name__/values":
            return _ok(["node_cpu_busy_pct_forecast", "node_load1_forecast"])
        if request.url.path == "/api/v1/series":
            match = request.url.params.get("match[]") or ""
            if "node_cpu_busy_pct_forecast" in match:
                return _ok(
                    [
                        # Sink copies.
                        {
                            "__name__": "node_cpu_busy_pct_forecast",
                            "group": "node_capacity",
                            "model": "AutoARIMA",
                            "instance": "node-a:9100",
                        },
                        # Scrape copies — extra ``job`` and changed ``instance``.
                        {
                            "__name__": "node_cpu_busy_pct_forecast",
                            "group": "node_capacity",
                            "model": "AutoARIMA",
                            "instance": "forecaster:9091",
                            "job": "promforecast",
                        },
                    ]
                )
            return _ok(
                [
                    {
                        "__name__": "node_load1_forecast",
                        "group": "node_capacity",
                        "model": "AutoARIMA",
                        "instance": "node-a:9100",
                    }
                ]
            )
        return httpx.Response(404)

    client = _make_transport(handler)
    out = io.StringIO()
    rc = run_diagnose_duplicates(
        datasource_url="http://vm:8428/",
        out=out,
        http_client=client,
    )
    assert rc == 1
    body = out.getvalue()
    assert "node_cpu_busy_pct_forecast" in body
    assert "node_load1_forecast" not in body.split("Remediation:")[0]
    assert "metric_relabel_configs" in body
    assert "/api/v1/admin/tsdb/delete_series" in body
    assert 'job="promforecast"' in body


def test_run_diagnose_ignores_intra_source_label_variance() -> None:
    """Two sink-shaped families differing only by an optional label are not
    a duplicate. Conformal calibration legitimately adds a ``calibrated``
    label to some emitted rows but not others — this is intra-source
    variance, not the dual-source state the detector targets."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/label/__name__/values":
            return _ok(["foo_forecast_upper"])
        if request.url.path == "/api/v1/series":
            return _ok(
                [
                    {"__name__": "foo_forecast_upper", "group": "g", "level": "80"},
                    {
                        "__name__": "foo_forecast_upper",
                        "group": "g",
                        "level": "80",
                        "calibrated": "true",
                    },
                ]
            )
        return httpx.Response(404)

    out = io.StringIO()
    rc = run_diagnose_duplicates(
        datasource_url="http://vm:8428",
        out=out,
        http_client=_make_transport(handler),
    )
    assert rc == 0, out.getvalue()
    assert "no duplicate forecast labelsets detected" in out.getvalue()


def test_run_diagnose_json_output_is_stable_and_machine_parseable() -> None:
    """JSON output is the CI integration point — assert the shape directly."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/label/__name__/values":
            return _ok(["foo_forecast"])
        if request.url.path == "/api/v1/series":
            return _ok(
                [
                    {"__name__": "foo_forecast", "group": "g", "instance": "a"},
                    {"__name__": "foo_forecast", "group": "g", "instance": "b", "job": "x"},
                ]
            )
        return httpx.Response(404)

    out = io.StringIO()
    rc = run_diagnose_duplicates(
        datasource_url="http://vm:8428",
        json_output=True,
        out=out,
        http_client=_make_transport(handler),
    )
    assert rc == 1
    payload = json.loads(out.getvalue())
    assert payload["has_duplicates"] is True
    finding = payload["findings"][0]
    assert finding["metric"] == "foo_forecast"
    assert finding["is_duplicate"] is True
    assert len(finding["families"]) == 2
    assert any(fam["looks_scrape_shaped"] for fam in finding["families"])


# ---------------------------------------------------------------------------
# Filter knobs
# ---------------------------------------------------------------------------


def test_run_diagnose_passes_selector_through_to_tsdb() -> None:
    """``--selector`` is spliced into every per-metric series lookup."""
    observed_matches: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/label/__name__/values":
            return _ok(["foo_forecast"])
        if request.url.path == "/api/v1/series":
            observed_matches.append(request.url.params.get("match[]") or "")
            return _ok([])
        return httpx.Response(404)

    rc = run_diagnose_duplicates(
        datasource_url="http://vm:8428",
        selector='{cluster="prod"}',
        out=io.StringIO(),
        http_client=_make_transport(handler),
    )
    assert rc == 0
    assert observed_matches == ['{__name__="foo_forecast",cluster="prod"}']


def test_run_diagnose_passes_since_window_as_start_end() -> None:
    """``--since 6h`` is converted to start/end query params."""
    observed_params: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/label/__name__/values":
            return _ok(["foo_forecast"])
        if request.url.path == "/api/v1/series":
            observed_params.append(dict(request.url.params))
            return _ok([])
        return httpx.Response(404)

    rc = run_diagnose_duplicates(
        datasource_url="http://vm:8428",
        since_seconds=3600,
        out=io.StringIO(),
        http_client=_make_transport(handler),
    )
    assert rc == 0
    assert "start" in observed_params[0]
    assert "end" in observed_params[0]
    assert int(observed_params[0]["end"]) - int(observed_params[0]["start"]) == 3600


# ---------------------------------------------------------------------------
# Server-side ``match[]`` filter on ``/__name__/values``
# ---------------------------------------------------------------------------


def test_label_values_request_carries_server_side_match_filter() -> None:
    """The label-values request must include the ``match[]`` regex.

    Pushes the forecast-metric filter to the server so a TSDB with tens
    of thousands of metric names returns a kilobyte-class response rather
    than the full catalogue.
    """
    observed_label_values_params: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/label/__name__/values":
            observed_label_values_params.append(dict(request.url.params))
            return _ok(["foo_forecast"])
        if request.url.path == "/api/v1/series":
            return _ok([])
        return httpx.Response(404)

    rc = run_diagnose_duplicates(
        datasource_url="http://vm:8428",
        out=io.StringIO(),
        http_client=_make_transport(handler),
    )
    assert rc == 0
    assert observed_label_values_params, "label-values endpoint was never called"
    match = observed_label_values_params[0].get("match[]") or ""
    assert "__name__" in match and "_forecast" in match


def test_label_values_falls_back_to_client_side_filter_when_server_ignores_match() -> None:
    """If the TSDB ignores ``match[]`` and returns the full catalogue, the
    client-side regex must still keep only forecast metrics."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/label/__name__/values":
            return _ok(
                [
                    "node_cpu_seconds_total",
                    "node_cpu_busy_pct_forecast",
                    "forecast_quality_score",
                    "forecast_failures_total",
                ]
            )
        if request.url.path == "/api/v1/series":
            return _ok([])
        return httpx.Response(404)

    out = io.StringIO()
    rc = run_diagnose_duplicates(
        datasource_url="http://vm:8428",
        out=out,
        http_client=_make_transport(handler),
        json_output=True,
    )
    assert rc == 0
    payload = json.loads(out.getvalue())
    # Only ``node_cpu_busy_pct_forecast`` matches; the operational
    # ``forecast_*`` metrics and the unrelated ``node_cpu_seconds_total``
    # are correctly filtered out by the client-side fallback.
    assert payload["metrics_scanned"] == 1


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_run_diagnose_returns_two_on_unreachable_datasource() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    rc = run_diagnose_duplicates(
        datasource_url="http://nowhere:8428",
        out=io.StringIO(),
        http_client=_make_transport(handler),
    )
    assert rc == 2


@pytest.mark.asyncio
async def test_scan_raises_on_non_success_status() -> None:
    """VM/Prometheus returns ``status: error`` on bad selectors."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "error", "errorType": "bad", "error": "nope"})

    with pytest.raises(DiagnoseError):
        await scan_for_duplicates(
            datasource_url="http://vm:8428",
            selector=None,
            since_seconds=None,
            http_client=_make_transport(handler),
        )


# ---------------------------------------------------------------------------
# Retry / backoff on transient failures
# ---------------------------------------------------------------------------


def test_label_values_retries_on_transient_5xx_and_then_succeeds() -> None:
    """A single 5xx response is smoothed over by the retry layer.

    Matches the behaviour of ``source.py``'s runtime PromQL client: 5xx
    and transport errors retry with exponential backoff up to 3 attempts;
    4xx raises immediately. Without this, a momentary TSDB hiccup during
    a diagnose run flips the exit code to ``2`` even though everything is
    fine seconds later.
    """
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/label/__name__/values":
            attempts["count"] += 1
            if attempts["count"] == 1:
                return httpx.Response(503, text="overloaded")
            return _ok(["foo_forecast"])
        if request.url.path == "/api/v1/series":
            return _ok([])
        return httpx.Response(404)

    rc = run_diagnose_duplicates(
        datasource_url="http://vm:8428",
        out=io.StringIO(),
        http_client=_make_transport(handler),
    )
    assert rc == 0
    # Second attempt is the success; 3 attempts is the cap so anything
    # between 2 and 3 is acceptable here.
    assert attempts["count"] >= 2


def test_4xx_fails_fast_without_retry() -> None:
    """4xx (auth, bad query) is *not* retried — operator must see the real error."""
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/label/__name__/values":
            attempts["count"] += 1
            return httpx.Response(401, text="unauthorized")
        return httpx.Response(404)

    rc = run_diagnose_duplicates(
        datasource_url="http://vm:8428",
        out=io.StringIO(),
        http_client=_make_transport(handler),
    )
    assert rc == 2
    assert attempts["count"] == 1


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_bearer_token_attaches_authorization_header() -> None:
    """``DiagnoseAuth(bearer_token=...)`` propagates as a Bearer header."""
    observed_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_auth.append(request.headers.get("Authorization"))
        if request.url.path == "/api/v1/label/__name__/values":
            return _ok([])
        return httpx.Response(404)

    # Build the auth via the dataclass, materialise the kwargs into the
    # transport-backed client manually so the test exercises exactly the
    # same path production uses (auth → kwargs → AsyncClient).
    auth = DiagnoseAuth(bearer_token="abc123")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), **auth.build_client_kwargs())
    rc = run_diagnose_duplicates(
        datasource_url="http://vm:8428",
        out=io.StringIO(),
        http_client=client,
    )
    assert rc == 0
    assert observed_auth and observed_auth[0] == "Bearer abc123"


def test_basic_auth_attaches_basic_credentials() -> None:
    """``DiagnoseAuth(basic_user=..., basic_password=...)`` propagates as Basic auth."""
    observed_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_auth.append(request.headers.get("Authorization"))
        if request.url.path == "/api/v1/label/__name__/values":
            return _ok([])
        return httpx.Response(404)

    auth = DiagnoseAuth(basic_user="alice", basic_password="s3cret")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), **auth.build_client_kwargs())
    rc = run_diagnose_duplicates(
        datasource_url="http://vm:8428",
        out=io.StringIO(),
        http_client=client,
    )
    assert rc == 0
    assert observed_auth and observed_auth[0] is not None
    assert observed_auth[0].startswith("Basic ")


def test_diagnose_auth_without_credentials_attaches_no_authorization() -> None:
    """The plain-HTTP path must not attach an Authorization header.

    A spurious header would be an information leak to operators expecting
    "no auth means no auth" in the dev stack.
    """
    observed_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_auth.append(request.headers.get("Authorization"))
        if request.url.path == "/api/v1/label/__name__/values":
            return _ok([])
        return httpx.Response(404)

    auth = DiagnoseAuth()
    kwargs = auth.build_client_kwargs()
    assert "auth" not in kwargs
    assert "headers" not in kwargs
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)
    rc = run_diagnose_duplicates(
        datasource_url="http://vm:8428",
        out=io.StringIO(),
        http_client=client,
    )
    assert rc == 0
    assert observed_auth and observed_auth[0] is None


def test_diagnose_auth_tls_knobs_resolve_to_httpx_kwargs() -> None:
    """TLS flags map onto ``cert`` / ``verify`` as httpx expects."""
    auth = DiagnoseAuth(tls_cert="/tmp/c.pem", tls_key="/tmp/k.pem", tls_ca="/tmp/ca.pem")
    kwargs = auth.build_client_kwargs()
    assert kwargs["cert"] == ("/tmp/c.pem", "/tmp/k.pem")
    assert kwargs["verify"] == "/tmp/ca.pem"

    skip = DiagnoseAuth(insecure_skip_verify=True, tls_ca="/tmp/ca.pem")
    # ``insecure_skip_verify`` takes precedence inside the dataclass; the
    # CLI layer enforces mutual exclusion, but the dataclass picks the
    # safer behaviour ("skip" wins over a partial CA-only setup).
    assert skip.build_client_kwargs()["verify"] is False


# ---------------------------------------------------------------------------
# Parallel fan-out
# ---------------------------------------------------------------------------


def test_per_metric_probes_run_in_parallel() -> None:
    """Many forecast metrics → many concurrent series probes.

    Implementation detail check: each metric's ``/api/v1/series`` probe
    runs under a shared semaphore. Concretely the assertion is that all
    expected metrics were probed in a single scan — the parallelism
    itself is hard to assert deterministically without timing.
    """
    metric_names = [f"m{i}_forecast" for i in range(12)]
    seen: set[str] = set()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/label/__name__/values":
            return _ok(metric_names)
        if request.url.path == "/api/v1/series":
            match = request.url.params.get("match[]") or ""
            for name in metric_names:
                if name in match:
                    seen.add(name)
                    break
            return _ok([])
        return httpx.Response(404)

    rc = run_diagnose_duplicates(
        datasource_url="http://vm:8428",
        out=io.StringIO(),
        http_client=_make_transport(handler),
        max_concurrent=4,
    )
    assert rc == 0
    assert seen == set(metric_names)


# ---------------------------------------------------------------------------
# Sample-rate subcommand
# ---------------------------------------------------------------------------


def test_sample_rate_clean_install_returns_zero() -> None:
    """Series count below the cap → no anomalies → exit code 0."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/label/__name__/values":
            return _ok(["foo_forecast"])
        if request.url.path == "/api/v1/query":
            return _vector(
                [
                    {
                        "metric": {"__name__": "foo_forecast", "group": "g", "instance": "a"},
                        "value": [1700000000, "2"],
                    }
                ]
            )
        return httpx.Response(404)

    out = io.StringIO()
    rc = run_diagnose_sample_rate(
        datasource_url="http://vm:8428",
        window_seconds=600,
        max_samples_per_window=5,
        out=out,
        http_client=_make_transport(handler),
    )
    assert rc == 0
    assert "no series exceeded" in out.getvalue()


def test_sample_rate_above_cap_returns_one_and_names_series() -> None:
    """Series count above the cap → flagged → exit code 1."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/label/__name__/values":
            return _ok(["foo_forecast"])
        if request.url.path == "/api/v1/query":
            return _vector(
                [
                    {
                        "metric": {"__name__": "foo_forecast", "group": "g", "instance": "a"},
                        "value": [1700000000, "40"],  # well above cap=5
                    },
                    {
                        "metric": {"__name__": "foo_forecast", "group": "g", "instance": "b"},
                        "value": [1700000000, "2"],  # below cap, not flagged
                    },
                ]
            )
        return httpx.Response(404)

    out = io.StringIO()
    rc = run_diagnose_sample_rate(
        datasource_url="http://vm:8428",
        window_seconds=600,
        max_samples_per_window=5,
        out=out,
        http_client=_make_transport(handler),
    )
    assert rc == 1
    body = out.getvalue()
    assert "foo_forecast" in body
    assert "instance" in body and "'a'" in body
    # The clean series must not be named in the anomaly listing.
    assert "'b'" not in body.split("Likely cause")[0]


def test_sample_rate_json_output_has_stable_shape() -> None:
    """JSON shape is the CI integration point; assert key fields exist."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/label/__name__/values":
            return _ok(["foo_forecast"])
        if request.url.path == "/api/v1/query":
            return _vector(
                [
                    {
                        "metric": {"__name__": "foo_forecast", "group": "g"},
                        "value": [1700000000, "99"],
                    }
                ]
            )
        return httpx.Response(404)

    out = io.StringIO()
    rc = run_diagnose_sample_rate(
        datasource_url="http://vm:8428",
        window_seconds=600,
        max_samples_per_window=5,
        json_output=True,
        out=out,
        http_client=_make_transport(handler),
    )
    assert rc == 1
    payload = json.loads(out.getvalue())
    assert payload["has_anomalies"] is True
    assert payload["window_seconds"] == 600
    assert payload["max_samples_per_window"] == 5
    assert payload["anomalies"][0]["metric"] == "foo_forecast"
    assert payload["anomalies"][0]["samples_in_window"] == 99


def test_sample_rate_query_uses_count_over_time_with_window() -> None:
    """The probe must use ``count_over_time(<metric>[<window>])``."""
    observed_queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/label/__name__/values":
            return _ok(["foo_forecast"])
        if request.url.path == "/api/v1/query":
            observed_queries.append(request.url.params.get("query") or "")
            return _vector([])
        return httpx.Response(404)

    rc = run_diagnose_sample_rate(
        datasource_url="http://vm:8428",
        window_seconds=600,
        max_samples_per_window=5,
        out=io.StringIO(),
        http_client=_make_transport(handler),
    )
    assert rc == 0
    assert observed_queries == ['count_over_time({__name__="foo_forecast"}[600s])']


def test_sample_rate_splices_user_selector() -> None:
    """``--selector '{cluster="prod"}'`` is forwarded into the PromQL query."""
    observed_queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/label/__name__/values":
            return _ok(["foo_forecast"])
        if request.url.path == "/api/v1/query":
            observed_queries.append(request.url.params.get("query") or "")
            return _vector([])
        return httpx.Response(404)

    rc = run_diagnose_sample_rate(
        datasource_url="http://vm:8428",
        window_seconds=600,
        max_samples_per_window=5,
        selector='{cluster="prod"}',
        out=io.StringIO(),
        http_client=_make_transport(handler),
    )
    assert rc == 0
    assert observed_queries == ['count_over_time({__name__="foo_forecast",cluster="prod"}[600s])']


def test_sample_rate_returns_two_on_unreachable_datasource() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    rc = run_diagnose_sample_rate(
        datasource_url="http://nowhere:8428",
        window_seconds=600,
        max_samples_per_window=5,
        out=io.StringIO(),
        http_client=_make_transport(handler),
    )
    assert rc == 2


@pytest.mark.asyncio
async def test_sample_rate_raises_on_non_vector_result_type() -> None:
    """``count_over_time`` must return a vector; a matrix/scalar means the
    query was malformed in a way the operator should see."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/label/__name__/values":
            return _ok(["foo_forecast"])
        if request.url.path == "/api/v1/query":
            return httpx.Response(
                200,
                json={"status": "success", "data": {"resultType": "scalar", "result": [1, "1"]}},
            )
        return httpx.Response(404)

    with pytest.raises(DiagnoseError):
        await scan_for_sample_rate(
            datasource_url="http://vm:8428",
            window_seconds=600,
            max_samples_per_window=5,
            selector=None,
            http_client=_make_transport(handler),
        )


# ---------------------------------------------------------------------------
# CLI resolver helpers (``main._resolve_diagnose_*``)
# ---------------------------------------------------------------------------
#
# These exercise the argparse → dataclass adapters that live in ``main.py``
# alongside ``cli()``. They're CLI glue rather than diagnose-internal code,
# but they belong here so all the diagnose-subcommand surface area is tested
# in one file.


def _make_args(**overrides: Any) -> Any:
    """Build a fake ``argparse.Namespace`` carrying every diagnose CLI flag.

    Defaults match what argparse would produce when no flag is set; tests
    override only the fields they care about. Keeping the helper local
    avoids importing argparse in every test that needs one of these.
    """
    import argparse  # noqa: PLC0415

    defaults = {
        "datasource": None,
        "config": None,
        "bearer_token": None,
        "basic_user": None,
        "basic_password": None,
        "tls_cert": None,
        "tls_key": None,
        "tls_ca": None,
        "tls_insecure_skip_verify": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_resolve_datasource_explicit_flag_wins(tmp_path: Any) -> None:
    """``--datasource`` overrides ``--config`` when both are given."""
    from pathlib import Path  # noqa: PLC0415

    from promforecast.main import _resolve_diagnose_datasource  # noqa: PLC0415

    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        "apiVersion: promforecast.io/v1\n"
        "datasource:\n"
        "  url: http://config-source:8428/\n"
        "server:\n"
        "  refresh_interval: 1h\n"
        "defaults:\n"
        "  models: [SeasonalNaive]\n"
        "groups: []\n"
    )
    args = _make_args(datasource="http://explicit:8428", config=Path(cfg_yaml))
    assert _resolve_diagnose_datasource(args) == "http://explicit:8428"


def test_resolve_datasource_inherits_from_config(tmp_path: Any) -> None:
    """``--config`` alone resolves to the YAML's ``datasource.url``."""
    from pathlib import Path  # noqa: PLC0415

    from promforecast.main import _resolve_diagnose_datasource  # noqa: PLC0415

    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        "apiVersion: promforecast.io/v1\n"
        "datasource:\n"
        "  url: http://from-config:8428/\n"
        "server:\n"
        "  refresh_interval: 1h\n"
        "defaults:\n"
        "  models: [SeasonalNaive]\n"
        "groups: []\n"
    )
    args = _make_args(config=Path(cfg_yaml))
    assert _resolve_diagnose_datasource(args) == "http://from-config:8428/"


def test_resolve_datasource_fails_when_neither_provided() -> None:
    """Neither ``--datasource`` nor ``--config`` → SystemExit with a clear error."""
    from promforecast.main import _resolve_diagnose_datasource  # noqa: PLC0415

    args = _make_args()
    with pytest.raises(SystemExit) as info:
        _resolve_diagnose_datasource(args)
    assert "one of --datasource or --config" in str(info.value)


def test_resolve_datasource_fails_with_targeted_error_on_bad_config(tmp_path: Any) -> None:
    """Broken YAML at ``--config`` → SystemExit with the path in the error.

    The operator must see *which* file failed; a bare ``ValidationError``
    from pydantic with no file path is the kind of UX the migration doc
    warns against.
    """
    from pathlib import Path  # noqa: PLC0415

    from promforecast.main import _resolve_diagnose_datasource  # noqa: PLC0415

    cfg_yaml = tmp_path / "broken.yaml"
    cfg_yaml.write_text("not: a: valid: yaml: at: all: :\n")
    args = _make_args(config=Path(cfg_yaml))
    with pytest.raises(SystemExit) as info:
        _resolve_diagnose_datasource(args)
    assert "failed to load --config" in str(info.value)
    assert "broken.yaml" in str(info.value)


def test_resolve_auth_picks_up_bearer_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``$PROMFORECAST_BEARER_TOKEN`` populates the bearer field when the flag is omitted."""
    from promforecast.main import _resolve_diagnose_auth  # noqa: PLC0415

    monkeypatch.setenv("PROMFORECAST_BEARER_TOKEN", "from-env")
    auth = _resolve_diagnose_auth(_make_args())
    assert auth.bearer_token == "from-env"


def test_resolve_auth_cli_flag_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit ``--bearer-token`` wins over the env var.

    The CLI flag is the per-invocation override; the env var is the
    operator's persistent default. Without this precedence a one-off
    invocation with a different token would need an ``unset`` first.
    """
    from promforecast.main import _resolve_diagnose_auth  # noqa: PLC0415

    monkeypatch.setenv("PROMFORECAST_BEARER_TOKEN", "from-env")
    auth = _resolve_diagnose_auth(_make_args(bearer_token="from-flag"))
    assert auth.bearer_token == "from-flag"


def test_resolve_auth_picks_up_basic_password_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``$PROMFORECAST_BASIC_PASSWORD`` populates the basic-auth password."""
    from promforecast.main import _resolve_diagnose_auth  # noqa: PLC0415

    monkeypatch.setenv("PROMFORECAST_BASIC_PASSWORD", "env-pass")
    auth = _resolve_diagnose_auth(_make_args(basic_user="alice"))
    assert auth.basic_user == "alice"
    assert auth.basic_password == "env-pass"


def test_resolve_auth_rejects_tls_ca_with_skip_verify() -> None:
    """``--tls-ca`` + ``--tls-insecure-skip-verify`` → SystemExit.

    Combining "use this CA bundle" with "don't verify TLS at all" is a
    contradiction the operator should see at argument parse time, not via
    a silent precedence rule inside the auth dataclass.
    """
    from pathlib import Path  # noqa: PLC0415

    from promforecast.main import _resolve_diagnose_auth  # noqa: PLC0415

    args = _make_args(tls_ca=Path("/tmp/ca.pem"), tls_insecure_skip_verify=True)
    with pytest.raises(SystemExit) as info:
        _resolve_diagnose_auth(args)
    assert "--tls-ca" in str(info.value)
    assert "--tls-insecure-skip-verify" in str(info.value)


def test_resolve_auth_rejects_tls_cert_without_tls_key() -> None:
    """``--tls-cert`` without ``--tls-key`` → SystemExit.

    httpx silently accepts a cert-only call and then fails at handshake
    time with an obscure error. Catching the half-configuration at parse
    time keeps the operator from chasing a TLS error that's really a
    missing flag.
    """
    from pathlib import Path  # noqa: PLC0415

    from promforecast.main import _resolve_diagnose_auth  # noqa: PLC0415

    args = _make_args(tls_cert=Path("/tmp/cert.pem"))
    with pytest.raises(SystemExit) as info:
        _resolve_diagnose_auth(args)
    assert "--tls-cert and --tls-key" in str(info.value)


def test_resolve_auth_rejects_tls_key_without_tls_cert() -> None:
    """Symmetric case: key without cert is also rejected."""
    from pathlib import Path  # noqa: PLC0415

    from promforecast.main import _resolve_diagnose_auth  # noqa: PLC0415

    args = _make_args(tls_key=Path("/tmp/key.pem"))
    with pytest.raises(SystemExit) as info:
        _resolve_diagnose_auth(args)
    assert "--tls-cert and --tls-key" in str(info.value)


def test_resolve_auth_returns_empty_bundle_when_no_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No flags + no env vars → all fields default to ``None`` / ``False``.

    The plain-HTTP dev-stack path must not synthesise any credentials
    when the operator hasn't asked for them. The env-var fallbacks are
    explicitly cleared so the test isn't affected by the developer's
    shell environment.
    """
    from promforecast.main import _resolve_diagnose_auth  # noqa: PLC0415

    monkeypatch.delenv("PROMFORECAST_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("PROMFORECAST_BASIC_PASSWORD", raising=False)
    auth = _resolve_diagnose_auth(_make_args())
    assert auth.bearer_token is None
    assert auth.basic_user is None
    assert auth.basic_password is None
    assert auth.tls_cert is None
    assert auth.tls_key is None
    assert auth.tls_ca is None
    assert auth.insecure_skip_verify is False
