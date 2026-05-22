"""Entrypoint: parses config, starts the HTTP server and the scheduler."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from . import config as config_module
from . import ha_cache as ha_cache_module
from . import preview as preview_module
from . import telemetry as telemetry_module
from . import validate as validate_module
from .cache import MemoryQueryCache, QueryCache, QueryCacheBackend, RedisQueryCache
from .exporter import Exporter
from .leader import KubernetesLeaseClient, LeaderElector, LeaderElectorConfig, default_identity
from .reload import ConfigMapWatcher, ConfigReloader, ReloadError
from .runner import Runner
from .sink import ForecastSink, RemoteWriteSink
from .source import PromSource

logger = structlog.get_logger(__name__)


def cli() -> None:
    """Console-script entrypoint.

    Two subcommands today: ``run`` (the default — boots the forecaster) and
    ``validate`` (CI/dev sanity check, prints a cardinality estimate and an
    optional sample fit). ``run`` is implicit when no subcommand is given so
    the existing container ``args: [--config=...]`` invocation keeps working
    without the chart having to learn about subcommands.
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "validate":
        rc = validate_module.run_validate(
            config_path=args.config,
            dry_run=args.dry_run,
            datasource_url_override=args.datasource_url,
        )
        sys.exit(rc)

    run(config_path=args.config, log_level=args.log_level)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="promforecast")
    sub = parser.add_subparsers(dest="command")

    # The default ``run`` arguments live on the top-level parser as well as
    # on the ``run`` subparser so old invocations like
    # ``promforecast --config foo.yaml`` keep working unchanged.
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/etc/promforecast/config.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )

    run_p = sub.add_parser("run", help="Run the forecaster (default).")
    run_p.add_argument("--config", type=Path, default=Path("/etc/promforecast/config.yaml"))
    run_p.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )

    validate_p = sub.add_parser(
        "validate",
        help="Validate a config file without starting the server.",
    )
    validate_p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file.",
    )
    validate_p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run a sample query + fit against the configured datasource and "
            "print the first forecast points and per-model MAPE preview."
        ),
    )
    validate_p.add_argument(
        "--datasource-url",
        default=None,
        help=(
            "Override datasource.url for the dry-run sample query. Useful "
            "when the production URL is unreachable from CI."
        ),
    )
    return parser


def run(config_path: Path, log_level: str) -> None:
    """Boot the forecaster.

    Parses config, validates metric IDs, builds the model registry, acquires
    the leader lease (HA only), starts FastAPI, and schedules per-group runs.
    """
    _configure_logging(log_level)
    cfg = config_module.load(config_path)
    logger.info("config_loaded", path=str(config_path), groups=len(cfg.groups))

    exporter = Exporter()
    # Snapshot the SHA-256 of the bytes we just loaded so /metrics carries
    # the running config's identity from the very first scrape. The reloader
    # refreshes this whenever a new config is applied; no other code path
    # writes the hash.
    exporter.set_config_hash(_compute_config_hash(config_path))

    # Shared Redis client: built once, shared between the snapshot cache
    # (HA mode) and the query cache (when backend resolves to Redis). When
    # neither feature wants it the client stays at ``None`` so import-time
    # cost of ``redis`` is paid only when actually needed.
    redis_client = _build_redis_client(cfg)
    snapshot_cache = _build_snapshot_cache(cfg, redis_client)
    query_cache = _build_query_cache(cfg, exporter, redis_client)

    source = PromSource(
        url=cfg.datasource.url,
        timeout_seconds=cfg.datasource.timeout.total_seconds(),
        max_concurrency=cfg.datasource.max_concurrent_requests,
        max_keepalive_connections=cfg.datasource.max_keepalive_connections,
        on_retry=exporter.increment_source_retry,
        cache=query_cache,
    )
    sink = _build_sink(cfg)
    # Build the tracer *before* the runner so spans cover the very first
    # group run. The disabled tracer is free, the real one carries SDK
    # init cost only when ``telemetry.enabled`` is true.
    tracer = telemetry_module.build_tracer(cfg.telemetry)
    runner = Runner(config=cfg, source=source, exporter=exporter, sink=sink, tracer=tracer)
    if snapshot_cache is not None:
        # Wire the leader's per-group snapshot hook so a successful run
        # publishes the rendered /metrics payload to Redis. Followers
        # read this on every scrape (see ``/metrics`` handler below).
        runner.set_snapshot_publisher(_make_snapshot_publisher(exporter, snapshot_cache))
    reloader = ConfigReloader(runner=runner, exporter=exporter, config_path=config_path)
    watcher = (
        ConfigMapWatcher(reloader)
        if cfg.server.reload.enabled and cfg.server.reload.watch_configmap
        else None
    )

    elector = _build_elector(cfg, exporter, runner)
    if elector is None:
        # No HA configured — record this replica as a permanent "leader"
        # so the metric is meaningful in single-replica deployments.
        exporter.set_leader_state(identity=default_identity(), is_leader=True)

    app = _build_app(
        exporter,
        runner,
        reloader,
        watcher=watcher,
        reload_enabled=cfg.server.reload.enabled,
        expose_config_endpoint=cfg.server.expose_config_endpoint,
        expose_preview_endpoint=cfg.server.expose_preview_endpoint,
        source=source,
        snapshot_cache=snapshot_cache,
        elector=elector,
        tracer=tracer,
    )
    host, port = _parse_listen(cfg.server.listen)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level=log_level,
            lifespan="on",
            # Disable uvicorn's default logging config so records flow through
            # our structlog-formatted root handler (set up in _configure_logging).
            log_config=None,
        )
    )
    asyncio.run(_serve(server, reloader=reloader, redis_client=redis_client))


@asynccontextmanager
async def _lifespan_factory(
    runner: Runner,
    watcher: ConfigMapWatcher | None,
    elector: LeaderElector | None,
    tracer: telemetry_module.Tracer | None = None,
) -> AsyncIterator[None]:
    """Start lifecycle, gated on leader election when HA mode is active.

    Without HA: behaves exactly as before — the runner starts immediately
    and stops on shutdown.

    With HA: the elector owns the runner. It runs in a background task and
    calls ``runner.start()`` / ``runner.stop()`` via callbacks set during
    construction; the lifespan only waits on the elector and the watcher.

    The OTel tracer (when enabled) is shut down *last* so any spans emitted
    by the runner's final stop sequence still have a chance to flush via
    the BatchSpanProcessor before the gRPC channel is torn down.
    """
    elector_task: asyncio.Task[None] | None = None
    if elector is None:
        await runner.start()
    else:
        elector_task = asyncio.create_task(elector.run())
    if watcher is not None:
        await watcher.start()
    try:
        yield
    finally:
        if watcher is not None:
            await watcher.stop()
        if elector is not None:
            await elector.stop()
            if elector_task is not None:
                import contextlib  # noqa: PLC0415

                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await elector_task
        await runner.stop()
        if tracer is not None:
            tracer.shutdown()


def _build_app(
    exporter: Exporter,
    runner: Runner,
    reloader: ConfigReloader,
    *,
    watcher: ConfigMapWatcher | None = None,
    reload_enabled: bool,
    expose_config_endpoint: bool = False,
    expose_preview_endpoint: bool = False,
    source: PromSource | None = None,
    snapshot_cache: ha_cache_module.SnapshotCache | None = None,
    elector: LeaderElector | None = None,
    tracer: telemetry_module.Tracer | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with _lifespan_factory(runner, watcher, elector, tracer=tracer):
            yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        # Surface the running config's identity on /healthz too: a quick
        # ``curl -s /healthz | jq .config_hash`` confirms the deployment is
        # running the expected revision without scraping /metrics first.
        timestamp, success, _ = exporter._reload_state()
        identity, is_leader, _ = exporter._leader_state()
        body: dict[str, object] = {
            "status": "ok",
            "config_hash": exporter._config_hash_state(),
            "config_last_reload_timestamp": timestamp,
            "config_last_reload_success": success,
        }
        if identity:
            body["identity"] = identity
            body["is_leader"] = is_leader
        return body

    @app.get("/metrics")
    async def metrics() -> Response:
        # HA followers serve the leader's cached snapshot. We always try
        # the local registry first — if the cache misses (Redis blip,
        # leader hasn't run yet) the local empty render is a better
        # signal than a 404 / 500.
        if snapshot_cache is not None and elector is not None and not elector.is_leader():
            cached = await snapshot_cache.get()
            if cached is not None:
                return Response(
                    content=cached,
                    media_type="text/plain; version=0.0.4; charset=utf-8",
                )
        return Response(
            content=exporter.render(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    if reload_enabled:

        @app.post("/-/reload")
        async def reload_endpoint() -> Response:
            """Trigger a validate-before-swap config reload.

            Mirrors Prometheus' own ``/-/reload`` for muscle memory. Returns
            ``200`` with the diff on success, ``400`` (config error) on
            validation failure with the running config left untouched.
            """
            try:
                outcome = await reloader.reload()
            except ReloadError as exc:
                return PlainTextResponse(str(exc), status_code=400)
            return JSONResponse(
                {
                    "status": "ok",
                    "added": outcome.result.added,
                    "removed": outcome.result.removed,
                    "changed": outcome.result.changed,
                    "timestamp": outcome.timestamp,
                }
            )

    if expose_config_endpoint:

        @app.get("/config")
        async def config_endpoint() -> Response:
            """Return the running, sanitised config plus its content hash.

            Off by default; opt-in via ``server.expose_config_endpoint`` so
            an operator must explicitly accept exposing the parsed config
            (with credentials redacted) on the unauthenticated metrics
            port. Useful for debugging "is the deployment actually running
            what I think it is?" without shelling into the pod.
            """
            return JSONResponse(
                {
                    "config_hash": exporter._config_hash_state(),
                    "config": config_module.sanitised_dump(runner._config),
                }
            )

    if expose_preview_endpoint and source is None:
        # Configured to expose the endpoint but the caller didn't wire a
        # source (typical for unit tests or partial deployments). The
        # endpoint can't function without one, so log a clear warning
        # instead of silently registering nothing.
        logger.warning(
            "preview_endpoint_disabled_no_source",
            reason="server.expose_preview_endpoint=true but no datasource was wired",
        )
    if expose_preview_endpoint and source is not None:
        from fastapi import Request  # noqa: PLC0415

        @app.post("/forecast/preview")
        async def preview_endpoint(req: Request) -> Response:
            """Run a single ad-hoc fit for the supplied query and return JSON.

            Stateless: nothing about this request touches ``/metrics``,
            the snapshot, or scheduled fits. Same network-level
            access-control model as ``/-/reload``. See ``docs/preview.md``
            for the request schema + worked examples.
            """
            try:
                payload = await req.json()
            except Exception as exc:
                return JSONResponse(
                    {"error": "invalid_body", "message": f"bad JSON: {exc}"},
                    status_code=400,
                )
            try:
                request = preview_module.parse_request(payload)
            except preview_module.PreviewError as exc:
                return JSONResponse(
                    {"error": exc.reason, "message": str(exc)},
                    status_code=exc.status_code,
                )
            try:
                result = await preview_module.run_preview(
                    config=runner._config,
                    source=source,
                    request=request,
                    served_by=default_identity(),
                )
            except preview_module.PreviewError as exc:
                return JSONResponse(
                    {"error": exc.reason, "message": str(exc)},
                    status_code=exc.status_code,
                )
            return JSONResponse(result)

    return app


def _compute_config_hash(path: Path) -> str:
    """SHA-256 of the raw config file bytes, hex-encoded.

    Returns an empty string if the file disappears between load and hash
    (a tight race that the standard load already guards against). The
    same digest function is used by the ConfigMap watcher; both stay in
    lockstep so the metric agrees with what the watcher sees on disk.
    """
    import hashlib  # noqa: PLC0415

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return ""


async def _serve(
    server: uvicorn.Server,
    *,
    reloader: ConfigReloader,
    redis_client: object | None = None,
) -> None:
    loop = asyncio.get_running_loop()

    def _shutdown() -> None:
        server.should_exit = True

    # Track in-flight SIGHUP reloads so the loop holds a strong reference;
    # asyncio drops weak references to fire-and-forget tasks, which would
    # let them be garbage-collected mid-reload.
    sighup_tasks: set[asyncio.Task[None]] = set()

    def _handle_sighup() -> None:
        # SIGHUP is the classic Unix "reload your config" signal. Schedule
        # the async reload from the signal handler — exceptions are swallowed
        # by the reloader's error path so the signal never crashes the loop.
        logger.info("sighup_received")
        task = asyncio.create_task(_safe_reload(reloader))
        sighup_tasks.add(task)
        task.add_done_callback(sighup_tasks.discard)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)
    loop.add_signal_handler(signal.SIGHUP, _handle_sighup)
    try:
        await server.serve()
    finally:
        # Close the shared Redis client once everything else has shut down.
        # The snapshot and query caches both hold references to it, so we
        # can't close earlier without breaking in-flight cache ops.
        if redis_client is not None:
            await ha_cache_module.close_redis_client(redis_client)


async def _safe_reload(reloader: ConfigReloader) -> None:
    # Reload errors are already logged + counted by the reloader. Don't
    # propagate — a signal-driven reload should never tear down the loop.
    import contextlib  # noqa: PLC0415

    with contextlib.suppress(ReloadError):
        await reloader.reload()


def _configure_logging(level: str) -> None:
    """Pipe uvicorn / FastAPI / library logs through structlog so all output
    is consistent JSON. ``foreign_pre_chain`` decorates records that didn't
    originate in structlog (e.g. uvicorn's stdlib loggers)."""
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    # uvicorn pre-installs handlers on its own loggers; clear them so records
    # propagate to root (and through our formatter) instead of double-printing.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


def _redis_needed(cfg: config_module.Config) -> bool:
    """Does this config want a shared Redis client?"""
    if cfg.high_availability.enabled:
        return True
    qc = cfg.query_cache
    if not qc.enabled:
        return False
    return qc.backend == "redis"


def _build_redis_client(cfg: config_module.Config) -> object | None:
    """Construct the shared async Redis client, or ``None`` if not needed.

    Both the HA snapshot cache and the (Redis-backed) query cache reuse
    this client so a deployment that opts into both pays for exactly one
    connection pool.
    """
    if not _redis_needed(cfg):
        return None
    address = cfg.high_availability.redis.address or cfg.query_cache.redis_address
    if not address:
        raise ValueError(
            "redis backend requested but no address configured; "
            "set highAvailability.redis.address or query_cache.redis_address"
        )
    password = cfg.high_availability.redis.password or cfg.query_cache.redis_password or None
    return ha_cache_module.build_redis_client(address, password=password)


def _build_snapshot_cache(
    cfg: config_module.Config, redis_client: object | None
) -> ha_cache_module.SnapshotCache | None:
    """Redis-backed shared snapshot cache for HA followers, or ``None``."""
    if not cfg.high_availability.enabled or redis_client is None:
        return None
    ttl_seconds = max(1, int(cfg.high_availability.redis.snapshot_ttl.total_seconds()))
    return ha_cache_module.SnapshotCache(redis_client, ttl_seconds=ttl_seconds)


def _build_query_cache(
    cfg: config_module.Config,
    exporter: Exporter,
    redis_client: object | None,
) -> QueryCache | None:
    """Resolve the query-cache backend and return a wired :class:`QueryCache`.

    Returns ``None`` when caching is disabled. Backend resolution:

    * ``backend: auto`` — Redis if HA is enabled, otherwise in-memory LRU.
    * ``backend: memory`` — always in-memory LRU.
    * ``backend: redis`` — use the shared client; requires one to be
      configured (HA enabled, or ``query_cache.redis_address`` set).
    """
    qc = cfg.query_cache
    if not qc.enabled:
        return None
    backend: QueryCacheBackend
    chosen = qc.backend
    if chosen == "auto":
        chosen = "redis" if redis_client is not None else "memory"
    if chosen == "memory":
        backend = MemoryQueryCache(max_entries=qc.max_entries)
    else:
        if redis_client is None:
            raise ValueError(
                "query_cache.backend=redis requires either highAvailability.enabled or "
                "query_cache.redis_address to be set"
            )
        backend = RedisQueryCache(redis_client)
    ttl_seconds = max(1, int(qc.ttl.total_seconds()))
    return QueryCache(
        backend,
        ttl_seconds=ttl_seconds,
        on_event=exporter.increment_query_cache_event,
    )


def _make_snapshot_publisher(
    exporter: Exporter,
    cache: ha_cache_module.SnapshotCache,
) -> Callable[[], Awaitable[None]]:
    """Build the per-run callback that publishes the rendered snapshot."""

    async def _publish() -> None:
        rendered = exporter.render()
        await cache.set(rendered)

    return _publish


def _build_elector(
    cfg: config_module.Config,
    exporter: Exporter,
    runner: Runner,
) -> LeaderElector | None:
    """Construct the leader elector for HA mode, or ``None`` when disabled.

    Wires the on-acquired / on-lost callbacks so the runner only schedules
    work while this replica holds the lease. Sets the exporter's leader
    state on every transition so ``forecast_leader_elected`` is accurate.
    """
    ha = cfg.high_availability
    if not ha.enabled:
        return None
    namespace = ha.namespace or os.environ.get("PROMFORECAST_NAMESPACE", "default")
    identity = ha.identity or default_identity()
    exporter.set_leader_state(identity=identity, is_leader=False)

    async def _on_acquired() -> None:
        exporter.set_leader_state(identity=identity, is_leader=True)
        await runner.start()

    async def _on_lost() -> None:
        exporter.set_leader_state(identity=identity, is_leader=False)
        await runner.pause()

    elector_config = LeaderElectorConfig(
        namespace=namespace,
        lease_name=ha.lease_name,
        identity=identity,
        lease_duration_seconds=ha.lease_duration_seconds,
        renew_deadline_seconds=ha.renew_deadline_seconds,
        retry_period_seconds=ha.retry_period_seconds,
    )
    return LeaderElector(
        config=elector_config,
        client=KubernetesLeaseClient(),
        on_started_leading=_on_acquired,
        on_stopped_leading=_on_lost,
    )


def _build_sink(cfg: config_module.Config) -> ForecastSink | None:
    """Construct the configured push sink, or ``None`` if disabled.

    Currently only ``remote_write`` is wired up; if more sinks are added
    later this is the dispatch point.
    """
    rw = cfg.sink.remote_write
    if not rw.enabled:
        return None
    logger.info(
        "remote_write_sink_enabled",
        url=rw.url,
        timeout_s=rw.timeout.total_seconds(),
    )
    return RemoteWriteSink(
        url=rw.url,
        timeout_seconds=rw.timeout.total_seconds(),
        headers=rw.headers,
        chunk_size=rw.chunk_size,
        max_connections=rw.max_connections,
        max_keepalive_connections=rw.max_keepalive_connections,
    )


def _parse_listen(listen: str) -> tuple[str, int]:
    """Accept ``:9091`` or ``host:9091``."""
    if listen.startswith(":"):
        return "0.0.0.0", int(listen[1:])
    host, _, port = listen.rpartition(":")
    return host or "0.0.0.0", int(port)


if __name__ == "__main__":
    cli()
