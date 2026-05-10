"""Entrypoint: parses config, starts the HTTP server and the scheduler."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from . import config as config_module
from . import validate as validate_module
from .exporter import Exporter
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
    source = PromSource(
        url=cfg.datasource.url,
        timeout_seconds=cfg.datasource.timeout.total_seconds(),
    )
    sink = _build_sink(cfg)
    runner = Runner(config=cfg, source=source, exporter=exporter, sink=sink)
    reloader = ConfigReloader(runner=runner, exporter=exporter, config_path=config_path)
    watcher = (
        ConfigMapWatcher(reloader)
        if cfg.server.reload.enabled and cfg.server.reload.watch_configmap
        else None
    )

    app = _build_app(
        exporter,
        runner,
        reloader,
        watcher=watcher,
        reload_enabled=cfg.server.reload.enabled,
        expose_config_endpoint=cfg.server.expose_config_endpoint,
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
    asyncio.run(_serve(server, reloader=reloader))


@asynccontextmanager
async def _lifespan_factory(
    runner: Runner,
    watcher: ConfigMapWatcher | None,
) -> AsyncIterator[None]:
    await runner.start()
    if watcher is not None:
        await watcher.start()
    try:
        yield
    finally:
        if watcher is not None:
            await watcher.stop()
        await runner.stop()


def _build_app(
    exporter: Exporter,
    runner: Runner,
    reloader: ConfigReloader,
    *,
    watcher: ConfigMapWatcher | None = None,
    reload_enabled: bool,
    expose_config_endpoint: bool = False,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with _lifespan_factory(runner, watcher):
            yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        # Surface the running config's identity on /healthz too: a quick
        # ``curl -s /healthz | jq .config_hash`` confirms the deployment is
        # running the expected revision without scraping /metrics first.
        timestamp, success, _ = exporter._reload_state()
        return {
            "status": "ok",
            "config_hash": exporter._config_hash_state(),
            "config_last_reload_timestamp": timestamp,
            "config_last_reload_success": success,
        }

    @app.get("/metrics")
    async def metrics() -> Response:
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
    await server.serve()


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
    )


def _parse_listen(listen: str) -> tuple[str, int]:
    """Accept ``:9091`` or ``host:9091``."""
    if listen.startswith(":"):
        return "0.0.0.0", int(listen[1:])
    host, _, port = listen.rpartition(":")
    return host or "0.0.0.0", int(port)


if __name__ == "__main__":
    cli()
