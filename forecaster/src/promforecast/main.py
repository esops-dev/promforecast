"""Entrypoint: parses config, starts the HTTP server and the scheduler."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI, Response

from . import config as config_module
from .exporter import Exporter
from .runner import Runner
from .source import PromSource

logger = structlog.get_logger(__name__)


def cli() -> None:
    """Console-script entrypoint."""
    parser = argparse.ArgumentParser(prog="promforecast")
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
    args = parser.parse_args()
    run(config_path=args.config, log_level=args.log_level)


def run(config_path: Path, log_level: str) -> None:
    """Boot the forecaster.

    Parses config, validates metric IDs, builds the model registry, acquires
    the leader lease (HA only), starts FastAPI, and schedules per-group runs.
    """
    _configure_logging(log_level)
    cfg = config_module.load(config_path)
    logger.info("config_loaded", path=str(config_path), groups=len(cfg.groups))

    exporter = Exporter()
    source = PromSource(
        url=cfg.datasource.url,
        timeout_seconds=cfg.datasource.timeout.total_seconds(),
    )
    runner = Runner(config=cfg, source=source, exporter=exporter)

    app = _build_app(exporter, runner)
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
    asyncio.run(_serve(server))


@asynccontextmanager
async def _lifespan_factory(
    runner: Runner,
) -> AsyncIterator[None]:
    await runner.start()
    try:
        yield
    finally:
        await runner.stop()


def _build_app(exporter: Exporter, runner: Runner) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with _lifespan_factory(runner):
            yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(
            content=exporter.render(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return app


async def _serve(server: uvicorn.Server) -> None:
    loop = asyncio.get_running_loop()

    def _shutdown() -> None:
        server.should_exit = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)
    await server.serve()


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


def _parse_listen(listen: str) -> tuple[str, int]:
    """Accept ``:9091`` or ``host:9091``."""
    if listen.startswith(":"):
        return "0.0.0.0", int(listen[1:])
    host, _, port = listen.rpartition(":")
    return host or "0.0.0.0", int(port)


if __name__ == "__main__":
    cli()
