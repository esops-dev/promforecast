"""Entrypoint: parses config, starts the HTTP server and the scheduler."""

from __future__ import annotations

import argparse
from pathlib import Path


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
    raise NotImplementedError


if __name__ == "__main__":
    cli()
