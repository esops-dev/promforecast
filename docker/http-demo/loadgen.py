# Tiny load generator: fires ~20-30 req/min by default at the http-demo app,
# with a slow sinusoidal modulation on top so the rate has shape (and the
# forecaster has a non-trivial signal to fit). Endpoints are picked from a
# weighted mix so http_requests_total{endpoint=...} differs per route.
from __future__ import annotations

import asyncio
import math
import os
import random
import time
from contextlib import suppress

import httpx

TARGET = os.getenv("TARGET", "http://http-demo:8000")
BASE_RPM = float(os.getenv("BASE_RPM", "25"))
AMPLITUDE_RPM = float(os.getenv("AMPLITUDE_RPM", "15"))
PERIOD_SECONDS = float(os.getenv("PERIOD_SECONDS", "600"))  # 10-minute "diurnal" cycle
JITTER = float(os.getenv("JITTER", "0.25"))

ENDPOINTS = [
    ("/api/users", 5),
    ("/api/products", 3),
    ("/api/slow", 1),
    ("/api/flaky", 2),
    ("/healthz", 1),
]


def pick_endpoint() -> str:
    paths, weights = zip(*ENDPOINTS, strict=True)
    return random.choices(paths, weights=weights, k=1)[0]


def current_rpm(t0: float) -> float:
    phase = 2 * math.pi * ((time.monotonic() - t0) % PERIOD_SECONDS) / PERIOD_SECONDS
    return max(1.0, BASE_RPM + AMPLITUDE_RPM * math.sin(phase))


async def hammer() -> None:
    t0 = time.monotonic()
    timeout = httpx.Timeout(5.0)
    async with httpx.AsyncClient(timeout=timeout, base_url=TARGET) as client:
        while True:
            rpm = current_rpm(t0)
            interval = 60.0 / rpm
            interval *= 1.0 + random.uniform(-JITTER, JITTER)
            with suppress(Exception):
                await client.get(pick_endpoint())
            await asyncio.sleep(max(0.05, interval))


if __name__ == "__main__":
    asyncio.run(hammer())
