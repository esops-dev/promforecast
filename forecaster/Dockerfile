# syntax=docker/dockerfile:1.7

# ---------- Builder ----------
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

# Bake the HA + OTel extras into the image so flipping
# ``highAvailability.enabled: true`` or ``telemetry.enabled: true``
# doesn't require a custom build. The combined size is small (~8 MiB for
# HA, ~10 MiB for the OTel exporter chain) — well below the base image
# noise floor.
RUN pip install --upgrade pip build \
 && pip wheel --wheel-dir /wheels ".[ha,otel]"

# ---------- Runtime ----------
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system --gid 65532 promforecast \
 && useradd  --system --uid 65532 --gid 65532 --no-create-home promforecast

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl \
 && rm -rf /wheels

USER 65532:65532

EXPOSE 9091

ENTRYPOINT ["promforecast"]
CMD ["--config", "/etc/promforecast/config.yaml"]
