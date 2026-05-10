"""Config hot-reload coordinator.

Wraps the parse + validate + swap dance behind a single ``reload()`` entry
point. The contract is *validate-before-swap*: if parsing or validation
fails, the running configuration is preserved and the caller is told why.
A successful reload is reflected in the exporter's reload telemetry
(``forecast_config_last_reload_timestamp_seconds``) and emits a structured
``config_reload_succeeded`` log line.

Triggers:

* HTTP ``/-/reload`` endpoint (registered in ``main``).
* ``SIGHUP`` (registered in ``main``).
* Optional ConfigMap inotify watcher (started in ``main`` when
  ``server.reload.watch_configmap`` is true).

All triggers funnel through ``ConfigReloader.reload`` so the validate-then-
swap semantics are identical regardless of source.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from . import config as config_module

if TYPE_CHECKING:
    from .exporter import Exporter
    from .runner import ReloadResult, Runner

logger = structlog.get_logger(__name__)


class ReloadError(RuntimeError):
    """Raised when a reload attempt fails validation.

    The running config is left untouched. The error message is suitable for
    surfacing to a human operator (e.g. as the body of a 400 response from
    ``/-/reload``).
    """


@dataclass(frozen=True)
class ReloadOutcome:
    """Result of a successful reload, surfaced to operators."""

    result: ReloadResult
    timestamp: float


class ConfigReloader:
    """Owns reload coordination for one Runner + config file pair.

    Holds an internal lock so concurrent reload triggers (HTTP + SIGHUP +
    inotify) serialise rather than racing each other. The lock is separate
    from the Runner's own reload lock so the failure path (validate-only)
    stays cheap and never has to wait for in-flight runs.
    """

    def __init__(self, runner: Runner, exporter: Exporter, config_path: Path) -> None:
        self._runner = runner
        self._exporter = exporter
        self._config_path = config_path
        self._lock = asyncio.Lock()

    @property
    def config_path(self) -> Path:
        return self._config_path

    async def reload(self) -> ReloadOutcome:
        """Re-parse, validate, and swap in the config at ``config_path``.

        On validation failure the current config is preserved, the failure
        is recorded on the exporter so dashboards can detect the rollback,
        and a :class:`ReloadError` is raised. The running config is *not*
        rolled forward to a partial state — either the whole new config is
        applied or none of it is.
        """
        async with self._lock:
            log = logger.bind(path=str(self._config_path))
            log.info("config_reload_start")
            try:
                new_config = config_module.load(self._config_path)
            except FileNotFoundError as exc:
                self._exporter.increment_reload_failure()
                self._exporter.set_reload_state(success=False, timestamp=time.time())
                log.error("config_reload_file_missing", error=str(exc))
                raise ReloadError(f"config file not found: {self._config_path}") from exc
            except Exception as exc:
                # Validation errors land here — pydantic's ValidationError,
                # YAML parse errors, the metric-name validator. They're all
                # recoverable: the in-flight config keeps running.
                self._exporter.increment_reload_failure()
                self._exporter.set_reload_state(success=False, timestamp=time.time())
                log.error("config_reload_validation_failed", error=str(exc))
                raise ReloadError(f"config validation failed: {exc}") from exc

            try:
                result = await self._runner.apply_config(new_config)
            except Exception as exc:
                # apply_config is best-effort idempotent, but if a task
                # cancellation hangs we still treat the reload as failed.
                # The runner's existing config remains active.
                self._exporter.increment_reload_failure()
                self._exporter.set_reload_state(success=False, timestamp=time.time())
                log.exception("config_reload_apply_failed")
                raise ReloadError(f"config apply failed: {exc}") from exc

            now = time.time()
            self._exporter.set_reload_state(success=True, timestamp=now)
            # Refresh the reported config hash *after* a successful swap so
            # /metrics tracks exactly what the runner is now executing. On
            # validation/apply failure we leave the previous hash intact —
            # the metric should reflect the running config, not the rejected
            # candidate.
            new_hash = _hash_config_file(self._config_path)
            if new_hash:
                self._exporter.set_config_hash(new_hash)
            log.info(
                "config_reload_succeeded",
                added=result.added,
                removed=result.removed,
                changed=result.changed,
                config_hash=new_hash,
            )
            return ReloadOutcome(result=result, timestamp=now)


class ConfigMapWatcher:
    """Polls the config file's mtime + content hash and triggers a reload on change.

    Kubernetes projects ConfigMaps via a symlink that flips atomically when
    the underlying data changes. We deliberately *do not* use ``inotify``
    directly: portability across kernels and over NFS / overlay-fs is
    spotty. A small polling loop is the lowest-friction option, costs one
    ``stat`` per second, and matches what kubelet itself does internally.

    Hashing the file content (in addition to mtime) protects against
    edits-in-place that don't change the timestamp, and against the kubelet
    eventually projecting a file with the same content.
    """

    def __init__(
        self,
        reloader: ConfigReloader,
        *,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        self._reloader = reloader
        self._poll_interval = poll_interval_seconds
        self._stopped = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        # Snapshot the baseline fingerprint synchronously at construction —
        # if we deferred until the task starts, a file change between
        # ``start()`` and the first poll could be misread as the initial
        # state and never trigger a reload.
        self._initial_stat = _stat_signature(reloader.config_path)
        self._initial_hash = (
            _content_hash(reloader.config_path) if self._initial_stat is not None else None
        )

    async def start(self) -> None:
        """Snapshot the current file fingerprint and start the poll loop."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        import contextlib  # noqa: PLC0415

        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None

    async def _run(self) -> None:
        import contextlib  # noqa: PLC0415

        path = self._reloader.config_path
        last_stat = self._initial_stat
        last_hash = self._initial_hash
        log = logger.bind(path=str(path))
        log.info("configmap_watch_started", interval_s=self._poll_interval)
        while not self._stopped.is_set():
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self._poll_interval)
                return
            except TimeoutError:
                pass
            # Two-stage fingerprint: stat() is ~microseconds, sha256 over a
            # ConfigMap-sized file (≤1 MiB in practice) is milliseconds. The
            # common case is "no change" so we short-circuit on stat alone
            # and only hash when (mtime, size) differs.
            current_stat = _stat_signature(path)
            if current_stat is None:
                # File temporarily missing during a ConfigMap re-projection;
                # don't trigger a reload on a transient absence.
                continue
            if current_stat == last_stat:
                continue
            current_hash = _content_hash(path)
            if current_hash is None:
                continue
            if current_hash == last_hash:
                # Touched without content change (rare but possible — e.g.
                # an editor that rewrites the same bytes). Update the stat
                # baseline so we don't keep re-hashing every poll.
                last_stat = current_stat
                continue
            last_stat = current_stat
            last_hash = current_hash
            log.info("configmap_change_detected")
            with contextlib.suppress(Exception):
                # Reload errors are already logged + counted by the
                # reloader; swallow here so the watcher keeps polling.
                await self._reloader.reload()


def _stat_signature(path: Path) -> tuple[float, int] | None:
    """Cheap (mtime, size) tuple — the no-change short-circuit.

    Returns ``None`` on FileNotFoundError so the caller can treat a missing
    file as "no signal" rather than retriggering on every poll.
    """
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return st.st_mtime, st.st_size


def _content_hash(path: Path) -> str | None:
    """sha256 of the file body, or ``None`` if it disappeared mid-read.

    Only called once stat-level change is detected, so this is off the hot
    path of polling — we accept the cost of reading the whole file when it
    actually changes.
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _hash_config_file(path: Path) -> str:
    """Hex SHA-256 of the on-disk config bytes; ``""`` if the file vanishes.

    Used by the reload coordinator to refresh the exporter-side
    ``forecast_config_hash`` value. Mirrors :func:`_content_hash` but
    returns the empty string on failure so the caller can ``if new_hash``
    cleanly without juggling ``Optional``.
    """
    digest = _content_hash(path)
    return digest if digest is not None else ""
