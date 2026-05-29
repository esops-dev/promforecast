"""HMAC signing and secret-reference resolution for webhooks.

Shared by the webhook *regressor* (which verifies the HMAC on a response it
pulls in) and the webhook *trigger* (which signs a request it pushes out).
The signature is computed over the raw request/response body with
HMAC-SHA256 and carried in the ``X-Promforecast-Signature: sha256=<hex>``
header — the de-facto convention GitHub, Stripe, and most webhook
producers use.

Secrets are never written into the config literally in production: a field
like ``secret_ref: ${WEBHOOK_SECRET}`` is resolved against the process
environment (the chart injects the variable from a Kubernetes Secret via
``secretKeyRef``). A literal value is accepted too, for local testing.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Mapping

#: The header outbound triggers set and inbound regressors verify.
SIGNATURE_HEADER = "X-Promforecast-Signature"

#: ``${NAME}`` references an environment variable; anything else is a literal.
_ENV_REF_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")

#: ``${NAME}`` anywhere inside a string (used for header-value expansion,
#: where the reference is typically embedded — e.g. ``Bearer ${TOKEN}``).
_ENV_SUBST_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env_refs(value: str) -> str:
    """Substitute every ``${NAME}`` in ``value`` with environment variable ``NAME``.

    Unlike :func:`resolve_secret_ref` (which matches a *whole-value*
    ``${NAME}`` and rejects an empty result), this expands references
    *embedded* in a larger string — the shape webhook/trigger ``headers``
    use (``Authorization: Bearer ${TOKEN}``). An unset variable expands to
    the empty string rather than raising: a header is delivered best-effort,
    and a failed auth degrades through the same graceful path as any other
    webhook failure. A value with no ``${...}`` is returned unchanged.
    """
    return _ENV_SUBST_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)


def expand_header_env_refs(headers: Mapping[str, str]) -> dict[str, str]:
    """Expand ``${ENV_VAR}`` references in every header *value*.

    Header *names* are left verbatim. Lets operators inject secrets into
    webhook/trigger headers from the environment (Kubernetes Secret →
    env var) the same way ``secret_ref`` works for signing, instead of
    baking a literal token into the config.
    """
    return {name: expand_env_refs(value) for name, value in headers.items()}


class SecretResolutionError(ValueError):
    """Raised when a ``${ENV_VAR}`` reference resolves to an empty value.

    A missing or empty secret must fail loudly rather than silently
    disabling signing — an unsigned payload that the operator believed was
    signed is worse than a hard error at startup.
    """


def resolve_secret_ref(ref: str) -> str:
    """Resolve a ``${ENV_VAR}`` reference (or pass a literal through).

    Mirrors the indirection the chart already uses for datasource
    credentials: a value of the form ``${NAME}`` reads environment variable
    ``NAME``; any other value is returned verbatim. An empty resolved
    secret raises :class:`SecretResolutionError`.
    """
    match = _ENV_REF_RE.match(ref.strip())
    value = ref if match is None else os.environ.get(match.group(1), "")
    if not value:
        raise SecretResolutionError(
            f"secret reference {ref!r} resolved to an empty value; "
            "set the environment variable (injected from a Kubernetes Secret) "
            "or provide a literal secret"
        )
    return value


def compute_signature(secret: str, body: bytes) -> str:
    """Return ``sha256=<hex>`` HMAC-SHA256 of ``body`` keyed by ``secret``."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_signature(secret: str, body: bytes, header_value: str | None) -> bool:
    """Constant-time check that ``header_value`` matches ``body``'s signature.

    Returns ``False`` for a missing or malformed header so the caller can
    treat it identically to a mismatch (both surface as a regressor
    ``signature_invalid`` failure).
    """
    if not header_value:
        return False
    expected = compute_signature(secret, body)
    return hmac.compare_digest(expected, header_value.strip())
