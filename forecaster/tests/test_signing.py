"""Tests for HMAC signing and secret-reference resolution."""

from __future__ import annotations

import hashlib
import hmac

import pytest

from promforecast import signing


def test_compute_signature_matches_stdlib_hmac() -> None:
    secret = "topsecret"
    body = b'{"hello": "world"}'
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert signing.compute_signature(secret, body) == expected


def test_verify_signature_round_trip() -> None:
    secret = "topsecret"
    body = b"payload"
    sig = signing.compute_signature(secret, body)
    assert signing.verify_signature(secret, body, sig)


def test_verify_signature_rejects_tampered_body() -> None:
    secret = "topsecret"
    sig = signing.compute_signature(secret, b"original")
    assert not signing.verify_signature(secret, b"tampered", sig)


def test_verify_signature_rejects_wrong_secret() -> None:
    body = b"payload"
    sig = signing.compute_signature("right", body)
    assert not signing.verify_signature("wrong", body, sig)


def test_verify_signature_missing_header_is_false() -> None:
    assert not signing.verify_signature("secret", b"payload", None)
    assert not signing.verify_signature("secret", b"payload", "")


def test_resolve_secret_ref_literal_passthrough() -> None:
    assert signing.resolve_secret_ref("literal-secret") == "literal-secret"


def test_resolve_secret_ref_env_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PF_TEST_SECRET", "from-env")
    assert signing.resolve_secret_ref("${PF_TEST_SECRET}") == "from-env"


def test_resolve_secret_ref_empty_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PF_MISSING_SECRET", raising=False)
    with pytest.raises(signing.SecretResolutionError):
        signing.resolve_secret_ref("${PF_MISSING_SECRET}")


def test_resolve_secret_ref_empty_literal_raises() -> None:
    with pytest.raises(signing.SecretResolutionError):
        signing.resolve_secret_ref("")


def test_expand_env_refs_embedded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PF_TEST_TOKEN", "abc123")
    assert signing.expand_env_refs("Bearer ${PF_TEST_TOKEN}") == "Bearer abc123"


def test_expand_env_refs_missing_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PF_MISSING_TOKEN", raising=False)
    assert signing.expand_env_refs("Bearer ${PF_MISSING_TOKEN}") == "Bearer "


def test_expand_env_refs_no_ref_passthrough() -> None:
    assert signing.expand_env_refs("plain-value") == "plain-value"


def test_expand_header_env_refs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PF_TEST_TOKEN", "xyz")
    out = signing.expand_header_env_refs({"Authorization": "Bearer ${PF_TEST_TOKEN}", "X": "lit"})
    assert out == {"Authorization": "Bearer xyz", "X": "lit"}
