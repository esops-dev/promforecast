"""Validation tests for the ``sharding`` config block."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from promforecast.config import (
    Config,
    DatasourceConfig,
    GroupConfig,
    HighAvailabilityConfig,
    QueryCacheConfig,
    QueryConfig,
    ShardingConfig,
)


def _base_config(**overrides: object) -> Config:
    return Config(
        datasource=DatasourceConfig(url="http://x"),
        groups=[GroupConfig(name="g", queries=[QueryConfig(id="m", promql="up")])],
        **overrides,
    )


def test_sharding_defaults_off() -> None:
    cfg = _base_config()
    assert cfg.sharding.enabled is False
    assert cfg.sharding.mode == "consistent_hash"
    assert cfg.sharding.replica_count == 1
    assert cfg.sharding.shard_index == -1


def test_sharding_consistent_hash_needs_no_redis() -> None:
    cfg = _base_config(
        sharding=ShardingConfig(enabled=True, mode="consistent_hash", replica_count=4)
    )
    assert cfg.sharding.enabled is True


def test_sharding_leader_assigned_requires_redis() -> None:
    with pytest.raises(ValidationError, match="leader_assigned requires a Redis backend"):
        _base_config(sharding=ShardingConfig(enabled=True, mode="leader_assigned", replica_count=3))


def test_sharding_leader_assigned_accepts_query_cache_redis() -> None:
    cfg = _base_config(
        sharding=ShardingConfig(enabled=True, mode="leader_assigned", replica_count=3),
        query_cache=QueryCacheConfig(redis_address="redis:6379"),
    )
    assert cfg.sharding.mode == "leader_assigned"


def test_sharding_and_ha_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        _base_config(
            sharding=ShardingConfig(enabled=True, mode="consistent_hash", replica_count=2),
            high_availability=HighAvailabilityConfig(enabled=True, redis={"address": "redis:6379"}),
        )


def test_sharding_shard_index_must_be_below_replica_count() -> None:
    with pytest.raises(ValidationError, match="must be < replica_count"):
        ShardingConfig(enabled=True, replica_count=3, shard_index=3)


def test_sharding_member_ttl_must_exceed_heartbeat() -> None:
    with pytest.raises(ValidationError, match="member_ttl_seconds must be >"):
        ShardingConfig(heartbeat_interval_seconds=30, member_ttl_seconds=30)


def test_sharding_renew_deadline_below_lease_duration() -> None:
    with pytest.raises(ValidationError, match="renew_deadline_seconds must be <"):
        ShardingConfig(lease_duration_seconds=20, renew_deadline_seconds=20)


def test_sharding_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError):
        ShardingConfig(mode="round_robin")  # type: ignore[arg-type]
