"""Configuration schema and validation.

The YAML config is the single source of truth. The pydantic models below and
``charts/promforecast/values.schema.json`` must agree — the CI helm-schema
check enforces this.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["none", "bearer", "basic", "mtls"] = "none"


class DatasourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    timeout: timedelta = timedelta(seconds=30)
    auth: AuthConfig = AuthConfig()


class ReloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    watch_configmap: bool = True


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    listen: str = ":9091"
    refresh_interval: timedelta = timedelta(hours=1)
    reload: ReloadConfig = ReloadConfig()


class SafetyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_series_per_query: int = 500
    max_total_series: int = 5000
    series_overflow: Literal["drop_lowest_priority", "reject", "sample"] = "drop_lowest_priority"
    fit_timeout: timedelta = timedelta(seconds=60)
    query_timeout: timedelta = timedelta(seconds=30)


class DefaultsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lookback: timedelta = timedelta(days=14)
    step: timedelta = timedelta(minutes=5)
    horizon: timedelta = timedelta(hours=24)
    models: list[str] = Field(default_factory=lambda: ["AutoARIMA", "SeasonalNaive"])
    season_length: int = 288
    confidence_levels: list[int] = Field(default_factory=lambda: [80, 95])
    min_points: int = 200
    evaluate: bool = True
    auto_select: bool | Literal["ensemble"] = False
    fallback_model: str = "SeasonalNaive"
    fallback_mape_threshold: float | None = 50.0
    emit_deviation: bool = True
    emit_quality: bool = True


class RegressorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    promql: str | None = None
    type: Literal["custom", "calendar"] = "custom"
    required: bool = False


class QueryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    promql: str
    regressors: list[RegressorConfig] = Field(default_factory=list)


class GroupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    priority: int = 1
    queries: list[QueryConfig]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasource: DatasourceConfig
    server: ServerConfig = ServerConfig()
    safety: SafetyConfig = SafetyConfig()
    defaults: DefaultsConfig = DefaultsConfig()
    groups: list[GroupConfig]


def load(path: str) -> Config:
    """Load and validate a YAML config file."""
    raise NotImplementedError
