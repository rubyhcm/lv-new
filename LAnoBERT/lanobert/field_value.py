"""Structured field-value extraction and deterministic constraint scoring."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LogFields(BaseModel):
    """Normalized physical fields exposed to the constraint engine."""

    model_config = ConfigDict(extra="ignore")

    timestamp: Optional[datetime] = None
    severity: Optional[str] = None
    host: Optional[str] = None
    service: Optional[str] = None
    event_code: Optional[str] = None
    response_time_ms: Optional[float] = Field(default=None, ge=0)

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, value: Any) -> Any:
        return value.upper().strip() if isinstance(value, str) else value


class ConstraintConfig(BaseModel):
    """Thresholds used by :class:`ConstraintEngine`."""

    max_response_time_ms: Optional[float] = Field(default=None, ge=0)
    max_heartbeat_interval_s: Optional[float] = Field(default=None, ge=0)
    critical_severities: frozenset[str] = frozenset({"CRITICAL", "FATAL", "ERROR"})
    blacklisted_hosts: frozenset[str] = frozenset()

    @field_validator("critical_severities", mode="before")
    @classmethod
    def normalize_severities(cls, value: Iterable[str]) -> frozenset[str]:
        return frozenset(item.upper().strip() for item in value)


class HardState(BaseModel):
    """Auditable output of the structured branch."""

    response_time_violation: bool = False
    heartbeat_violation: bool = False
    critical_severity: bool = False
    blacklisted_host: bool = False
    hard_anomaly: bool = False

    @property
    def vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in (
            "response_time_violation",
            "heartbeat_violation",
            "critical_severity",
            "blacklisted_host",
        ))


class ConstraintEngine:
    """Evaluate deterministic constraints without invoking an ML model."""

    def __init__(self, config: ConstraintConfig):
        self.config = config

    def evaluate(
        self,
        fields: LogFields | Mapping[str, Any],
        *,
        previous_timestamp: Optional[datetime] = None,
    ) -> HardState:
        current = fields if isinstance(fields, LogFields) else LogFields.model_validate(fields)
        response_violation = (
            self.config.max_response_time_ms is not None
            and current.response_time_ms is not None
            and current.response_time_ms > self.config.max_response_time_ms
        )
        heartbeat_violation = False
        if (
            self.config.max_heartbeat_interval_s is not None
            and previous_timestamp is not None
            and current.timestamp is not None
        ):
            heartbeat_violation = (
                current.timestamp - previous_timestamp
            ).total_seconds() > self.config.max_heartbeat_interval_s
        severity_violation = (
            current.severity is not None
            and current.severity in self.config.critical_severities
        )
        host_violation = (
            current.host is not None and current.host in self.config.blacklisted_hosts
        )
        return HardState(
            response_time_violation=response_violation,
            heartbeat_violation=heartbeat_violation,
            critical_severity=severity_violation,
            blacklisted_host=host_violation,
            hard_anomaly=any((
                response_violation,
                heartbeat_violation,
                severity_violation,
                host_violation,
            )),
        )
