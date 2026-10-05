"""Extract structured fields from common system-log representations."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Mapping

from .field_value import LogFields

_KEY_VALUE = re.compile(r"(?P<key>[A-Za-z][\w.-]*)=(?P<value>\"[^\"]*\"|\S+)")
_BGL_TIMESTAMP = re.compile(
    r"(?P<value>\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+)"
)
_BGL_SEVERITY = re.compile(r"\b(?P<value>TRACE|DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\b", re.I)
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")


def _parse_timestamp(value: str) -> datetime | None:
    for pattern in ("%Y-%m-%d-%H.%M.%S.%f", "%Y-%m-%dT%H:%M:%S.%f",
                    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, pattern)
        except ValueError:
            continue
    return None


def extract_fields(line: str | Mapping[str, Any]) -> LogFields:
    """Extract fields without failing the log stream on missing optional data."""
    if isinstance(line, Mapping):
        return LogFields.model_validate(line)
    values: dict[str, Any] = {}
    for match in _KEY_VALUE.finditer(line):
        key = match.group("key").lower().replace("-", "_").replace(".", "_")
        values[key] = match.group("value").strip('"')
    timestamp = values.get("timestamp") or values.get("time")
    if timestamp:
        values["timestamp"] = _parse_timestamp(str(timestamp))
    else:
        match = _BGL_TIMESTAMP.search(line)
        if match:
            values["timestamp"] = _parse_timestamp(match.group("value"))
    severity = values.get("severity") or values.get("level")
    if not severity:
        match = _BGL_SEVERITY.search(line)
        severity = match.group("value") if match else None
    values["severity"] = severity
    if "host" not in values:
        ip = _IP.search(line)
        if ip:
            values["host"] = ip.group()
    for key in ("response_time_ms", "latency", "response_time"):
        if key in values:
            values["response_time_ms"] = values[key]
            break
    return LogFields.model_validate(values)
