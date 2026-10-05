"""Metrics for evaluating whether anomalies are detected before an incident."""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np


def early_warning_rate(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
    warning_horizon: int = 1,
) -> float:
    """Return the fraction of incidents preceded by a warning.

    An incident is a contiguous positive run. A warning counts only when it
    occurs within ``warning_horizon`` positions before that run.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    starts = np.flatnonzero((labels == 1) & np.r_[True, labels[:-1] == 0])
    if len(starts) == 0:
        return 0.0
    detected = sum(
        np.any(scores[max(0, start - warning_horizon):start] >= threshold)
        for start in starts
    )
    return float(detected / len(starts))


def detection_lead_time(
    timestamps: Sequence[datetime],
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> dict[str, float]:
    """Summarize time between first warning and each incident start."""
    if not (len(timestamps) == len(scores) == len(labels)):
        raise ValueError("timestamps, scores, and labels must have equal length")
    labels_array = np.asarray(labels, dtype=int)
    starts = np.flatnonzero((labels_array == 1) & np.r_[True, labels_array[:-1] == 0])
    lead_seconds = []
    for start in starts:
        warnings = np.flatnonzero(np.asarray(scores[:start]) >= threshold)
        if len(warnings):
            lead_seconds.append((timestamps[start] - timestamps[warnings[-1]]).total_seconds())
    return {
        "incident_count": float(len(starts)),
        "detected_incident_count": float(len(lead_seconds)),
        "early_warning_rate": float(len(lead_seconds) / len(starts)) if len(starts) else 0.0,
        "mean_lead_time_seconds": float(np.mean(lead_seconds)) if lead_seconds else 0.0,
        "median_lead_time_seconds": float(np.median(lead_seconds)) if lead_seconds else 0.0,
    }
