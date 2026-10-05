from datetime import datetime, timedelta

from lanobert.early_detection import detection_lead_time, early_warning_rate
from lanobert.structured import extract_fields


def test_extract_fields_supports_bgl_and_key_value_logs():
    fields = extract_fields(
        "2005-10-18-14.20.00.000 host=compute-1 severity=error response_time_ms=250"
    )
    assert fields.timestamp == datetime(2005, 10, 18, 14, 20)
    assert fields.host == "compute-1"
    assert fields.severity == "ERROR"
    assert fields.response_time_ms == 250


def test_early_metrics_measure_warning_before_incident():
    timestamps = [datetime(2026, 1, 1) + timedelta(minutes=i) for i in range(5)]
    labels = [0, 0, 1, 1, 0]
    scores = [0.1, 0.8, 0.9, 0.95, 0.1]
    result = detection_lead_time(timestamps, scores, labels, threshold=0.7)
    assert result["early_warning_rate"] == 1.0
    assert result["mean_lead_time_seconds"] == 60.0
    assert early_warning_rate(scores, labels, threshold=0.7, warning_horizon=2) == 1.0
