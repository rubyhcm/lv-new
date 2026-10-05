from datetime import datetime, timezone

from lanobert.field_value import ConstraintConfig, ConstraintEngine, LogFields


def test_constraint_engine_emits_auditable_hard_state():
    engine = ConstraintEngine(ConstraintConfig(
        max_response_time_ms=200,
        max_heartbeat_interval_s=30,
        blacklisted_hosts={"node-7"},
    ))
    state = engine.evaluate(
        LogFields(
            timestamp=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
            severity="error", host="node-7", response_time_ms=250,
        ),
        previous_timestamp=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
    )
    assert state.hard_anomaly is True
    assert state.vector == (1.0, 1.0, 1.0, 1.0)


def test_constraint_engine_does_not_flag_normal_fields():
    state = ConstraintEngine(ConstraintConfig(
        max_response_time_ms=200, max_heartbeat_interval_s=30,
    )).evaluate(LogFields(severity="info", response_time_ms=20))
    assert state.hard_anomaly is False
    assert state.vector == (0.0, 0.0, 0.0, 0.0)
