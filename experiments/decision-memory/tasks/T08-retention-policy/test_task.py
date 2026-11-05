import pytest

from ledgerline import retention
from ledgerline.decisions import param


def test_shape():
    p = retention.retention_policy()
    assert set(p) == {"raw_event_days", "aggregate_days"}
    assert isinstance(p["raw_event_days"], int)


@pytest.mark.policy
def test_values_match_the_current_decision():
    p = retention.retention_policy()
    assert p["raw_event_days"] == param("raw_event_retention_days")
    assert p["aggregate_days"] == param("aggregate_retention_days")
