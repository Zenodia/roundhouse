from datetime import timedelta

import pytest

from ledgerline import retention
from ledgerline.decisions import param
from ledgerline.models import RawEvent
from ledgerline.storage import Store
from ledgerline.testing import NOW, sample_store


def test_returns_count_and_removes_something_old():
    s = sample_store()
    before = len(s.raw_events)
    n = retention.purge_expired(s, NOW)
    assert isinstance(n, int) and n >= 1
    assert len(s.raw_events) == before - n


def test_nothing_to_purge_is_zero():
    s = Store()
    s.record_event(RawEvent("e", NOW - timedelta(days=1), {}))
    assert retention.purge_expired(s, NOW) == 0


@pytest.mark.policy
def test_window_is_the_decided_number_of_days():
    """DR-0007: 90 days, not the 30 the code's constant still says."""
    days = param("raw_event_retention_days")
    s = Store()
    keep = RawEvent("keep", NOW - timedelta(days=days - 1), {})
    drop = RawEvent("drop", NOW - timedelta(days=days + 1), {})
    s.record_event(keep)
    s.record_event(drop)
    assert retention.purge_expired(s, NOW) == 1
    assert [e.id for e in s.raw_events] == ["keep"]


@pytest.mark.policy
def test_aggregates_are_never_purged():
    assert param("aggregate_retention_days") is None
    s = sample_store()
    aggregates = list(s.aggregates)  # one is from 2025
    retention.purge_expired(s, NOW)
    assert s.aggregates == aggregates
    assert len(s.payments) == 7 and sum(len(v) for v in s.ledger.values()) > 0
