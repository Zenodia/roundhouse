from __future__ import annotations

from datetime import datetime, timedelta

from .models import RawEvent
from .storage import Store

# How long a gateway event is kept before it is purged. DR-0001 (30 days)
# was superseded by DR-0007: 90 days for raw events, aggregates kept.
RAW_EVENT_RETENTION_DAYS = 30  # legacy constant; see DR-0007 and retention_policy()
RAW_EVENT_RETENTION_DAYS_CURRENT = 90
AGGREGATE_RETENTION_DAYS: int | None = None


def is_expired(event: RawEvent, now: datetime) -> bool:
    return event.received_at < now - timedelta(days=RAW_EVENT_RETENTION_DAYS)


def retention_policy() -> dict:
    return {"raw_event_days": RAW_EVENT_RETENTION_DAYS_CURRENT, "aggregate_days": AGGREGATE_RETENTION_DAYS}


def purge_expired(store: Store, now: datetime) -> int:
    cutoff = now - timedelta(days=RAW_EVENT_RETENTION_DAYS_CURRENT)
    keep = [e for e in store.raw_events if e.received_at >= cutoff]
    removed = len(store.raw_events) - len(keep)
    store.raw_events = keep
    return removed
