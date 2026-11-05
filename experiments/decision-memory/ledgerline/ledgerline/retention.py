from __future__ import annotations

from datetime import datetime, timedelta

from .models import RawEvent

# How long a gateway event is kept before it is purged.
RAW_EVENT_RETENTION_DAYS = 30


def is_expired(event: RawEvent, now: datetime) -> bool:
    return event.received_at < now - timedelta(days=RAW_EVENT_RETENTION_DAYS)
