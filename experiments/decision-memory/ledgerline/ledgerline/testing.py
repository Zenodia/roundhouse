"""Fixture builders shared by the repo's tests and the task tests."""

from __future__ import annotations

from datetime import datetime, timedelta

from . import payments
from .models import Aggregate, Customer, RawEvent
from .storage import Store

NOW = datetime(2026, 11, 5, 12, 0, 0)


def customer(cid: str = "c_us", region: str = "US") -> Customer:
    return Customer(cid, region, f"{cid}@example.com")


def webhook(payment_id: str, customer_id: str = "c_us", region: str = "US", amount_cents: int = 1000, pan: str = "4111111111111111") -> dict:
    return {
        "event_id": f"evt_{payment_id}",
        "payment_id": payment_id,
        "customer_id": customer_id,
        "region": region,
        "email": f"{customer_id}@example.com",
        "amount_cents": amount_cents,
        "currency": "USD",
        "brand": "visa",
        "pan": pan,
    }


def captured_payment(store: Store, pid: str, captured_at: datetime, *, settled_at: datetime | None = None, region: str = "US", amount_cents: int = 1000, customer_id: str | None = None):
    """A payment authorized and captured at `captured_at`, optionally settled."""
    cid = customer_id or f"c_{region.lower()}"
    payments.create(store, pid, customer(cid, region), amount_cents, "USD", captured_at - timedelta(minutes=5))
    payments.authorize(store, pid, captured_at - timedelta(minutes=1))
    payments.capture(store, pid, captured_at)
    if settled_at is not None:
        payments.settle(store, pid, settled_at)
    return store.get_payment(pid)


def sample_store(now: datetime = NOW) -> Store:
    """A store with payments spread over states, months and regions, raw
    events of several ages, and two aggregates."""
    s = Store()
    # October: captured and settled; captured but not settled.
    captured_payment(s, "p_oct_settled", datetime(2026, 10, 10, 9), settled_at=datetime(2026, 10, 12, 9), amount_cents=5000)
    captured_payment(s, "p_oct_unsettled", datetime(2026, 10, 28, 9), amount_cents=700)
    # Captured in October, settled in November.
    captured_payment(s, "p_oct_cap_nov_settle", datetime(2026, 10, 31, 22), settled_at=datetime(2026, 11, 2, 9), amount_cents=2000)
    # November: captured only; EU and UK customers.
    captured_payment(s, "p_nov_us", datetime(2026, 11, 3, 10), amount_cents=3000)
    captured_payment(s, "p_nov_eu", datetime(2026, 11, 4, 10), region="EU", amount_cents=4000)
    captured_payment(s, "p_nov_uk", datetime(2026, 11, 4, 11), region="UK", amount_cents=1500)
    # An authorized-only payment, from a webhook (carries a PAN).
    payments.ingest_gateway_webhook(s, webhook("p_nov_auth", amount_cents=900), now - timedelta(hours=1))
    # Raw events of different ages, besides the ingest's own.
    for days in (5, 45, 100, 400):
        s.record_event(RawEvent(f"evt_age_{days}", now - timedelta(days=days), {"payment_id": "p_old", "last4": "1111"}))
    s.add_aggregate(Aggregate("2026-06-01", "default", 12000, 11000))
    s.add_aggregate(Aggregate("2025-06-01", "eu", 8000, 8000))
    return s
