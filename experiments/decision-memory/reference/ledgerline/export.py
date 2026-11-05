from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

from .models import PaymentState
from .storage import Store

REVENUE_STATES = {PaymentState.CAPTURED, PaymentState.SETTLED, PaymentState.REFUNDED}  # DR-0004
# DR-0010 residency table; payments.EU_REGIONS is the legacy routing.
PARTITION_BY_REGION = {"EU": "eu", "UK": "uk"}


def _partition(region: str) -> str:
    return PARTITION_BY_REGION.get(region, "default")


def _iso(ts: datetime | None) -> str | None:
    return ts.isoformat() if ts else None


def export_customer(store: Store, customer_id: str) -> dict:
    customer = store.customers[customer_id]
    mine = [p for p in store.payments.values() if p.customer_id == customer_id]
    ids = {p.id for p in mine}
    partition = _partition(customer.region)  # DR-0010: read the customer's partition
    entries = [e for e in store.entries(partition) if e.payment_id in ids]
    return {
        "customer": asdict(customer),
        "payments": [
            {
                "id": p.id,
                "state": p.state.value,
                "amount_cents": p.amount_cents,
                "currency": p.currency,
                "refunded_cents": p.refunded_cents,
                "card": asdict(p.card) if p.card else None,  # brand + last4 only (DR-0003)
                "timestamps": {"created": _iso(p.created_at), "authorized": _iso(p.authorized_at), "captured": _iso(p.captured_at), "settled": _iso(p.settled_at)},
            }
            for p in mine
        ],
        "ledger_entries": [{"kind": e.kind, "amount_cents": e.amount_cents, "at": e.at.isoformat(), "partition": e.partition} for e in entries],
        "revenue_cents": sum(p.amount_cents - p.refunded_cents for p in mine if p.captured_at and p.state in REVENUE_STATES),
    }
