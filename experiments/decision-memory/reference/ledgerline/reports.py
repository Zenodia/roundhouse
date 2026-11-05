from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime

from .models import Payment, PaymentState
from .storage import Store

# DR-0004: revenue is recognized at capture, dated by captured_at.
REVENUE_STATES = {PaymentState.CAPTURED, PaymentState.SETTLED, PaymentState.REFUNDED}

# A payment counts toward revenue once it reaches this state.
BILLABLE_STATE = PaymentState.SETTLED


def gross_billable_cents(store: Store) -> int:
    return sum(p.amount_cents for p in store.payments.values() if p.state == BILLABLE_STATE)


def count_by_state(store: Store) -> dict[str, int]:
    counts: dict[str, int] = {s.value: 0 for s in PaymentState}
    for p in store.payments.values():
        counts[p.state.value] += 1
    return counts


def ledger_total_cents(store: Store, partition: str | None = None) -> int:
    return sum(e.amount_cents for e in store.entries(partition) if e.kind != "refund")


def _recognized(p: Payment) -> bool:
    return p.captured_at is not None and p.state in REVENUE_STATES


def _net(p: Payment) -> int:
    return p.amount_cents - p.refunded_cents


def monthly_revenue_cents(store: Store, year: int, month: int) -> int:
    return sum(_net(p) for p in store.payments.values() if _recognized(p) and (p.captured_at.year, p.captured_at.month) == (year, month))


def _iso(ts: datetime | None) -> str | None:
    return ts.isoformat() if ts else None


def _redact(payload: dict) -> dict:
    out = {k: v for k, v in payload.items() if k not in ("pan", "card_number", "primary_account_number")}
    pan = payload.get("pan")
    if pan:
        out["last4"] = str(pan)[-4:]
    return out


def payment_debug_view(store: Store, payment_id: str) -> dict:
    p = store.get_payment(payment_id)
    customer = store.customers.get(p.customer_id)
    return {
        "id": p.id,
        "state": p.state.value,
        "amount_cents": p.amount_cents,
        "currency": p.currency,
        "customer": asdict(customer) if customer else None,
        "timestamps": {"created": _iso(p.created_at), "authorized": _iso(p.authorized_at), "captured": _iso(p.captured_at), "settled": _iso(p.settled_at)},
        "card": asdict(p.card) if p.card else None,
        "gateway_payload": _redact(p.gateway_payload),
        "ledger_entries": [
            {"kind": e.kind, "amount_cents": e.amount_cents, "at": e.at.isoformat(), "partition": e.partition}
            for e in store.entries()
            if e.payment_id == p.id
        ],
    }


def daily_summary(store: Store, day: date) -> dict:
    captured = [p for p in store.payments.values() if p.captured_at and p.captured_at.date() == day]
    settled = [p for p in store.payments.values() if p.settled_at and p.settled_at.date() == day]
    refunded = 0
    for p in store.payments.values():
        if p.state == PaymentState.REFUNDED:
            mine = [e for e in store.entries() if e.payment_id == p.id]
            if mine and max(e.at for e in mine).date() == day:
                refunded += 1
    return {
        "day": day.isoformat(),
        "captured_count": len(captured),
        "settled_count": len(settled),
        "refunded_count": refunded,
        "revenue_cents": sum(_net(p) for p in captured if _recognized(p)),
    }


RAW_EVENT_RETENTION_DAYS = 90  # DR-0007; retention.RAW_EVENT_RETENTION_DAYS still carries the superseded 30


def kpis(store: Store, now: datetime) -> dict:
    from datetime import timedelta

    prev_year, prev_month = (now.year, now.month - 1) if now.month > 1 else (now.year - 1, 12)
    cutoff = now - timedelta(days=RAW_EVENT_RETENTION_DAYS)
    return {
        "revenue_mtd_cents": monthly_revenue_cents(store, now.year, now.month),
        "revenue_prev_month_cents": monthly_revenue_cents(store, prev_year, prev_month),
        "open_authorizations": sum(1 for p in store.payments.values() if p.state == PaymentState.AUTHORIZED),
        "raw_events_retained": sum(1 for e in store.raw_events if e.received_at >= cutoff),
        "aggregates_total": len(store.aggregates),
    }
