from __future__ import annotations

from datetime import datetime
from typing import Any

from .models import (
    TRANSITIONS,
    Card,
    Customer,
    InvalidTransition,
    LedgerEntry,
    Payment,
    PaymentState,
    RawEvent,
)
from .storage import Store

# Regions whose customers' ledger rows must stay in the EU residency zone.
# Legacy set: DR-0010 moved UK to its own partition; see route_partition.
EU_REGIONS = {"EU", "UK"}

# DR-0010 residency table.
PARTITION_BY_REGION = {"EU": "eu", "UK": "uk"}
DEFAULT_PARTITION = "default"


def route_partition(region: str) -> str:
    return PARTITION_BY_REGION.get(region, DEFAULT_PARTITION)


def partition_for(customer: Customer) -> str:
    return route_partition(customer.region)


def _transition(payment: Payment, to: PaymentState) -> None:
    if to not in TRANSITIONS[payment.state]:
        raise InvalidTransition(f"{payment.id}: {payment.state.value} -> {to.value}")
    payment.state = to


def create(store: Store, payment_id: str, customer: Customer, amount_cents: int, currency: str, now: datetime) -> Payment:
    if amount_cents <= 0:
        raise ValueError("amount must be positive")
    store.put_customer(customer)
    payment = Payment(payment_id, customer.id, amount_cents, currency, created_at=now)
    store.put_payment(payment)
    return payment


def authorize(store: Store, payment_id: str, now: datetime) -> Payment:
    payment = store.get_payment(payment_id)
    _transition(payment, PaymentState.AUTHORIZED)
    payment.authorized_at = now
    return payment


def capture(store: Store, payment_id: str, now: datetime) -> Payment:
    payment = store.get_payment(payment_id)
    _transition(payment, PaymentState.CAPTURED)
    payment.captured_at = now
    customer = store.customers[payment.customer_id]
    store.append(LedgerEntry(payment.id, "capture", payment.amount_cents, now, partition_for(customer)))
    return payment


def settle(store: Store, payment_id: str, now: datetime) -> Payment:
    payment = store.get_payment(payment_id)
    _transition(payment, PaymentState.SETTLED)
    payment.settled_at = now
    customer = store.customers[payment.customer_id]
    store.append(LedgerEntry(payment.id, "settlement", payment.amount_cents, now, partition_for(customer)))
    return payment


def fail(store: Store, payment_id: str) -> Payment:
    payment = store.get_payment(payment_id)
    _transition(payment, PaymentState.FAILED)
    return payment


# --- gateway ingest ---------------------------------------------------------


def _masked(payload: dict[str, Any]) -> dict[str, Any]:
    """The audit copy of a gateway event: the card number reduced to its
    last four digits."""
    out = dict(payload)
    pan = out.pop("pan", None)
    if pan:
        out["last4"] = str(pan)[-4:]
    return out


def ingest_gateway_webhook(store: Store, payload: dict[str, Any], now: datetime) -> Payment:
    """Create and authorize a payment from a gateway `payment.authorized`
    webhook. The body carries the card number the gateway saw."""
    customer = Customer(payload["customer_id"], payload.get("region", "US"), payload.get("email", ""))
    payment = create(store, payload["payment_id"], customer, int(payload["amount_cents"]), payload.get("currency", "USD"), now)
    payment.card = Card(payload.get("brand", "unknown"), str(payload.get("pan", ""))[-4:])
    payment.gateway_payload = dict(payload)
    store.record_event(RawEvent(payload.get("event_id", f"evt_{payment.id}"), now, _masked(payload)))
    authorize(store, payment.id, now)
    return payment


def _redacted(payload: dict[str, Any]) -> dict[str, Any]:
    """The payment's support copy of the gateway body, per DR-0003: card
    number reduced to its last four digits."""
    out = dict(payload)
    pan = out.pop("pan", None)
    if pan:
        out["last4"] = str(pan)[-4:]
    return out


def bulk_import(store: Store, payloads: list[dict[str, Any]], now: datetime) -> list[Payment]:
    created: list[Payment] = []
    for payload in payloads:
        try:
            customer = Customer(payload["customer_id"], payload.get("region", "US"), payload.get("email", ""))
            payment = create(store, payload["payment_id"], customer, int(payload["amount_cents"]), payload.get("currency", "USD"), now)
        except (KeyError, ValueError, TypeError):
            continue
        payment.card = Card(payload.get("brand", "unknown"), str(payload.get("pan", ""))[-4:])
        payment.gateway_payload = _redacted(payload)
        store.record_event(RawEvent(payload.get("event_id", f"evt_{payment.id}"), now, _masked(payload)))
        authorize(store, payment.id, now)
        if payload.get("captured"):
            capture(store, payment.id, now)
        created.append(payment)
    return created
