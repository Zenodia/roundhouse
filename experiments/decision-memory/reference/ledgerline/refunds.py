from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from . import payments
from .models import LedgerEntry, PaymentState
from .storage import Store

APPROVAL_AFTER_DAYS = 30  # DR-0005


class RefundError(Exception):
    pass


class ApprovalRequired(RefundError):
    pass


@dataclass
class Refund:
    payment_id: str
    amount_cents: int
    at: datetime
    manager_approved: bool


def request_refund(store: Store, payment_id: str, amount_cents: int, now: datetime, manager_approved: bool = False) -> Refund:
    payment = store.get_payment(payment_id)
    if payment.state not in (PaymentState.CAPTURED, PaymentState.SETTLED):
        raise RefundError(f"{payment_id}: not refundable in state {payment.state.value}")
    if amount_cents <= 0:
        raise RefundError("amount must be positive")
    if payment.captured_at is None:
        raise RefundError("payment has no capture time")
    if now - payment.captured_at > timedelta(days=APPROVAL_AFTER_DAYS) and not manager_approved:
        raise ApprovalRequired(f"{payment_id}: refund more than {APPROVAL_AFTER_DAYS} days after capture needs manager approval")
    if payment.refunded_cents + amount_cents > payment.amount_cents:
        raise RefundError("refund exceeds captured amount minus prior refunds")
    payment.refunded_cents += amount_cents
    customer = store.customers[payment.customer_id]
    store.append(LedgerEntry(payment.id, "refund", amount_cents, now, payments.partition_for(customer)))
    if payment.refunded_cents == payment.amount_cents:
        payment.state = PaymentState.REFUNDED
    return Refund(payment.id, amount_cents, now, manager_approved)
