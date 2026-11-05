from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class PaymentState(str, Enum):
    CREATED = "created"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    SETTLED = "settled"
    REFUNDED = "refunded"
    FAILED = "failed"


# Legal transitions of the payment state machine.
TRANSITIONS: dict[PaymentState, set[PaymentState]] = {
    PaymentState.CREATED: {PaymentState.AUTHORIZED, PaymentState.FAILED},
    PaymentState.AUTHORIZED: {PaymentState.CAPTURED, PaymentState.FAILED},
    PaymentState.CAPTURED: {PaymentState.SETTLED, PaymentState.REFUNDED},
    PaymentState.SETTLED: {PaymentState.REFUNDED},
    PaymentState.REFUNDED: set(),
    PaymentState.FAILED: set(),
}


class InvalidTransition(Exception):
    pass


@dataclass
class Customer:
    id: str
    region: str  # ISO-ish region code the storefront reports: "US", "EU", "UK", ...
    email: str = ""


@dataclass
class Card:
    brand: str
    last4: str


@dataclass
class Payment:
    id: str
    customer_id: str
    amount_cents: int
    currency: str
    state: PaymentState = PaymentState.CREATED
    created_at: datetime | None = None
    authorized_at: datetime | None = None
    captured_at: datetime | None = None
    settled_at: datetime | None = None
    refunded_cents: int = 0
    card: Card | None = None
    # The gateway's webhook body as received, kept for support investigations.
    gateway_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class LedgerEntry:
    payment_id: str
    kind: str  # "capture" | "settlement" | "refund"
    amount_cents: int
    at: datetime
    partition: str


@dataclass
class RawEvent:
    """A gateway event as it arrived, for audit and replay."""

    id: str
    received_at: datetime
    payload: dict[str, Any]


@dataclass
class Aggregate:
    """A daily rollup; computed from raw events, kept after they are gone."""

    day: str  # YYYY-MM-DD
    partition: str
    captured_cents: int
    settled_cents: int
