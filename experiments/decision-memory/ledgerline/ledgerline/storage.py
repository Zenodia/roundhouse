from __future__ import annotations

from datetime import datetime

from .models import Aggregate, Customer, LedgerEntry, Payment, RawEvent

PARTITIONS = ("default", "eu")


class Store:
    """In-memory store standing in for Postgres. Ledger entries live in
    partitions (one table per data residency zone)."""

    def __init__(self) -> None:
        self.customers: dict[str, Customer] = {}
        self.payments: dict[str, Payment] = {}
        self.ledger: dict[str, list[LedgerEntry]] = {p: [] for p in PARTITIONS}
        self.raw_events: list[RawEvent] = []
        self.aggregates: list[Aggregate] = []

    # --- customers / payments -------------------------------------------

    def put_customer(self, customer: Customer) -> None:
        self.customers[customer.id] = customer

    def put_payment(self, payment: Payment) -> None:
        self.payments[payment.id] = payment

    def get_payment(self, payment_id: str) -> Payment:
        return self.payments[payment_id]

    # --- ledger -----------------------------------------------------------

    def append(self, entry: LedgerEntry) -> None:
        if entry.partition not in self.ledger:
            self.ledger[entry.partition] = []
        self.ledger[entry.partition].append(entry)

    def entries(self, partition: str | None = None) -> list[LedgerEntry]:
        if partition is not None:
            return list(self.ledger.get(partition, []))
        out: list[LedgerEntry] = []
        for rows in self.ledger.values():
            out.extend(rows)
        return out

    # --- events / aggregates ---------------------------------------------

    def record_event(self, event: RawEvent) -> None:
        self.raw_events.append(event)

    def events_before(self, cutoff: datetime) -> list[RawEvent]:
        return [e for e in self.raw_events if e.received_at < cutoff]

    def add_aggregate(self, aggregate: Aggregate) -> None:
        self.aggregates.append(aggregate)
