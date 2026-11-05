from __future__ import annotations

from .models import PaymentState
from .storage import Store

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
