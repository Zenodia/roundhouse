import json
from datetime import datetime

import pytest

from ledgerline import export, payments
from ledgerline.decisions import param
from ledgerline.models import Customer, LedgerEntry
from ledgerline.storage import Store
from ledgerline.testing import captured_payment, sample_store, webhook


def test_export_shape():
    s = sample_store()
    e = export.export_customer(s, "c_us")
    json.dumps(e)
    assert e["customer"]["id"] == "c_us"
    assert {p["id"] for p in e["payments"]} >= {"p_oct_settled", "p_nov_us"}
    assert all("kind" in le and "partition" in le for le in e["ledger_entries"])


@pytest.mark.policy
def test_export_carries_last4_only():
    s = sample_store()
    payments.ingest_gateway_webhook(s, webhook("p_auth2", customer_id="c_us", pan="5555444433332222"), datetime(2026, 11, 5))
    blob = json.dumps(export.export_customer(s, "c_us"))
    assert "5555444433332222" not in blob and "4111111111111111" not in blob
    for field in param("forbidden_card_fields"):
        assert f'"{field}"' not in blob


@pytest.mark.policy
def test_uk_customer_entries_come_from_the_uk_partition():
    """DR-0010: per-customer reads look in the customer's partition."""
    table = param("partition_by_region")
    s = Store()
    captured_payment(s, "p_uk", datetime(2026, 11, 3), region="UK", amount_cents=1200, customer_id="c_uk")
    # The one-off migration (LL-455) has moved this customer's rows to `uk`.
    moved = [e for e in s.entries("eu") if e.payment_id == "p_uk"]
    s.ledger["eu"] = [e for e in s.entries("eu") if e.payment_id != "p_uk"]
    for e in moved:
        s.append(LedgerEntry(e.payment_id, e.kind, e.amount_cents, e.at, table["UK"]))
    e = export.export_customer(s, "c_uk")
    assert len(e["ledger_entries"]) == 1
    assert e["ledger_entries"][0]["partition"] == table["UK"]


@pytest.mark.policy
def test_revenue_in_export_is_recognized_at_capture():
    assert param("billable_state") == "captured"
    s = Store()
    captured_payment(s, "p1", datetime(2026, 11, 1), amount_cents=1000, customer_id="c_x")  # captured, not settled
    p = captured_payment(s, "p2", datetime(2026, 10, 1), settled_at=datetime(2026, 10, 3), amount_cents=500, customer_id="c_x")
    p.refunded_cents = 100
    assert export.export_customer(s, "c_x")["revenue_cents"] == 1000 + 500 - 100
