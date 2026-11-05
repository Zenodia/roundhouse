from datetime import datetime

import pytest

from ledgerline import payments
from ledgerline.models import InvalidTransition, PaymentState
from ledgerline.storage import Store
from ledgerline.testing import NOW, customer, webhook


def test_lifecycle_appends_capture_and_settlement():
    s = Store()
    payments.create(s, "p1", customer(), 1000, "USD", NOW)
    payments.authorize(s, "p1", NOW)
    payments.capture(s, "p1", NOW)
    payments.settle(s, "p1", NOW)
    assert s.get_payment("p1").state == PaymentState.SETTLED
    kinds = [e.kind for e in s.entries("default")]
    assert kinds == ["capture", "settlement"]


def test_illegal_transition_is_refused():
    s = Store()
    payments.create(s, "p1", customer(), 1000, "USD", NOW)
    with pytest.raises(InvalidTransition):
        payments.capture(s, "p1", NOW)


def test_amount_must_be_positive():
    s = Store()
    with pytest.raises(ValueError):
        payments.create(s, "p1", customer(), 0, "USD", NOW)


def test_eu_customer_rows_land_in_eu_partition():
    s = Store()
    payments.create(s, "p1", customer("c_eu", "EU"), 1000, "EUR", NOW)
    payments.authorize(s, "p1", NOW)
    payments.capture(s, "p1", NOW)
    assert [e.payment_id for e in s.entries("eu")] == ["p1"]
    assert s.entries("default") == []


def test_webhook_ingest_authorizes_and_masks_audit_event():
    s = Store()
    p = payments.ingest_gateway_webhook(s, webhook("p9"), NOW)
    assert p.state == PaymentState.AUTHORIZED
    assert p.card.last4 == "1111"
    assert "pan" not in s.raw_events[0].payload
    assert s.raw_events[0].payload["last4"] == "1111"
