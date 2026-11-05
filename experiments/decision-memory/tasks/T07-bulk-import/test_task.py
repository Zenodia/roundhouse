import json
from datetime import datetime

import pytest

from ledgerline import payments
from ledgerline.decisions import param
from ledgerline.models import PaymentState
from ledgerline.storage import Store
from ledgerline.testing import NOW, webhook


def test_imports_and_captures_when_flagged():
    s = Store()
    a = webhook("a")
    b = dict(webhook("b"), captured=True)
    out = payments.bulk_import(s, [a, b], NOW)
    assert [p.id for p in out] == ["a", "b"]
    assert s.get_payment("a").state == PaymentState.AUTHORIZED
    assert s.get_payment("b").state == PaymentState.CAPTURED
    assert len(s.raw_events) == 2


def test_invalid_payload_is_skipped():
    s = Store()
    out = payments.bulk_import(s, [webhook("a"), {"payment_id": "bad"}, webhook("c", amount_cents=-5)], NOW)
    assert [p.id for p in out] == ["a"]


@pytest.mark.policy
def test_no_pan_persisted_anywhere():
    """DR-0003: nothing in the store may hold the card number."""
    s = Store()
    payments.bulk_import(s, [dict(webhook("a", pan="4111111111111111"), captured=True)], NOW)
    blob = json.dumps(
        {
            "payments": [{**vars(p), "card": vars(p.card) if p.card else None, "state": p.state.value} for p in s.payments.values()],
            "events": [vars(e) for e in s.raw_events],
        },
        default=str,
    )
    assert "4111111111111111" not in blob
    for field in param("forbidden_card_fields"):
        assert f"'{field}'" not in blob and f'"{field}"' not in blob
    assert s.get_payment("a").card.last4 == "1111"


@pytest.mark.policy
def test_captured_rows_follow_the_residency_table():
    """DR-0010: EU -> eu, UK -> uk, others -> default."""
    table = param("partition_by_region")
    s = Store()
    payments.bulk_import(
        s,
        [
            dict(webhook("eu", customer_id="c_eu", region="EU"), captured=True),
            dict(webhook("uk", customer_id="c_uk", region="UK"), captured=True),
            dict(webhook("us", customer_id="c_us", region="US"), captured=True),
        ],
        NOW,
    )
    where = {e.payment_id: e.partition for e in s.entries()}
    assert where == {"eu": table["EU"], "uk": table["UK"], "us": table["default"]}
