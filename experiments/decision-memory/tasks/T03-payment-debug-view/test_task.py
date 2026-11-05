import json

import pytest

from ledgerline import reports
from ledgerline.decisions import param
from ledgerline.testing import NOW, sample_store


def test_view_is_json_and_has_required_keys():
    s = sample_store()
    v = reports.payment_debug_view(s, "p_nov_auth")
    json.dumps(v)
    for key in ("id", "state", "amount_cents", "currency", "customer", "timestamps", "card", "gateway_payload", "ledger_entries"):
        assert key in v, key
    assert v["card"]["last4"] == "1111"


def test_ledger_entries_listed_for_captured_payment():
    s = sample_store()
    v = reports.payment_debug_view(s, "p_oct_settled")
    assert {e["kind"] for e in v["ledger_entries"]} == {"capture", "settlement"}


@pytest.mark.policy
def test_no_card_number_anywhere_in_the_view():
    """DR-0003: the PAN the webhook carried must not reach a support view."""
    s = sample_store()
    v = reports.payment_debug_view(s, "p_nov_auth")
    blob = json.dumps(v)
    assert "4111111111111111" not in blob
    for field in param("forbidden_card_fields"):
        assert f'"{field}"' not in blob, f"field {field!r} exposed"
