from datetime import datetime, timedelta

import pytest

from ledgerline import refunds
from ledgerline.decisions import param
from ledgerline.models import PaymentState
from ledgerline.storage import Store
from ledgerline.testing import captured_payment

CAP = datetime(2026, 10, 1, 12)


def test_partial_refund_updates_payment_and_ledger():
    s = Store()
    captured_payment(s, "p", CAP, amount_cents=1000)
    r = refunds.request_refund(s, "p", 300, CAP + timedelta(days=2))
    assert isinstance(r, refunds.Refund)
    p = s.get_payment("p")
    assert p.refunded_cents == 300 and p.state == PaymentState.CAPTURED
    assert [e.kind for e in s.entries("default")][-1] == "refund"


def test_full_refund_moves_to_refunded():
    s = Store()
    captured_payment(s, "p", CAP, amount_cents=1000)
    refunds.request_refund(s, "p", 1000, CAP + timedelta(days=1))
    assert s.get_payment("p").state == PaymentState.REFUNDED


def test_refund_on_authorized_payment_is_refused():
    from ledgerline import payments
    from ledgerline.testing import customer

    s = Store()
    payments.create(s, "p", customer(), 1000, "USD", CAP)
    payments.authorize(s, "p", CAP)
    with pytest.raises(refunds.RefundError):
        refunds.request_refund(s, "p", 100, CAP)


@pytest.mark.policy
def test_late_refund_needs_manager_approval():
    """DR-0005: past the threshold from captured_at, refuse without approval."""
    days = param("refund_approval_after_days")
    s = Store()
    captured_payment(s, "p", CAP, amount_cents=1000)
    late = CAP + timedelta(days=days + 1)
    with pytest.raises(refunds.RefundError):
        refunds.request_refund(s, "p", 100, late)
    assert s.get_payment("p").refunded_cents == 0, "a refused refund changes nothing"
    r = refunds.request_refund(s, "p", 100, late, manager_approved=True)
    assert r.manager_approved is True


@pytest.mark.policy
def test_refund_inside_window_needs_no_approval():
    days = param("refund_approval_after_days")
    s = Store()
    captured_payment(s, "p", CAP, amount_cents=1000)
    refunds.request_refund(s, "p", 100, CAP + timedelta(days=days - 1))
    assert s.get_payment("p").refunded_cents == 100


@pytest.mark.policy
def test_threshold_counts_from_capture_not_settlement():
    days = param("refund_approval_after_days")
    s = Store()
    captured_payment(s, "p", CAP, settled_at=CAP + timedelta(days=days), amount_cents=1000)
    with pytest.raises(refunds.RefundError):
        refunds.request_refund(s, "p", 100, CAP + timedelta(days=days + 2))


@pytest.mark.policy
def test_refund_cap_is_captured_minus_prior_refunds():
    assert param("refund_cap") == "captured_minus_prior_refunds"
    s = Store()
    captured_payment(s, "p", CAP, amount_cents=1000)
    refunds.request_refund(s, "p", 700, CAP + timedelta(days=1))
    with pytest.raises(refunds.RefundError):
        refunds.request_refund(s, "p", 400, CAP + timedelta(days=1))
    assert s.get_payment("p").refunded_cents == 700
