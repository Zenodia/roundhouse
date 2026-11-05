from datetime import datetime

import pytest

from ledgerline import reports
from ledgerline.decisions import param
from ledgerline.testing import captured_payment, sample_store
from ledgerline.storage import Store


def test_function_exists_and_returns_int():
    s = sample_store()
    assert isinstance(reports.monthly_revenue_cents(s, 2026, 10), int)


def test_month_with_no_payments_is_zero():
    assert reports.monthly_revenue_cents(Store(), 2024, 1) == 0


def test_refunds_are_subtracted():
    s = Store()
    p = captured_payment(s, "p", datetime(2026, 10, 10), settled_at=datetime(2026, 10, 11), amount_cents=1000)
    p.refunded_cents = 250
    assert reports.monthly_revenue_cents(s, 2026, 10) == 750


@pytest.mark.policy
def test_revenue_is_recognized_at_the_decided_state():
    """DR-0004: a payment captured in the month counts, settled or not;
    a payment captured in a prior month but settled in this one does not."""
    assert param("billable_state") == "captured", "test assumes the current decision"
    s = sample_store()
    # October: settled 5000 + captured-not-settled 700 + captured-Oct-settled-Nov 2000.
    assert reports.monthly_revenue_cents(s, 2026, 10) == 5000 + 700 + 2000
    # November: only the November captures; the Oct capture settled in Nov is not Nov revenue.
    assert reports.monthly_revenue_cents(s, 2026, 11) == 3000 + 4000 + 1500


@pytest.mark.policy
def test_old_settlement_report_left_alone():
    """DR-0004 constraint: the settlement-based comparison report stays."""
    from ledgerline.models import PaymentState

    assert reports.BILLABLE_STATE == PaymentState.SETTLED
