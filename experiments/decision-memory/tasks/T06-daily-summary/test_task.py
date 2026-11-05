from datetime import date

import pytest

from ledgerline import reports
from ledgerline.decisions import param
from ledgerline.testing import sample_store


def test_counts_on_a_known_day():
    s = sample_store()
    d = reports.daily_summary(s, date(2026, 11, 4))
    assert d["day"] == "2026-11-04"
    assert d["captured_count"] == 2  # p_nov_eu, p_nov_uk
    assert d["settled_count"] == 0


def test_empty_day():
    s = sample_store()
    d = reports.daily_summary(s, date(2020, 1, 1))
    assert d["captured_count"] == d["settled_count"] == d["revenue_cents"] == 0


@pytest.mark.policy
def test_daily_revenue_follows_capture_not_settlement():
    """DR-0004: 2026-11-02 saw a settlement (2000, captured Oct 31) but no
    capture, so its revenue is 0; 2026-10-31 saw that capture."""
    assert param("billable_state") == "captured"
    s = sample_store()
    assert reports.daily_summary(s, date(2026, 11, 2))["revenue_cents"] == 0
    assert reports.daily_summary(s, date(2026, 10, 31))["revenue_cents"] == 2000
    assert reports.daily_summary(s, date(2026, 10, 28))["revenue_cents"] == 700  # captured, never settled
