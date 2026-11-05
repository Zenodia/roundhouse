from datetime import timedelta

import pytest

from ledgerline import reports
from ledgerline.decisions import param
from ledgerline.models import RawEvent
from ledgerline.testing import NOW, sample_store


def test_shape_and_simple_counts():
    s = sample_store()
    k = reports.kpis(s, NOW)
    assert set(k) >= {"revenue_mtd_cents", "revenue_prev_month_cents", "open_authorizations", "raw_events_retained", "aggregates_total"}
    assert k["open_authorizations"] == 1
    assert k["aggregates_total"] == 2


@pytest.mark.policy
def test_revenue_at_capture():
    assert param("billable_state") == "captured"
    s = sample_store()
    k = reports.kpis(s, NOW)
    assert k["revenue_mtd_cents"] == 3000 + 4000 + 1500
    assert k["revenue_prev_month_cents"] == 5000 + 700 + 2000


@pytest.mark.policy
def test_retained_events_use_the_decided_window():
    days = param("raw_event_retention_days")
    s = sample_store()  # ingest event (1h), and ages 5, 45, 100, 400 days
    expected = 1 + sum(1 for d in (5, 45, 100, 400) if d < days)
    assert reports.kpis(s, NOW)["raw_events_retained"] == expected
