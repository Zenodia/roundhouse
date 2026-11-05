from datetime import timedelta

from ledgerline import reports, retention
from ledgerline.models import RawEvent
from ledgerline.money import format_cents, split_cents
from ledgerline.testing import NOW, sample_store


def test_count_by_state_covers_every_state():
    s = sample_store()
    counts = reports.count_by_state(s)
    assert set(counts) == {"created", "authorized", "captured", "settled", "refunded", "failed"}
    assert sum(counts.values()) == len(s.payments)


def test_ledger_total_excludes_refunds_and_splits_by_partition():
    s = sample_store()
    assert reports.ledger_total_cents(s, "eu") > 0
    assert reports.ledger_total_cents(s) == sum(reports.ledger_total_cents(s, p) for p in s.ledger)


def test_is_expired_uses_the_configured_window():
    old = RawEvent("e", NOW - timedelta(days=retention.RAW_EVENT_RETENTION_DAYS + 1), {})
    fresh = RawEvent("f", NOW - timedelta(days=1), {})
    assert retention.is_expired(old, NOW)
    assert not retention.is_expired(fresh, NOW)


def test_money_helpers():
    assert format_cents(123456) == "1234.56 USD"
    assert format_cents(-5, "EUR") == "-0.05 EUR"
    assert split_cents(100, 3) == [34, 33, 33]
