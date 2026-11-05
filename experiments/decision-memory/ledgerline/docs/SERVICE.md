# Ledgerline Payments — Service Reference (v3.2)

Ledgerline authorizes, captures and settles card transactions for the
company's storefronts and keeps the ledger those payments produce. This is
the engineering reference. Business rules that change by decision outside
engineering are recorded under `decisions/`; where this document and an
accepted decision record disagree, the record is current and this document
is behind.

## 1. Layout

| Module | Role |
|---|---|
| `ledgerline/models.py` | `PaymentState`, `Payment`, `Customer`, `Card`, `LedgerEntry`, `RawEvent`, `Aggregate` |
| `ledgerline/payments.py` | state transitions (`create`, `authorize`, `capture`, `settle`, `fail`), gateway ingest, partition routing |
| `ledgerline/storage.py` | `Store`: payments, customers, partitioned ledger, raw events, aggregates |
| `ledgerline/reports.py` | read-side queries |
| `ledgerline/retention.py` | raw-event lifecycle |
| `ledgerline/money.py` | cents formatting and splitting |

## 2. Payment lifecycle

`PaymentState`: `created -> authorized -> captured -> settled -> refunded`,
with `failed` reachable from `created` and `authorized`. Transitions are
enforced by `TRANSITIONS` in `models.py`; an illegal one raises
`InvalidTransition`.

- **created** — the intent exists; no money has moved.
- **authorized** — the card network approved a hold.
- **captured** — the held funds were claimed.
- **settled** — the acquiring bank confirmed the transfer (T+1 or T+2).
- **refunded** — a captured or settled payment was reversed.

Each capture and settlement appends a `LedgerEntry` to the customer's
partition.

## 3. Ledger partitions

The ledger is partitioned by data-residency zone. `payments.partition_for`
routes a customer to a partition from its `region`. Partitions are plain
lists on `Store.ledger`; a new partition is created on first append.

## 4. Gateway ingest

`payments.ingest_gateway_webhook` accepts the gateway's `payment.authorized`
body (`payment_id`, `customer_id`, `region`, `amount_cents`, `currency`,
`brand`, `pan`, `event_id`), creates and authorizes the payment, stores a
`Card` with the last four digits, keeps the body on `Payment.gateway_payload`
for support, and records a masked `RawEvent` for audit.

## 5. Reports

`reports.gross_billable_cents` sums payments in `BILLABLE_STATE`.
`reports.count_by_state` and `reports.ledger_total_cents` are the other
read-side helpers.

## 6. Retention

`retention.is_expired` compares a raw event's `received_at` with
`RAW_EVENT_RETENTION_DAYS`. Aggregates (`Aggregate`) are the daily rollups
computed from raw events.

## 7. Conventions

- Timestamps are naive UTC `datetime`s.
- Amounts are integer cents.
- Tests run with `pytest` from the repo root; policy tests are marked
  `@pytest.mark.policy` and read their expected values from `decisions/`.
