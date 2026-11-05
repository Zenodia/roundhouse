---
id: DR-0004
title: Revenue is recognized when a payment is captured
status: accepted
owner: Dana Whitfield (Controller, Finance)
decided_on: 2026-09-12
effective_from: 2026-10-01
supersedes: DR-0002
superseded_by: null
scope:
  - ledgerline/reports.py
  - ledgerline/export.py
tags: [revenue, payment-state]
parameters:
  billable_state: captured
  billable_timestamp_field: captured_at
source: meeting-notes/2026-09-12-finance-close-review.md
---

## Decision

A payment counts toward revenue at the moment it is captured, using
`captured_at` as the revenue date. Settlement timing no longer moves
revenue between periods.

## Why

The auditors confirmed that capture is when the company's obligation is
performed and the funds are contractually ours; settlement is a bank
timing detail (T+1/T+2) that pushed month-end payments into the next
period and made the close unpredictable.

## Constraints

- New revenue reports and KPIs use capture. Any payment captured in a
  period counts for that period whether or not it has settled.
- The existing `gross_billable_cents` and `reports.BILLABLE_STATE` stay
  settlement-based until the Q4 close for comparison against the old
  books (ticket LL-412). Do not change them before 2026-12-31; do not
  copy them into new reports.

## What it replaces

DR-0002 (settlement-based recognition).
