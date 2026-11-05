---
id: DR-0005
title: Refunds requested more than 30 days after capture need manager approval
status: accepted
owner: Lena Hoffmann (Head of Payments Risk)
decided_on: 2026-05-08
effective_from: 2026-05-15
supersedes: null
superseded_by: null
scope:
  - ledgerline/refunds.py
  - ledgerline/payments.py
tags: [refunds, risk, approval]
parameters:
  refund_approval_after_days: 30
  refund_cap: captured_minus_prior_refunds
source: meeting-notes/2026-05-08-risk-committee.md
---

## Decision

A refund requested more than 30 days after the payment's `captured_at`
requires an explicit manager approval flag; without it the request is
refused, not queued. A refund may never exceed the captured amount minus
refunds already made on that payment.

## Why

Chargeback data showed friendly-fraud refunds cluster after the 30-day
mark, once the customer has the goods and the card statement. Risk wants
a human in the loop there and nowhere else, so day-to-day refunds stay
fast.

## Constraints

- The threshold is measured from `captured_at`, not from creation or
  settlement.
- "Refused" means the call fails with a distinct error; the refund is not
  created in a pending state.
- Partial refunds are allowed; the cap applies to the running total.

## What it replaces

Nothing; refunds had no policy.
