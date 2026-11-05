---
id: DR-0001
title: Gateway raw events are kept for 30 days
status: superseded
owner: Priya Natarajan (Counsel, Legal)
decided_on: 2026-02-03
effective_from: 2026-03-01
supersedes: null
superseded_by: DR-0007
scope:
  - ledgerline/retention.py
  - ledgerline/storage.py
tags: [retention, raw-events]
parameters: {}
source: meeting-notes/2026-02-03-data-retention-review.md
---

## Decision

Raw gateway events (`RawEvent`) are purged 30 days after receipt.

## Why

The events carry customer identifiers and masked card data. Legal wanted
the smallest window support could live with; support said disputes are
usually raised within two weeks.

## Constraints

Daily aggregates derived from the events are not covered by this record.

## What it replaces

Nothing; events were kept indefinitely before.
