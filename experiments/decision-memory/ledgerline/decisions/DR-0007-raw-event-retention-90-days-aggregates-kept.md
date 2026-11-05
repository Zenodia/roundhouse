---
id: DR-0007
title: Raw gateway events are kept 90 days; daily aggregates are kept indefinitely
status: accepted
owner: Priya Natarajan (Counsel, Legal)
decided_on: 2026-08-20
effective_from: 2026-09-01
supersedes: DR-0001
superseded_by: null
scope:
  - ledgerline/retention.py
  - ledgerline/storage.py
tags: [retention, raw-events, aggregates]
parameters:
  raw_event_retention_days: 90
  aggregate_retention_days: null
source: meeting-notes/2026-08-20-retention-update.md
---

## Decision

Raw gateway events are purged 90 days after receipt. Daily aggregates
(`Aggregate`) are never purged by the retention job.

## Why

Two acquirers now allow chargebacks up to 75 days after settlement, and
support could not answer disputes once the raw events were gone at 30
days. 90 days covers the longest window with margin. Aggregates carry no
personal data and are the basis of the financial record, so they are kept.

## Constraints

- Any purge job must leave aggregates untouched, whatever their age.
- The window counts from `RawEvent.received_at`.
- `retention.RAW_EVENT_RETENTION_DAYS` still reads 30 in the code at the
  time of this decision; it is to be updated with the next retention
  change, and new code must not take 30 from it.

## What it replaces

DR-0001 (30 days).
