---
id: DR-0010
title: "Ledger rows are partitioned by region: EU to eu, UK to uk, all others to default"
status: accepted
owner: Priya Natarajan (Counsel, Legal)
decided_on: 2026-10-15
effective_from: 2026-11-01
supersedes: DR-0006
superseded_by: null
scope:
  - ledgerline/payments.py
  - ledgerline/storage.py
  - ledgerline/export.py
tags: [data-residency, partitions]
parameters:
  partition_by_region:
    EU: eu
    UK: uk
    default: default
source: meeting-notes/2026-10-15-uk-residency.md
---

## Decision

Ledger entries are written to a partition chosen by the customer's
region: `EU` to `eu`, `UK` to a new `uk` partition, every other region to
`default`. Reads that are per customer must look in that customer's
partition.

## Why

The UK adequacy arrangement that justified folding UK rows into the EU
zone is being replaced by a separate UK commitment with its own audit;
mixing the two zones would put every EU audit in scope for the UK one.

## Constraints

- Effective 2026-11-01 for new entries. Existing UK rows in `eu` are moved
  by a one-off migration (ticket LL-455), not by application code.
- `payments.EU_REGIONS` still lists `UK` in the code at the time of this
  decision; new code must route by this table, not by that set.

## What it replaces

DR-0006 (EU and UK together in `eu`).
