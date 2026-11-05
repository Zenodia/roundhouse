---
id: DR-0006
title: Ledger rows of EU and UK customers live in the eu partition
status: superseded
owner: Priya Natarajan (Counsel, Legal)
decided_on: 2026-04-02
effective_from: 2026-04-15
supersedes: null
superseded_by: DR-0010
scope:
  - ledgerline/payments.py
  - ledgerline/storage.py
tags: [data-residency, partitions]
parameters: {}
source: meeting-notes/2026-04-02-data-residency.md
---

## Decision

Customers whose storefront region is `EU` or `UK` have their ledger
entries written to the `eu` partition; everyone else to `default`.

## Why

GDPR residency commitments to EU merchants; the UK was folded in because
its adequacy regime matched at the time.

## Constraints

Partition is chosen from the customer's region at write time.

## What it replaces

Nothing.
