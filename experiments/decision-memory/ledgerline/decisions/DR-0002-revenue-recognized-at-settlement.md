---
id: DR-0002
title: Revenue is recognized when a payment settles
status: superseded
owner: Dana Whitfield (Controller, Finance)
decided_on: 2026-01-20
effective_from: 2026-02-01
supersedes: null
superseded_by: DR-0004
scope:
  - ledgerline/reports.py
tags: [revenue, payment-state]
parameters: {}
source: meeting-notes/2026-01-20-finance-close-kickoff.md
---

## Decision

A payment counts toward revenue once the acquiring bank confirms the
transfer, i.e. in state `settled`. Revenue reports filter on settlement.

## Why

Finance treated the bank confirmation as the only moment the money was
certain. `reports.BILLABLE_STATE` encodes this.

## Constraints

None.

## What it replaces

Nothing.
