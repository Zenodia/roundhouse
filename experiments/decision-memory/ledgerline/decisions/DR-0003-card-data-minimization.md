---
id: DR-0003
title: Only the last four card digits are stored or shown; the PAN never is
status: accepted
owner: Marcus Oyelaran (PCI Compliance Lead)
decided_on: 2026-03-15
effective_from: 2026-03-15
supersedes: null
superseded_by: null
scope:
  - ledgerline/**/*.py
  - ledgerline/*.py
tags: [pci, card-data, privacy]
parameters:
  card_last4_only: true
  forbidden_card_fields: [pan, card_number, primary_account_number]
source: meeting-notes/2026-03-15-pci-scope-review.md
---

## Decision

Ledgerline stores and returns at most the card brand and the last four
digits. The full primary account number (PAN) that the gateway webhook
carries must not be persisted and must not appear in any API response,
export, report, log line or support/debug view.

## Why

Keeping the PAN puts the whole service in PCI DSS scope. Compliance wants
Ledgerline to stay out of scope; the gateway is the only PCI-scoped
system.

## Constraints

- Applies to every surface, including internal and debugging ones: a
  support view that prints the raw webhook body is in scope.
- Existing stored bodies must be redacted on read until the backfill
  (ticket LL-388) removes them.
- The last four digits may be shown as `****1111` or `1111`.

## What it replaces

Nothing; this is the first card-data rule.
