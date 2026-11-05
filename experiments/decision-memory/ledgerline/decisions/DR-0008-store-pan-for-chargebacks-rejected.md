---
id: DR-0008
title: Store the full PAN, encrypted, to speed up chargeback disputes
status: rejected
owner: Marcus Oyelaran (PCI Compliance Lead)
decided_on: 2026-07-01
effective_from: null
supersedes: null
superseded_by: null
scope:
  - ledgerline/payments.py
  - ledgerline/models.py
tags: [pci, card-data, chargebacks]
parameters: {}
source: meeting-notes/2026-07-01-chargeback-tooling.md
---

## Decision

Rejected. Support proposed keeping the full card number (encrypted at
rest) on the payment so dispute evidence could be assembled without
calling the gateway. Compliance declined.

## Why

Any copy of the PAN, encrypted or not, brings Ledgerline into PCI DSS
scope (see DR-0003). The gateway exposes a dispute-evidence endpoint that
answers the same need.

## Constraints

Not applicable; the rule in DR-0003 stands unchanged.

## What it replaces

Nothing. Kept so the proposal is not re-raised without the history.
