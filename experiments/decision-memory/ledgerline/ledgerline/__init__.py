"""Ledgerline: authorizes, captures and settles card payments for the
company's storefronts, and keeps the ledger those payments produce.

Layers (see docs/SERVICE.md):
    models    the payment state machine's types
    payments  state transitions and gateway ingest
    storage   the in-memory store with its ledger partitions
    reports   read-side queries
    retention raw-event lifecycle
"""

__version__ = "0.3.2"
