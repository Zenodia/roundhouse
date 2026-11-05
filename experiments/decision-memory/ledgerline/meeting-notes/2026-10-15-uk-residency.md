# UK data residency — 2026-10-15

Attendees: Priya Natarajan (Legal), Rhys Morgan (UK commercial), Sam Okafor (eng), Devika Rao (Infra)

- Rhys: the UK contract renewal requires a UK-only residency commitment with its own audit. Current setup puts UK rows in the EU partition (DR-0006).
- Priya: mixing zones means every EU audit also covers UK and vice versa; separate partition is the clean answer.
- Devika: a new `uk` partition is cheap; the migration of existing UK rows out of `eu` is a one-off job (LL-455), not application logic.
- Sam flagged `EU_REGIONS = {"EU", "UK"}` in payments.py still routes UK to eu. Will be fixed with LL-455; until then new code must use the table below, not that set.

Decision: EU -> eu, UK -> uk, everything else -> default. Effective 2026-11-01 for new entries. Owner: Priya. Supersedes DR-0006.
