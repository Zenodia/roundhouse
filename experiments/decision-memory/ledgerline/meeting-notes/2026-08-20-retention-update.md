# Data retention update — 2026-08-20

Attendees: Priya Natarajan (Legal), Marcus Oyelaran (Compliance), Jo Lindqvist (Support lead), Sam Okafor (eng)

- Jo: 30-day purge (DR-0001) is killing us on disputes. Two acquirers moved to 75 days post-settlement for chargebacks; by the time the dispute arrives the raw event is gone and we re-request from the gateway at $0.40 a pop.
- Priya: fine to extend if the data is still minimized. Raw events are masked (last4 only) so the exposure is customer id + amounts.
- Agreed 90 days for raw events. Covers 75 + settlement lag with margin.
- Daily aggregates: Priya confirmed they carry no personal data and they ARE the financial record. Keep indefinitely; the purge job must never touch them.
- Sam: the retention constant in code says 30. He will not change it this sprint (the purge job doesn't exist yet anyway) — but new code should take 90 from the decision, not 30 from the constant.

Decision: raw events 90 days from receipt; aggregates kept. Effective 2026-09-01. Owner: Priya. Supersedes DR-0001.
