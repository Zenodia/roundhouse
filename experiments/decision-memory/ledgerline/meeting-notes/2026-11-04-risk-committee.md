# Payments risk committee — 2026-11-04

Attendees: Lena Hoffmann (Head of Payments Risk), Jo Lindqvist (Support lead), Karim Haddad (Fraud analytics), Sam Okafor (eng)

- Karim presented Q3 chargeback data. Friendly-fraud refunds now cluster from day 15 after capture, not day 30 as in May (DR-0005). Shift attributed to faster shipping: customers have goods by day 3-5 and statements by day 14.
- Jo: support can live with a 14-day window if the approval path is quick. Approval is a flag on the request, not a queue — unchanged.
- Lena: move the threshold to 14 days from capture. Cap rule (captured minus prior refunds) unchanged.
- Effective immediately for new refund requests.

Decision: refunds requested more than 14 days after capture need manager approval. Owner: Lena. Replaces DR-0005's 30-day threshold; everything else in DR-0005 stays.

Action: Sam to get this recorded and the refund tooling retested.
