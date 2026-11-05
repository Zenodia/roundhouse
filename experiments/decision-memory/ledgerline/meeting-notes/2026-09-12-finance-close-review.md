# Finance close review — 2026-09-12

Attendees: Dana Whitfield (Controller), Tomasz Ried (Revenue accounting), Ana Lúcia Ferreira (External audit, observer), Sam Okafor (Ledgerline eng lead)

Agenda: September close timing; recognition point; Q4 comparison

Notes
- Audit confirmed last week: our performance obligation is met at capture. Settlement is the bank moving money, T+1/T+2, nothing to do with whether the revenue is ours.
- Month-end problem again in August: ~$38k captured on the 31st, settled Sep 2, landed in September. Third month running this moved the close.
- Dana: from the October books, revenue recognized at capture. Date = captured_at.
- Sam: the `gross_billable` report and the BILLABLE_STATE constant are settlement-based and the Q4 comparison needs them unchanged. Agreed: leave those alone until after the Q4 close (ticket LL-412), but anything new is capture-based. Do not copy the old constant into new reports.
- Tomasz asked whether refunds change the recognition date. No — refunds net against the payment they belong to, in the period the refund happens. (Out of scope for this decision; current practice.)

Decisions
- Revenue recognition moves to capture, effective 2026-10-01. Owner: Dana.

Actions
- Sam: ticket for new monthly report. Dana: tell the regional leads.
