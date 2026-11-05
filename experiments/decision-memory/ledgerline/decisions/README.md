# Decision records

The digital copy of decisions made outside the code: in finance reviews,
risk meetings, legal sign-offs. Each file is one decision with its owner,
date, scope and rationale. `status: accepted` records are the current rules;
`proposed` is not yet a decision, `rejected` never was one, `superseded`
is history.

How a decision gets here:

1. Someone decides something in a meeting. The meeting note lands in
   `meeting-notes/`.
2. `capture/draft_record.py` (or a person) drafts a record from the note
   with `status: proposed` and opens a pull request.
3. The decision owner — not an engineer — approves the PR. `CODEOWNERS`
   routes `decisions/` to the owners. Merging with `status: accepted` is
   the moment the decision exists for every system and agent reading this
   folder.
4. When a decision changes, a new record supersedes the old one; the old
   one is kept with `status: superseded`.

`python -m ledgerline.decisions validate` checks the folder is consistent;
CI runs it. Policy tests (`@pytest.mark.policy`) read their expected values
from `parameters:` here, so a changed decision changes the tests.

A record informs. It does not grant anyone permission to act, and it does
not resolve a conflict between two accepted records — `validate` refuses
that state.
