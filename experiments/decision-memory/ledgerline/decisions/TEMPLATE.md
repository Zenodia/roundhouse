---
id: DR-XXXX
title: One sentence, present tense, stating the rule
status: proposed            # proposed | accepted | superseded | rejected
owner: Name (role, team)    # the person who can approve or revoke this
decided_on: YYYY-MM-DD      # when the humans decided
effective_from: YYYY-MM-DD  # when systems must follow it (may be later)
supersedes: null            # DR-id this replaces, or null
superseded_by: null         # filled in when a later record replaces this one
scope:                      # files/globs this rule governs; used for retrieval
  - ledgerline/example.py
tags: []
parameters: {}              # machine-readable values policy tests read
source: meeting-notes/YYYY-MM-DD-topic.md
---

## Decision

What was decided, as a rule a system can follow.

## Why

The reasoning, in the words of the people who decided. This is what the
agent uses to interpret cases the rule does not spell out.

## Constraints

Limits, exceptions, transition arrangements, deadlines.

## What it replaces

The previous rule and why it no longer holds. Link the superseded record.
