# Work plan

This is the fixed handoff document for the seller-quality-intelligence repository.
The user chooses the planner, executor and reviewer by prompt. Read AGENTS.md first.

## Task and handoff

- Task ID: Not assigned
- Status: AWAITING_TASK
- Repository: sachin2967/seller-quality-intelligence
- Branch: Planner records the task branch before READY
- Base commit: Planner records the full starting SHA before READY
- Current owner/role: User selects the next planner
- PR: Planner/executor records the existing task PR

No implementation task is queued. Do not execute this empty template.
Status flow: AWAITING_TASK -> READY -> RUNNING -> NEEDS_REVIEW -> APPROVED.
Review may request CHANGES_REQUESTED; missing inputs may cause BLOCKED.
User cancellation is recorded as CANCELLED.

## Goal and context — planner

Record the requested outcome, relevant code/docs, constraints and current behavior.

## Allowed changes and boundaries — planner

Name the exact files/areas to change and any data, API, training or budget limits.
Preserve unrelated edits, existing evaluation evidence and repository contracts.

## Implementation steps — planner

Write small, ordered steps with enough decisions for the executor to proceed.

## Acceptance criteria and checks — planner

Define observable behavior, relevant edge cases, exact commands, expected outcomes
and required inputs. For ML tasks, specify splits, baselines, leakage checks,
metrics and reproducibility before training. READY requires all these sections.

## Execution results — executor

Not started. Preserve the plan above. Record changed files, actual check commands
and results, failures, deviations needing review, remaining gates and publication.
Report the final pushed commit and PR in the handoff message; record code commits
here when available. Do not invent a SHA for the commit containing this record.
Finish at NEEDS_REVIEW or BLOCKED, never self-approve.

## Review and corrections — reviewer

Not reviewed. Inspect the actual base-to-head diff and check evidence. Record
findings with file locations, required corrections, checks performed and the
decision: APPROVED, CHANGES_REQUESTED or BLOCKED. Review does not implement fixes.

## Blockers and next action — current owner

Waiting for the user's first concrete task. Keep the fixed filename; use Git
history for older approved/cancelled tasks. Only one model writes at a time.
