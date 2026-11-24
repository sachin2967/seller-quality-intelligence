# Repository instructions

Read README.md and the relevant operations/evaluation documentation before changing behavior. Preserve the Python API/UI stack, public-data provenance and declared model limitations. This repository has an independent Git history; stage and publish only its own task files.

## Manual planner, executor and reviewer roles

The user selects the role in their prompt; it is independent of model or vendor.
Use the fixed root [WORK_PLAN.md](WORK_PLAN.md) for the active task. The planner
inspects the repository and writes a bounded plan, then marks it READY without
implementing application code. The executor follows that plan, preserves its scope
and acceptance criteria, records actual results and blockers, and marks NEEDS_REVIEW.
The reviewer reads the actual diff and check evidence, then records APPROVED or
CHANGES_REQUESTED; passing tests alone are not approval. With requested corrections,
the executor edits only the reviewer's specified fixes and returns for review.

Do not build a launcher, agent dispatcher, background watcher or model API bridge
for this workflow unless the user explicitly asks. Do not execute an empty or
AWAITING_TASK plan. Keep one active task per repository; use Git history for older
tasks instead of creating changing task-file paths. See the [workflow guide PR](https://github.com/sachin2967/ml-portfolio/pull/2)
for copyable role prompts. A user role-switch hands over the same task branch once
the previous model has stopped. Sequential roles may use the same checkout;
concurrent writers require separate tasks/checkouts and explicit ownership.

## Shared agent GitHub workflow

Owner-approved on 24 November 2026. GitHub is the handoff source of truth for these
independent repositories. The owner authorizes normal commits and pushes of work
within the requested task; repeat permission is not needed for routine publication.

1. Before editing, inspect local status, fetch origin, and read open PRs, their head
   commits, comments and CI. Preserve all existing local and remote work. Do not
   assume that a default branch or a previous agent's summary is current.
2. Use one task branch per repository: `codex/<task>`, `claude/<task>` or `gemini/<task>`.
   Keep one active writer per branch. Sequential roles may share one checkout;
   concurrent work uses separate worktrees or clones. Never switch another agent's
   checkout or change its index, files or uncommitted work.
3. The implementing agent owns the PR. Create one draft PR per task/repository and
   update it with normal commits. The other agent reviews that PR rather than
   opening a duplicate PR for the same branch. Write access changes only after an
   explicit handoff; before taking over, fetch and verify the latest PR head.
4. Run the checks appropriate to each change. Inspect the staged diff for other
   agents' edits, secrets, raw data, model binaries and runtime state. Do not commit
   credentials or private submissions. Keep frozen evaluation inputs and measured
   limitations intact; never describe offline/scripted checks as live quality.
5. Commit and push every meaningful tested milestone, and publish any remaining
   intended changes before ending a turn. Report the PR URL, pushed commit, actual
   check results and unresolved issues. If a push is blocked, state that explicitly;
   never report local-only changes as available to the other agent on GitHub.
6. Fetch immediately before pushing. If the remote moved, preserve both histories;
   merge non-rewriting updates when appropriate, inspect conflicts and rerun checks
   affected by the integration. Never force-push, amend, reset, drop another agent's
   commits or bypass hooks/signing without explicit owner authorization.
7. Keep the default branch as the integration branch. Do not push routine task
   changes directly to it or merge PRs automatically. The owner decides merges
   unless a particular merge is explicitly authorized.
8. Handoffs must state scope, base/head commit, validation, known failures, external
   inputs and the next action. Verify CI against the latest pushed head; pending,
   failed and passed are distinct. A started workflow is not a successful check.
