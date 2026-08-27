# Two-Minute Schedule Proof Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox ('- [ ]') syntax for tracking.

**Goal:** Prove that GitHub itself automatically triggers the paper workflow, then restore the permanent daily schedule.

**Architecture:** Temporarily replace the existing cron with the current UTC hour and a minute approximately two minutes ahead, push it to the default branch, and cancel the push preview so it cannot block the concurrency group. Observe a new Actions run whose event is exactly 'schedule', verify its data delta and deployment, then restore '17 4 * * *'.

**Tech Stack:** GitHub Actions cron, GitHub REST API, PowerShell, git.

## Global Constraints

- A manual dispatch is not acceptable evidence for this test.
- The observed run event must be 'schedule'.
- The scheduled run performs the real update and remains capped at 20 papers.
- The final repository cron must be '17 4 * * *'.
- GitHub credentials and DeepSeek secrets must never be printed.

---

### Task 1: Publish the temporary schedule

**Files:**
- Modify: '.github/workflows/deploy.yml'

**Interfaces:**
- Consumes: current UTC hour and minute.
- Produces: a temporary daily cron aimed approximately two minutes after the config push.

- [ ] Compute the UTC target with '[DateTimeOffset]::UtcNow.AddMinutes(2)'.
- [ ] Replace only the cron line with '<target-minute> <target-hour> * * *' and update its comment.
- [ ] Run 'git diff --check' and the 49-test unittest suite.
- [ ] Commit with 'test: schedule automatic update proof' and push to 'origin/master'.
- [ ] Query recent workflow runs and cancel only the push-triggered preview for this commit.

### Task 2: Observe and verify the automatic run

**Files:**
- Generated: 'papers.md'
- Generated: 'site/assets/data.json'
- Generated: 'site/assets/paper-images/**'
- Generated: 'site/papers/**'

**Interfaces:**
- Consumes: GitHub's scheduled event.
- Produces: a successful run with 'event=schedule' and a verified public deployment.

- [ ] Poll the workflow runs API until a run with the temporary commit and 'event=schedule' appears.
- [ ] Poll its jobs until crawler, DeepSeek Flash, images, build, upload, and deploy all conclude 'success'.
- [ ] Fetch 'origin/master' and verify the paper count changes from 144 to 164.
- [ ] Verify all 20 new rows have summaries, full images, thumbnails, and detail pages.
- [ ] Verify the public 'assets/data.json' reports 164 papers.

### Task 3: Restore the permanent schedule

**Files:**
- Modify: '.github/workflows/deploy.yml'

**Interfaces:**
- Consumes: successful schedule proof.
- Produces: the permanent Beijing 12:17 schedule.

- [ ] Restore the comment and cron line to '17 4 * * *'.
- [ ] Run the focused workflow regression test and 'git diff --check'.
- [ ] Commit with 'ci: restore daily 12:17 schedule' and push.
- [ ] Cancel only the restore commit's push-triggered preview.
- [ ] Verify the remote workflow is active, the remote cron is '17 4 * * *', local and remote are synchronized, and the working tree is clean.
