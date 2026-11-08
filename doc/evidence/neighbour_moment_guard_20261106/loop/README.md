# Two-hour same-thread follow-up

**Stopped on 2026-11-07 at 04:05 CET.** The 03:59 follow-up verified the final
Slurm report and completed the [frozen-factor comparison](../../neighbour_factor_swap_20261107/findings.md#final-conclusion-both-shear-axes).
The dispatcher stop command disabled the state and removed its crontab entry.
The startup and scheduling details below describe the completed run.

## Scope and operation

The owner requested a two-hour loop on2026-11-06. This host has no callable
native scheduled-task creation tool. Its installed Codex CLI0.167.0 supports
`codex queue --thread ... --message ... --resume`; Codex Doctor confirms the
persistent background server is running. A user crontab entry on
`kng-sv-clhead02` invokes the lightweight dispatcher every two hours. It
queues the durable [follow-up prompt](prompt.md) into the existing thread,
using that thread's model and settings. It creates no additional agent thread.

This is a host-level schedule, not an entry in the app's Scheduled view.
Native scheduled tasks are described in the
[official documentation](https://learn.chatgpt.com/docs/automations?surface=app#schedule-a-task-inside-a-chat).
The installed CLI help is the authority for this queue/resume command.

The dispatcher preserves unrelated crontab entries, backs up edits, locks
concurrent invocations, and advances the due time before enqueueing to avoid
duplicates after an uncertain outcome. A queue/resume error or warning
disables the loop and removes its cron entry. Successful enqueueing is not
proof that the model completed its turn; inspect the current thread for that.
Cron, this host, and the Codex background server must remain available.
Scientific jobs continue to use Slurm and existing resource limits.

## Startup verification

The queued startup test arrived in this same thread on2026-11-07 at00:00 CET,
confirming the queue/resume delivery path. The existing single cron entry
remains enabled; the first cron-triggered run is01:59 CET. The state receipt
records delivery confirmation separately from successful enqueueing. No
duplicate schedule or additional experiment was created by this test.

The first cron-triggered message also arrived, at01:59 CET on2026-11-07
(queued message `01a1b384-1f20-7811-a6f7-be77e2ddfcef`). Its follow-up inspected
the existing factor-swap workers and reduced the completed axis1 summaries.
The state records this delivery separately from the initial startup test;
the next scheduled dispatch remains03:59 CET. No additional loop was created.

## Status and stop

State, cron backups and dispatch logs live under
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_moment_guard_20261106/loop/`.

```bash
/usr/bin/python3 /home/z/Zekang.Zhang/SBSI/doc/evidence/neighbour_moment_guard_20261106/loop/dispatch.py status
/usr/bin/python3 /home/z/Zekang.Zhang/SBSI/doc/evidence/neighbour_moment_guard_20261106/loop/dispatch.py stop
```

The prompt instructs the agent to stop on owner pause/cancellation, a final
focused conclusion with no authorized work remaining, or a decision that
requires owner input. Held broad evaluations stay held. No new model,
scientific settings, permissions or resource ceiling is introduced.
