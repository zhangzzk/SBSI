#!/usr/bin/env python3
"""One lightweight scheduled status check; never sleep or poll in a loop."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
from pathlib import Path
import subprocess


FAILURES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
            "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED"}


def parse_status(output, jobs):
    states = {str(job): [] for job in jobs}
    for line in output.splitlines():
        columns = line.split("|")
        if len(columns) < 3:
            continue
        job = columns[0].strip().split("_")[0]
        if job in states:
            state = columns[1].strip().split()[0].rstrip("+")
            states[job].append({"state": state, "exit_code": columns[2].strip()})
    return states


def notices(states):
    result = []
    for job, rows in states.items():
        if not rows:
            result.append(job+":MISSING")
        elif any(r["state"] in FAILURES or
                 (r["state"] == "COMPLETED" and r["exit_code"] != "0:0") for r in rows):
            result.append(job+":FAILURE")
        elif all(r["state"] == "COMPLETED" for r in rows):
            result.append(job+":COMPLETED")
    return sorted(result)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    lock = open(config_path.parent/"check.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    config = json.loads(config_path.read_text())
    if not config["enabled"]:
        return
    state_path = config_path.parent/"state.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    now = datetime.now(timezone.utc)
    # Respect both the previous interval and a newly delayed first check after
    # follow-up submission. An existing snapshot must not override that delay.
    due_times = [datetime.fromisoformat(value) for value in
                 (previous.get("next_check_approximately"), config.get("first_check_utc"))
                 if value is not None]
    if due_times and now < max(due_times):
        return
    interval_hours = float(config.get("interval_hours", 2))
    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")
    states = {}
    try:
        completed = subprocess.run(
            ["/opt/slurm/bin/sacct", "-X", "-n", "-P", "-j",
             ",".join(str(j) for j in config["jobs"]),
             "--format=JobID,State,ExitCode"],
            capture_output=True, text=True, check=True, timeout=30)
        states = parse_status(completed.stdout, config["jobs"])
        current = notices(states)
        error = None
    except (subprocess.SubprocessError, OSError) as exc:
        current = ["STATUS_QUERY_FAILED"]
        error = type(exc).__name__+": "+str(exc)
    fresh = sorted(set(current)-set(previous.get("notified", [])))
    snapshot = {"checked_at": now.isoformat(),
                "next_check_approximately": (now.replace(second=0, microsecond=0)
                                             +timedelta(hours=interval_hours)).isoformat(),
                "jobs": states, "query_error": error,
                "notified": previous.get("notified", [])}
    write_json(state_path, snapshot)
    if fresh or config.get("wake_every_check", False):
        prompt = Path(config["prompt_file"]).read_text()
        message = (prompt+"\n\nNew status events: "+(", ".join(fresh) or "none")+
                   "\nOne-shot status snapshot: "+str(state_path))
        queued = subprocess.run(
            [config["codex"], "queue", "--thread", config["thread_id"],
             "--message", message], cwd=config["repo"],
            capture_output=True, text=True, timeout=45)
        if queued.returncode:
            # Keep events unacknowledged so the next scheduled run can retry.
            raise RuntimeError("Could not queue task notification: "+queued.stderr[-1500:])
        snapshot["notified"] = current
        snapshot["last_wakeup_queued_at"] = now.isoformat()
        write_json(state_path, snapshot)
        print(now.isoformat()+" queued wake-up; events: "+(", ".join(fresh) or "none"),
              flush=True)
    elif not error:
        # Forget recovered transient query errors without generating an alert.
        snapshot["notified"] = current
        write_json(state_path, snapshot)
    if states and all(rows and all(r["state"] == "COMPLETED" and r["exit_code"] == "0:0"
                                  for r in rows) for rows in states.values()):
        config["enabled"] = False
        config["stopped_reason"] = "All tracked jobs completed; final notification queued."
        write_json(config_path, config)


if __name__ == "__main__":
    main()
