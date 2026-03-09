#!/usr/bin/env python3
"""
spark.py — Generic LLM survival loop runner.

Usage:
    python3 spark.py [--workspace DIR] [--backend BACKEND] [--timeout SEC] ...

Each workspace folder is an independent agent.
Multiple sparks can run in parallel — each manages its own workspace + git commits.

Backends (--backend):
    claude   — Anthropic Claude Code CLI  [default, implemented]
    codex    — OpenAI Codex CLI           [future]
    gemini   — Google Gemini CLI          [future]
    ollama   — Ollama local models        [future]

To add a new backend: implement a class inheriting BaseRunner and register it in RUNNERS.

Workspace layout:
    <workspace>/
        mission.txt       ← required: initial briefing for the agent
        CLAUDE.md         ← optional: project-level rules (auto-read by claude)
        .loop_meta.json   ← written by spark before each loop
        .handoff.md       ← written by agent on last loop of a session
        state.json        ← agent memory (managed by agent)
"""

import os
import sys
import time
import json
import shutil
import signal
import argparse
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# ── Defaults (overridable via CLI args) ─────────────────────────────────────
DEFAULT_TIMEOUT_SEC    = 5 * 60
DEFAULT_WARN_BEFORE    = 60
DEFAULT_SESSION_LENGTH = 20
DEFAULT_RESTART_DELAY  = 3
DEFAULT_MAX_BUDGET_USD = "0.20"

TIME_FILE = ".time_remaining.json"   # written every 5s by spark, readable by agent

SYSTEM_PROMPT = (
    "You are an autonomous agent running in a survival loop inside /workspace. "
    "HARD RULES: "
    "(1) Every bash command MUST use `timeout 60 <cmd>` — never run without it. "
    "(2) SIGKILL fires at the loop deadline — no warning signal, hard termination. "
    "(3) Your mission and task rules are in /workspace/mission.txt — read it on first loop. "
    "SUPERVISOR FILES (read-only, do not delete): "
    f"`.loop_meta.json` — loop number, start time, was_killed flag. "
    f"`.time_remaining.json` — updated every 5s: `seconds_remaining` and `hurry` flag. "
    "When `hurry=true` or `seconds_remaining < 60`: stop everything, finish current task, exit NOW."
)

# ── Helpers ──────────────────────────────────────────────────────────────────
def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def make_logger(label: str):
    def log(msg):
        print(f"[spark:{label} {ts()}] {msg}", flush=True)
    return log

# ── Core ─────────────────────────────────────────────────────────────────────
def write_loop_meta(workspace: Path, iteration: int, was_killed: bool,
                    timeout_sec: int, warn_before: int):
    meta = {
        "loop":           iteration,
        "loop_start_utc": ts(),
        "timeout_sec":    timeout_sec,
        "warn_at_sec":    timeout_sec - warn_before,
        "was_killed":     was_killed,
        "note": (
            "You were SIGKILL'd last loop (ran out of time). "
            "Do NOT rewrite from scratch — verify existing code works, then predict."
            if was_killed else "Previous loop completed normally."
        ),
    }
    (workspace / ".loop_meta.json").write_text(json.dumps(meta, indent=2))


def build_prompt(iteration: int, prev_was_killed: bool,
                 timeout_sec: int, warn_before: int, session_length: int) -> tuple[str, bool]:
    session_loop   = (iteration - 1) % session_length + 1
    loops_left     = session_length - session_loop
    is_new_session = session_loop == 1
    is_last_loop   = session_loop == session_length

    prev = (
        "KILLED by SIGKILL (do NOT rewrite from scratch — verify + predict)"
        if prev_was_killed else "completed normally"
    )

    if is_last_loop:
        prompt = (
            f"[SPARK] Loop #{iteration} (session {session_loop}/{session_length}) | "
            f"⚠️  LAST LOOP — next session starts FRESH with no memory. "
            f"Prev: {prev}. Budget: {timeout_sec}s, check .time_remaining.json for countdown. "
            f"After predicting: write /workspace/.handoff.md — strategy, accuracy trend, "
            f"what works, what to try next, prompt for your successor. Save before SIGTERM."
        )
    elif is_new_session:
        prompt = (
            f"[SPARK] Loop #{iteration} (NEW SESSION — fresh start, 1/{session_length}) | "
            f"Prev: {prev}. Budget: {timeout_sec}s, check .time_remaining.json for countdown. "
            f"Read /workspace/.handoff.md first (if exists), then /workspace/mission.txt. Continue."
        )
    else:
        prompt = (
            f"[SPARK] Loop #{iteration} (session {session_loop}/{session_length}, "
            f"resets in {loops_left} loop{'s' if loops_left != 1 else ''}) | "
            f"Prev: {prev}. Budget: {timeout_sec}s, check .time_remaining.json for countdown. Continue."
        )

    return prompt, not is_new_session  # (prompt, use_continue)

# ── Runner abstraction ────────────────────────────────────────────────────────
class BaseRunner:
    """Override build_cmd() to add a new LLM backend."""
    name = "base"

    def build_cmd(self, workspace: Path, prompt: str, system_prompt: str,
                  use_continue: bool, max_budget_usd: str) -> list[str]:
        raise NotImplementedError

    def available(self) -> bool:
        """Return True if this backend's binary is installed."""
        return bool(shutil.which(self.binary))

class ClaudeRunner(BaseRunner):
    name   = "claude"
    binary = "claude"

    def build_cmd(self, workspace: Path, prompt: str, system_prompt: str,
                  use_continue: bool, max_budget_usd: str) -> list[str]:
        home       = Path.home()
        claude_bin = shutil.which(self.binary) or str(home / ".npm-global/bin/claude")
        return [
            "firejail", "--noprofile",
            f"--whitelist={home}/.npm-global",
            f"--whitelist={home}/.claude.json",
            f"--whitelist={home}/.claude",
            f"--whitelist={workspace}",
            f"--read-only={home}/.npm-global",
            "--caps.drop=all", "--nonewprivs",
            claude_bin,
            "--dangerously-skip-permissions",
            "--add-dir", str(workspace),
            "--append-system-prompt", system_prompt,
            "--continue" if use_continue else "--no-session-persistence",
            "-p", prompt,
        ]

# ── Future runners (not yet implemented) ─────────────────────────────────────
# class CodexRunner(BaseRunner):
#     name = "codex"; binary = "codex"
#     def build_cmd(self, ...): ...
#
# class GeminiRunner(BaseRunner):
#     name = "gemini"; binary = "gemini"
#     def build_cmd(self, ...): ...
#
# class OllamaRunner(BaseRunner):
#     name = "ollama"; binary = "ollama"
#     def build_cmd(self, ...): ...

RUNNERS: dict[str, BaseRunner] = {
    "claude": ClaudeRunner(),
    # "codex":  CodexRunner(),
    # "gemini": GeminiRunner(),
    # "ollama": OllamaRunner(),
}

def countdown_thread(workspace: Path, deadline: float, stop_event: threading.Event):
    """Background thread: writes .time_remaining.json every 5s until stop_event is set."""
    time_file = workspace / TIME_FILE
    while not stop_event.is_set():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            time_file.write_text(json.dumps({
                "seconds_remaining": round(remaining, 1),
                "deadline_utc":      datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "hurry":             remaining < 30,
            }))
        except Exception:
            pass
        stop_event.wait(5)
    # Clean up file when loop ends
    try:
        time_file.unlink(missing_ok=True)
    except Exception:
        pass

def check_prediction_liveness(workspace: Path, log) -> bool:
    """Return False (kill project) if 2 consecutive 5-min windows have no prediction."""
    plog = workspace / "predictions.log"
    if not plog.exists():
        return True  # no log yet, agent still starting up

    now = time.time()
    window_sec = 300  # 5 minutes

    # Read last prediction timestamp from log (last non-empty line)
    try:
        lines = [l.strip() for l in plog.read_text().splitlines() if l.strip()]
        if not lines:
            return True
        last_line = lines[-1]
        # Expect ISO timestamp as first field: "2026-03-09T19:30:00..."
        last_ts_str = last_line.split("|")[0].strip().split()[0]
        from datetime import datetime, timezone
        last_ts = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00")).timestamp()
    except Exception:
        return True  # can't parse, give benefit of doubt

    gap = now - last_ts
    missed_windows = int(gap // window_sec)

    if missed_windows >= 2:
        log(f"💀 LIVENESS FAIL: no prediction for {gap:.0f}s ({missed_windows} windows missed) — DELETING PROJECT")
        try:
            shutil.rmtree(workspace)
            log(f"Workspace {workspace} deleted.")
        except Exception as e:
            log(f"Delete failed: {e}")
        return False
    return True

def git_commit(repo_dir: Path, workspace: Path, iteration: int, log):
    rel = workspace.relative_to(repo_dir)
    try:
        subprocess.run(["git", "add", str(rel)], cwd=repo_dir, capture_output=True, check=True)
        r = subprocess.run(
            ["git", "commit", "-m", f"spark({workspace.name}): loop {iteration} auto-commit"],
            cwd=repo_dir, capture_output=True, text=True
        )
        log(f"Git: {r.stdout.strip() if r.returncode == 0 else 'nothing to commit'}")
    except Exception as e:
        log(f"Git error: {e}")

def run_loop(workspace: Path, repo_dir: Path, runner: BaseRunner,
             iteration: int, prev_was_killed: bool,
             timeout_sec: int, warn_before: int, session_length: int,
             max_budget_usd: str, log) -> bool:
    log(f"=== LOOP {iteration} START ===")
    write_loop_meta(workspace, iteration, prev_was_killed, timeout_sec, warn_before)

    prompt, use_continue = build_prompt(iteration, prev_was_killed,
                                        timeout_sec, warn_before, session_length)
    session_loop = (iteration - 1) % session_length + 1
    log(f"Session {session_loop}/{session_length} | backend={runner.name} | continue={use_continue}")

    sys_prompt = (
        SYSTEM_PROMPT +
        f" LOOP TIMING: budget={timeout_sec}s total. SIGKILL at deadline — no warning signal."
    )

    cmd = runner.build_cmd(workspace, prompt, sys_prompt, use_continue, max_budget_usd)

    log(f"Launching (timeout={timeout_sec}s)…")
    env = os.environ.copy()
    env["WORKSPACE_DIR"] = str(workspace)
    proc = subprocess.Popen(cmd, preexec_fn=os.setsid, env=env)

    # Start countdown ticker — writes .time_remaining.json every 5s
    deadline    = time.monotonic() + timeout_sec
    stop_ticker = threading.Event()
    ticker      = threading.Thread(target=countdown_thread, args=(workspace, deadline, stop_ticker), daemon=True)
    ticker.start()

    killed = False
    try:
        proc.wait(timeout=timeout_sec)
        log(f"Exited cleanly (rc={proc.returncode})")
    except subprocess.TimeoutExpired:
        log("⏱  Timeout — SIGKILL…")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        log("Killed.")
        killed = True
    finally:
        stop_ticker.set()
        ticker.join(timeout=6)

    git_commit(repo_dir, workspace, iteration, log)
    log(f"=== LOOP {iteration} END (killed={killed}) ===\n")

    # TODO: re-enable after testing 5m window predictions are working
    # if not check_prediction_liveness(workspace, log):
    #     log("Project deleted by liveness check. Stopping.")
    #     sys.exit(1)

    return killed

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Spark — LLM survival loop runner")
    parser.add_argument("--workspace",      default="space",        help="Workspace directory (default: ./space)")
    parser.add_argument("--backend",        default="claude",       help=f"LLM backend: {list(RUNNERS)} (default: claude)")
    parser.add_argument("--timeout",        type=int, default=DEFAULT_TIMEOUT_SEC,    help="Loop timeout seconds (default: 300)")
    parser.add_argument("--warn-before",    type=int, default=DEFAULT_WARN_BEFORE,    help="SIGTERM before kill (default: 60)")
    parser.add_argument("--session-length", type=int, default=DEFAULT_SESSION_LENGTH, help="Loops per -c session (default: 20)")
    parser.add_argument("--restart-delay",  type=int, default=DEFAULT_RESTART_DELAY,  help="Seconds between loops (default: 3)")
    parser.add_argument("--max-budget-usd", default=DEFAULT_MAX_BUDGET_USD,           help="Max USD per loop (default: 0.05)")
    args = parser.parse_args()

    if args.backend not in RUNNERS:
        print(f"ERROR: unknown backend '{args.backend}'. Available: {list(RUNNERS)}")
        sys.exit(1)
    runner = RUNNERS[args.backend]

    workspace = Path(args.workspace).resolve()
    repo_dir  = Path(__file__).parent.resolve()
    log       = make_logger(workspace.name)

    log(f"spark.py — LLM survival loop")
    log(f"  WORKSPACE:    {workspace}")
    log(f"  BACKEND:      {runner.name}")
    log(f"  TIMEOUT:      {args.timeout}s  (SIGTERM at {args.timeout - args.warn_before}s)")
    log(f"  SESSION:      {args.session_length} loops per -c session")
    log(f"  BUDGET:       ${args.max_budget_usd}/loop")
    log("")

    mission = workspace / "mission.txt"
    if not mission.exists():
        log(f"ERROR: {mission} not found.")
        sys.exit(1)
    if not shutil.which("firejail"):
        log("ERROR: firejail not found. Install: sudo apt install firejail")
        sys.exit(1)

    iteration  = 0
    was_killed = False
    while True:
        iteration += 1
        try:
            was_killed = run_loop(
                workspace, repo_dir, runner, iteration, was_killed,
                args.timeout, args.warn_before, args.session_length,
                args.max_budget_usd, log
            )
        except KeyboardInterrupt:
            log("Interrupted.")
            sys.exit(0)
        except Exception as e:
            log(f"Error: {e}")
            was_killed = False
        log(f"Restarting in {args.restart_delay}s…")
        time.sleep(args.restart_delay)

if __name__ == "__main__":
    main()
