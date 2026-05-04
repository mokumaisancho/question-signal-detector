# RCA: Repeated Memory Management Failures During Model Benchmarking

**Date:** 2026-05-04
**Severity:** Critical — blocked all benchmark testing, wasted 45+ minutes

---

## Timeline of Failures

| Time | What I Did | What Happened | Why It Failed |
|------|-----------|---------------|---------------|
| 09:21 | Started 4 parallel huggingface-cli downloads | 2 hung on cache locks | No lock contention handling |
| 09:22 | Started 4 parallel wget downloads as fallback | All 4 ran, 3 completed | OK |
| 09:38 | First benchmark attempt (python3) | Missing llama_cpp | Wrong Python binary |
| 09:40 | Second attempt (~/.venv/bin/python3) | OOM killed at model load | 2 hung HF processes still consuming 2GB |
| 09:52 | Killed HF processes, retried | OOM killed again | 2 NEW zombie Python processes from failed attempts |
| 10:04 | Killed zombies, retried | OOM killed again | PID 66938 (2.5GB) from 10:04 still alive, unnoticed |
| 10:10 | Retried again | OOM killed again | PID 68896 (3.3GB) spawned ON TOP of 66938 |
| 10:14 | Killed both, waited for recovery | OOM killed again | System swap at 5.3GB, load at 58 |
| 11:04 | "Final" attempt with memory guard | User killed it | 8.4GB zombie (PID 85008) spawned and went unnoticed |

**Total zombie processes created:** 8+
**Peak wasted memory:** 8.4GB in a single zombie
**Peak swap:** 7.1GB
**Peak load average:** 140 (on 8 cores)

---

## Root Causes

### RCA-1: Background Task Orphan Problem

**Problem:** When `run_in_background` spawns a Python process that gets OOM-killed, the **parent zsh shell stays alive** holding the allocated memory. The shell snapshot mechanism (`~/.claude/shell-snapshots/`) creates a zsh process that doesn't die when the Python child dies.

**Why I missed it:** The background task reports "failed" but the actual process tree (zsh → python) means the zsh parent keeps the memory mapped. My cleanup only checked for `benchmark_model` in the process name, not for orphaned zsh shells holding model memory.

**Evidence:** Every failed attempt left a zombie at the same memory level as the model size (~4GB).

### RCA-2: No Pre-Flight Process Scan

**Problem:** Before each benchmark attempt, I never checked for existing high-memory Python processes. I assumed previous attempts had been fully cleaned up.

**Why I missed it:** I checked `ps aux | grep benchmark_model` which only matches the command name. The actual processes were Python interpreters (`/opt/homebrew/Cellar/python@3.14/...`) that don't contain "benchmark_model" in their comm field.

**Fix:** Memory guard must scan by RSS size + Python comm, not by command-line pattern.

### RCA-3: Sequential Retries Without Verification

**Problem:** I launched 5+ sequential benchmark attempts, each taking 30-60 seconds to fail. Between attempts, I only checked the output file (which showed "Loading...") and didn't verify the process tree was clean.

**Why I missed it:** The output file from a background task doesn't update in real-time. By the time I read "Loading...", the process was already dead, and a new attempt was spawning alongside the zombie from the previous attempt.

### RCA-4: No Feedback Loop Between Monitor and Action

**Problem:** I set up memory monitors that reported high swap/load, but I didn't use those reports to gate the next action. I saw "swap=7.1GB, load=140" and still launched a 4GB model load.

**Why I missed it:** The monitor events arrived as notifications, not as blocking conditions. I treated them as informational rather than as hard stops.

---

## The Fix: Memory Safety Protocol

### Principle
**Never launch a model load without first verifying:**
1. No orphan Python processes exist with >500MB RSS
2. Swap is below 3GB
3. Load average is below 20
4. Model size < 50% of (free RAM + inactive pages)

### Implementation

```python
class MemorySafetyGuard:
    """Enforce memory safety before model operations."""

    MAX_SWAP_GB = 3.0
    MAX_LOAD = 20.0
    MAX_ORPHAN_RSS_MB = 500

    @staticmethod
    def audit() -> dict:
        """Full memory and process audit."""
        # 1. Get system stats
        # 2. Find all Python processes with RSS > 500MB
        # 3. Find all zsh children of claude that might be orphans
        # 4. Return report

    @staticmethod
    def cleanup() -> int:
        """Kill orphan processes. Returns count killed."""
        # 1. Kill Python processes >500MB that aren't current process
        # 2. Kill zsh shells whose Python children are dead
        # 3. Wait 2 seconds
        # 4. gc.collect()
        # 5. Return cleanup count

    @staticmethod
    def can_load_model(model_path: str) -> tuple[bool, str]:
        """Check if model can safely be loaded."""
        # 1. audit()
        # 2. cleanup()
        # 3. Check swap < threshold
        # 4. Check load < threshold
        # 5. Check model_size < available_memory
        # 6. Return (can_load, reason)
```

### Process: IDENTIFY → RCA → SAFELY KILL → FIX → RESUME

1. **IDENTIFY**: `ps aux | grep Python | awk '$6 > 500000'` — find any Python process >500MB
2. **RCA**: Check if it's a model-loading process (has Metal/GGUF files open via lsof)
3. **SAFELY KILL**: `kill <pid>` then verify with `ps -p <pid>` after 2 seconds
4. **FIX**: `gc.collect()`, wait for swap to drain
5. **RESUME**: Only after `can_load_model()` returns True

---

## Lessons Learned

1. **Never assume background task cleanup.** The `run_in_background` mechanism does NOT guarantee child process cleanup on failure.
2. **Always scan by RSS, not by command name.** High-memory Python processes are the threat, regardless of what they're running.
3. **Gate model loads on system state.** Swap >3GB or load >20 means DON'T TRY.
4. **Kill before load, not after.** Proactive orphan cleanup before every model load attempt.
5. **One model at a time.** Never have multiple model-loading processes alive simultaneously.
