"""MemorySafetyGuard — enforces IDENTIFY → RCA → SAFELY KILL → FIX → RESUME.

Prevents zombie process accumulation and OOM kills during model benchmarking.
"""
from __future__ import annotations

import gc
import os
import signal
import subprocess
import time


class MemorySafetyGuard:
    MAX_SWAP_GB = 3.0
    MAX_LOAD_1M = 20.0
    MIN_FREE_PCT = 40.0
    ORPHAN_RSS_THRESHOLD_KB = 500_000  # 500MB

    @staticmethod
    def audit() -> dict:
        """Full system memory + process audit."""
        # Swap
        vm = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            capture_output=True, text=True,
        )
        parts = vm.stdout.strip().split("used = ")
        swap_mb = float(parts[1].split("M")[0]) if len(parts) > 1 else 0.0

        # Load
        ld = subprocess.run(
            ["sysctl", "-n", "vm.loadavg"],
            capture_output=True, text=True,
        )
        nums = ld.stdout.strip().strip("{}").split()
        load_1m = float(nums[0]) if nums else 0.0

        # Memory pressure
        mp = subprocess.run(
            ["memory_pressure"],
            capture_output=True, text=True,
        )
        free_pct = 0.0
        for line in mp.stdout.splitlines():
            if "free percentage" in line:
                free_pct = float(line.split(":")[-1].strip().rstrip("%"))

        # Python orphans
        orphans = MemorySafetyGuard._find_orphans()

        return {
            "swap_mb": swap_mb,
            "swap_gb": swap_mb / 1024,
            "load_1m": load_1m,
            "free_pct": free_pct,
            "orphans": orphans,
            "safe": (
                swap_mb / 1024 < MemorySafetyGuard.MAX_SWAP_GB
                and load_1m < MemorySafetyGuard.MAX_LOAD_1M
                and free_pct > MemorySafetyGuard.MIN_FREE_PCT
                and len(orphans) == 0
            ),
        }

    @staticmethod
    def _find_orphans() -> list[dict]:
        """Find Python processes exceeding RSS threshold."""
        result = subprocess.run(
            ["ps", "-eo", "pid,rss,ppid,comm"],
            capture_output=True, text=True,
        )
        my_pid = os.getpid()
        orphans = []
        for line in result.stdout.strip().split("\n")[1:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            pid, rss, ppid, comm = int(parts[0]), int(parts[1]), int(parts[2]), parts[3]
            if "Python" not in comm and "python" not in comm:
                continue
            if pid == my_pid:
                continue
            if rss < MemorySafetyGuard.ORPHAN_RSS_THRESHOLD_KB:
                continue
            orphans.append({"pid": pid, "rss_mb": rss // 1024, "comm": comm})
        return orphans

    @staticmethod
    def cleanup() -> int:
        """Kill orphan Python processes >500MB. Returns count killed."""
        orphans = MemorySafetyGuard._find_orphans()
        killed = 0
        for proc in orphans:
            print(f"[MemGuard] Killing orphan PID={proc['pid']} ({proc['rss_mb']}MB)", flush=True)
            try:
                os.kill(proc["pid"], signal.SIGTERM)
                killed += 1
            except ProcessLookupError:
                pass
        if killed:
            time.sleep(3)
            # Verify they're gone
            for proc in orphans:
                try:
                    os.kill(proc["pid"], 0)  # Check if still alive
                    # Still alive after SIGTERM, use SIGKILL
                    print(f"[MemGuard] PID={proc['pid']} survived SIGTERM, sending SIGKILL", flush=True)
                    os.kill(proc["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(2)
            gc.collect()
        return killed

    @staticmethod
    def can_load_model(model_path: str) -> tuple[bool, str]:
        """Check if a model can safely be loaded. Runs cleanup first."""
        # Phase 1: Kill orphans
        killed = MemorySafetyGuard.cleanup()
        if killed:
            print(f"[MemGuard] Cleaned up {killed} orphan(s)", flush=True)

        # Phase 2: Audit system
        state = MemorySafetyGuard.audit()

        if state["swap_gb"] > MemorySafetyGuard.MAX_SWAP_GB:
            return False, f"Swap {state['swap_gb']:.1f}GB > {MemorySafetyGuard.MAX_SWAP_GB}GB threshold"
        if state["load_1m"] > MemorySafetyGuard.MAX_LOAD_1M:
            return False, f"Load {state['load_1m']:.1f} > {MemorySafetyGuard.MAX_LOAD_1M} threshold"
        if state["free_pct"] < MemorySafetyGuard.MIN_FREE_PCT:
            return False, f"Free memory {state['free_pct']:.0f}% < {MemorySafetyGuard.MIN_FREE_PCT}% threshold"

        # Phase 3: Check model fits
        if not os.path.exists(model_path):
            return False, f"Model not found: {model_path}"

        model_gb = os.path.getsize(model_path) / (1024**3)
        # Need model_size * 1.3 free (model + runtime overhead)
        required_gb = model_gb * 1.3
        # Estimate free RAM: 16GB * free_pct / 100
        free_ram_gb = 16.0 * state["free_pct"] / 100.0

        if required_gb > free_ram_gb:
            return False, f"Model {model_gb:.1f}GB (needs {required_gb:.1f}GB) > {free_ram_gb:.1f}GB free RAM"

        return True, f"OK: swap={state['swap_gb']:.1f}GB load={state['load_1m']:.1f} free={state['free_pct']:.0f}% model={model_gb:.1f}GB"

    @staticmethod
    def report() -> str:
        """Human-readable audit report."""
        state = MemorySafetyGuard.audit()
        lines = [
            f"Swap: {state['swap_gb']:.1f}GB (limit {MemorySafetyGuard.MAX_SWAP_GB}GB)",
            f"Load: {state['load_1m']:.1f} (limit {MemorySafetyGuard.MAX_LOAD_1M})",
            f"Free: {state['free_pct']:.0f}% (min {MemorySafetyGuard.MIN_FREE_PCT}%)",
            f"Orphans: {len(state['orphans'])}",
        ]
        for o in state["orphans"]:
            lines.append(f"  PID={o['pid']} RSS={o['rss_mb']}MB")
        lines.append(f"SAFE: {state['safe']}")
        return "\n".join(lines)
