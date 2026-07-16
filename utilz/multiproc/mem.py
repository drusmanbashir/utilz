"""System memory fraction helpers."""
from __future__ import annotations


def mem_frac() -> float:  #AI
    """Return used RAM fraction from /proc/meminfo (1 - MemAvailable/MemTotal)."""
    total = None
    available = None
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                total = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                available = int(line.split()[1])
            if total is not None and available is not None:
                break
    return 1.0 - (available / total)


def mem_over(threshold: float) -> bool:  #AI
    return mem_frac() > threshold
