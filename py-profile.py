#!/usr/bin/env python3
"""Simple single-PID profiler for Linux using /proc."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


POLICY_NAMES = {
    0: "SCHED_OTHER",
    1: "SCHED_FIFO",
    2: "SCHED_RR",
    3: "SCHED_BATCH",
    5: "SCHED_IDLE",
    6: "SCHED_DEADLINE",
}


@dataclass
class ProcSnapshot:
    ts: float
    total_cpu_ticks: int
    rss_bytes: int
    vms_bytes: int
    threads: int
    read_bytes: Optional[int]
    write_bytes: Optional[int]
    vol_ctx: Optional[int]
    invol_ctx: Optional[int]
    sched_exec_ns: Optional[int]
    sched_wait_ns: Optional[int]
    rt_priority: Optional[int]
    policy: Optional[int]


@dataclass
class Sample:
    t: float
    cpu_percent_total: float
    cpu_percent_normalized: float
    rss_mb: float
    vms_mb: float
    threads: int
    read_kbps: Optional[float]
    write_kbps: Optional[float]
    vol_ctx_rate: Optional[float]
    invol_ctx_rate: Optional[float]
    sched_exec_ms_per_s: Optional[float]
    sched_wait_ms_per_s: Optional[float]
    rt_priority: Optional[int]
    policy: Optional[int]


def parse_proc_stat(raw: str) -> Dict[str, int]:
    end_comm = raw.rfind(")")
    if end_comm < 0:
        raise ValueError("Malformed /proc/<pid>/stat")
    rest = raw[end_comm + 2 :].split()
    if len(rest) < 39:
        raise ValueError("Unexpected /proc/<pid>/stat field count")

    return {
        "utime": int(rest[11]),
        "stime": int(rest[12]),
        "threads": int(rest[17]),
        "rt_priority": int(rest[37]),
        "policy": int(rest[38]),
    }


def read_kv_file(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not path.exists():
        return out
    try:
        for line in path.read_text().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip().split()[0]
            if value.isdigit():
                out[key.strip()] = int(value)
    except (OSError, ValueError):
        pass
    return out


def read_schedstat(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    try:
        parts = path.read_text().strip().split()
        if len(parts) >= 2:
            return {
                "exec_ns": int(parts[0]),
                "wait_ns": int(parts[1]),
            }
    except (OSError, ValueError):
        pass
    return {}


def read_snapshot(pid: int, pagesize: int) -> ProcSnapshot:
    base = Path(f"/proc/{pid}")
    stat_raw = (base / "stat").read_text().strip()
    stat = parse_proc_stat(stat_raw)

    statm = (base / "statm").read_text().strip().split()
    vms_bytes = int(statm[0]) * pagesize
    rss_bytes = int(statm[1]) * pagesize

    io_stats = read_kv_file(base / "io")
    status_stats = read_kv_file(base / "status")
    schedstat = read_schedstat(base / "schedstat")

    return ProcSnapshot(
        ts=time.time(),
        total_cpu_ticks=stat["utime"] + stat["stime"],
        rss_bytes=rss_bytes,
        vms_bytes=vms_bytes,
        threads=stat["threads"],
        read_bytes=io_stats.get("read_bytes"),
        write_bytes=io_stats.get("write_bytes"),
        vol_ctx=status_stats.get("voluntary_ctxt_switches"),
        invol_ctx=status_stats.get("nonvoluntary_ctxt_switches"),
        sched_exec_ns=schedstat.get("exec_ns"),
        sched_wait_ns=schedstat.get("wait_ns"),
        rt_priority=stat.get("rt_priority"),
        policy=stat.get("policy"),
    )


def build_sample(prev: ProcSnapshot, curr: ProcSnapshot, clk_tck: int, cpu_count: int) -> Sample:
    dt = max(curr.ts - prev.ts, 1e-9)

    cpu_secs = (curr.total_cpu_ticks - prev.total_cpu_ticks) / float(clk_tck)
    cpu_total = (cpu_secs / dt) * 100.0
    cpu_normalized = cpu_total / max(cpu_count, 1)

    def delta_rate(new: Optional[int], old: Optional[int]) -> Optional[float]:
        if new is None or old is None:
            return None
        return (new - old) / dt

    read_kbps = delta_rate(curr.read_bytes, prev.read_bytes)
    write_kbps = delta_rate(curr.write_bytes, prev.write_bytes)
    if read_kbps is not None:
        read_kbps /= 1024.0
    if write_kbps is not None:
        write_kbps /= 1024.0

    sched_exec = delta_rate(curr.sched_exec_ns, prev.sched_exec_ns)
    sched_wait = delta_rate(curr.sched_wait_ns, prev.sched_wait_ns)
    if sched_exec is not None:
        sched_exec /= 1e6  # ms per second
    if sched_wait is not None:
        sched_wait /= 1e6  # ms per second

    return Sample(
        t=curr.ts,
        cpu_percent_total=cpu_total,
        cpu_percent_normalized=cpu_normalized,
        rss_mb=curr.rss_bytes / (1024.0 * 1024.0),
        vms_mb=curr.vms_bytes / (1024.0 * 1024.0),
        threads=curr.threads,
        read_kbps=read_kbps,
        write_kbps=write_kbps,
        vol_ctx_rate=delta_rate(curr.vol_ctx, prev.vol_ctx),
        invol_ctx_rate=delta_rate(curr.invol_ctx, prev.invol_ctx),
        sched_exec_ms_per_s=sched_exec,
        sched_wait_ms_per_s=sched_wait,
        rt_priority=curr.rt_priority,
        policy=curr.policy,
    )


def read_rt_kernel_config() -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {
        "sched_rt_runtime_us": None,
        "sched_rt_period_us": None,
    }
    for key in out:
        path = Path(f"/proc/sys/kernel/{key}")
        try:
            out[key] = int(path.read_text().strip())
        except (OSError, ValueError):
            out[key] = None
    return out


def summarize(samples: List[Sample]) -> None:
    if not samples:
        print("No samples captured.")
        return

    def avg(vals: List[float]) -> float:
        return sum(vals) / len(vals)

    cpu_total = [s.cpu_percent_total for s in samples]
    cpu_norm = [s.cpu_percent_normalized for s in samples]
    rss = [s.rss_mb for s in samples]

    print("\nSummary")
    print(f"  Samples: {len(samples)}")
    print(f"  CPU % (total): avg={avg(cpu_total):.2f} max={max(cpu_total):.2f}")
    print(f"  CPU % (normalized): avg={avg(cpu_norm):.2f} max={max(cpu_norm):.2f}")
    print(f"  RSS MB: avg={avg(rss):.2f} max={max(rss):.2f}")


def plot_samples(samples: List[Sample], pid: int, rt_cfg: Dict[str, Optional[int]]) -> None:
    if plt is None:
        print("matplotlib is not installed. Install it with: pip install matplotlib")
        return
    if not samples:
        print("No samples to plot.")
        return

    t0 = samples[0].t
    t = [s.t - t0 for s in samples]

    def values(attr: str) -> List[float]:
        return [getattr(s, attr) for s in samples]

    def values_or_nan(attr: str) -> List[float]:
        out = []
        for s in samples:
            v = getattr(s, attr)
            out.append(float("nan") if v is None else v)
        return out

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    ax = axes.ravel()

    ax[0].plot(t, values("cpu_percent_total"), label="CPU % (all cores)")
    ax[0].plot(t, values("cpu_percent_normalized"), label="CPU % (normalized)")
    ax[0].set_ylabel("CPU %")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, values("rss_mb"), label="RSS MB")
    ax[1].plot(t, values("vms_mb"), label="VMS MB")
    ax[1].set_ylabel("Memory (MB)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(t, values_or_nan("read_kbps"), label="Read KB/s")
    ax[2].plot(t, values_or_nan("write_kbps"), label="Write KB/s")
    ax[2].set_ylabel("I/O KB/s")
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)

    ax[3].plot(t, values_or_nan("vol_ctx_rate"), label="Voluntary ctx/s")
    ax[3].plot(t, values_or_nan("invol_ctx_rate"), label="Nonvoluntary ctx/s")
    ax[3].set_ylabel("Switches/s")
    ax[3].legend()
    ax[3].grid(True, alpha=0.3)

    ax[4].plot(t, values_or_nan("sched_exec_ms_per_s"), label="On-CPU ms/s")
    ax[4].plot(t, values_or_nan("sched_wait_ms_per_s"), label="Runqueue wait ms/s")
    ax[4].set_ylabel("ms/s")
    ax[4].legend()
    ax[4].grid(True, alpha=0.3)

    ax[5].plot(t, values("threads"), label="Threads")
    ax[5].plot(t, values_or_nan("rt_priority"), label="RT priority")
    ax[5].set_ylabel("Count")
    ax[5].legend()
    ax[5].grid(True, alpha=0.3)

    runtime = rt_cfg["sched_rt_runtime_us"]
    period = rt_cfg["sched_rt_period_us"]
    first_policy = samples[0].policy
    policy_name = POLICY_NAMES.get(first_policy, str(first_policy))

    fig.suptitle(
        f"PID {pid} profile | policy={policy_name} | "
        f"sched_rt_runtime_us={runtime} sched_rt_period_us={period}",
        fontsize=11,
    )

    for a in ax:
        a.set_xlabel("Time (s)")

    fig.tight_layout()
    plt.show()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple Linux single-PID profiler")
    parser.add_argument("pid", type=int, help="PID to profile")
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=0.25,
        help="Sampling interval in seconds (default: 0.25)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=15.0,
        help="Max profiling duration in seconds (default: 15)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting and print summary only",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    pid = args.pid

    proc_dir = Path(f"/proc/{pid}")
    if not proc_dir.exists():
        print(f"PID {pid} does not exist.", file=sys.stderr)
        return 1

    if args.interval <= 0:
        print("Interval must be > 0.", file=sys.stderr)
        return 1
    if args.duration <= 0:
        print("Duration must be > 0.", file=sys.stderr)
        return 1

    pagesize = os.sysconf("SC_PAGE_SIZE")
    clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    cpu_count = os.cpu_count() or 1

    rt_cfg = read_rt_kernel_config()

    try:
        prev = read_snapshot(pid, pagesize)
    except FileNotFoundError:
        print(f"PID {pid} exited before sampling started.")
        return 1

    print(
        f"Profiling PID {pid} for up to {args.duration:.1f}s "
        f"at {args.interval:.3f}s intervals..."
    )

    samples: List[Sample] = []
    end_time = time.time() + args.duration

    while time.time() < end_time:
        time.sleep(args.interval)
        try:
            curr = read_snapshot(pid, pagesize)
        except FileNotFoundError:
            print("Process exited during profiling.")
            break

        samples.append(build_sample(prev, curr, clk_tck, cpu_count))
        prev = curr

    summarize(samples)

    if not args.no_plot:
        plot_samples(samples, pid, rt_cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
