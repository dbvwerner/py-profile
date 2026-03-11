# py-profile

Simple Linux single-PID profiler in Python.

It samples `/proc/<pid>` and plots:
- CPU utilization over time
- Memory usage (RSS + virtual memory) over time
- I/O throughput over time
- Context switches over time
- RT-related scheduler metrics (`schedstat` wait/runtime, RT priority, policy)

## Requirements

- Linux
- Python 3.9+
- `matplotlib` for plots

Install plotting dependency:

```bash
pip install matplotlib
```

## Usage

```bash
python3 py-profile.py <pid> [-i 0.25] [-d 15]
```

Examples:

```bash
python3 py-profile.py 1234
python3 py-profile.py 1234 --interval 0.1 --duration 30
python3 py-profile.py 1234 --no-plot
```

## Notes

- Some metrics (I/O counters) can require extra permissions depending on process ownership.
- RT kernel settings shown in the title are read from:
  - `/proc/sys/kernel/sched_rt_runtime_us`
  - `/proc/sys/kernel/sched_rt_period_us`
