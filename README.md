# tab_pfn_estimations

A back-of-the-envelope simulator for running **TabPFN-3** end-to-end on a Cerebras CS
system. It estimates, stage by stage, how long inference takes and whether each stage is
**compute-bound** or **memory-bound** (limited by streaming data from MemoryX).

## Main file

**`compute_memory_estimates.py`** — the whole model lives here.

- **Hardware model:** 30 GB on-chip SRAM (treated as free), MemoryX at 0.15 TB/s,
  compute at 1.25e16 usable FLOP/s (10% of peak).
- **Stages modeled:** S1 cell/feature embedding → S2 feature aggregation (train + test)
  → S3 ICL transformer (fit builds the KV cache, predict cross-attends) → S4 output head.
- Chunk-by-chunk transfers **double-buffer** (overlap) with compute; the ICL KV cache
  streams one layer at a time.

Tunable knobs live in `TabPFNEstimator(...)`: `N` (train rows), `M` (test rows), `C`
(features), `E` (estimators), `B` (batch), `task` (`multiclass` | `regression`).

## Setup

Only needs `rich` (console output) and `matplotlib` + `numpy` (plots):

```bash
uv run python compute_memory_estimates.py
```

(or `pip install rich matplotlib numpy` then `python compute_memory_estimates.py`)

## Output

1. **Console** — a per-stage timeline (IN / OUT / COMPUTE / WALL, with the bottleneck
   flagged) plus a summary table and grand total, for the default `N=1M, M=10k, C=200,
   E=8` run.
2. **`plots/stage_timings.png`** — a 3×4 grid sweeping training rows (100 → 1M, log–log).
   Rows = features `C ∈ {10, 100, 500}`, columns = estimators `E ∈ {1, 2, 4, 8}`,
   `M = 0.01·N`. Each panel shows one line per stage + a Total; **square markers =
   compute-bound, circles = memory-bound**, and a `max:` label gives the peak total time.
   Regenerated on every run.
