# tab_pfn_estimations

A back-of-the-envelope simulator for running **TabPFN-3** end-to-end on a **Cerebras CS-3**
or an **NVIDIA H100**. It estimates, stage by stage, how long inference takes and whether
each stage is **compute-bound** or **memory-bound** (limited by streaming data across the
device's slow-tier memory).

## Main file

**`compute_memory_estimates.py`** — the whole model lives here.

- **Hardware model:** a `Device` bundles the four numbers that distinguish accelerators —
  slow-tier bandwidth, free (SRAM) capacity, usable FLOP/s, and how estimators pack in.
  - **CS-3:** 30 GB SRAM (free), MemoryX at 0.15 TB/s, 1.25e16 usable FLOP/s (10% of peak).
    The whole working set stays resident, so only the raw table + KV cache cross MemoryX.
  - **H100:** ~50 MB SRAM (too small to hold the working set), 80 GB HBM3 at 3.35 TB/s,
    dense FP16 at **46.5% utilization — calibrated so the model reproduces the paper's
    measured ~107 s at 1M rows** (see the `H100 =` comment). Because SRAM can't hold them,
    row-embeddings and summaries also round-trip HBM here; estimators are wave-packed into
    the 80 GB budget.
- **Same physics both ways:** the FLOP formulas are device-independent; a single rule —
  *a cross-stage tensor is billed as traffic iff it doesn't fit the free tier* — decides
  what CS-3 gets for free but H100 must pay for.
- **Stages modeled:** S1 cell/feature embedding → S2 feature aggregation (train + test)
  → S3 ICL transformer (fit builds the KV cache, predict cross-attends) → S4 output head.
- Chunk-by-chunk transfers **double-buffer** (overlap) with compute; the ICL KV cache
  streams one layer at a time.

Tunable knobs live in `TabPFNEstimator(...)`: `N` (train rows), `M` (test rows), `C`
(features), `E` (estimators), `B` (batch), `task` (`multiclass` | `regression`),
`device` (`CS3` | `H100`).

## Setup

Only needs `rich` (console output) and `matplotlib` + `numpy` (plots):

```bash
uv run python compute_memory_estimates.py --device CS3     # or --device H100
uv run python compute_memory_estimates.py --device H100 --output-file-path plots/gpu.png
```

(or `pip install rich matplotlib numpy` then `python compute_memory_estimates.py ...`)

## Output

1. **Console** — a per-stage timeline (IN / OUT / COMPUTE / WALL, with the bottleneck
   flagged) plus a summary table and grand total, for the default `N=1M, M=10k, C=200,
   E=8` run on the chosen device.
2. **`plots/<device>_stage_timings.png`** — a 3×4 grid sweeping training rows
   (100 → 1M, log–log). Rows = features `C ∈ {10, 100, 500}`, columns = estimators
   `E ∈ {1, 2, 4, 8}`, `M = 0.01·N`. Each panel shows one line per stage + a Total;
   **square markers = compute-bound, circles = memory-bound**, and a `max:` label gives
   the peak total time. Regenerated on every run.
