# TabPFN-3 — Cold Fit vs Predict: FLOPs & Memory

## Model REPR
  ```
  TabPFNV3(
    (x_embed): Linear(in_features=6, out_features=128, bias=True)
    (col_y_encoder): TrainableOrthogonalEmbedding(
      (embedding): Embedding(160, 128)
    )
    (feature_distribution_embedder): FeatureDistributionEmbedder(
      (layers): ModuleList(
        (0-2): 3 x InducedSelfAttentionBlock(
          (cross_attn_block1): CrossAttentionBlock(
            (attn): CrossAttention(
              (softmax_scaling_layer): SoftmaxScalingMLP(
                (base_mlp): Sequential(
                  (0): Linear(in_features=1, out_features=64, bias=True)
                  (1): GELU(approximate='none')
                  (2): Linear(in_features=64, out_features=128, bias=True)
                )
                (query_mlp): Sequential(
                  (0): Linear(in_features=16, out_features=64, bias=True)
                  (1): GELU(approximate='none')
                  (2): Linear(in_features=64, out_features=16, bias=True)
                )
              )
              (q_projection): Linear(in_features=128, out_features=128, bias=False)
              (k_projection): Linear(in_features=128, out_features=128, bias=False)
              (v_projection): Linear(in_features=128, out_features=128, bias=False)
              (out_projection): Linear(in_features=128, out_features=128, bias=False)
            )
            (mlp): MLP(
              (0): Linear(in_features=128, out_features=256, bias=False)
              (1): GELU(approximate='none')
              (2): Linear(in_features=256, out_features=128, bias=False)
            )
            (layernorm_q): _DtypeMatchingRMSNorm((128,), eps=None, elementwise_affine=True)
            (layernorm_kv): _DtypeMatchingRMSNorm((128,), eps=None, elementwise_affine=True)
            (layernorm2): _DtypeMatchingRMSNorm((128,), eps=None, elementwise_affine=True)
          )
          (cross_attn_block2): CrossAttentionBlock(
            (attn): CrossAttention(
              (q_projection): Linear(in_features=128, out_features=128, bias=False)
              (k_projection): Linear(in_features=128, out_features=128, bias=False)
              (v_projection): Linear(in_features=128, out_features=128, bias=False)
              (out_projection): Linear(in_features=128, out_features=128, bias=False)
            )
            (mlp): MLP(
              (0): Linear(in_features=128, out_features=256, bias=False)
              (1): GELU(approximate='none')
              (2): Linear(in_features=256, out_features=128, bias=False)
            )
            (layernorm_q): _DtypeMatchingRMSNorm((128,), eps=None, elementwise_affine=True)
            (layernorm_kv): _DtypeMatchingRMSNorm((128,), eps=None, elementwise_affine=True)
            (layernorm2): _DtypeMatchingRMSNorm((128,), eps=None, elementwise_affine=True)
          )
        )
      )
    )
    (column_aggregator): ColumnAggregator(
      (blocks): ModuleList(
        (0-2): 3 x TransformerBlock(
          (attention): Attention(
            (q_projection): Linear(in_features=128, out_features=128, bias=False)
            (k_projection): Linear(in_features=128, out_features=128, bias=False)
            (v_projection): Linear(in_features=128, out_features=128, bias=False)
            (out_projection): Linear(in_features=128, out_features=128, bias=False)
          )
          (layernorm): _DtypeMatchingRMSNorm((128,), eps=None, elementwise_affine=True)
          (layernorm_mlp): _DtypeMatchingRMSNorm((128,), eps=None, elementwise_affine=True)
          (mlp): MLP(
            (0): Linear(in_features=128, out_features=256, bias=False)
            (1): GELU(approximate='none')
            (2): Linear(in_features=256, out_features=128, bias=False)
          )
        )
      )
      (rope): RotaryEmbedding()
      (out_ln): _DtypeMatchingRMSNorm((128,), eps=None, elementwise_affine=True)
    )
    (icl_y_encoder): TrainableOrthogonalEmbedding(
      (embedding): Embedding(160, 512)
    )
    (icl_blocks): ModuleList(
      (0-23): 24 x ICLTransformerBlock(
        (icl_attention): ICLAttention(
          (softmax_scaling_layer): SoftmaxScalingMLP(
            (base_mlp): Sequential(
              (0): Linear(in_features=1, out_features=64, bias=True)
              (1): GELU(approximate='none')
              (2): Linear(in_features=64, out_features=512, bias=True)
            )
            (query_mlp): Sequential(
              (0): Linear(in_features=64, out_features=64, bias=True)
              (1): GELU(approximate='none')
              (2): Linear(in_features=64, out_features=64, bias=True)
            )
          )
          (q_projection): Linear(in_features=512, out_features=512, bias=False)
          (out_projection): Linear(in_features=512, out_features=512, bias=False)
          (k_projection): Linear(in_features=512, out_features=512, bias=False)
          (v_projection): Linear(in_features=512, out_features=512, bias=False)
        )
        (layernorm): _DtypeMatchingRMSNorm((512,), eps=None, elementwise_affine=True)
        (layernorm_mlp): _DtypeMatchingRMSNorm((512,), eps=None, elementwise_affine=True)
        (mlp): MLP(
          (0): Linear(in_features=512, out_features=1024, bias=False)
          (1): GELU(approximate='none')
          (2): Linear(in_features=1024, out_features=512, bias=False)
        )
      )
    )
    (output_norm): _DtypeMatchingRMSNorm((512,), eps=None, elementwise_affine=True)
    (many_class_decoder): ManyClassDecoder(
      (q_projection): Linear(in_features=512, out_features=384, bias=True)
      (k_projection): Linear(in_features=512, out_features=384, bias=True)
      (softmax_scaling_layer): SoftmaxScalingMLP(
        (base_mlp): Sequential(
          (0): Linear(in_features=1, out_features=64, bias=True)
          (1): GELU(approximate='none')
          (2): Linear(in_features=64, out_features=384, bias=True)
        )
        (query_mlp): Sequential(
          (0): Linear(in_features=64, out_features=64, bias=True)
          (1): GELU(approximate='none')
          (2): Linear(in_features=64, out_features=64, bias=True)
        )
      )
    )
  ```

## Introduction
A layer-by-layer accounting of the compute and memory of the **TabPFN-3** model
(the checkpoint in `model/tabpfn-v3-classifier-v3_20260417_binary.ckpt`), for the
two inference stages **cold fit** and **predict**, under both `fit_with_cache`
and `low_memory` settings. Classification first, then regression, then a
side-by-side.

All formulas are derived directly from
`.venv/.../tabpfn/architectures/tabpfn_v3.py` and
`.venv/.../tabpfn/inference.py`, and the running numbers come from
`flops_calc.py` (in this repo). Cache sizes were cross-checked against the
library's own `get_cache_size()` (exact match).

---

## 0. Setup, conventions, and the one thing to keep in mind

**Variables**

| Symbol | Meaning | Running example |
|---|---|---|
| `N` = `X_train` | training rows (a fixed dataset property) | **100,000** |
| `M` = `X_test` | test rows (a fixed dataset property) | **1,000** |
| `R` | rows *flowing through the shared layers on a given pass* — a generic placeholder, **not** a dataset property: `R = N` (cold fit), `R = M` (cached predict), or `R = N+M` (`low_memory` full pass) | varies |
| `B` | batch size = number of independent datasets stacked into one forward pass (the leading `1` in every shape). Ordinary single-dataset fit/predict → `B = 1`; only the batched engine uses `B > 1`. **Not** the estimator count `E` — the 8 estimators are separate `B = 1` passes | **1** |
| `F` | raw input features | **100** |
| `C` | columns the model actually sees | **≈100** (see caveat §13) |
| `E` | estimators | **8** |
| `P` | predict runs | **10** |

- Precision: **FP16 (2 bytes)**, **no KV-cache quantization**.
- Chunking: `inference_row_chunk_size = 2048`, `inference_col_chunk_size = 4`.
- Feature grouping (`_group_features`) uses `torch.roll` → **one triplet per
  feature → C = F**. We use C = 100 in the walk-through. (Real preprocessing
  adds a fingerprint column + `svd_quarter` SVD components, so 4 of 8 estimators
  actually run at C≈126 and 4 at C≈101. That touches only Stages 0–2 and the
  inducing cache — both tiny — so it barely moves totals. See §13.)
- Batch dim: single dataset → `B = 1` per estimator. The 8 estimators are 8
  separate forward passes that **share one model copy** but differ in
  preprocessing (and each gets its own KV cache).

**FLOP convention:** 1 multiply-add = 2 FLOPs. `Linear(in→out)` on `t` tokens
= `2·t·in·out`. Attention core (QK^T + A·V) = `4·S_q·S_k·d_model`. RMSNorm,
softmax, and the small softmax-scaling MLPs are omitted (each <~2% of a block).

**The single most important fact.** Every row is compressed to *one* 512-dim
vector before the expensive part, so the 24-layer **ICL transformer** operates
on a sequence of length = number of rows, and its self-attention is **O(N²)**.
That term dominates everything: in the example **ICL is 501.6 of the 526 TFLOP
per training pass (95%)**, and the ICL weights are **97% of the model**.
Everything upstream (turning features into row-vectors) is cheap by comparison.

---

## 1. The pipeline (shared by classification and regression)

```
X  (R rows × C cols)
 → [Stage 0] group + cell-embed            → (1, R, C, 128)
 → [Stage 1] FeatureDistributionEmbedder   → (1, R, C, 128)     (shape preserved)
             (per-column set-transformer, ×3 ISAB blocks)
 → [Stage 2] ColumnAggregator (×3)         → (1, R, 4, 128) → flatten → (1, R, 512)
             (per-row, 4 CLS tokens)          << the C axis disappears here
 → [Stage 3] ICL transformer (×24, D=512)     << the O(N²) part; KV cache lives here
 → output_norm
 → [Head]  classification: ManyClassDecoder → logits (M, 160)
           regression:     output_projection → bucket logits (M, 5000)
```

`R` = rows flowing through = `N` (fit), `M` (cached predict), or `N+M`
(low_memory full pass).

Architecture constants (from the checkpoint config): `embed_dim d=128`,
`icl_emsize D=512` (=128×4 CLS), `K=128` inducing points, ICL heads 8 (head_dim
64), **test KV heads = 1** (`icl_num_kv_heads_test=1`, multi-query), 24 ICL
layers, `ff_factor=2`.

---

## 2. The two "stages" and the two modes

"Cold fit" and "predict" are not different layers — they are *which rows flow
through the same layers and what gets saved*, controlled by `fit_mode`:

- **`fit_with_cache`** → `InferenceEngineExplicitKVCache`. **Fit** runs the full
  pipeline over the **train** rows once and *◊saves a KV cache*; **predict** runs
  **test-only** rows and reuses the cache (`x_is_test_only=True`).
- **`low_memory`** → `InferenceEngineOnDemand`. **Fit** does *no model compute*
  (only CPU preprocessing, no cache). **Every predict** re-runs the **full
  train+test** pipeline from scratch.

---

## 3. Stage-by-stage walk-through (one estimator)

Each stage: REPR, what it does, the shape that moves on (and why), FLOPs
(formula + FIT number at N=100k, C=100), weights, activations.

### Stage 0 — Cell embedding
```
(x_embed): Linear(in_features=6, out_features=128)
```
- **What:** each feature is grouped with 2 cyclically-shifted neighbors (3
  values) + 3 NaN/Inf indicators → **6 inputs**; a linear maps 6→128. The
  train-row label embedding is added to train cells only.
- **Shape out:** `(1, R, C, 128)` — one 128-vec per *cell*; C preserved because
  columns are embedded independently.
- **FLOPs:** `2·(R·C)·6·128`. FIT: <span style="color:yellow">**0.015 TF** (trivial)</span>
- **Weights:** x_embed 896 params (+ y-encoder, see §3/head).
- **Activation:** `R·C·128` unchunked = **2.4 GB** at N=100k → this is exactly
  why Stages 0–2 are **row-chunked at 2048** (peak ≈ **<span style="color:cyan">50 MB/chunk).</span>**

### Stage 1 — FeatureDistributionEmbedder
```
3 × InducedSelfAttentionBlock:
   cross_attn_block1: CrossAttention(128,128,128) + MLP(128→256→128)  # inducing ← rows
   cross_attn_block2: CrossAttention(128,128,128) + MLP(128→256→128)  # rows ← inducing
   inducing_vectors: (128 × 128)
```
- **What:** per column (reshape to `B·C` sequences of length R). `block1`: 128
  inducing points attend to the **train rows** → a 128×128 summary of that
  column's distribution. `block2`: **all rows** attend to those inducing points.
  This ISAB replaces O(N²) per-column attention with two O(N·128) attentions.
- **Shape out:** `(1, R, C, 128)` — **preserved** (refines each cell using
  column statistics; does not mix columns).
- **The cacheable object:** `block1`'s output = the **`inducing_hidden`** cache.
  Test rows skip `block1` and run only `block2`.
- **FLOPs (FIT):** `block1`×3 over N + `block2`×5 over N (×3 in the row loop +
  ×2 recomputed in `_compute_all_inducing_hidden`). = **<span style="color:yellow">17.05 TF.</span>**
- **Weights:** 869,616 params.


### Stage 2 — ColumnAggregator
```
3 × TransformerBlock: Attention(128,128,128) + MLP(128→256→128)   # RoPE
cls_tokens: (4 × 128) ; out_ln
```
- **What:** per row (sequence = 4 CLS tokens prepended to the C feature-vectors).
  Blocks 0–1 = full self-attention over the `C+4` sequence (columns finally
  interact). Last block = cross-attention readout: 4 CLS tokens query the full
  sequence.
- **Shape out:** `(1, R, 4, 128)` → **flattened to `(1, R, 512)`**. The crucial
  collapse: **the C axis vanishes**, every row becomes one 512-dim token, and
  everything downstream is independent of feature count.
- **FLOPs (FIT):** **<span style="color:yellow">7.34 TF.</span>**
- **Weights:** 394,632 params.

### Stage 3 — ICL transformer (the expensive part)
```
24 × ICLTransformerBlock:
   ICLAttention: q,k,v,out = Linear(512→512)   # 8 heads × 64
   MLP(512→1024→512)
icl_y_encoder (added to train rows)
```
- **What:** rows attend to **train rows only**. Train rows do full 8-head
  self-attention; test rows use **1 KV head** (multi-query) — this is what
  shrinks the cache 8×.
- **Shape out:** `(1, R, 512)`, unchanged across all 24 layers.
- **FLOPs — one layer at FIT (N=100k):** attention `4·N²·512` = **20.48 TF**;
  projections+MLP = **0.42 TF**. Attention is **98%** of the layer and is
  **O(N²)**.
- **24-layer FIT total: <span style="color:yellow">501.6 TF/estimator**</span> — 95% of the whole training pass.
- **Weights:** 51,357,696 params (**97% of the model**).

### Head (task-specific) — see §3-clf and §8-reg below.

---

# PART A — CLASSIFICATION

Head = `ManyClassDecoder`:
```
q_projection: Linear(512→384) ; k_projection: Linear(512→384)   # 6 heads × 64
```
- **What:** test embeddings query, **train** embeddings are keys, **one-hot
  train labels are values**. Output = attention-weighted average of one-hot
  labels → `log(clamp(...))` → logits. Non-parametric in class count (values
  chunked into 64-wide pieces; 160 classes → 3 chunks).
- **Shape:** query `(M,512)`, keys `(N,512)`, one-hot values `(N,160)` → logits
  `(M,160)`. **Attends to all N train rows → O(M·N).**
- **FLOPs:** `2·M·512·384 + 2·N·512·384 + 6·3·4·M·N·64` ≈ **0.50 TF** per
  predict. Runs at **predict only** (at fit M=0 → early return).
- **Weights:** 427,392 params. Total model = **53,153,144 params → 101.4 MB (fp16)**.

## A.4 COLD FIT

### `fit_with_cache` — one full forward over N train rows, saves a cache

| Layer | FLOPs / estimator | why |
|---|---:|---|
| Stage 0 cell-embed | 0.015 TF | linear in N·C |
| Stage 1 dist-embed | 17.05 TF | linear in N·C |
| Stage 2 col-agg | 7.34 TF | linear in N |
| **Stage 3 ICL ×24** | **501.59 TF** | **O(N²)** |
| Head (decoder) | 0 (M=0) | — |
| **Total / estimator** | **525.99 TF** | |
| **× 8 estimators** | **4,207.96 TF** | paid once |

**Cache stored per estimator** (FP16, no quant — `TabPFNV3Cache`):

| Component | Shape | Size |
|---|---|---:|
| ICL KV (24 × K,V × **1** test head × 64) | `24·2·N·1·64` | **585.9 MB** |
| `train_embeddings` (decoder keys) | `N·512` | 97.7 MB |
| `inducing_hidden` (3 × C × 128 × 128) | `3·C·128·128` | 9.4 MB |
| scaler stats | `2·C` | ~0 |
| **Per estimator** | | **693.0 MB** |
| **× 8 (persistent)** | | **5.41 GB** |

The KV cache is **1 head, not 8** (`k_cache[:, :, :1]`), scaling with **rows,
not rows×features** → 585.9 MB at N=100k, ~5.9 GB at N=1M (paper's "≈7 GB").

**Fit peak activation ≈ 0.4 GB/est** (the `(1,N,512)`=98 MB ICL sequence +
transient q/k/v; flash attention → no N×N materialization), plus its 693 MB
cache growing during the 24 layers.

### `low_memory` cold fit
**Zero model FLOPs, zero cache.** Fit only fits CPU preprocessors; all work is
deferred to predict.

## A.5 PREDICT

### `fit_with_cache` — test-only (M=1000), reuse cache

M<2048 → no row-chunking. Stage 1 **skips block1** (cached inducing). ICL
**skips K/V projection** (cached) and does test→train cross-attention `M×N`.

| Layer | FLOPs / est / run | why |
|---|---:|---|
| Stage 0 | 0.0002 TF | M tiny |
| Stage 1 (block2 only) | 0.081 TF | block1 cached |
| Stage 2 | 0.073 TF | M rows |
| **Stage 3 ICL ×24** | **4.99 TF** | `4·M·N·D` — **O(M·N)** |
| Head (decoder) | 0.50 TF | one-hot retrieval, O(M·N) |
| **Total / est / run** | **5.65 TF** | |
| **× 8 est × 10 runs** | **451.7 TF** | |

One cached-ICL layer = **0.205 TF** vs **20.48 TF** at fit → a **100× drop**
(exactly N/M). The cache converts O(N²) fit work into O(M·N) predict work.

**Predict peak activation ≈ 0.18 GB/est**, dominated by the decoder's one-hot
value tensor `(N, 6, 160)` = 183 MB. Plus the **5.41 GB resident cache**.

### `low_memory` — full recompute every run (R = N+M = 101,000)

ICL query-chunked by `save_peak_memory_factor=8`, **nothing cached**.

| Layer | FLOPs / est / run | why |
|---|---:|---|
| Stage 0–2 | 24.56 TF | full N+M |
| **Stage 3 ICL ×24** | **506.58 TF** | **O(N²) — recomputed** |
| Head (decoder) | 0.50 TF | |
| **Total / est / run** | **531.64 TF** | |
| **× 8 est × 10 runs** | **42,531 TF** | |

Every predict redoes the whole O(N²) training attention. **Peak activation ≈
0.29 GB/est** (ICL sequence + full 8-head train K/V), **no persistent cache**.

## A.6 Classification side-by-side

**FLOPs (TFLOP)** — N=100k, M=1k, C=100, E=8, P=10

| Attribute | `fit_with_cache` | `low_memory` |
|---|---:|---:|
| Cold-fit compute (once) | **4,207.96** | 0 |
| Predict compute (×10) | **451.68** | **42,531.03** |
| **Grand total** | **4,659.6** | **42,531.0** |
| per predict run (8 est) | 45.2 | 4,253 |
| ratio | **1×** | **9.1×** |

**Memory**

| Attribute | `fit_with_cache` | `low_memory` |
|---|---:|---:|
| Model weights (fp16, shared) | 101.4 MB | 101.4 MB |
| Persistent KV cache | **5.41 GB** | **0** |
| Peak activation (fit) | ~0.40 GB/est | — |
| Peak activation (predict) | ~0.18 GB/est | ~0.29 GB/est |
| **Resident peak (predict)** | **≈ 5.7 GB** | **≈ 0.4 GB** |

**Grand totals per attribute, per stage (classification):**

| | fit_cache · FIT | fit_cache · PREDICT(×10) | low_memory · FIT | low_memory · PREDICT(×10) |
|---|---:|---:|---:|---:|
| FLOPs | 4,207.96 TF | 451.68 TF | 0 | 42,531.03 TF |
| Weights | 101.4 MB | 101.4 MB | 101.4 MB | 101.4 MB |
| KV cache written/held | +5.41 GB | 5.41 GB held | 0 | 0 |
| Peak activation | ~0.40 GB/est | ~0.18 GB/est | — | ~0.29 GB/est |

**One-line trade-off:** `fit_with_cache` spends **5.4 GB** to make prediction
**~9× cheaper overall** and each individual predict **~100× cheaper in the ICL
layers**; `low_memory` keeps resident memory under **~0.5 GB** but recomputes the
O(N²) training attention on every predict.

---

# PART B — REGRESSION

## B.1 What changes vs classification

Stages 0–3 and the ICL/KV-cache machinery are **identical**. Only the target
encoders and the output head differ:

| Piece | Classification | Regression |
|---|---|---|
| `col_y_encoder` | `Embedding(160→128)` (lookup, free) | `Linear(1→128)` (256 params) |
| `icl_y_encoder` | `Embedding(160→512)` | `Linear(1→512)` (1,024 params) |
| Head | `ManyClassDecoder` (retrieval attn, **O(M·N)**) | `output_projection` MLP |
| Head shape | q/k `Linear(512→384)`, one-hot values | `Linear(512→1024)→GELU→Linear(1024→5000)` |
| Output | logits `(M, 160)` | **bucket logits `(M, 5000)`** (bar distribution) |
| Head params | 427,392 | **5,650,312** |
| **Total params** | 53,153,144 → **101.4 MB** | **58,274,944 → 111.2 MB** |

The two consequences that matter:

1. **The regression head does NOT attend to train.** It is a per-test-row MLP:
   `output_projection(test_emb)`. FLOPs are **O(M)** with **no N-dependence** —
   unlike the classification decoder's O(M·N) retrieval. So the head is *cheaper
   in FLOPs* (0.011 vs 0.50 TF) but *heavier in weights* (5.65M vs 0.43M) and
   produces a 5000-wide bucket distribution per row (bar-distribution head;
   quantiles/mean decoded from the CDF post-hoc via `regression_borders`).

2. **`train_embeddings` (97.7 MB/est) is still cached but ignored** by the
   regression head (`get_cache_size` docstring: "regression stores but ignores
   them"). So the **KV cache size is identical to classification: 693 MB/est →
   5.41 GB total.**

Verified by instantiating the regression architecture: total **58,274,944
params**, head = `Linear(512→1024)→GELU→Linear(1024→5000)`, output `(M, 5000)`.

## B.2 COLD FIT

### `fit_with_cache` — one full forward over N train rows

| Layer | FLOPs / estimator | why |
|---|---:|---|
| Stage 0 + y-encoders | 0.015 TF | linear; y-enc `Linear(1→d)` negligible |
| Stage 1 dist-embed | 17.05 TF | |
| Stage 2 col-agg | 7.34 TF | |
| **Stage 3 ICL ×24** | **501.59 TF** | **O(N²)** — identical to clf |
| Head | 0 (M=0) | — |
| **Total / estimator** | **526.00 TF** | (≈ identical to clf) |
| **× 8 estimators** | **4,207.96 TF** | |

Cache identical to classification: **693 MB/est → 5.41 GB total** (the cached
`train_embeddings` are dead weight for regression). Fit peak activation ~0.4
GB/est.

### `low_memory` cold fit
Zero model FLOPs, zero cache (same as classification).

## B.3 PREDICT

### `fit_with_cache` — test-only (M=1000), reuse cache

| Layer | FLOPs / est / run | why |
|---|---:|---|
| Stage 0–2 | 0.155 TF | test-only, block1 cached |
| **Stage 3 ICL ×24** | **4.99 TF** | `4·M·N·D` — O(M·N) |
| Head (`output_projection`) | **0.011 TF** | **O(M) only, no N** |
| **Total / est / run** | **5.16 TF** | |
| **× 8 est × 10 runs** | **412.5 TF** | |

**Predict peak activation ≈ 34 MB/est** — much smaller than classification's
183 MB, because there is **no `(N,6,160)` one-hot tensor**; the biggest head
activation is the `(M, 5000)` bucket logits = 10 MB. Plus the **5.41 GB resident
cache**.

### `low_memory` — full recompute every run (R = N+M)

| Layer | FLOPs / est / run | why |
|---|---:|---|
| Stage 0–2 + y-enc | 24.56 TF | full N+M |
| **Stage 3 ICL ×24** | **506.58 TF** | **O(N²) — recomputed** |
| Head | 0.011 TF | |
| **Total / est / run** | **531.15 TF** | |
| **× 8 est × 10 runs** | **42,491.9 TF** | |

Peak activation ~0.29 GB/est (ICL seq + train K/V), no persistent cache.

## B.4 Regression side-by-side

**FLOPs (TFLOP)** — N=100k, M=1k, C=100, E=8, P=10

| Attribute | `fit_with_cache` | `low_memory` |
|---|---:|---:|
| Cold-fit compute (once) | **4,207.96** | 0 |
| Predict compute (×10) | **412.54** | **42,491.90** |
| **Grand total** | **4,620.5** | **42,491.9** |
| per predict run (8 est) | 41.3 | 4,249 |
| ratio | **1×** | **9.2×** |

**Memory**

| Attribute | `fit_with_cache` | `low_memory` |
|---|---:|---:|
| Model weights (fp16, shared) | 111.2 MB | 111.2 MB |
| Persistent KV cache | **5.41 GB** | **0** |
| Peak activation (fit) | ~0.40 GB/est | — |
| Peak activation (predict) | ~34 MB/est | ~0.29 GB/est |
| **Resident peak (predict)** | **≈ 5.6 GB** | **≈ 0.4 GB** |

**Grand totals per attribute, per stage (regression):**

| | fit_cache · FIT | fit_cache · PREDICT(×10) | low_memory · FIT | low_memory · PREDICT(×10) |
|---|---:|---:|---:|---:|
| FLOPs | 4,207.96 TF | 412.54 TF | 0 | 42,491.90 TF |
| Weights | 111.2 MB | 111.2 MB | 111.2 MB | 111.2 MB |
| KV cache written/held | +5.41 GB | 5.41 GB held | 0 | 0 |
| Peak activation | ~0.40 GB/est | ~34 MB/est | — | ~0.29 GB/est |

---

# PART C — Classification vs Regression, at a glance

| | Classification | Regression |
|---|---:|---:|
| Model weights (fp16) | 101.4 MB | 111.2 MB |
| Head params | 0.43 M | 5.65 M |
| Head compute per predict | O(M·N) = **0.50 TF** | O(M) = **0.011 TF** |
| Output | logits (M, 160) | bucket logits (M, 5000) |
| KV cache total | 5.41 GB | 5.41 GB (train_emb wasted) |
| **fit_cache total FLOPs** | **4,659.6 TF** | **4,620.5 TF** |
| **low_memory total FLOPs** | **42,531.0 TF** | **42,491.9 TF** |
| Predict peak activation (fit_cache) | ~183 MB/est (one-hot) | ~34 MB/est (no one-hot) |

The two tasks are within ~1% on total FLOPs and identical on cache, because the
O(N²) ICL stage dominates and is shared. The head choice only changes (a) head
weight size, (b) whether the head attends to train (classification O(M·N) vs
regression O(M)), and (c) predict-time peak activation (the classification
one-hot value tensor is the biggest predict-time allocation; regression has none).

---

# §13. Caveats (so you can trust / scale the numbers)

1. **Feature inflation.** Real preprocessing makes 4 estimators run at C≈126
   (adds `svd_quarter` ≈F/4 + fingerprint column) and 4 at C≈101. This inflates
   **only Stages 0–2 and `inducing_hidden`** by ~13% — invisible against the ICL
   total. (Verified: F=30 → per-estimator columns `[38,38,38,38,31,31,31,31]`.)
2. **Flash/SDPA assumed.** Memory figures assume a memory-efficient attention
   backend, so N×N score matrices are never materialized. On a CPU/MPS machine
   with no flash kernel, the N²=10¹⁰ score matrix would OOM — **100k rows is not
   runnable on a 16 GB laptop**; these are analytical (H100/FA3 regime). Shapes
   and cache sizes were validated on the tiny breast-cancer case.
3. **Omitted terms:** RMSNorm, softmax, softmax-scaling MLPs (<~2% each). All
   matmuls and attention cores are included.
4. **Regression y-preprocessing** (`REGRESSION_Y_PREPROCESS_TRANSFORMS`,
   safepower etc.) and classification decision-threshold/temperature tuning are
   CPU post/pre-processing — negligible model FLOPs, not counted.
5. **FLOP convention:** 1 MAC = 2 FLOPs. Halve for MAC counts.

---

# §14. Reproduce

```bash
uv run python flops_calc.py
```

`flops_calc.py` encodes the per-layer formulas above (classification +
regression). Cache sizes cross-checked against:

```python
from tabpfn.architectures.tabpfn_v3 import get_cache_size, TabPFNV3Config
import torch
cfg = TabPFNV3Config(icl_num_kv_heads_test=1)
get_cache_size(n_train=100_000, n_features=100, model_config=cfg,
               base_dtype=torch.float16, quantize_kv_cache=False)
# -> 693.0 MB / estimator  (matches the hand table)
```

Key source references:
- Model forward + stages: `tabpfn/architectures/tabpfn_v3.py`
  (`TabPFNV3.forward`, `_stages_0_to_2`, `ICLAttention`, `ManyClassDecoder`).
- Cache sizing: `get_cache_size`, `TabPFNV3Cache`.
- Fit/predict engines: `tabpfn/inference.py`
  (`InferenceEngineExplicitKVCache` = `fit_with_cache`;
  `InferenceEngineOnDemand` = `low_memory`).

---

# PART D — Sensitivity to the real per-estimator column mix

The flat `C=100` assumption vs the real mix (4 estimators at C≈126 from
`svd_quarter`+fingerprint, 4 at C≈101), classification, N=100k, M=1k, P=10:

| Assumption | fit_cache total FLOPs | of which fit / predict | KV cache (×8) |
|---|---:|---:|---:|
| flat C=100 | 4,659.6 TF | 4,207.96 / 451.68 | 5.414 GB |
| mix 4×126 / 4×101 | 4,688.8 TF | 4,235.36 / 453.45 | 5.424 GB |
| **difference** | **+0.6%** | +0.7% / +0.4% | **+0.2%** |

**Conclusion:** the column-count detail is a rounding error. C only enters
Stages 0–2 and the `inducing_hidden` cache; the O(N²) ICL stage (which dominates
both FLOPs and cache) is independent of C. Flat C=100 is fine for all practical
purposes.

---

# PART E — N-sweep and the fit_cache ↔ low_memory crossover

Classification, M=1000, C=100, E=8, P=10 predict runs:

| N (train) | fit_cache TOTAL | low_memory TOTAL | ratio | KV cache (×8, fp16) | breakeven P\* |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 26.0 TF | 55.0 TF | 2.1× | 0.13 GB | 1.01 |
| 10,000 | 128.8 TF | 731.4 TF | 5.7× | 0.61 GB | 1.00 |
| 100,000 | 4,659.6 TF | 42,531 TF | 9.1× | 5.41 GB | 1.00 |
| 1,000,000 | 400,324 TF | 3,964,082 TF | 9.9× | **53.5 GB** | 1.00 |

Two things to read off this:

1. **Breakeven `P* ≈ 1` at every N.** `P* = FIT / (full − cached_predict)`.
   Building the cache costs almost exactly one full forward (`FIT ≈ full`), and a
   cached predict is ~100× cheaper, so `full − cached ≈ FIT` → **P\* ≈ 1**. In
   words: **`fit_with_cache` pays for itself after the very first prediction.**
   If you predict more than once, caching wins on compute — always.

2. **The compute win grows with N** (2.1× → 9.9×) because `low_memory` re-runs
   the O(N²) training attention on *every* predict, while cached predict is only
   O(M·N). With P=10 runs, the ratio approaches P as N grows.

3. **Memory is the real constraint, not compute.** The KV cache is O(N): at
   N=1M it is **53.5 GB** in fp16 (×8 estimators) — it will not fit on one 80 GB
   H100 alongside activations. That is exactly why the deployed engine
   **int8-quantizes the KV cache by default** (→ ~31 GB) and/or you drop
   estimators. So in practice you choose `low_memory` (or int8 quant, or fewer
   estimators) **because of memory, never because of compute.**

---

# PART F — Is this model compute-bound or I/O-bound?

Roofline: compare arithmetic intensity **AI = FLOPs / bytes moved** to the
hardware "ridge point" (H100 fp16 ≈ 990 TFLOP/s ÷ 3.35 TB/s HBM ≈
**~300 FLOP/byte**). AI ≫ ridge → compute-bound; AI ≪ ridge → memory-bandwidth
(I/O) bound.

| Operation | AI (FLOP/byte) | Verdict |
|---|---:|---|
| FIT ICL attention, N=10k | 5,000 | **compute-bound** |
| FIT ICL attention, N=100k | 50,000 | **compute-bound** |
| FIT ICL attention, N=1M | 500,000 | **compute-bound** |
| Cached predict, whole model, M=1 (online) | ~57 | **memory-bound** |
| Cached predict, M=8 | ~104 | **memory-bound** |
| Cached predict, M=40 | ~319 | compute-bound |
| Cached predict, M=1000 (batched) | ~6,778 | **compute-bound** |

**The answer is: it depends on the phase — and it mirrors LLM prefill vs decode.**

- **Fit / cache-build, and any full forward (including every `low_memory`
  predict): strongly compute-bound.** The O(N²) attention has AI = N/2, which is
  thousands-to-millions of FLOP/byte — orders of magnitude above the ridge.
  Weights (~106 MB) are read once and reused across all N rows, so weight
  traffic is negligible. This is the "prefill" regime.

- **Cached predict: bounded by *what you're waiting on*, set by the test batch
  size M.** The KV-cache read is a **fixed O(N) cost regardless of M** (you must
  stream all ~586 MB/estimator of keys/values), while compute scales with M.
  - Small batch / online serving (M ≲ 38 rows/call): **memory-bandwidth bound** —
    you drag the whole KV cache across HBM to score a handful of rows. This is
    the "decode" regime.
  - Batched serving (M ≳ 40, e.g. our M=1000): **compute-bound** again, because
    the streamed KV is amortized over many queries.
  - Crossover ≈ **M > ~38 test rows per call** (AI = 8·M = ridge → M ≈ 38).

**Why the architecture is built the way it is.** The **multi-query 1-head KV
cache** (`icl_num_kv_heads_test=1`) is precisely a *memory-bandwidth*
optimization for the memory-bound small-batch predict regime: it cuts
bytes-streamed-per-token by 8×, which both shrinks the cache (7 GB vs 56 GB at
1M rows) and directly speeds up online prediction (and lowers the compute/memory
crossover M). Int8 KV quantization is the same lever again — halving bytes moved.
Row-chunking, by contrast, targets the *compute/activation-memory* side of the
fit pass. FlashAttention-3 raises the compute ceiling for the compute-bound fit
and batched-predict phases.

**Caveat for your laptop.** All of the above is the GPU (H100 / flash) picture.
On CPU/MPS with no flash kernel there is no N×N-free attention path, so large-N
fits are infeasible regardless of the roofline, and the practical bottleneck
becomes latency/compute on small problems.

Effectively, the balance between IO and Compute

Total Attention FLOPs, Total FF FLOPs, KV Cache  (assuming KV cache is being streamed in)
Compute KV Cache at all for longer sequence lengths (Does current attention tiling already happens - We currently run with max 200K)