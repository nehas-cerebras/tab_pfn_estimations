"""
TabPFN-3 end-to-end timeline simulator for a Cerebras CS system.

Physical model
--------------
* On-chip memory (SRAM): 30 GB at ~21 PB/s -> treated as free (never the bottleneck).
* MemoryX: everything above 30 GB, streamed at 0.15 TB/s (= 150 GB/s).
* Compute: 1.25e17 peak FLOP/s at 10% utilization -> 1.25e16 usable FLOP/s.

Two-phase pre-ICL feature aggregation
-------------------------------------
Phase 1 (column-chunked, fit only): stream the raw table one column-band at a time,
    build the 3 per-layer inducing summaries.  block1 x3 + block2 x2.
Phase 2 (row-chunked, train then test): stream one row-band at a time, run the 3
    fan-outs against the cached summaries + the column aggregator, emit the 512-dim
    per-row embedding.  block2 x3 + column-aggregator.

Then the 24-layer ICL transformer (fit: O(N^2) self-attention; predict: M x N cross).

WHAT CROSSES MemoryX: the raw featurised table (N x C x G, fp16), streamed once PER
estimator/batch element (each uses a different feature-subsample + row-bootstrap, so
the cells feeding downstream differ) -> transfer scales with E*B just like compute.
The 128-dim embeddings are produced on-chip and never leave; summaries and
row-embeddings fit in 30 GB.  Chunk size is set by the on-chip *activation* peak.

Overlap: double-buffered prefetch -> load(chunk0) then compute(i) || load(i+1).

FLOP CONVENTION
---------------
* 1 multiply-accumulate = 2 FLOPs.
* A Linear(in -> out) applied to T tokens costs 2 * T * in * out.
* Attention scores (Q@K^T) over q_len x k_len, summed across heads, cost
      2 * q_len * k_len * (H * head_dim) = 2 * q_len * k_len * width,
  because H * head_dim == width; the head_dim itself is the contracted axis and
  never appears in the score *shape*.  A@V costs the same.
* Every _*_flops() helper below returns the cost for ONE estimator/batch element.
  The E*B factor is applied exactly once, at the call site, via self.mult.
"""

from rich.console import Console
from rich.table import Table
from rich import box

console = Console(width=104)

# ---------------------------------------------------------------- hardware ----
MEMORY_BANDWIDTH = 0.15 * 10**12        # 0.15 TB/s = 150 GB/s
ONCHIP_MEMORY    = 30 * 10**9           # 30 GB SRAM
PEAK_FLOPS       = 1.25 * 10**17
UTILIZATION      = 0.10
USABLE_FLOPS     = PEAK_FLOPS * UTILIZATION


# --------------------------------------------------------------- formatters ---
def fmt_bytes(b):
    if b >= 1e9:  return f"{b/1e9:.2f} GB"
    if b >= 1e6:  return f"{b/1e6:.2f} MB"
    if b >= 1e3:  return f"{b/1e3:.2f} KB"
    return f"{b:.0f} B"


def fmt_flops(f):
    if f >= 1e12: return f"{f/1e12:.2f} TFLOP"
    if f >= 1e9:  return f"{f/1e9:.2f} GFLOP"
    return f"{f/1e6:.2f} MFLOP"


def fmt_t(s):
    if s <= 0:    return "0"
    if s >= 1:    return f"{s:.3f} s"
    if s >= 1e-3: return f"{s*1e3:.2f} ms"
    return f"{s*1e6:.1f} us"

# ---------------------------------------------- pre-ICL (stage 1/2) constants -
EMB      = 128     # per-column embedding width (dist-embed + column-aggregator)
G        = 6       # feature_group_size: raw input width per column
KIND     = 128     # number of inducing points / summary tokens (== EMB numerically)
CLS      = 4       # column-aggregator CLS tokens prepended per row
FF_STAGE = 2 * EMB # stage-1/2 feed-forward width (ff_factor = 2)  -> 256

# ------------------------------------------------------- ICL (stage 3) consts -
D             = 512    # icl_emsize (4 CLS x 128)
FF            = 1024   # icl feed-forward width
HDIM          = 64     # icl head dim
KV_HEADS_TEST = 1      # multi-query KV heads at predict (memory saving)
NLAYERS       = 24     # icl transformer depth

# ------------------------------------------ output head (stage 4) constants ---
# The config selects ONE head: many-class decoder (multiclass) OR MLP (regression).
MAX_NUM_CLASSES  = 160   # decoder one-hot width (fixed capacity, even for binary)
DEC_HEAD_DIM     = 64    # decoder_head_dim
DEC_NUM_HEADS    = 6     # decoder_num_heads  -> attention_size = 64*6 = 384
NUM_BUCKETS      = 5000  # regression bar-distribution bins (= n_out for regression)

# ----------------------------------------------------------- memory model -----
BYTES           = 2    # fp16
ACT_PEAK_COPIES = 3    # x + K/Q + V/out co-resident (flash attn -> no score matrix)


def _t_flops(flops):
    return flops / USABLE_FLOPS


def _t_mem(nbytes):
    return nbytes / MEMORY_BANDWIDTH


def _pipeline_time(n_chunks, t_compute, t_transfer):
    """Double-buffered: load(0) then compute(i) overlapped with load(i+1)."""
    if n_chunks <= 0:
        return 0.0
    if n_chunks == 1:
        return t_transfer + t_compute
    return t_transfer + (n_chunks - 1) * max(t_compute, t_transfer) + t_compute


# =============================================================================
# FLOP formulas  (all per single estimator/batch element)
# =============================================================================
def _embed_flops(rows, cols):
    """Cell embedder: Linear(G -> EMB) applied to every (row, col) cell."""
    tokens = rows * cols
    return 2 * tokens * G * EMB


def _block1_flops(rows, cols):
    """
    Fan-IN (InducedSelfAttentionBlock.cross_attn_block1): the KIND=128 inducing
    points attend OVER all `rows` -> distils rows into a 128x128 summary.

    Query length  = KIND (inducing points)      -> constant in rows
    Key/Val length= rows                         -> the expensive axis

    Only the row-scaling terms are kept; the Q-proj / out-proj / MLP act on the
    128 inducing points (constant in rows) and are dropped as negligible.
    """
    k_proj = 2 * cols * rows * EMB * EMB   # K = Linear(EMB->EMB) over `rows` tokens
    v_proj = 2 * cols * rows * EMB * EMB   # V = Linear(EMB->EMB) over `rows` tokens
    scores = 2 * cols * KIND * rows * EMB  # Q@K^T : (KIND x rows), width EMB contracted
    av     = 2 * cols * KIND * rows * EMB  # A@V   : same shape
    # -> with KIND == EMB == 128 this is 8 * cols * rows * EMB^2
    return k_proj + v_proj + scores + av


def _block2_flops(rows, cols):
    """
    Fan-OUT (InducedSelfAttentionBlock.cross_attn_block2): all `rows` attend over
    the 128-token summary -> re-expands the summary back onto every row.

    Query length  = rows                         -> the expensive axis
    Key/Val length= KIND (summary)               -> constant in rows

    K/V projections act on the 128 summary tokens (constant in rows) -> dropped.
    Everything else touches all `rows`.
    """
    q_proj   = 2 * cols * rows * EMB * EMB          # Q = Linear(EMB->EMB) over `rows`
    scores   = 2 * cols * rows * KIND * EMB         # Q@K^T : (rows x KIND)
    av       = 2 * cols * rows * KIND * EMB         # A@V
    out_proj = 2 * cols * rows * EMB * EMB          # output Linear(EMB->EMB) over `rows`
    mlp      = (2 * cols * rows * EMB * FF_STAGE    # Linear(EMB->256) over `rows`
                + 2 * cols * rows * FF_STAGE * EMB) # Linear(256->EMB) over `rows`
    # -> with KIND == EMB and FF_STAGE == 2*EMB this is 16 * cols * rows * EMB^2
    return q_proj + scores + av + out_proj + mlp


def _col_agg_flops(rows, C):
    """
    Column aggregator: a tiny transformer run PER ROW over S = CLS + C tokens
    (4 CLS + C column embeddings), width EMB, 3 blocks.  Collapses the C columns
    into the 4 CLS tokens (-> 4*128 = 512-dim icl input).

    Blocks 1 & 2 are full self-attention over all S tokens (quadratic in S).
    Block 3 is cross-attention: only the CLS=4 tokens query the full S tokens, so
    it is linear in S -- only its K/V projection (over S tokens) survives; the
    Q-proj / out-proj / MLP / attention act on 4 tokens and are dropped.
    """
    S = CLS + C

    # ---- one FULL self-attention block (blocks 1 and 2) ----
    qkv_proj   = 3 * 2 * S * EMB * EMB              # Q,K,V = 3 x Linear(EMB->EMB) over S
    out_proj   = 2 * S * EMB * EMB                  # output Linear(EMB->EMB) over S
    mlp        = (2 * S * EMB * FF_STAGE            # Linear(EMB->256) over S
                  + 2 * S * FF_STAGE * EMB)         # Linear(256->EMB) over S
    full_dense = qkv_proj + out_proj + mlp          # = 16 * S * EMB^2
    full_attn  = 2 * S * S * EMB + 2 * S * S * EMB  # Q@K^T + A@V = 4 * S^2 * EMB

    two_full = 2 * (full_dense + full_attn)         # blocks 1 + 2

    # ---- block 3: cross-attn, CLS=4 tokens query the full S ----
    # only the K/V projections scale with S (4 * S * EMB^2); Q/out/MLP/attn ~ CLS -> dropped
    cross_kv_proj = 2 * 2 * S * EMB * EMB           # K and V = 2 x Linear(EMB->EMB) over S

    per_row = two_full + cross_kv_proj              # = 36*S*EMB^2 + 8*S^2*EMB
    return rows * per_row


def _icl_layer_fit_flops(N):
    """One ICL layer during fit: full self-attention over the N train rows."""
    scores = 2 * N * N * D                    # Q@K^T over (N x N), width D
    av     = 2 * N * N * D                    # A@V
    proj   = 4 * (2 * N * D * D)              # Q,K,V,out = 4 x Linear(D->D) over N
    mlp    = 2 * N * D * FF + 2 * N * FF * D  # Linear(D->FF) + Linear(FF->D) over N
    return scores + av + proj + mlp          # = 4*N^2*D + 8*N*D^2 + 4*N*D*FF


def _icl_layer_predict_flops(M, N):
    """One ICL layer during predict: M test rows cross-attend to N cached train rows."""
    scores = 2 * M * N * D                    # Q@K^T over (M x N)
    av     = 2 * M * N * D                    # A@V
    proj   = 2 * (2 * M * D * D)              # Q + out = 2 x Linear(D->D) over M
    #                                           (K,V come from the cache -> no proj here)
    mlp    = 2 * M * D * FF + 2 * M * FF * D  # over M
    return scores + av + proj + mlp          # = 4*M*N*D + 4*M*D^2 + 4*M*D*FF


def _kv_cache_bytes(N, mult):
    """KV cache: NLAYERS x (K,V) x N rows x head_dim x kv_heads, fp16, x estimators."""
    return mult * NLAYERS * 2 * N * HDIM * KV_HEADS_TEST * BYTES


def _head_flops(M, N, task, n_classes):
    """
    Final prediction head -- exactly one path is built, chosen by task_type.

    MULTICLASS (ManyClassDecoder): attention-based retrieval.  The M test rows
    query the N train rows; the "values" are the one-hot train targets, chunked
    into DEC_HEAD_DIM-wide pieces (num_chunks = ceil(n_classes / DEC_HEAD_DIM))
    and folded into the batch, so Q@K^T / A@V are each replicated per chunk.
    Cost is O(M*N) -- the head re-attends over the whole train set.

    REGRESSION (MLP head): Linear(D -> D*ff) -> GELU -> Linear(D*ff -> num_buckets),
    applied ONLY to the M test rows.  Cost is O(M) -- no train dependence at all.
    """
    if task == "multiclass":
        attn_size  = DEC_HEAD_DIM * DEC_NUM_HEADS           # 64 * 6 = 384
        num_chunks = (n_classes + DEC_HEAD_DIM - 1) // DEC_HEAD_DIM  # class dim folded

        q_proj = 2 * M * D * attn_size                      # test queries, over M
        k_proj = 2 * N * D * attn_size                      # train keys, over ALL N
        scores = 2 * num_chunks * M * N * attn_size         # Q@K^T, replicated per chunk
        av     = 2 * num_chunks * M * N * attn_size         # A@V over one-hot value chunks
        return q_proj + k_proj + scores + av

    # regression: two Linear layers over the M test rows only
    hidden = D * (FF // D)                                  # icl_emsize * ff_factor = 1024
    lin1 = 2 * M * D * hidden                               # Linear(D -> 1024)
    lin2 = 2 * M * hidden * NUM_BUCKETS                     # Linear(1024 -> num_buckets)
    return lin1 + lin2


class TabPFNEstimator:
    def __init__(self, **kwargs):
        self.N = kwargs.get('N', 1_000_000)   # train rows
        self.M = kwargs.get('M', 10_000)      # test rows
        self.E = kwargs.get('E', 8)           # estimators
        self.B = kwargs.get('B', 1)           # batch
        self.C = kwargs.get('C', 200)         # columns / features
        self.task = kwargs.get('task', 'multiclass')          # 'multiclass' | 'regression'
        self.n_classes = kwargs.get('n_classes', MAX_NUM_CLASSES)  # decoder one-hot width
        self.mult = self.E * self.B           # concurrent copies on-chip
        self.stages = []

    # ------------------------------------------------- chunk-size helpers ------
    def _max_cols(self, rows):
        """Largest column band whose activation peak (3 x rows x Cc x EMB) fits."""
        denom = ACT_PEAK_COPIES * BYTES * rows * EMB * self.mult
        return max(1, int(ONCHIP_MEMORY / denom))

    def _max_rows(self, cols, reserve):
        """Largest row band whose activation peak fits in (30GB - reserve)."""
        denom = ACT_PEAK_COPIES * BYTES * cols * EMB * self.mult
        return max(1, int((ONCHIP_MEMORY - reserve) / denom))

    def _record(self, rec):
        """Store a per-stage record (dict) for narration + the summary table."""
        self.stages.append(rec)
        return rec['wall']

    # ------------------------------------ Phase 1: column-chunked summaries ----
    def phase1(self):
        # chunk columns so the (rows x Cc x EMB) activation peak fits on-chip
        Cc = min(self.C, self._max_cols(self.N))
        n_chunks = (self.C + Cc - 1) // Cc


        # why this many chunks: each column costs a fixed activation peak; pack as
        # many as fit in 30 GB, then ceil-divide the C columns over that width.
        peak_per_col = ACT_PEAK_COPIES * BYTES * self.N * EMB * self.mult
        why = (f"peak = {ACT_PEAK_COPIES}x(N x 128) fp16 x{self.mult} = "
               f"{fmt_bytes(peak_per_col)}/col")

        # IN (MemX -> CSX): raw cells, N x C x G, one copy per estimator. Bill the
        # TRUE total (work is linear in columns) -- NOT Cc*n_chunks, which over-counts
        # the partial last chunk and makes the wall jump each time n_chunks steps.
        in_total = BYTES * self.N * self.C * G * self.mult
        # OUT (CSX -> MemX): 3 per-layer summaries (inducing_hidden), kept for phase 2
        out_bytes = 3 * self.C * EMB * EMB * BYTES * self.mult
        # compute: cell embed + 3 fan-ins (build summaries) + 2 fan-outs, over full C
        flops_total = (_embed_flops(self.N, self.C)
                       + 3 * _block1_flops(self.N, self.C)
                       + 2 * _block2_flops(self.N, self.C)) * self.mult
        # per-chunk pieces feed ONLY the double-buffer fill/drain term
        in_chunk, flops_chunk = in_total / n_chunks, flops_total / n_chunks

        t_in_chunk, t_cmp_chunk = _t_mem(in_chunk), _t_flops(flops_chunk)
        t_in, t_cmp = _t_mem(in_total), _t_flops(flops_total)
        t_out = _t_mem(out_bytes)
        wall = _pipeline_time(n_chunks, t_cmp_chunk, t_in_chunk) + t_out
        return self._record({
            'title': "Stage 1 - Cell + Feature Embedding, Phase 1 (column-wise chunking)",
            'tag': "S1 embed (col-chunk)",
            'axis': "columns", 'n_chunks': n_chunks, 'chunk_desc': f"{Cc} cols", 'why': why,
            'in_desc': f"raw cells  N x C={self.C} x G={G}  x {self.mult} est",
            'in_chunk': in_chunk, 'in_total': in_total, 't_in': t_in,
            'out_desc': "3 inducing summaries (kept for Phase 2)",
            'out_bytes': out_bytes, 't_out': t_out,
            'flops_chunk': flops_chunk, 'flops_total': flops_total, 't_cmp': t_cmp,
            'wall': wall, 'pipelined': True,
            'bound': "memory" if t_in_chunk > t_cmp_chunk else "compute",
        })

    # -------------------------- Phase 2: row-chunked fan-out + aggregation -----
    def phase2(self, rows, label):
        # the produced 512-dim row embeddings stay on-chip -> reserve their space
        reserve = BYTES * rows * D * self.mult
        Rc = min(rows, self._max_rows(self.C, reserve))
        n_chunks = (rows + Rc - 1) // Rc
        # why this many chunks: the 512-d row embeddings must also sit on-chip, so
        # only (30 GB - that reserve) is left for the per-row activation peak.
        peak_per_row = ACT_PEAK_COPIES * BYTES * self.C * EMB * self.mult
        why = (f"reserve {fmt_bytes(reserve)} for {rows:,}x{D}-d embeddings; "
               f"peak = {ACT_PEAK_COPIES}x(C x 128) fp16 x{self.mult} = "
               f"{fmt_bytes(peak_per_row)}/row; Rc = floor((30GB-reserve)/that) = {Rc}; "
               f"n = ceil({rows:,}/Rc) = {n_chunks}")

        # IN (MemX -> CSX): raw cells, rows x C x G, one copy per estimator. Bill the
        # TRUE total (work is linear in rows) -- NOT Rc*n_chunks, which over-counts the
        # partial last chunk and makes the wall jump each time n_chunks steps.
        in_total = BYTES * rows * self.C * G * self.mult
        # OUT: 512-dim row embeddings, retained on-chip for ICL -> no MemX write
        out_bytes = 0
        # compute: cell embed + 3 fan-outs (over full C) + column aggregator, over all rows
        flops_total = (_embed_flops(rows, self.C)
                       + 3 * _block2_flops(rows, self.C)
                       + _col_agg_flops(rows, self.C)) * self.mult
        # per-chunk pieces feed ONLY the double-buffer fill/drain term
        in_chunk, flops_chunk = in_total / n_chunks, flops_total / n_chunks

        t_in_chunk, t_cmp_chunk = _t_mem(in_chunk), _t_flops(flops_chunk)
        t_in, t_cmp = _t_mem(in_total), _t_flops(flops_total)
        wall = _pipeline_time(n_chunks, t_cmp_chunk, t_in_chunk)
        return self._record({
            'title': f"Stage 2 - Feature Aggregation, Phase 2, {label.upper()} rows (row-wise chunking)",
            'tag': f"S2 aggregate {label.upper()} (row-chunk)",
            'axis': "rows", 'n_chunks': n_chunks, 'chunk_desc': f"{Rc} rows", 'why': why,
            'in_desc': f"raw cells  {rows:,} x C={self.C} x G={G}  x {self.mult} est",
            'in_chunk': in_chunk, 'in_total': in_total, 't_in': t_in,
            'out_desc': f"{rows:,} x {D}-d row embeddings retained on-chip (no MemX write)",
            'out_bytes': out_bytes, 't_out': 0.0,
            'flops_chunk': flops_chunk, 'flops_total': flops_total, 't_cmp': t_cmp,
            'wall': wall, 'pipelined': True,
            'bound': "memory" if t_in_chunk > t_cmp_chunk else "compute",
        })

    # ------------------------------------------------- ICL fit (24 layers) -----
    def icl_fit(self):
        # Pipeline over the 24 layers: the residual stream stays on-chip, and each
        # layer's KV block is flushed to MemX as soon as it is computed, overlapping
        # with the NEXT layer's compute (double-buffer). We never hold all 24
        # layers' KV on-chip at once.
        flops_layer = self.mult * _icl_layer_fit_flops(self.N)
        t_cmp_layer = _t_flops(flops_layer)

        kv_bytes = _kv_cache_bytes(self.N, self.mult)   # total across all layers
        kv_layer = kv_bytes / NLAYERS                   # one layer's (K,V) block
        t_out_layer = _t_mem(kv_layer)

        flops = flops_layer * NLAYERS
        t_cmp = t_cmp_layer * NLAYERS
        t_out = t_out_layer * NLAYERS
        wall = _pipeline_time(NLAYERS, t_cmp_layer, t_out_layer)

        why = (f"pipeline over {NLAYERS} layers; per-layer flush "
               f"{fmt_bytes(kv_layer)} overlaps next layer's compute "
               f"({fmt_t(t_cmp_layer)} cmp vs {fmt_t(t_out_layer)} out)")

        return self._record({
            'title': "Stage 3 - ICL Transformer, FIT (build KV cache)",
            'tag': "S3 ICL fit (N^2 self-attn)",
            'axis': "layers", 'n_chunks': NLAYERS, 'chunk_desc': "1 layer", 'why': why,
            'in_desc': "row embeddings already on-chip (from Phase 2)",
            'in_chunk': 0, 'in_total': 0, 't_in': 0.0,
            'out_desc': f"KV cache  {NLAYERS} layers x (K,V) x N x {HDIM} x {KV_HEADS_TEST}kv "
                        f"(flushed 1 layer at a time)",
            'out_bytes': kv_bytes, 't_out': t_out,
            'flops_chunk': flops_layer, 'flops_total': flops, 't_cmp': t_cmp,
            'wall': wall, 'pipelined': True,
            'bound': "memory" if t_out_layer > t_cmp_layer else "compute",
        })

    # --------------------------------------------- ICL predict (24 layers) -----
    def icl_predict(self):
        # Pipeline over the 24 layers: layer L's KV block is streamed in from MemX
        # while layer L-1 is being attended (double-buffer). We prefetch layer L+1's
        # KV during layer L's cross-attention, so only ~2 layers' KV sit on-chip.
        flops_layer = self.mult * _icl_layer_predict_flops(self.M, self.N)
        t_cmp_layer = _t_flops(flops_layer)

        kv_bytes = _kv_cache_bytes(self.N, self.mult)   # total across all layers
        kv_layer = kv_bytes / NLAYERS                   # one layer's (K,V) block
        t_in_layer = _t_mem(kv_layer)

        flops = flops_layer * NLAYERS
        t_cmp = t_cmp_layer * NLAYERS
        t_in = t_in_layer * NLAYERS
        wall = _pipeline_time(NLAYERS, t_cmp_layer, t_in_layer)

        why = (f"pipeline over {NLAYERS} layers; per-layer load "
               f"{fmt_bytes(kv_layer)} overlaps prev layer's compute "
               f"({fmt_t(t_cmp_layer)} cmp vs {fmt_t(t_in_layer)} in)")

        return self._record({
            'title': "Stage 3 - ICL Transformer, PREDICT (M x N cross-attention)",
            'tag': "S3 ICL predict (M x N)",
            'axis': "layers", 'n_chunks': NLAYERS, 'chunk_desc': "1 layer", 'why': why,
            'in_desc': f"KV cache read back 1 layer at a time "
                       f"({fmt_bytes(kv_layer)}/layer, {fmt_bytes(kv_bytes)} total)",
            'in_chunk': kv_layer, 'in_total': kv_bytes, 't_in': t_in,
            'out_desc': "logits (tiny)",
            'out_bytes': 0, 't_out': 0.0,
            'flops_chunk': flops_layer, 'flops_total': flops, 't_cmp': t_cmp,
            'wall': wall, 'pipelined': True,
            'bound': "memory" if t_in_layer > t_cmp_layer else "compute",
        })

    # ------------------------------------------------ output head (stage 4) ----
    def output_head(self):
        flops = self.mult * _head_flops(self.M, self.N, self.task, self.n_classes)
        t_cmp = _t_flops(flops)

        if self.task == "multiclass":
            nchunks = (self.n_classes + DEC_HEAD_DIM - 1) // DEC_HEAD_DIM
            in_desc = f"train embeddings on-chip (N x {D}); retrieval attn M x N"
            out_bytes = BYTES * self.M * self.n_classes * self.mult
            out_desc = (f"logits M x {self.n_classes}  "
                        f"[{nchunks} class-chunks, O(M*N)]")
        else:
            in_desc = f"test embeddings on-chip (M x {D})"
            out_bytes = BYTES * self.M * NUM_BUCKETS * self.mult
            out_desc = (f"bar-dist logits M x {NUM_BUCKETS}  "
                        f"[MLP {D}->{FF}->{NUM_BUCKETS}, O(M)]")

        return self._record({
            'title': f"Stage 4 - Output Head ({self.task})",
            'tag': f"S4 head ({self.task})",
            'axis': None, 'n_chunks': 1, 'chunk_desc': "-",
            'in_desc': in_desc, 'in_chunk': 0, 'in_total': 0, 't_in': 0.0,
            'out_desc': out_desc, 'out_bytes': out_bytes, 't_out': 0.0,
            'flops_chunk': flops, 'flops_total': flops, 't_cmp': t_cmp,
            'wall': t_cmp, 'pipelined': False, 'bound': "compute",
        })

    # -------------------------------------------------------------- driver -----
    def simulate(self):
        """Run all stages, populate self.stages, return them (no console output)."""
        self.stages = []
        self.phase1()
        self.phase2(self.N, "train")
        self.phase2(self.M, "test")
        self.icl_fit()
        self.icl_predict()
        self.output_head()
        return self.stages

    def run(self):
        self.simulate()
        self._render()

    # ------------------------------------------------------------- rendering ---
    def _header(self):
        console.rule("[bold white]TabPFN-3  wafer timeline")
        console.print(
            f"[bold]N[/]={self.N:,}  [bold]M[/]={self.M:,}  [bold]C[/]={self.C}  "
            f"[bold]E[/]={self.E}  [bold]B[/]={self.B}  [bold]task[/]={self.task}\n"
            f"[dim]MemX {MEMORY_BANDWIDTH/1e12:.2f} TB/s  |  on-chip {ONCHIP_MEMORY/1e9:.0f} GB  |  "
            f"{USABLE_FLOPS/1e15:.1f} PF/s usable ({UTILIZATION:.0%} of {PEAK_FLOPS/1e15:.0f} PF/s)[/]\n")

    def _narrate(self, r):
        col = "red" if r['bound'] == "memory" else "green"
        console.print(f"[bold cyan]{r['title']}[/]")
        if r['axis']:
            console.print(f"   Processing done in [bold]{r['n_chunks']}[/] chunks "
                          f"({r['chunk_desc']} each, chunked over [bold]{r['axis']}[/]).")
            if r.get('why'):
                console.print(f"     [dim]why {r['n_chunks']} chunks: {r['why']}[/]")
        # IN
        if r['in_total'] > 0:
            if r['n_chunks'] > 1:
                console.print(
                    f"   [green]IN  MemX->CSX[/]: {r['in_desc']}\n"
                    f"        = [bold]{fmt_bytes(r['in_chunk'])}[/]/chunk -> "
                    f"[bold]{fmt_t(r['t_in']/r['n_chunks'])}[/]/chunk   "
                    f"(total {fmt_bytes(r['in_total'])} -> {fmt_t(r['t_in'])})")
            else:
                console.print(f"   [green]IN  MemX->CSX[/]: {r['in_desc']} "
                              f"= [bold]{fmt_bytes(r['in_total'])}[/] -> [bold]{fmt_t(r['t_in'])}[/]")
        else:
            console.print(f"   [green]IN  MemX->CSX[/]: {r['in_desc']}  [dim](no transfer)[/]")
        # OUT
        if r['out_bytes'] > 0:
            console.print(f"   [yellow]OUT CSX->MemX[/]: {r['out_desc']} "
                          f"= [bold]{fmt_bytes(r['out_bytes'])}[/] -> [bold]{fmt_t(r['t_out'])}[/]")
        else:
            console.print(f"   [yellow]OUT CSX->MemX[/]: {r['out_desc']}")
        # compute
        cmp_chunk = f" ([bold]{fmt_flops(r['flops_chunk'])}[/]/chunk)" if r['n_chunks'] > 1 else ""
        console.print(f"   [blue]COMPUTE[/]: {fmt_flops(r['flops_total'])}{cmp_chunk} -> "
                      f"[bold]{fmt_t(r['t_cmp'])}[/]")
        # wall + bound
        overlap = "overlapped" if r['pipelined'] else "sequential"
        console.print(f"   -> [bold]WALL {fmt_t(r['wall'])}[/]  "
                      f"[dim]({overlap}, bound by[/] [bold {col}]{r['bound']}[/][dim])[/]\n")

    def _render(self):
        self._header()
        for r in self.stages:
            self._narrate(r)

        table = Table(title="Summary", box=box.SIMPLE_HEAVY, title_style="bold")
        table.add_column("stage", justify="left", style="cyan", no_wrap=True)
        table.add_column("chunks", justify="right")
        table.add_column("IN", justify="right", style="green")
        table.add_column("OUT", justify="right", style="yellow")
        table.add_column("compute", justify="right", style="blue")
        table.add_column("wall", justify="right", style="bold")
        table.add_column("bound", justify="center")

        grand = 0.0
        for r in self.stages:
            grand += r['wall']
            bound_col = "red" if r['bound'] == "memory" else "green"
            table.add_row(
                r['tag'],
                str(r['n_chunks']) if r['axis'] else "-",
                fmt_t(r['t_in']), fmt_t(r['t_out']), fmt_t(r['t_cmp']),
                fmt_t(r['wall']), f"[{bound_col}]{r['bound']}[/]")
        table.add_section()
        table.add_row("GRAND TOTAL", "", "", "", "", fmt_t(grand), "", style="bold white")
        console.print(table)


# =============================================================================
# Plotting: per-stage wall time vs training rows, small-multiples grid
# =============================================================================
# Colours: Okabe-Ito colourblind-safe qualitative palette, assigned in fixed
# order (identity follows the stage, never cycled). Yellow (#F0E442) is omitted
# as it washes out on a white surface.
STAGE_SERIES = [
    ("S1",                    "#E69F00"),   # orange
    ("S2 train",              "#56B4E9"),   # sky blue
    ("S2 test",               "#0072B2"),   # blue
    ("S3 ICL fit",            "#009E73"),   # bluish green
    ("S3 ICL predict",        "#D55E00"),   # vermillion
    ("S4 head (multiclass)",  "#CC79A7"),   # reddish purple
]
TOTAL_COLOR = "#000000"


def plot_stage_timings(out_dir="plots", n_points=30):
    """3x4 small-multiples grid of per-stage wall time vs training rows.

    Rows    -> n_features C in {10, 100, 500}
    Columns -> n_estimators E in {1, 2, 4, 8}
    Each panel: one log-log line per stage (S1, S2 train/test, ICL fit/predict,
    head) plus the Total. Test rows are pinned to M = 0.01 * N_train.
    Written to <out_dir>/stage_timings.png every time the script runs.
    """
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")                       # headless: just write the file
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(out_dir, exist_ok=True)

    feature_rows   = [10, 100, 500]             # C, one per grid row
    estimator_cols = [1, 2, 4, 8]               # E, one per grid column
    N_grid = np.logspace(2, 6, n_points)        # 100 .. 1,000,000 training rows

    fig, axes = plt.subplots(
        len(feature_rows), len(estimator_cols),
        figsize=(20, 12), sharex=True, sharey=True,
    )

    for i, C in enumerate(feature_rows):
        for j, E in enumerate(estimator_cols):
            ax = axes[i][j]
            series = {lbl: [] for lbl, _ in STAGE_SERIES}   # ms per point
            bounds = {lbl: [] for lbl, _ in STAGE_SERIES}   # 'compute' | 'memory'
            totals = []
            for Nf in N_grid:
                N = int(round(Nf))
                M = max(1, int(round(0.01 * N)))            # predict rows = 1% of train
                stages = TabPFNEstimator(
                    N=N, M=M, C=C, E=E, B=1, task='multiclass').simulate()
                walls = [r['wall'] for r in stages]         # fixed order (see simulate)
                for k, (lbl, _) in enumerate(STAGE_SERIES):
                    series[lbl].append(walls[k] * 1e3)      # s -> ms
                    bounds[lbl].append(stages[k]['bound'])
                totals.append(sum(walls) * 1e3)

            # thin the markers: ~1 every `mstep` points (line stays continuous)
            mstep = max(1, n_points // 8)
            show = (np.arange(n_points) % mstep == 0)
            for lbl, color in STAGE_SERIES:
                y = np.array(series[lbl])
                b = np.array(bounds[lbl])
                ax.plot(N_grid, y, color=color, lw=1.6, label=lbl, zorder=2)
                # marker shape encodes the bottleneck: square = compute, circle = memory
                cmp_m = (b == "compute") & show
                mem_m = (b == "memory") & show
                ax.scatter(N_grid[cmp_m], y[cmp_m], marker="s", s=26, color=color,
                           edgecolors="white", linewidths=0.6, zorder=3)
                ax.scatter(N_grid[mem_m], y[mem_m], marker="o", s=26, color=color,
                           edgecolors="white", linewidths=0.6, zorder=3)
            ax.plot(N_grid, totals, color=TOTAL_COLOR, lw=2.6, label="Total", zorder=4)

            # report the peak Total time (in seconds) in the top-left corner -- the
            # one region every panel leaves empty (all lines start low at N=100), so
            # it never brushes the curve or the frame.
            totals_arr = np.array(totals)
            peak_s = totals_arr.max() / 1e3
            ax.text(0.04, 0.95, f"max: {peak_s:.2f} s",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=11, fontweight="bold", color=TOTAL_COLOR, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7",
                              lw=0.6, alpha=0.9))

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.4)
            ax.tick_params(labelsize=9)
            if i == 0:
                ax.set_title(f"E = {E} estimator{'s' if E > 1 else ''}",
                             fontsize=13, fontweight="bold")
            if j == 0:
                ax.set_ylabel("time (ms)", fontsize=11)
                ax.annotate(f"C = {C} features", xy=(-0.34, 0.5),
                            xycoords="axes fraction", ha="center", va="center",
                            rotation=90, fontsize=13, fontweight="bold")
            if i == len(feature_rows) - 1:
                ax.set_xlabel("training rows  N", fontsize=11)

    handles, labels = axes[0][0].get_legend_handles_labels()
    # append a shape key: square = compute-bound point, circle = memory-bound point
    handles += [
        Line2D([], [], marker="s", ls="none", color="0.35", markeredgecolor="white",
               markersize=8, label="compute-bound"),
        Line2D([], [], marker="o", ls="none", color="0.35", markeredgecolor="white",
               markersize=8, label="memory-bound"),
    ]
    labels += ["compute-bound", "memory-bound"]
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("TabPFN-3 per-stage wall time  (M = 0.01·N, log–log)",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0.02, 0, 1, 0.95])

    out = os.path.join(out_dir, "stage_timings.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == '__main__':
    TabPFNEstimator(N=1_000_000, M=10_000, C=200, E=8, B=1, task='multiclass').run()
    print()
    # TabPFNEstimator(N=1_000_000, M=10_000, C=200, E=8, B=1, task='regression').run()
    plot_stage_timings()
