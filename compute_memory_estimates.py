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
  The E*B factor is applied exactly once, at the call site, via self.mult_time.
"""

from rich.console import Console
from rich.table import Table
from rich import box

console = Console(width=104)

# ---------------------------------------------------------------- hardware ----
# A Device bundles the four numbers that make one accelerator different from
# another.  The stage math below is written against these fields, never against
# literal constants, so swapping devices is just swapping this object.
#
#   mem_bw      slow-tier bandwidth (B/s).  The ONLY bandwidth that costs time;
#               the on-chip tier is treated as free (see onchip_free).
#   onchip_free capacity of the free/fast tier (SRAM).  A cross-stage tensor is
#               kept here for free iff it fits; otherwise it spills to the slow
#               tier and its transfer is billed at mem_bw.  This single number
#               is what makes CS3 and H100 bill different things (see _resident).
#   dram_cap    resident budget used for estimator wave-packing (see below).
#               inf for CS3, whose KV/summaries stream to MemoryX instead of
#               staying resident.
#   usable_flops   peak FLOP/s x utilization -- the compute rate actually seen.
#   concurrent_estimators   CS3 fits all E estimators on the wafer at once;
#               H100 packs as many as fit in dram_cap and runs the rest in waves.
from dataclasses import dataclass


@dataclass(frozen=True)
class Device:
    name: str
    mem_bw: float             # slow-tier bandwidth (B/s)
    onchip_free: float        # free/SRAM tier capacity (B)
    dram_cap: float           # resident budget for wave-packing (B; inf = streams)
    peak_flops: float         # peak FP16 tensor-core throughput (FLOP/s)
    util: float               # sustained fraction of peak on this workload
    concurrent_estimators: bool
    slow_tier: str            # name of the slow tier, for narration (MemX / HBM)
    label: str = ""           # human-friendly full name for the plot panel
    note: str = ""            # one-line modelling caveat shown on the plot panel

    @property
    def usable_flops(self):
        return self.peak_flops * self.util


# Cerebras CS-3: 30 GB SRAM at ~21 PB/s (free); MemoryX streams at 0.15 TB/s;
# 125 PF/s peak FP16 at 10% utilization.  Whole working set is resident.
CS3 = Device(
    name="CS3", mem_bw=0.15e12, onchip_free=30e9, dram_cap=float("inf"),
    peak_flops=1.25e16, util=0.10, concurrent_estimators=True, slow_tier="MemX",
    label="Cerebras CS-3",
    note="30 GB SRAM holds the whole working set, so only the raw table and KV "
         "cache cross MemoryX.",
)

# NVIDIA H100 SXM5: ~50 MB on-chip (L2) -- too small to hold the working set, so
# the 80 GB HBM3 at 3.35 TB/s is the tier data must cross.  Dense FP16 tensor
# core is 989 TF/s.
#
# Utilization is CALIBRATED, not guessed.  At N=1M a single estimator's runtime
# is 99.9% the ICL N^2 self-attention -- a hard 4.93e16-FLOP count with no free
# parameters except sustained throughput.  Prior Labs measure this exact point
# at ~107 s on an H100 (TabPFN-3 paper, Fig. 8, single estimator, no preproc.,
# FlashAttention-3 on).  Pinning util so the model reproduces 107 s gives 0.465,
# squarely in FA3's realistic MFU band (FA3 tops out near 75% of peak); our
# earlier 40% was conservative and made the GPU look ~17% slower than measured.
H100 = Device(
    name="H100", mem_bw=3.35e12, onchip_free=50e6, dram_cap=80e9,
    peak_flops=989e12, util=0.465, concurrent_estimators=False, slow_tier="HBM",
    label="NVIDIA H100 SXM5",
    note="",
)

DEVICES = {d.name: d for d in (CS3, H100)}


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
        self.dev = kwargs.get('device', CS3)  # which accelerator we are modelling

        # Two different "how many estimators at once" numbers:
        #   mult_time  -- total work.  ALL E*B estimator/batch copies must be
        #                 computed and streamed, whichever device, so FLOP and
        #                 byte TOTALS always scale by this.
        #   mult_mem   -- copies co-resident on-chip *at the same time*, which is
        #                 what competes for the free tier and thus sizes chunks.
        # CS3 holds all E*B at once (mult_mem == mult_time).  H100 packs as many
        # estimators as fit in its 80 GB HBM and runs the rest in later waves, so
        # only n_concurrent*B copies are resident together.
        self.mult_time = self.E * self.B
        self.n_concurrent, self.n_waves = self._plan_waves()
        self.mult_mem = self.n_concurrent * self.B
        self.stages = []

    # -------------------------------------------- estimator wave-packing -------
    def _plan_waves(self):
        """How many estimators run concurrently, and in how many waves.

        CS3 runs all E at once (its KV/summaries spill to MemoryX rather than
        occupying the resident budget).  H100 must keep each estimator's KV
        cache + train row-embeddings resident in 80 GB HBM, so it fits
        floor(budget / per-estimator footprint) at a time and waves the rest.
        Total wall time is unchanged by this (it is set by mult_time), but the
        concurrency correctly sizes the on-chip chunks for each device.
        """
        if self.dev.concurrent_estimators:
            return self.E, 1
        per_est = (_kv_cache_bytes(self.N, 1)            # ~7 GB @ 1M rows
                   + BYTES * self.N * D)                 # train row-embeddings
        fits = max(1, int(self.dev.dram_cap / per_est))
        n_concurrent = min(self.E, fits)
        n_waves = (self.E + n_concurrent - 1) // n_concurrent
        return n_concurrent, n_waves

    # ------------------------------------------------------ time & tier --------
    def _t_flops(self, flops):
        return flops / self.dev.usable_flops

    def _t_mem(self, nbytes):
        return nbytes / self.dev.mem_bw

    def _resident(self, per_est_bytes):
        """True iff a cross-stage tensor stays on-chip for free: all co-resident
        copies fit in the free SRAM tier.  On CS3 (30 GB) nearly everything does;
        on H100 (50 MB) nearly nothing does."""
        return per_est_bytes * self.mult_mem <= self.dev.onchip_free

    def _stream_bytes(self, per_est_bytes):
        """Bytes a cross-stage tensor forces across the slow tier: zero while it
        stays resident on-chip, else every copy (mult_time) must be written/read.
        This is the ONE rule that makes CS3 bill only raw cells + KV, while H100
        also bills the row-embedding and summary round-trips it cannot hold."""
        return 0 if self._resident(per_est_bytes) else per_est_bytes * self.mult_time

    # ------------------------------------------------- chunk-size helpers ------
    def _max_cols(self, rows):
        """Largest column band whose activation peak (3 x rows x Cc x EMB) fits."""
        denom = ACT_PEAK_COPIES * BYTES * rows * EMB * self.mult_mem
        return max(1, int(self.dev.onchip_free / denom))

    def _max_rows(self, cols, reserve):
        """Largest row band whose activation peak fits in (free tier - reserve)."""
        denom = ACT_PEAK_COPIES * BYTES * cols * EMB * self.mult_mem
        return max(1, int((self.dev.onchip_free - reserve) / denom))

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
        # many as fit in the free tier, then ceil-divide the C columns over it.
        peak_per_col = ACT_PEAK_COPIES * BYTES * self.N * EMB * self.mult_mem
        why = (f"peak = {ACT_PEAK_COPIES}x(N x 128) fp16 x{self.mult_mem} = "
               f"{fmt_bytes(peak_per_col)}/col")

        # IN (slow tier -> chip): raw cells, N x C x G, one copy per estimator.
        # Bill the TRUE total (work is linear in columns) -- NOT Cc*n_chunks, which
        # over-counts the partial last chunk and jumps the wall each n_chunks step.
        in_total = BYTES * self.N * self.C * G * self.mult_time
        # OUT (chip -> slow tier): the 3 per-block inducing summaries are flushed
        # for phase 2 to reuse.  This checkpoint is billed on both devices.
        out_bytes = 3 * self.C * EMB * EMB * BYTES * self.mult_time
        # compute: cell embed + 3 fan-ins (build summaries) + 2 fan-outs, over full C
        flops_total = (_embed_flops(self.N, self.C)
                       + 3 * _block1_flops(self.N, self.C)
                       + 2 * _block2_flops(self.N, self.C)) * self.mult_time
        # per-chunk pieces feed ONLY the double-buffer fill/drain term
        in_chunk, flops_chunk = in_total / n_chunks, flops_total / n_chunks

        t_in_chunk, t_cmp_chunk = self._t_mem(in_chunk), self._t_flops(flops_chunk)
        t_in, t_cmp = self._t_mem(in_total), self._t_flops(flops_total)
        t_out = self._t_mem(out_bytes)
        wall = _pipeline_time(n_chunks, t_cmp_chunk, t_in_chunk) + t_out
        return self._record({
            'title': "Stage 1 - Cell + Feature Embedding, Phase 1 (column-wise chunking)",
            'tag': "S1 embed (col-chunk)",
            'axis': "columns", 'n_chunks': n_chunks, 'chunk_desc': f"{Cc} cols", 'why': why,
            'in_desc': f"raw cells  N x C={self.C} x G={G}  x {self.mult_time} est",
            'in_chunk': in_chunk, 'in_total': in_total, 't_in': t_in,
            'out_desc': "3 inducing summaries (kept for Phase 2)",
            'out_bytes': out_bytes, 't_out': t_out,
            'flops_chunk': flops_chunk, 'flops_total': flops_total, 't_cmp': t_cmp,
            'wall': wall, 'pipelined': True,
            'bound': "memory" if t_in_chunk > t_cmp_chunk else "compute",
        })

    # -------------------------- Phase 2: row-chunked fan-out + aggregation -----
    def phase2(self, rows, label):
        # The 512-d row embeddings are kept on-chip only if they fit the free tier
        # (they do on CS3, they do not on H100).  When resident we reserve their
        # space and pay no transfer; otherwise they live in the slow tier (billed
        # as rowemb_out below) and free the reserve for a bigger activation band.
        rowemb_per_est = BYTES * rows * D
        reserve = rowemb_per_est * self.mult_mem if self._resident(rowemb_per_est) else 0
        Rc = min(rows, self._max_rows(self.C, reserve))
        n_chunks = (rows + Rc - 1) // Rc
        peak_per_row = ACT_PEAK_COPIES * BYTES * self.C * EMB * self.mult_mem
        why = (f"reserve {fmt_bytes(reserve)} for {rows:,}x{D}-d embeddings; "
               f"peak = {ACT_PEAK_COPIES}x(C x 128) fp16 x{self.mult_mem} = "
               f"{fmt_bytes(peak_per_row)}/row; Rc = floor((free-reserve)/that) = {Rc}; "
               f"n = ceil({rows:,}/Rc) = {n_chunks}")

        # IN (slow tier -> chip): raw cells, rows x C x G, one copy per estimator.
        # Bill the TRUE total (work is linear in rows) -- see phase1.  On H100 the
        # cached inducing summaries no longer fit on-chip, so add their read-back;
        # on CS3 that term is zero (they stay resident from phase1).
        raw_in = BYTES * rows * self.C * G * self.mult_time
        summary_per_est = 3 * self.C * EMB * EMB * BYTES
        summary_read = self._stream_bytes(summary_per_est)     # 0 on CS3
        in_total = raw_in + summary_read
        # OUT: row embeddings.  Free on CS3 (resident); a real HBM write on H100.
        out_bytes = self._stream_bytes(rowemb_per_est)         # 0 on CS3
        # compute: cell embed + 3 fan-outs (over full C) + column aggregator, over all rows
        flops_total = (_embed_flops(rows, self.C)
                       + 3 * _block2_flops(rows, self.C)
                       + _col_agg_flops(rows, self.C)) * self.mult_time
        # per-chunk pieces feed ONLY the double-buffer fill/drain term (raw stream only)
        in_chunk, flops_chunk = raw_in / n_chunks, flops_total / n_chunks

        t_in_chunk, t_cmp_chunk = self._t_mem(in_chunk), self._t_flops(flops_chunk)
        t_in, t_cmp = self._t_mem(in_total), self._t_flops(flops_total)
        t_out = self._t_mem(out_bytes)
        # one-time summary read is a prologue; row-embedding write drains at the end
        wall = self._t_mem(summary_read) + _pipeline_time(n_chunks, t_cmp_chunk, t_in_chunk) + t_out
        onchip_note = "resident" if out_bytes == 0 else f"{fmt_bytes(out_bytes)} -> slow tier"
        return self._record({
            'title': f"Stage 2 - Feature Aggregation, Phase 2, {label.upper()} rows (row-wise chunking)",
            'tag': f"S2 aggregate {label.upper()} (row-chunk)",
            'axis': "rows", 'n_chunks': n_chunks, 'chunk_desc': f"{Rc} rows", 'why': why,
            'in_desc': f"raw cells  {rows:,} x C={self.C} x G={G}  x {self.mult_time} est",
            'in_chunk': in_chunk, 'in_total': in_total, 't_in': t_in,
            'out_desc': f"{rows:,} x {D}-d row embeddings ({onchip_note})",
            'out_bytes': out_bytes, 't_out': t_out,
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
        flops_layer = self.mult_time * _icl_layer_fit_flops(self.N)
        t_cmp_layer = self._t_flops(flops_layer)

        kv_bytes = _kv_cache_bytes(self.N, self.mult_time)   # total across all layers
        kv_layer = kv_bytes / NLAYERS                        # one layer's (K,V) block
        t_out_layer = self._t_mem(kv_layer)

        # IN: the train row-embeddings from phase 2.  Resident (free) on CS3; on
        # H100 they were spilled to HBM, so read them back once as a prologue.
        rowemb_in = self._stream_bytes(BYTES * self.N * D)    # 0 on CS3
        t_in = self._t_mem(rowemb_in)

        flops = flops_layer * NLAYERS
        t_cmp = t_cmp_layer * NLAYERS
        t_out = t_out_layer * NLAYERS
        wall = t_in + _pipeline_time(NLAYERS, t_cmp_layer, t_out_layer)

        why = (f"pipeline over {NLAYERS} layers; per-layer flush "
               f"{fmt_bytes(kv_layer)} overlaps next layer's compute "
               f"({fmt_t(t_cmp_layer)} cmp vs {fmt_t(t_out_layer)} out)")

        in_desc = ("row embeddings already on-chip (from Phase 2)" if rowemb_in == 0
                   else f"train row-embeddings read back ({fmt_bytes(rowemb_in)})")
        return self._record({
            'title': "Stage 3 - ICL Transformer, FIT (build KV cache)",
            'tag': "S3 ICL fit (N^2 self-attn)",
            'axis': "layers", 'n_chunks': NLAYERS, 'chunk_desc': "1 layer", 'why': why,
            'in_desc': in_desc,
            'in_chunk': 0, 'in_total': rowemb_in, 't_in': t_in,
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
        flops_layer = self.mult_time * _icl_layer_predict_flops(self.M, self.N)
        t_cmp_layer = self._t_flops(flops_layer)

        kv_bytes = _kv_cache_bytes(self.N, self.mult_time)   # total across all layers
        kv_layer = kv_bytes / NLAYERS                        # one layer's (K,V) block
        t_in_layer = self._t_mem(kv_layer)

        # test row-embeddings feed the queries; tiny (M = 1% of N) and usually fit
        # on-chip even on H100, so this prologue is typically zero.
        testemb_in = self._stream_bytes(BYTES * self.M * D)

        flops = flops_layer * NLAYERS
        t_cmp = t_cmp_layer * NLAYERS
        t_in = t_in_layer * NLAYERS
        wall = self._t_mem(testemb_in) + _pipeline_time(NLAYERS, t_cmp_layer, t_in_layer)

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
        flops = self.mult_time * _head_flops(self.M, self.N, self.task, self.n_classes)
        t_cmp = self._t_flops(flops)

        if self.task == "multiclass":
            nchunks = (self.n_classes + DEC_HEAD_DIM - 1) // DEC_HEAD_DIM
            # retrieval decoder re-attends over the N train embeddings -- resident
            # on CS3, a fresh read from HBM on H100.
            in_bytes = self._stream_bytes(BYTES * self.N * D)
            in_desc = f"train embeddings (N x {D}); retrieval attn M x N"
            out_bytes = BYTES * self.M * self.n_classes * self.mult_time
            out_desc = (f"logits M x {self.n_classes}  "
                        f"[{nchunks} class-chunks, O(M*N)]")
        else:
            # regression MLP touches only the M test embeddings (tiny)
            in_bytes = self._stream_bytes(BYTES * self.M * D)
            in_desc = f"test embeddings (M x {D})"
            out_bytes = BYTES * self.M * NUM_BUCKETS * self.mult_time
            out_desc = (f"bar-dist logits M x {NUM_BUCKETS}  "
                        f"[MLP {D}->{FF}->{NUM_BUCKETS}, O(M)]")

        # single fused op: compute overlaps the input read (roofline max)
        t_in = self._t_mem(in_bytes)
        wall = max(t_cmp, t_in)
        return self._record({
            'title': f"Stage 4 - Output Head ({self.task})",
            'tag': f"S4 head ({self.task})",
            'axis': None, 'n_chunks': 1, 'chunk_desc': "-",
            'in_desc': in_desc, 'in_chunk': 0, 'in_total': in_bytes, 't_in': t_in,
            'out_desc': out_desc, 'out_bytes': out_bytes, 't_out': 0.0,
            'flops_chunk': flops, 'flops_total': flops, 't_cmp': t_cmp,
            'wall': wall, 'pipelined': False,
            'bound': "memory" if t_in > t_cmp else "compute",
        })

    # -------------------------------------------------------------- driver -----
    def simulate(self):
        """Run all stages, populate self.stages, return them (no console output)."""
        self.stages = []
        self.phase1()
        self.phase2(self.N, "train")
        self.icl_fit()
        self.phase2(self.M, "test")
        self.icl_predict()
        self.output_head()
        return self.stages

    def run(self):
        self.simulate()
        self._render()

    # ------------------------------------------------------------- rendering ---
    def _header(self):
        d = self.dev
        console.rule(f"[bold white]TabPFN-3  {d.name} timeline")
        onchip = "inf" if d.onchip_free >= 1e12 else fmt_bytes(d.onchip_free)
        waves = (f"  |  {self.n_concurrent}/{self.E} estimators concurrent "
                 f"({self.n_waves} wave{'s' if self.n_waves > 1 else ''})"
                 if not d.concurrent_estimators else "")
        console.print(
            f"[bold]N[/]={self.N:,}  [bold]M[/]={self.M:,}  [bold]C[/]={self.C}  "
            f"[bold]E[/]={self.E}  [bold]B[/]={self.B}  [bold]task[/]={self.task}\n"
            f"[dim]{d.name}: slow-tier {d.mem_bw/1e12:.2f} TB/s  |  on-chip {onchip}  |  "
            f"{d.usable_flops/1e15:.3f} PF/s usable{waves}[/]\n")

    def _narrate(self, r):
        col = "red" if r['bound'] == "memory" else "green"
        tier = self.dev.slow_tier                       # "MemX" (CS3) or "HBM" (H100)
        in_lbl, out_lbl = f"IN  {tier}->chip", f"OUT chip->{tier}"
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
                    f"   [green]{in_lbl}[/]: {r['in_desc']}\n"
                    f"        = [bold]{fmt_bytes(r['in_chunk'])}[/]/chunk -> "
                    f"[bold]{fmt_t(r['t_in']/r['n_chunks'])}[/]/chunk   "
                    f"(total {fmt_bytes(r['in_total'])} -> {fmt_t(r['t_in'])})")
            else:
                console.print(f"   [green]{in_lbl}[/]: {r['in_desc']} "
                              f"= [bold]{fmt_bytes(r['in_total'])}[/] -> [bold]{fmt_t(r['t_in'])}[/]")
        else:
            console.print(f"   [green]{in_lbl}[/]: {r['in_desc']}  [dim](no transfer)[/]")
        # OUT
        if r['out_bytes'] > 0:
            console.print(f"   [yellow]{out_lbl}[/]: {r['out_desc']} "
                          f"= [bold]{fmt_bytes(r['out_bytes'])}[/] -> [bold]{fmt_t(r['t_out'])}[/]")
        else:
            console.print(f"   [yellow]{out_lbl}[/]: {r['out_desc']}")
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
# Colours: Cerebras brand palette, assigned in fixed order (identity follows the
# stage, never cycled). Paired stages share a colour family: the two Row Agg
# stages use the teal family, the two ICL stages the purple family; Feature Embed
# and head take the remaining accents. Total keeps the near-black.
STAGE_SERIES = [
    ("Feature Embed",         "#F05A28"),   # orange
    ("Row Agg Train",         "#2AA88C"),   # teal  (teal family, lighter; darkened from mint #81E2C1 for legibility on white)
    ("ICL Train",             "#6639B7"),   # purple
    ("Row Agg Test",          "#00B0CA"),   # cyan  (teal family, darker)
    ("ICL Test",              "#D12DB1"),   # magenta
    ("head (multiclass)",     "#69BE28"),   # green
]
TOTAL_COLOR = "#231F20"                     # near-black


def _fmt_rate(flops):
    """FLOP/s as PF/s or TF/s, whichever reads cleaner."""
    return f"{flops/1e15:.4g} PF/s" if flops >= 1e15 else f"{flops/1e12:.4g} TF/s"


def _draw_hw_panel(ax, dev):
    """Render the device's hardware spec into `ax` (a blank, row-spanning axes).

    Everything is read straight off the Device, so the panel can never drift out
    of sync with the numbers actually driving the model.
    """
    import textwrap
    from matplotlib.patches import Rectangle

    onchip = "unbounded" if dev.onchip_free >= 1e12 else fmt_bytes(dev.onchip_free)
    cap = "streams" if dev.dram_cap == float("inf") else fmt_bytes(dev.dram_cap)
    est = ("all E co-resident" if dev.concurrent_estimators
           else f"wave-packed in {fmt_bytes(dev.dram_cap)}")
    rows = [
        ("Compute", None),
        ("peak FP16",   _fmt_rate(dev.peak_flops)),
        ("utilization", f"{dev.util:.1%}"),
        ("usable",      _fmt_rate(dev.usable_flops)),
        ("Memory", None),
        (dev.slow_tier, f"{dev.mem_bw/1e12:.2f} TB/s"),
        ("on-chip free",     onchip),
        ("capacity",    cap),
        # ("Estimators", None),
        (est, ""),
    ]

    # a boxed panel: title bar, then aligned key/value rows, then the caveat note
    ax.add_patch(Rectangle((0.04, 0.02), 0.92, 0.96, transform=ax.transAxes,
                           fill=True, fc="#F5F5F5", ec="0.7", lw=1.0, zorder=0))
    ax.text(0.5, 0.955, "HARDWARE MODEL", transform=ax.transAxes, ha="center",
            va="top", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.905, dev.label or dev.name, transform=ax.transAxes, ha="center",
            va="top", fontsize=12, color="#F05A28", fontweight="bold")

    y = 0.855
    for key, val in rows:
        if val is None:                          # section header
            y -= 0.012
            ax.text(0.10, y, key, transform=ax.transAxes, ha="left", va="top",
                    fontsize=11, fontweight="bold")
        else:
            ax.text(0.15, y, key, transform=ax.transAxes, ha="left", va="top",
                    fontsize=10.5, family="monospace")
            ax.text(0.90, y, val, transform=ax.transAxes, ha="right", va="top",
                    fontsize=10.5, family="monospace", fontweight="bold")
        y -= 0.052

    if dev.note:
        y -= 0.02
        wrapped = textwrap.fill(dev.note, width=24)
        ax.text(0.10, y, wrapped, transform=ax.transAxes, ha="left", va="top",
                fontsize=9, style="italic", color="0.3")


def _draw_timing_panel(ax, device, C, E, N_grid, n_points, fs):
    """Plot every stage curve + the Total onto `ax` for one (C, E) config.

    Shared by the full grid and the slide layout so the two never drift; `fs`
    is a dict of size knobs (line width, marker size, annotation/tick font) that
    lets the slide draw the same panel at a legible scale.  Returns the peak
    Total wall time in seconds (for the corner annotation / callers).
    """
    import numpy as np

    series = {lbl: [] for lbl, _ in STAGE_SERIES}   # ms per point
    bounds = {lbl: [] for lbl, _ in STAGE_SERIES}   # 'compute' | 'memory'
    totals = []
    for Nf in N_grid:
        N = int(round(Nf))
        M = max(1, int(round(0.01 * N)))            # predict rows = 1% of train
        stages = TabPFNEstimator(
            N=N, M=M, C=C, E=E, B=1, task='multiclass', device=device).simulate()
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
        ax.plot(N_grid, y, color=color, lw=fs['lw'], label=lbl, zorder=2)
        # marker encodes the bottleneck: plain square = compute, black X = memory
        cmp_m = (b == "compute") & show
        mem_m = (b == "memory") & show
        ax.scatter(N_grid[cmp_m], y[cmp_m], marker="s", s=fs['ms'], color=color,
                   edgecolors="white", linewidths=0.6, zorder=3)
        ax.scatter(N_grid[mem_m], y[mem_m], marker="x", s=fs['ms'], color="black",
                   linewidths=1.4, zorder=5)
    ax.plot(N_grid, totals, color=TOTAL_COLOR, lw=fs['lw_total'], label="Total", zorder=4)

    # report the peak Total time (in seconds) in the top-left corner -- the one
    # region every panel leaves empty (all lines start low at N=100), so it
    # never brushes the curve or the frame.
    peak_s = np.array(totals).max() / 1e3
    ax.text(0.04, 0.95, f"max: {peak_s:.2f} s",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=fs['annot'], fontweight="bold", color=TOTAL_COLOR, zorder=5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7",
                      lw=0.6, alpha=0.9))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1e6)   # ms; fixed across all panels for comparability
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.4)
    ax.tick_params(labelsize=fs['tick'])
    return peak_s


# font/size knobs for the dense 3x4 grid
_FS_GRID  = dict(lw=1.6, lw_total=2.6, ms=26, annot=11, tick=9)
# larger knobs for the 1x3 slide layout (readable when dropped on a slide)
_FS_SLIDE = dict(lw=2.4, lw_total=3.6, ms=52, annot=16, tick=13)


def plot_stage_timings_slide(device=CS3, out_path=None, n_points=30,
                             features=(10, 100, 500), estimators=(1,),
                             hw_panel=True):
    """Slide-ready row of per-stage wall-time-vs-N panels; one panel per column.

    Exactly one of `features` / `estimators` is the swept axis (the longer list);
    the other is held fixed at its single value.  Sweep C -> panels titled by
    feature count at fixed E; sweep E -> panels titled by estimator count at
    fixed C.  When `hw_panel` is True a hardware-spec column is appended.
    `out_path` defaults to plots/<device>_slide_vary{C,E}.png.
    """
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")                       # headless: just write the file
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    vary_E = len(estimators) > 1                 # which axis is swept across columns
    cols = list(estimators) if vary_E else list(features)
    fixed_C = features[0]
    fixed_E = estimators[0]

    if out_path is None:
        tag = "varyE" if vary_E else "varyC"
        out_path = os.path.join("plots", f"{device.name}_slide_{tag}.png")
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    N_grid = np.logspace(2, 6, n_points)        # 100 .. 1,000,000 training rows
    ncol = len(cols)
    # append a narrow spec column only when requested
    width_ratios = [1] * ncol + ([0.5] if hw_panel else [])
    fig = plt.figure(figsize=(6.4 * ncol + (2.6 if hw_panel else 0), 6.2))
    gs = fig.add_gridspec(1, ncol + (1 if hw_panel else 0), width_ratios=width_ratios)
    axes, base = [], None
    for j in range(ncol):
        ax = fig.add_subplot(gs[0, j], sharex=base, sharey=base)
        base = base or ax
        axes.append(ax)

    for j, v in enumerate(cols):
        ax = axes[j]
        C, E = (fixed_C, v) if vary_E else (v, fixed_E)
        _draw_timing_panel(ax, device, C, E, N_grid, n_points, _FS_SLIDE)
        title = (f"E = {E} estimator{'s' if E > 1 else ''}" if vary_E
                 else f"C = {C} features")
        ax.set_title(title, fontsize=17, fontweight="bold")
        ax.set_xlabel("training rows  N", fontsize=14)
        if j == 0:
            ax.set_ylabel("time (ms)", fontsize=14)

    if hw_panel:
        hw_ax = fig.add_subplot(gs[0, ncol])
        hw_ax.axis("off")
        _draw_hw_panel(hw_ax, device)

    handles, labels = axes[0].get_legend_handles_labels()
    handles += [
        Line2D([], [], marker="s", ls="none", color="0.35", markeredgecolor="white",
               markersize=10, label="compute-bound"),
        Line2D([], [], marker="x", ls="none", color="black",
               markeredgewidth=1.6, markersize=10, label="memory-bound"),
    ]
    labels += ["compute-bound", "memory-bound"]
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, fontsize=13, bbox_to_anchor=(0.5, 1.0))
    fixed_desc = f"C = {fixed_C}" if vary_E else f"E = {fixed_E}"
    fig.suptitle(f"TabPFN-3 per-stage wall time on {device.name}  "
                 f"({fixed_desc}, M = 0.01·N, log–log)",
                 fontsize=18, fontweight="bold", y=1.10)
    # the fact the collapsed axis folds away, stated once (matches the swept axis)
    caption = ("Total scales linearly with estimators E (×E) — the shape is fixed; "
               "only the vertical offset moves." if vary_E else
               "ICL's N² dominates at large N, so feature count C barely shifts the "
               "Total; it mainly lifts the Feature Embed / Row Agg stages.")
    fig.text(0.02, -0.02, caption, ha="left", va="top",
             fontsize=12, style="italic", color="0.3")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.90])

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_stage_timings(device=CS3, out_path=None, n_points=30):
    """3x4 small-multiples grid of per-stage wall time vs training rows.

    Rows    -> n_features C in {10, 100, 500}
    Columns -> n_estimators E in {1, 2, 4, 8}
    Each panel: one log-log line per stage (Feature Embed, Row Agg train/test,
    ICL train/test, head) plus the Total. Test rows are pinned to M = 0.01 * N_train.
    `device` selects the hardware model; `out_path` defaults to
    plots/<device>_stage_timings.png.
    """
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")                       # headless: just write the file
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if out_path is None:
        out_path = os.path.join("plots", f"{device.name}_stage_timings.png")
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    feature_rows   = [10, 100, 500]             # C, one per grid row
    estimator_cols = [1, 2, 4, 8]               # E, one per grid column
    N_grid = np.logspace(2, 6, n_points)        # 100 .. 1,000,000 training rows

    # 3x4 panel grid + a narrow 5th column (spanning all rows) for the hardware
    # spec of the device being plotted.
    nrow, ncol = len(feature_rows), len(estimator_cols)
    fig = plt.figure(figsize=(23, 12))
    gs = fig.add_gridspec(nrow, ncol + 1, width_ratios=[1] * ncol + [0.62])
    axes, base = [], None
    for i in range(nrow):
        row = []
        for j in range(ncol):
            ax = fig.add_subplot(gs[i, j], sharex=base, sharey=base)
            base = base or ax
            row.append(ax)
        axes.append(row)
    hw_ax = fig.add_subplot(gs[:, ncol])        # spans all 3 rows
    hw_ax.axis("off")

    for i, C in enumerate(feature_rows):
        for j, E in enumerate(estimator_cols):
            ax = axes[i][j]
            _draw_timing_panel(ax, device, C, E, N_grid, n_points, _FS_GRID)
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

    # ---- hardware-spec panel (right column, spans all rows) ----------------
    _draw_hw_panel(hw_ax, device)

    handles, labels = axes[0][0].get_legend_handles_labels()
    # append a shape key: square = compute-bound point, circle = memory-bound point
    handles += [
        Line2D([], [], marker="s", ls="none", color="0.35", markeredgecolor="white",
               markersize=8, label="compute-bound"),
        Line2D([], [], marker="x", ls="none", color="black",
               markeredgewidth=1.6, markersize=8, label="memory-bound"),
    ]
    labels += ["compute-bound", "memory-bound"]
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(f"TabPFN-3 per-stage wall time on {device.name}  (M = 0.01·N, log–log)",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0.02, 0, 1, 0.95])

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# =============================================================================
# Serving demo: what the user experience feels like under caching.
#
# In TabPFN "fit" = building the KV cache from the training rows.  The first
# three stages (Feature Embed, Row Agg Train, ICL Train) depend ONLY on the
# training data, so they are cacheable and reusable; the last three (Row Agg
# Test, ICL Test, head) are per-query and always run.  So every serving policy
# collapses to ONE number -- the cache hit-rate h, the fraction of requests
# that reuse an already-built fit and skip the train prefix:
#
#     cold  = all 6 stages          (new dataset: pay the O(N^2) ICL fit)
#     warm  = last 3 stages         (cached fit: straight to the query)
#     L(h)  = h*warm + (1-h)*cold   (blended per-request latency)
#
# The four scenarios are illustrative points on that single h axis.
_N_PREFIX = 3   # count of cacheable train-only stages (matches simulate() order)

SERVING_SCENARIOS = [
    ("Fresh every time",       0.00),   # streaming / unique context -- no reuse
    ("Pin on demand",          0.30),   # user opts in per dataset
    ("Warm the popular ones",  0.60),   # cache large / frequent datasets only
    ("Fit-once, query-many",   0.95),   # analyst explores one dataset
]

# one representative serving request (a "medium" dataset); tune via the CLI
_SERVE_POINT = dict(N=100_000, C=100, E=8)
_DEV_COLOR   = {"CS3": "#F05A28", "H100": "#76B900"}   # Cerebras orange / NVIDIA green


def _serving_split(device, N, C, E, task='multiclass'):
    """Per-stage walls (s), plus cold (all 6) and warm (last 3) latency."""
    M = max(1, N // 100)                               # predict rows = 1% of train
    stages = TabPFNEstimator(N=N, M=M, C=C, E=E, B=1, task=task,
                             device=device).simulate()
    walls = [r['wall'] for r in stages]                # seconds, simulate() order
    return walls, sum(walls), sum(walls[_N_PREFIX:])


def _mute(color, sat=0.68, val=0.9):
    """Desaturate + slightly darken a hex colour so the bars read softer."""
    import colorsys
    from matplotlib.colors import to_rgb, to_hex
    h, s, v = colorsys.rgb_to_hsv(*to_rgb(color))
    return to_hex(colorsys.hsv_to_rgb(h, s * sat, v * val))


def plot_serving_breakdown(devices=(CS3, H100), out_path=None, **pt):
    """Fig 1 -- the felt experience: one request, cold vs warm stage breakdown.

    Devices stack vertically (2x1) on a SHARED linear x-axis (ms), so bar
    lengths are directly comparable across CS3 and H100 (H100 cold dwarfs CS3).
    Stacked segment widths stay honestly proportional -- ICL Train visibly
    dominates cold, and warm removes it.  Two horizontal stacked bars per device
    (cold = all 6 stages, warm = the 3 per-query stages); the title reports the
    speedup.  Colours are the (muted) stage palette; totals annotated in ms.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    p = {**_SERVE_POINT, **pt}
    N, C, E = p['N'], p['C'], p['E']
    if out_path is None:
        out_path = os.path.join("plots", "serving_cold_vs_warm.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    colors = [_mute(c) for _, c in STAGE_SERIES]
    rows = [("cold\n(new dataset)", range(6)),
            ("warm\n(cached fit)",  range(_N_PREFIX, 6))]

    fig, axes = plt.subplots(len(devices), 1, figsize=(9, 2.9 * len(devices)),
                             sharex=True, squeeze=False)
    for ax, dev in zip(axes[:, 0], devices):
        walls, cold, warm = _serving_split(dev, N, C, E)
        walls_ms = [w * 1e3 for w in walls]
        for row, (_name, keep) in enumerate(rows):
            left = 0.0
            for k in keep:
                ax.barh(row, walls_ms[k], left=left, color=colors[k],
                        edgecolor="white", linewidth=0.8, height=0.6, zorder=2)
                left += walls_ms[k]
            total = cold * 1e3 if row == 0 else warm * 1e3
            ax.text(left, row, f"  {total:,.0f} ms", va="center", ha="left",
                    fontsize=11, fontweight="bold", color="0.15")
        speedup = cold / warm if warm else float('inf')
        ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows])
        ax.invert_yaxis()
        ax.set_title(f"{dev.name}  —  warm is {speedup:.0f}× faster", fontweight="bold")
        ax.grid(True, axis="x", ls=":", lw=0.4, alpha=0.4)

    axes[-1, 0].margins(x=0.16)                        # headroom for the labels
    axes[-1, 0].set_xlabel("per-request latency (ms)", fontsize=11)

    handles = [Patch(fc=c, ec="white") for c in colors]
    fig.legend(handles, [lbl for lbl, _ in STAGE_SERIES], loc="upper center",
               ncol=6, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"One request, cold vs warm   (N={N:,}, C={C}, E={E}, M=1%·N)",
                 fontsize=14, fontweight="bold", y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_serving_hitrate(devices=(CS3, H100), out_path=None, **pt):
    """Fig 2 -- the economics: blended latency & throughput vs hit-rate h.

    One line per device (log latency, left axis); the right axis reads the same
    curve as throughput (1000/ms = requests/s).  The four scenarios are marked
    as dots and labelled along the top.
    """
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = {**_SERVE_POINT, **pt}
    N, C, E = p['N'], p['C'], p['E']
    if out_path is None:
        out_path = os.path.join("plots", "serving_hitrate.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    h = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for dev in devices:
        _, cold, warm = _serving_split(dev, N, C, E)
        L = (h * warm + (1 - h) * cold) * 1e3          # ms
        color = _DEV_COLOR.get(dev.name, "#333")
        ax.plot(h, L, color=color, lw=2.6, label=dev.name, zorder=3)
        hv = np.array([s[1] for s in SERVING_SCENARIOS])
        Lv = (hv * warm + (1 - hv) * cold) * 1e3
        ax.scatter(hv, Lv, color=color, s=45, edgecolors="white",
                   linewidths=0.8, zorder=4)

    for _name, hval in SERVING_SCENARIOS:
        ax.axvline(hval, ls=":", lw=0.7, color="0.6", zorder=1)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("cache hit-rate  h  (fraction of requests reusing a cached fit)",
                  fontsize=12)
    ax.set_ylabel("blended per-request latency (ms)", fontsize=12)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.4)

    # top axis: name each scenario at its h; right axis: same curve as throughput
    top = ax.secondary_xaxis("top")
    top.set_xticks([s[1] for s in SERVING_SCENARIOS])
    top.set_xticklabels([s[0] for s in SERVING_SCENARIOS], rotation=25,
                        ha="left", fontsize=9)
    # reciprocal transform; guard the 0 that matplotlib probes at the axis edge
    _recip = lambda v: 1e3 / np.where(np.asarray(v, float) == 0, np.nan, v)
    rt = ax.secondary_yaxis("right", functions=(_recip, _recip))
    rt.set_ylabel("throughput (requests / s)", fontsize=12)

    ax.legend(title="device", fontsize=11, loc="upper right")
    fig.suptitle(f"Latency & throughput vs cache hit-rate   (N={N:,}, C={C}, E={E})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=list(DEVICES), default="CS3",
                        help="accelerator to model (default: CS3)")
    parser.add_argument("--output-file-path", default=None,
                        help="plot destination (default: plots/<device>_stage_timings[_slide].png)")
    parser.add_argument("--slide", action="store_true",
                        help="emit the slide-ready 1x3 layout (E=8) instead of the full 3x4 grid")
    parser.add_argument("--serving", action="store_true",
                        help="emit the two serving-demo figures (cold-vs-warm + hit-rate) "
                             "comparing CS3 and H100, instead of the per-stage sweep")
    parser.add_argument("--serve-N", type=int, default=_SERVE_POINT['N'],
                        help=f"serving-demo train rows N (default: {_SERVE_POINT['N']:,})")
    parser.add_argument("--serve-C", type=int, default=_SERVE_POINT['C'],
                        help=f"serving-demo feature count C (default: {_SERVE_POINT['C']})")
    parser.add_argument("--serve-E", type=int, default=_SERVE_POINT['E'],
                        help=f"serving-demo estimators E (default: {_SERVE_POINT['E']})")
    args = parser.parse_args()
    dev = DEVICES[args.device]

    if args.serving:
        pt = dict(N=args.serve_N, C=args.serve_C, E=args.serve_E)
        # both devices on one figure so the client sees the CS3 vs H100 contrast
        for d in (CS3, H100):
            _, cold, warm = _serving_split(d, **pt)
            print(f"{d.name:5s}  cold {fmt_t(cold):>10s}  warm {fmt_t(warm):>10s}  "
                  f"({cold / warm:.0f}× faster warm)" if warm else f"{d.name}: warm=0")
        print()
        plot_serving_breakdown(**pt)
        plot_serving_hitrate(**pt)
        import sys; sys.exit(0)

    # one worked example to the console, then the sweep to the plot
    TabPFNEstimator(N=1_000_000, M=10_000, C=100, E=1, B=1,
                    task='multiclass', device=dev).run()
    print()
    if args.slide:
        # two slide figures: (1) vary C at E=1 with the hardware panel;
        # (2) vary E at C=200 without it.  --output-file-path is ignored here
        # since two files are written to their default names.
        plot_stage_timings_slide(device=dev, features=(10, 100, 500),
                                 estimators=(1,), hw_panel=True)
        plot_stage_timings_slide(device=dev, features=(200,),
                                 estimators=(1, 2, 4, 8), hw_panel=False)
    else:
        plot_stage_timings(device=dev, out_path=args.output_file_path)
