"""FLOP / memory calculator for TabPFN-3 (classification), derived from
tabpfn/architectures/tabpfn_v3.py. Convention: 1 multiply-add = 2 FLOPs.
Norms + softmax + softmax-scaling MLP are omitted (each <~2% of a block).
Memory assumes a memory-efficient (flash/SDPA) attention backend, so N x N
score matrices are never materialised."""

# ---- architecture constants (from the loaded checkpoint config) ----
d   = 128      # embed_dim (stages 0-2)
D   = 512      # icl_emsize = 128 * 4 CLS
K   = 128      # inducing points
Cl  = 4        # CLS tokens
Gin = 6        # cell-embed input (3 feats + 3 nan indicators)
dist_blocks = 3
colagg_blocks = 3
icl_blocks = 24
dec_heads, dec_hd, Tdec = 6, 64, 160   # many-class decoder
FP16 = 2       # bytes

def lin(tok, din, dout):      return 2*tok*din*dout          # matmul MAC*2
def attn(sq, sk, dmodel):     return 4*sq*sk*dmodel          # QK^T + A V
def mlp(tok, dm):             return lin(tok, dm, 2*dm) + lin(tok, 2*dm, dm)

# ---------- Stage 0: cell embedding ----------
def stage0(rows, C):          return lin(rows*C, Gin, d)     # Linear(6->128)

# ---------- Stage 1: FeatureDistributionEmbedder (per column) ----------
# block1 = inducing(K) <- rows(keys) ; block2 = rows(query) <- inducing(K)
def isab_block1(nkeys, C):    # K queries attend to nkeys
    return C*( lin(K,d,d)+lin(nkeys,d,d)+lin(nkeys,d,d)+lin(K,d,d)  # q,k,v,out
               + attn(K, nkeys, d) + mlp(K, d) )
def isab_block2(nq, C):       # nq rows attend to K inducing
    return C*( lin(nq,d,d)+lin(K,d,d)+lin(K,d,d)+lin(nq,d,d)
               + attn(nq, K, d) + mlp(nq, d) )

def stage1_fit(N, C):
    # phase1 (compute inducing over N): block1 x3, block2 x2 (blocks 0,1) on N
    f  = 3*isab_block1(N, C) + 2*isab_block2(N, C)
    # phase2 row-chunk loop: block2 x3 on N (block1 cached)
    f += 3*isab_block2(N, C)
    return f
def stage1_predict_cached(M, C):     # test-only, block1 cached -> block2 x3 on M
    return 3*isab_block2(M, C)
def stage1_full(R, N, C):            # low_memory: full R rows, no cache
    f  = 3*isab_block1(N, C) + 2*isab_block2(N, C)   # phase1 over N train
    f += 3*isab_block2(R, C)                          # phase2 over all R
    return f

# ---------- Stage 2: ColumnAggregator (per row, seq = C + CLS) ----------
def stage2(rows, C):
    S = C + Cl
    per_row = 0.0
    # blocks 0,1: full self-attn over S
    for _ in range(colagg_blocks-1):
        per_row += lin(S,d,d)*4 + attn(S,S,d) + mlp(S,d)
    # last block: CLS(4) query attends to full S
    per_row += lin(Cl,d,d)*2 + lin(S,d,d)*2 + attn(Cl,S,d) + mlp(Cl,d)
    return rows*per_row

# ---------- Stage 3: ICL transformer (24 layers, D=512) ----------
def icl_fit(N):        # train<->train self attn, full 8 heads
    per = lin(N,D,D)*4 + attn(N,N,D) + mlp(N,D)
    return icl_blocks*per
def icl_predict_cached(M, N):   # test->train, K/V from cache (no k/v proj)
    per = lin(M,D,D)*2 + attn(M,N,D) + mlp(M,D)     # only q_proj + out_proj
    return icl_blocks*per
def icl_full(R, N):    # low_memory: all R query, N train keys
    per = lin(R,D,D)*2 + lin(N,D,D)*2 + attn(R,N,D) + mlp(R,D)
    return icl_blocks*per

# ---------- Decoder (many-class) ----------
def decoder(M, N):
    nchunks = -(-Tdec//dec_hd)          # ceil(160/64)=3
    f  = lin(M,D,dec_heads*dec_hd) + lin(N,D,dec_heads*dec_hd)   # q,k proj
    f += dec_heads*nchunks*attn(M, N, dec_hd)                    # QK^T+AV per head,chunk
    return f

# ---------- Regression head + y-encoders ----------
Nbuckets = 5000
def yenc_fit(N):   return lin(N,1,d) + lin(N,1,D)        # col Linear(1->128) + icl Linear(1->512), train only
def reg_head(M):   return lin(M,D,2*D) + lin(M,2*D,Nbuckets)  # 512->1024->5000, per test row, no N-dependence

# ============================================================
N, M, C = 100_000, 1_000, 100      # train, test, model columns
E, P    = 8, 10                    # estimators, predict runs
def pf(x): return f"{x/1e12:8.3f} TFLOP"
def gb(x): return f"{x/1024**3:6.3f} GB"
def mb(x): return f"{x/1024**2:7.1f} MB"

print("="*70); print("PER-ESTIMATOR, PER-CALL FLOPs (C=%d, N=%d, M=%d)"%(C,N,M)); print("="*70)
fit_s012  = stage0(N,C)+stage1_fit(N,C)+stage2(N,C)
fit_icl   = icl_fit(N)
print("FIT (build cache): stage0-2 =",pf(fit_s012),"| ICL =",pf(fit_icl),
      "| total =",pf(fit_s012+fit_icl))
pr_s012   = stage0(M,C)+stage1_predict_cached(M,C)+stage2(M,C)
pr_icl    = icl_predict_cached(M,N)
pr_dec    = decoder(M,N)
print("PREDICT cached  : stage0-2 =",pf(pr_s012),"| ICL =",pf(pr_icl),
      "| dec =",pf(pr_dec),"| total =",pf(pr_s012+pr_icl+pr_dec))
R=N+M
lm_s012   = stage0(R,C)+stage1_full(R,N,C)+stage2(R,C)
lm_icl    = icl_full(R,N)
lm_dec    = decoder(M,N)
print("LOWMEM full pass: stage0-2 =",pf(lm_s012),"| ICL =",pf(lm_icl),
      "| dec =",pf(lm_dec),"| total =",pf(lm_s012+lm_icl+lm_dec))

print("\n"+"="*70); print("GRAND TOTAL FLOPs  (E=%d estimators, P=%d predict runs)"%(E,P)); print("="*70)
fc_fit   = E*(fit_s012+fit_icl)
fc_pred  = E*P*(pr_s012+pr_icl+pr_dec)
lm_fit   = 0.0
lm_pred  = E*P*(lm_s012+lm_icl+lm_dec)
print("fit_cache : FIT =",pf(fc_fit)," + PREDICT(%dx) ="%P,pf(fc_pred),
      " => TOTAL =",pf(fc_fit+fc_pred))
print("low_memory: FIT =",pf(lm_fit)," + PREDICT(%dx) ="%P,pf(lm_pred),
      " => TOTAL =",pf(lm_pred))
print("low_memory / fit_cache compute ratio = %.1fx"%((lm_pred)/(fc_fit+fc_pred)))

print("\n"+"="*70); print("MEMORY"); print("="*70)
weights = 53_153_144*FP16
print("model weights (fp16, one shared copy):", mb(weights))
# KV cache per estimator (fit_cache only), fp16 no-quant:
kv   = icl_blocks*2*N*1*dec_hd*FP16          # 1 test KV head, head_dim 64
temb = N*D*FP16
ind  = dist_blocks*C*K*d*FP16
cache_est = kv+temb+ind
print("fit_cache KV-cache / estimator:", mb(cache_est),
      " (KV",mb(kv)," train_emb",mb(temb)," inducing",mb(ind),")")
print("fit_cache KV-cache TOTAL (x%d):"%E, gb(E*cache_est))
# rough activation peaks
print("fit_cache FIT   peak act/est ~", mb(N*D*FP16*4), "(ICL seq + qkv, flash)")
print("fit_cache PRED  peak act/est ~", mb(N*dec_heads*Tdec*FP16), "(decoder one-hot values)")
print("lowmem   PRED  peak act/est ~", mb((N*D*FP16) + 2*(N*D*FP16)), "(ICL seq + train K/V, flash)")

# =====================================================================
print("\n\n"+"#"*70); print("# REGRESSION"); print("#"*70)
rfit_s012 = stage0(N,C)+stage1_fit(N,C)+stage2(N,C)+yenc_fit(N)
rfit_icl  = icl_fit(N)
print("FIT (build cache): stage0-2+yenc =",pf(rfit_s012),"| ICL =",pf(rfit_icl),
      "| total =",pf(rfit_s012+rfit_icl))
rpr_s012  = stage0(M,C)+stage1_predict_cached(M,C)+stage2(M,C)
rpr_icl   = icl_predict_cached(M,N)
rpr_head  = reg_head(M)
print("PREDICT cached  : stage0-2 =",pf(rpr_s012),"| ICL =",pf(rpr_icl),
      "| head =",pf(rpr_head),"| total =",pf(rpr_s012+rpr_icl+rpr_head))
rlm_s012  = stage0(R,C)+stage1_full(R,N,C)+stage2(R,C)+yenc_fit(N)
rlm_icl   = icl_full(R,N)
rlm_head  = reg_head(M)
print("LOWMEM full pass: stage0-2+yenc =",pf(rlm_s012),"| ICL =",pf(rlm_icl),
      "| head =",pf(rlm_head),"| total =",pf(rlm_s012+rlm_icl+rlm_head))

print("\nGRAND TOTAL FLOPs (regression, E=%d, P=%d)"%(E,P))
rfc_fit  = E*(rfit_s012+rfit_icl)
rfc_pred = E*P*(rpr_s012+rpr_icl+rpr_head)
rlm_pred = E*P*(rlm_s012+rlm_icl+rlm_head)
print("fit_cache : FIT =",pf(rfc_fit)," + PREDICT(%dx) ="%P,pf(rfc_pred)," => TOTAL =",pf(rfc_fit+rfc_pred))
print("low_memory: FIT =",pf(0.0)," + PREDICT(%dx) ="%P,pf(rlm_pred)," => TOTAL =",pf(rlm_pred))

print("\nMEMORY (regression)")
rweights = 58_274_944*FP16
print("model weights (fp16):", mb(rweights))
print("KV-cache / estimator (same as clf):", mb(cache_est)," TOTAL x%d:"%E, gb(E*cache_est))
print("PRED peak act/est ~", mb(M*Nbuckets*FP16 + M*C*d*FP16), "(bucket logits + stage0-2 test; no one-hot)")


# =====================================================================
# (a) real per-estimator column mix, (b) N-sweep, (c) roofline
# =====================================================================
def fit_e(N, C):        return stage0(N,C)+stage1_fit(N,C)+stage2(N,C)+icl_fit(N)          # clf/reg identical at fit
def pred_e_clf(N,M,C):  return stage0(M,C)+stage1_predict_cached(M,C)+stage2(M,C)+icl_predict_cached(M,N)+decoder(M,N)
def lm_e_clf(N,M,C):    R=N+M; return stage0(R,C)+stage1_full(R,N,C)+stage2(R,C)+icl_full(R,N)+decoder(M,N)
def pred_e_reg(N,M,C):  return stage0(M,C)+stage1_predict_cached(M,C)+stage2(M,C)+icl_predict_cached(M,N)+reg_head(M)
def lm_e_reg(N,M,C):    R=N+M; return stage0(R,C)+stage1_full(R,N,C)+stage2(R,C)+icl_full(R,N)+reg_head(M)
def cache_e(N,C):       return icl_blocks*2*N*1*dec_hd*FP16 + N*D*FP16 + dist_blocks*C*K*d*FP16

MIX = [126,126,126,126,101,101,101,101]          # 4 SVD-quarter estimators + 4 plain
FLAT = [100]*8

print("\n\n"+"#"*70); print("# (a) REAL COLUMN MIX vs FLAT C=100  (clf, N=100k, M=1k, P=10)"); print("#"*70)
for label, cols in [("flat C=100", FLAT), ("mix 4x126/4x101", MIX)]:
    fit  = sum(fit_e(N,c) for c in cols)
    pred = P*sum(pred_e_clf(N,M,c) for c in cols)
    cache= sum(cache_e(N,c) for c in cols)
    print(f"{label:18s}: fit_cache total {pf(fit+pred)} (fit {pf(fit)} + pred {pf(pred)}) | cache {gb(cache)}")

print("\n"+"#"*70); print("# (b) N-SWEEP (clf, M=1000, C=100, E=8, P=10)  crossover P* = FIT/(full-pred)"); print("#"*70)
print(f"{'N':>9} | {'fit_cache TOT':>14} | {'low_mem TOT':>13} | {'ratio':>6} | {'KV cache x8':>11} | {'P* breakeven':>12}")
for Nx in [1_000, 10_000, 100_000, 1_000_000]:
    fit  = E*fit_e(Nx,100)
    predc= E*P*pred_e_clf(Nx,M,100)
    lm   = E*P*lm_e_clf(Nx,M,100)
    cache= E*cache_e(Nx,100)
    Pstar= fit_e(Nx,100)/(lm_e_clf(Nx,M,100)-pred_e_clf(Nx,M,100))
    print(f"{Nx:>9} | {pf(fit+predc):>14} | {pf(lm):>13} | {lm/(fit+predc):>5.1f}x | {gb(cache):>11} | {Pstar:>10.2f}")

print("\n"+"#"*70); print("# (c) ROOFLINE: arithmetic intensity (FLOP/byte). H100 fp16 ridge ~= 300"); print("#"*70)
RIDGE = 300
# fit ICL one layer: flash attn, bytes = Q,K,V read + O write = 4*N*D*2
for Nx in [10_000, 100_000, 1_000_000]:
    fl = attn(Nx,Nx,D); by = 4*Nx*D*FP16
    print(f"FIT ICL attn  N={Nx:>9}: AI={fl/by:>10.0f} FLOP/B  -> {'COMPUTE' if fl/by>RIDGE else 'MEMORY'}-bound")
# cached predict, whole model, per est: bytes = KV cache read + weights + train_emb (clf)
for Mx in [1, 8, 40, 1000]:
    fl = pred_e_clf(N,Mx,100)
    by = cache_e(N,100) + 53_153_144*FP16     # read KV cache + weights once
    print(f"PRED cached   M={Mx:>9}: AI={fl/by:>10.1f} FLOP/B  -> {'COMPUTE' if fl/by>RIDGE else 'MEMORY'}-bound  (per-est, N=100k)")
# crossover M for cached-predict attention only: AI = 8*M ; =RIDGE -> M
print(f"cached-predict attention becomes compute-bound at M > ~{RIDGE/8:.0f} test rows/call")
