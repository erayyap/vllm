"""attn512: paged, causal, GQA attention for head_dim = 512 (Gemma 4 global layers).

Target: H100 (SM90).  Developed against Triton 3.8.0: the gluon kernel uses Triton's
experimental Gluon API (pinned to 3.8); the triton kernel should work on Triton >= 3.4.

Semantics (same as vLLM's flash_attn_varlen_func with a block_table):
  * q            [num_tokens, Hq, D]  bf16/fp16 (token-major, varlen via cu_seqlens_q)
  * k/v_cache    [num_pages, page_size, Hkv, D]  bf16/fp16 or float8_e4m3fn, any strides
                 with stride(-1) == 1 (vLLM passes kv_cache.transpose(1, 2).split(D, -1),
                 i.e. K and V interleaved per (token, head) for the LBNHC layout).
  * block_table  [B, max_pages] int32
  * cu_seqlens_q [B + 1] int32, seqused_k [B] int32 (seqused_k INCLUDES the new tokens;
                 the q tokens of sequence b are its LAST q_len positions).
  * causal (bottom-right aligned), no softcap / window / sinks.
  * k_descale / v_descale: None, python float, 0-d/1-elem tensor, or [B, Hkv] tensor of any
    strides (vLLM passes layer._k_scale.expand(B, Hkv), i.e. strides (0, 0)).  They are
    folded into the score scale / output scale exactly like FA4's SM90 fp8-KV path.

Two implementations of the forward kernel share the split-KV combine kernel below:

  impl="gluon"  (kernel_gluon.py, the fast path; explicit layouts, SMEM ring and async wgmma -
                see that file's docstring).  128-key tiles, S^T = K Q^T with each warpgroup
                owning 64 keys (FlashMLA's per-warpgroup KV blocks), K as register operand with
                in-register FP8 dequant, O split by columns over the 2 warpgroups, a 1-barrier-
                per-chunk cp.async ring for K/V, FP8 V dequantised into a 3-deep fp16 buffer.
  impl="triton" (this file, portable fallback; Triton's automatic pipeliner):
    * GQA packing: one CTA owns BLOCK_M = 64 rows = (64 / G) tokens x the G query heads of
      one KV head, so every K/V byte fetched feeds 64 rows; the M-blocks of a
      (seq, kv_head) are adjacent in the grid -> they stream the same pages through L2.
    * S^T = sum_c K_c Q_c^T over D chunks of DC columns (Q^T chunks resident in SMEM),
      O_c += P V_c with the 64x512 fp32 O split across the 2 warpgroups.
    * FP8 KV: e4m3 -> fp16 (as FA4 on SM90); Q bf16 -> fp16; P fp16; fp32 accumulation.
      Optional QK_FP8 mode quantises Q per row to e4m3 and runs Q.K^T as an FP8 wgmma
      directly on the FP8 K tile - approximate, opt-in.
  Both: split-KV over each sequence's causal tile range + combine (split count from a
  wave-quantisation model), masking only on the diagonal tiles, page-table rows prefetched
  one tile ahead.  impl="auto" (default) runs a one-off self-test of the Gluon kernel against
  a PyTorch reference per (kv dtype, page size, device) and falls back to Triton on failure.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import asdict, dataclass, replace

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634
_FP8_TYPES = tuple(
    t for t in (getattr(torch, "float8_e4m3fn", None),) if t is not None
)


# --------------------------------------------------------------------------------------
# Triton kernels
# --------------------------------------------------------------------------------------
@triton.jit
def _fetch_pages(bt_row, tile, last_pg, PAGE_SIZE: tl.constexpr, BLOCK_N: tl.constexpr):
    """Block-table entries covering key tile `tile` (clamped to the sequence's last page).

    PAGE_SIZE >= BLOCK_N -> scalar page id; else a [BLOCK_N // PAGE_SIZE] vector.
    """
    if PAGE_SIZE >= BLOCK_N:
        pg = tl.minimum((tile * BLOCK_N) // PAGE_SIZE, last_pg)
        pages = tl.load(bt_row + pg)
    else:
        PPT: tl.constexpr = BLOCK_N // PAGE_SIZE
        pg = tl.minimum(tile * PPT + tl.arange(0, PPT), last_pg)
        pages = tl.load(bt_row + pg)
    return pages


@triton.jit
def _row_offsets(pages, n0, stride_p, stride_t, PAGE_SIZE: tl.constexpr, BLOCK_N: tl.constexpr):
    """int64 element offsets of cache rows [n0, n0 + BLOCK_N) for head 0."""
    if PAGE_SIZE >= BLOCK_N:
        in_page = (n0 % PAGE_SIZE) + tl.arange(0, BLOCK_N)
        rows = pages.to(tl.int64) * stride_p + in_page.to(tl.int64) * stride_t
    else:
        PPT: tl.constexpr = BLOCK_N // PAGE_SIZE
        r2 = pages.to(tl.int64)[:, None] * stride_p \
            + (tl.arange(0, PAGE_SIZE).to(tl.int64) * stride_t)[None, :]
        rows = tl.reshape(r2, [BLOCK_N])
    return rows


@triton.jit
def _attn_inner(accs, m_i, l_i, qTs, q_rscale, qk_scale,
                k_head, v_head, bt_row, tile_lo, tile_hi, kv_len, q_pos,
                stride_kp, stride_kt, stride_vp, stride_vt,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DC: tl.constexpr,
                NUM_DC: tl.constexpr, PAGE_SIZE: tl.constexpr,
                KV_FP8: tl.constexpr, QK_FP8: tl.constexpr, MASK: tl.constexpr,
                PV_DTYPE: tl.constexpr):
    # S is computed transposed (S^T = K Q^T, [BLOCK_N keys, BLOCK_M rows]) so that Triton's
    # chained-dot heuristic (warpsPerCTA = [num_warps, 1]) splits the KEYS across the two
    # warpgroups (64 keys each) instead of replicating a 64-row Q.K^T in both of them, and
    # so that K is the register (A) operand: FP8 K is dequantised in registers and never
    # round-trips through shared memory; only Q^T (resident) is read from SMEM.
    offs_d = tl.arange(0, DC)
    # Block-table entries are prefetched one tile ahead (loop-carried) so that the K/V
    # addresses of a tile never depend on a load issued in the same iteration; this is
    # what lets Triton's software pipeliner keep the next tile's cp.asyncs in flight.
    last_pg = (kv_len - 1) // PAGE_SIZE
    pages_nx = _fetch_pages(bt_row, tile_lo, last_pg, PAGE_SIZE, BLOCK_N)
    for tile in range(tile_lo, tile_hi):
        n0 = tile * BLOCK_N
        offs_n = n0 + tl.arange(0, BLOCK_N)
        pages = pages_nx
        pages_nx = _fetch_pages(bt_row, tile + 1, last_pg, PAGE_SIZE, BLOCK_N)
        k_rows = _row_offsets(pages, n0, stride_kp, stride_kt, PAGE_SIZE, BLOCK_N)
        v_rows = _row_offsets(pages, n0, stride_vp, stride_vt, PAGE_SIZE, BLOCK_N)
        if MASK:
            n_ok = offs_n < kv_len

        # ---- S^T = K Q^T, accumulated over D chunks --------------------------------
        sT = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
        for c in tl.static_range(NUM_DC):
            kp = k_head + k_rows[:, None] + (c * DC + offs_d)[None, :]
            if MASK:
                k = tl.load(kp, mask=n_ok[:, None], other=0.0)
            else:
                k = tl.load(kp)
            if KV_FP8:
                if not QK_FP8:
                    k = k.to(tl.float16)
            sT = tl.dot(k, qTs[c], sT)
        if QK_FP8:
            sT = sT * (q_rscale * qk_scale)[None, :]
        else:
            sT = sT * qk_scale
        if MASK:
            ok = (offs_n[:, None] <= q_pos[None, :]) & n_ok[:, None]
            sT = tl.where(ok, sT, float("-inf"))

        # ---- online softmax over keys (axis 0), log2 domain --------------------------
        m_new = tl.maximum(m_i, tl.max(sT, 0))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        pT = tl.math.exp2(sT - m_use[None, :])
        alpha = tl.math.exp2(m_i - m_use)
        l_i = l_i * alpha + tl.sum(pT, 0)
        m_i = m_new
        p = tl.trans(pT.to(PV_DTYPE))          # [BLOCK_M, BLOCK_N]

        # ---- O_c = O_c * alpha + P V_c ----------------------------------------------
        new_accs = ()
        for c in tl.static_range(NUM_DC):
            vp = v_head + v_rows[:, None] + (c * DC + offs_d)[None, :]
            if MASK:
                v = tl.load(vp, mask=n_ok[:, None], other=0.0)
            else:
                v = tl.load(vp)
            if KV_FP8:
                v = v.to(tl.float16)
            new_accs = new_accs + (tl.dot(p, v, accs[c] * alpha[:, None]),)
        accs = new_accs
    return accs, m_i, l_i


@triton.jit
def _attn512_fwd_kernel(
    Q, K, V, O, OP, LSE,
    BT, CU_Q, SEQ_K, KDS, VDS,
    sm_scale_log2,
    stride_qt, stride_qh,
    stride_kp, stride_kt, stride_kh,
    stride_vp, stride_vt, stride_vh,
    stride_ot, stride_oh,
    stride_ops, stride_opt, stride_oph,
    stride_ls, stride_lt,
    stride_bt,
    stride_kds_b, stride_kds_h, stride_vds_b, stride_vds_h,
    num_kv_heads, num_m_blocks, num_splits,
    G: tl.constexpr, HEAD_DIM: tl.constexpr, DC: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PAGE_SIZE: tl.constexpr,
    KV_FP8: tl.constexpr, QK_FP8: tl.constexpr, SPLIT: tl.constexpr,
    HAS_KDS: tl.constexpr, HAS_VDS: tl.constexpr,
):
    NUM_DC: tl.constexpr = HEAD_DIM // DC
    TPB: tl.constexpr = BLOCK_M // G  # query tokens per M-block

    pid = tl.program_id(0)
    split = tl.program_id(1)
    m_block = pid % num_m_blocks
    bh = pid // num_m_blocks
    h = bh % num_kv_heads
    b = bh // num_kv_heads

    q_start = tl.load(CU_Q + b)
    q_len = tl.load(CU_Q + b + 1) - q_start
    t0 = m_block * TPB
    if t0 < q_len:
        kv_len = tl.load(SEQ_K + b)
        ctx = kv_len - q_len                       # keys before the first new token
        t_last = tl.minimum(t0 + TPB, q_len) - 1
        kv_end = ctx + t_last + 1                  # exclusive key bound of this M-block
        kv_full = ctx + t0 + 1                     # keys < kv_full are visible to all rows
        n_tiles = tl.cdiv(kv_end, BLOCK_N)
        if SPLIT:
            tiles_per = tl.cdiv(n_tiles, num_splits)
            tile_lo = tl.minimum(split * tiles_per, n_tiles)
            tile_hi = tl.minimum(tile_lo + tiles_per, n_tiles)
        else:
            tile_lo = 0
            tile_hi = n_tiles
        n_full = tl.maximum(kv_full, 0) // BLOCK_N
        tile_mid = tl.minimum(tl.maximum(n_full, tile_lo), tile_hi)

        # ---- rows: r -> (token t0 + r // G, head h*G + r % G) -------------------------
        offs_m = tl.arange(0, BLOCK_M)
        t_row = t0 + offs_m // G
        row_ok = t_row < q_len
        tok = (q_start + t_row).to(tl.int64)
        head = h * G + offs_m % G
        q_pos = ctx + t_row                        # last visible key (inclusive)
        offs_d = tl.arange(0, DC)

        if KV_FP8:
            PV_DTYPE: tl.constexpr = tl.float16
        else:
            PV_DTYPE: tl.constexpr = K.dtype.element_ty

        q_rows = Q + tok * stride_qt + head * stride_qh
        qs = ()
        if QK_FP8:
            amax = tl.zeros([BLOCK_M], dtype=tl.float32)
            for c in tl.static_range(NUM_DC):
                qc = tl.load(q_rows[:, None] + (c * DC + offs_d)[None, :],
                             mask=row_ok[:, None], other=0.0).to(tl.float32)
                amax = tl.maximum(amax, tl.max(tl.abs(qc), 1))
            q_rscale = tl.where(amax > 0, amax / 448.0, 1.0)
            for c in tl.static_range(NUM_DC):
                qc = tl.load(q_rows[:, None] + (c * DC + offs_d)[None, :],
                             mask=row_ok[:, None], other=0.0).to(tl.float32)
                qs = qs + (tl.trans((qc / q_rscale[:, None]).to(K.dtype.element_ty)),)
        else:
            q_rscale = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
            for c in tl.static_range(NUM_DC):
                qc = tl.load(q_rows[:, None] + (c * DC + offs_d)[None, :],
                             mask=row_ok[:, None], other=0.0)
                qs = qs + (tl.trans(qc.to(PV_DTYPE)),)

        qk_scale = sm_scale_log2
        if HAS_KDS:
            qk_scale = qk_scale * tl.load(KDS + b * stride_kds_b + h * stride_kds_h)
        o_scale = 1.0
        if HAS_VDS:
            o_scale = tl.load(VDS + b * stride_vds_b + h * stride_vds_h)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        accs = ()
        for c in tl.static_range(NUM_DC):
            accs = accs + (tl.zeros([BLOCK_M, DC], dtype=tl.float32),)

        bt_row = BT + b.to(tl.int64) * stride_bt
        k_head = K + h * stride_kh
        v_head = V + h * stride_vh

        # fully visible tiles: no masking
        accs, m_i, l_i = _attn_inner(
            accs, m_i, l_i, qs, q_rscale, qk_scale, k_head, v_head, bt_row,
            tile_lo, tile_mid, kv_len, q_pos, stride_kp, stride_kt, stride_vp, stride_vt,
            BLOCK_M, BLOCK_N, DC, NUM_DC, PAGE_SIZE, KV_FP8, QK_FP8, False, PV_DTYPE)
        # diagonal / tail tiles: causal + length mask
        accs, m_i, l_i = _attn_inner(
            accs, m_i, l_i, qs, q_rscale, qk_scale, k_head, v_head, bt_row,
            tile_mid, tile_hi, kv_len, q_pos, stride_kp, stride_kt, stride_vp, stride_vt,
            BLOCK_M, BLOCK_N, DC, NUM_DC, PAGE_SIZE, KV_FP8, QK_FP8, True, PV_DTYPE)

        # ---- epilogue -------------------------------------------------------------
        l_ok = l_i > 0
        inv_l = tl.where(l_ok, o_scale / tl.where(l_ok, l_i, 1.0), 0.0)
        if SPLIT:
            lse = tl.where(l_ok, m_i + tl.math.log2(tl.where(l_ok, l_i, 1.0)), float("-inf"))
            tl.store(LSE + split.to(tl.int64) * stride_ls + tok * stride_lt + head, lse,
                     mask=row_ok)
            op_rows = (OP + split.to(tl.int64) * stride_ops + tok * stride_opt
                       + head * stride_oph)
            for c in tl.static_range(NUM_DC):
                tl.store(op_rows[:, None] + (c * DC + offs_d)[None, :],
                         accs[c] * inv_l[:, None], mask=row_ok[:, None])
        else:
            o_rows = O + tok * stride_ot + head * stride_oh
            for c in tl.static_range(NUM_DC):
                tl.store(o_rows[:, None] + (c * DC + offs_d)[None, :],
                         (accs[c] * inv_l[:, None]).to(O.dtype.element_ty),
                         mask=row_ok[:, None])


@triton.jit(do_not_specialize=["num_seqs", "num_splits"])  # no mid-inference recompiles
def _attn512_combine_kernel(
    OP, LSE, O, CU_Q,
    num_seqs, num_heads, num_splits,
    stride_ops, stride_opt, stride_oph,
    stride_ls, stride_lt,
    stride_ot, stride_oh,
    HEAD_DIM: tl.constexpr, BLOCK_H: tl.constexpr,
):
    tok = tl.program_id(0)
    hb = tl.program_id(1)
    total = tl.load(CU_Q + num_seqs)
    if tok < total:
        tok64 = tok.to(tl.int64)
        offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        h_ok = offs_h < num_heads
        offs_d = tl.arange(0, HEAD_DIM)
        m = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
        # split offsets accumulate in the (64-bit) pointers: no int32 overflow
        lse_ptrs = LSE + tok64 * stride_lt + offs_h
        for s in range(num_splits):
            lse = tl.load(lse_ptrs, mask=h_ok, other=float("-inf"))
            m = tl.maximum(m, lse)
            lse_ptrs += stride_ls
        m_use = tl.where(m == float("-inf"), 0.0, m)
        acc = tl.zeros([BLOCK_H, HEAD_DIM], dtype=tl.float32)
        den = tl.zeros([BLOCK_H], dtype=tl.float32)
        lse_ptrs = LSE + tok64 * stride_lt + offs_h
        o_ptrs = OP + tok64 * stride_opt + offs_h[:, None] * stride_oph + offs_d[None, :]
        for s in range(num_splits):
            lse = tl.load(lse_ptrs, mask=h_ok, other=float("-inf"))
            w = tl.math.exp2(lse - m_use)
            use = h_ok & (w > 0)
            o = tl.load(o_ptrs, mask=use[:, None], other=0.0)
            acc += tl.where(use[:, None], o * w[:, None], 0.0)
            den += tl.where(use, w, 0.0)
            lse_ptrs += stride_ls
            o_ptrs += stride_ops
        out = acc / tl.where(den > 0, den, 1.0)[:, None]
        tl.store(O + tok64 * stride_ot + offs_h[:, None] * stride_oh + offs_d[None, :],
                 out.to(O.dtype.element_ty), mask=h_ok[:, None])


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Attn512Config:
    block_n: int = 64      # keys per tile (gluon: must be 128)
    dc: int = 64           # D chunk; HEAD_DIM // dc chunks (gluon: 64)
    num_warps: int = 8     # (gluon: must be 8)
    num_stages: int = 3    # triton software-pipelining depth
    ring: int = 8          # gluon: K/V chunk ring slots (must divide 2 * HEAD_DIM / dc)
    lag: int = 0           # gluon: ring slots not prefetched into; 0 = auto (fp8 1, bf16 2)
    qk_fp8: bool = False   # triton, approximate: FP8 Q.K^T (fp8 KV only)
    max_splits: int = 64
    split_overhead_tiles: float = 3.0   # split cost model: fixed cost per CTA in tile units
    min_tiles_per_split: int = 4


# Defaults chosen from offline SM90 compiles (SMEM <= 227 KB, register/spill checks, async
# copies overlapping, no wgmma serialisation).  Run `python3 bench.py --tune` on the H100 to
# measure and write attn512_tuned.json next to this file (picked up automatically).
#  * triton/fp8 : 64-key tiles, 3-stage cp.async pipeline (213 KB SMEM).
#  * triton/bf16: 128-key tiles do not fit a 2-stage pipeline in SMEM -> unpipelined.
#  * gluon/fp8  : 8-slot ring (7 chunks = 56 KB in flight), 197 KB SMEM.
#  * gluon/bf16 : 8-slot ring (6 chunks = 96 KB in flight), 213 KB SMEM.
_DEFAULTS = {
    ("triton", "fp8"): Attn512Config(block_n=64, dc=64, num_stages=3),
    ("triton", "bf16"): Attn512Config(block_n=128, dc=64, num_stages=1),
    ("gluon", "fp8"): Attn512Config(block_n=128, dc=64, ring=8, lag=1),
    ("gluon", "bf16"): Attn512Config(block_n=128, dc=64, ring=8, lag=2),
}

_TUNED_PATH = os.environ.get(
    "ATTN512_TUNED_JSON",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn512_tuned.json"))


# impl preferred by impl="auto" per KV kind (gluon still has to pass its self-test);
# bench.py --tune records the measured winner in attn512_tuned.json ("auto" key).
_AUTO_PREF = {"fp8": "gluon", "bf16": "gluon"}


def _load_tuned():
    cfgs = dict(_DEFAULTS)
    fields_ok = Attn512Config.__dataclass_fields__
    try:
        with open(_TUNED_PATH) as f:
            data = json.load(f)
        for key, val in data.items():  # keys like "triton/fp8"; "auto": {"fp8": impl, ...}
            if key == "auto" and isinstance(val, dict):
                _AUTO_PREF.update({k: v for k, v in val.items()
                                   if k in _AUTO_PREF and v in ("gluon", "triton")})
                continue
            if "/" not in key or not isinstance(val, dict):
                continue  # e.g. "_meta"
            impl, kind = key.split("/", 1)
            if (impl, kind) in cfgs:
                cfgs[(impl, kind)] = replace(cfgs[(impl, kind)],
                                             **{k: v for k, v in val.items() if k in fields_ok})
    except FileNotFoundError:
        pass
    except Exception as e:  # never break the model on a bad file
        print(f"[attn512] ignoring bad tuned config {_TUNED_PATH}: {e}")
    env = os.environ.get("ATTN512_CONFIG")  # JSON overrides applied to every config
    if env:
        fields = json.loads(env)
        for key in cfgs:
            cfgs[key] = replace(cfgs[key], **fields)
    return cfgs


_CONFIGS = _load_tuned()


def default_impl() -> str:
    """ATTN512_IMPL = auto (default) | gluon | triton."""
    return os.environ.get("ATTN512_IMPL", "auto")


def get_config(kv_fp8: bool, impl: str | None = None) -> Attn512Config:
    impl = impl or default_impl()
    if impl == "auto":
        impl = "gluon"
    return _CONFIGS[(impl, "fp8" if kv_fp8 else "bf16")]


_SM_COUNT: dict = {}


def _num_sms(device) -> int:
    idx = device.index if device.index is not None else 0
    if device.type != "cuda":
        return 132
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_COUNT[idx]


def choose_num_splits(num_ctas_base: int, max_seqlen_k: int, cfg: Attn512Config,
                      num_sms: int) -> int:
    """Wave-quantization model: minimise ceil(ctas / SMs) * (tiles/split + overhead)."""
    n_tiles = max(1, triton.cdiv(max_seqlen_k, cfg.block_n))
    best_s, best_cost = 1, None
    for s in range(1, cfg.max_splits + 1):
        if s > 1 and n_tiles / s < cfg.min_tiles_per_split:
            break
        waves = math.ceil(num_ctas_base * s / num_sms)
        cost = waves * (n_tiles / s + cfg.split_overhead_tiles) + (0.0 if s == 1 else 1.0)
        if best_cost is None or cost < best_cost * 0.98:
            best_s, best_cost = s, cost
    return best_s


# --------------------------------------------------------------------------------------
# Public op
# --------------------------------------------------------------------------------------
def _as_descale(x, device):
    """-> (tensor or None, stride_b, stride_h)."""
    if x is None:
        return None, 0, 0
    if not isinstance(x, torch.Tensor):
        x = float(x)
        if x == 1.0:
            return None, 0, 0
        return torch.full((1,), x, dtype=torch.float32, device=device), 0, 0
    assert x.dtype == torch.float32, "descale must be float32"
    if x.numel() == 1:
        return x.reshape(1) if x.dim() != 1 else x, 0, 0
    assert x.dim() == 2, "descale must be scalar or [num_seqs, num_kv_heads]"
    return x, x.stride(0), x.stride(1)


_GLUON_MOD = None
_GLUON_IMPORT_ERR: str | None = None
_SELFTEST: dict = {}     # (impl, kv dtype, page_size, device index) -> bool
_WARNED: set = set()


def _warn_once(msg: str):
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f"[attn512] {msg}", flush=True)


def _gluon_module():
    """Import kernel_gluon.py from this directory (works as a package or a plain module)."""
    global _GLUON_MOD, _GLUON_IMPORT_ERR
    if _GLUON_MOD is None and _GLUON_IMPORT_ERR is None:
        try:
            import importlib.util
            import sys
            name = "attn512_kernel_gluon"
            if name in sys.modules:
                _GLUON_MOD = sys.modules[name]
            else:
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_gluon.py")
                spec = importlib.util.spec_from_file_location(name, path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                try:
                    spec.loader.exec_module(mod)
                except BaseException:
                    del sys.modules[name]
                    raise
                _GLUON_MOD = mod
        except Exception as e:  # Gluon missing / API drift in another Triton version
            _GLUON_IMPORT_ERR = f"{type(e).__name__}: {e}"
            _warn_once(f"Gluon kernel unavailable ({_GLUON_IMPORT_ERR}); using the Triton kernel")
    return _GLUON_MOD


def _gluon_smem_bytes(cfg: Attn512Config, D: int, kv_bytes: int) -> int:
    q = 64 * D * 2
    ring = cfg.ring * 128 * cfg.dc * kv_bytes
    v16 = 3 * 128 * cfg.dc * 2 if kv_bytes == 1 else 0
    p = 128 * 64 * 2
    return q + ring + v16 + p + 4096   # + reduction / layout-conversion scratch


def fit_gluon_config(cfg: Attn512Config, D: int) -> Attn512Config:
    """Shrink dc / ring for head dims below 512 (tests); a no-op for D = 512."""
    dc = min(cfg.dc, max(16, D // 2))
    nch = 2 * D // dc
    ring = cfg.ring
    while nch % ring:
        ring -= 1
    return replace(cfg, dc=dc, ring=ring)


def _check_gluon_cfg(cfg: Attn512Config, D: int, kv_fp8: bool):
    lag = cfg.lag or (1 if kv_fp8 else 2)
    nch = 2 * D // cfg.dc
    if cfg.block_n != 128 or cfg.num_warps != 8 or D % cfg.dc or D // cfg.dc < 2:
        raise ValueError(f"gluon kernel needs block_n=128, num_warps=8, 2 <= D/dc: {cfg}")
    if nch % cfg.ring or not (1 <= cfg.ring - lag <= nch) or lag < (1 if kv_fp8 else 2):
        raise ValueError(f"gluon: ring must divide {nch} and exceed lag (fp8 lag>=1, bf16 "
                         f"lag>=2): {cfg}")
    smem = _gluon_smem_bytes(cfg, D, 1 if kv_fp8 else 2)
    if smem > 227 * 1024:
        raise ValueError(f"gluon config needs ~{smem // 1024} KB SMEM > 227 KB: {cfg}")
    return lag


def _launch_fwd(impl, cfg, grid, q, k_cache, v_cache, o_main, op, lse, block_table, cu_seqlens_q,
                seqused_k, kds, vds, scale_log2, out, split, kds_sb, kds_sh, vds_sb, vds_sh,
                Hkv, num_m_blocks, num_splits, G, D, page_size, kv_fp8, qk_fp8,
                k_norm_weight=None, cos_sin_cache=None, dec_bm=0):
    common = (
        q, k_cache, v_cache, o_main, op, lse,
        block_table, cu_seqlens_q, seqused_k,
        kds if kds is not None else q, vds if vds is not None else q,
        scale_log2,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        out.stride(0), out.stride(1),
        op.stride(0) if split else 0, op.stride(1) if split else 0, op.stride(2) if split else 0,
        lse.stride(0) if split else 0, lse.stride(1) if split else 0,
        block_table.stride(0),
        kds_sb, kds_sh, vds_sb, vds_sh,
        Hkv, num_m_blocks, num_splits,
    )
    if impl == "gluon" and dec_bm:
        k_only = 0 if k_norm_weight is None else 2
        pair = int(os.environ.get("ATTN512_DEC_PAIR", "0"))
        _gluon_module()._attn512_gluon_dec_kernel[grid](
            *common,
            k_norm_weight if k_only else q, q, 0,
            G=G, HEAD_DIM=D, DC=64, BLOCK_M=dec_bm, BLOCK_N=128, PAGE_SIZE=page_size,
            RING=int(os.environ.get("ATTN512_DEC_RING", "8")),
            LAG=max(int(os.environ.get("ATTN512_DEC_LAG", "1")), 2 if pair else 1),
            KV_FP8=kv_fp8, SPLIT=split,
            HAS_KDS=kds is not None, HAS_VDS=vds is not None, K_ONLY=k_only, PAIR=pair,
            num_warps=8,
        )
    elif impl == "gluon":
        lag = _check_gluon_cfg(cfg, D, kv_fp8)
        k_only = 0 if k_norm_weight is None else (2 if cos_sin_cache is None else 1)
        _gluon_module()._attn512_gluon_kernel[grid](
            *common,
            k_norm_weight if k_only else q, cos_sin_cache if k_only == 1 else q,
            cos_sin_cache.stride(0) if k_only == 1 else 0,
            G=G, HEAD_DIM=D, DC=cfg.dc, BLOCK_M=64, BLOCK_N=128, PAGE_SIZE=page_size,
            RING=cfg.ring, LAG=lag, KV_FP8=kv_fp8, SPLIT=split,
            HAS_KDS=kds is not None, HAS_VDS=vds is not None, K_ONLY=k_only,
            KO_NOROPE=int(os.environ.get("ATTN512_KO_NOROPE", "0")),
            num_warps=8,
        )
    else:
        assert k_norm_weight is None, "K-only cache needs the gluon kernel"
        dc = min(cfg.dc, D)
        assert D % dc == 0 and (dc & (dc - 1)) == 0, "head_dim must be a multiple of a pow2 chunk"
        assert page_size % cfg.block_n == 0 or cfg.block_n % page_size == 0, \
            "triton kernel: page_size and block_n must divide one another"
        _attn512_fwd_kernel[grid](
            *common,
            G=G, HEAD_DIM=D, DC=dc, BLOCK_M=64, BLOCK_N=cfg.block_n, PAGE_SIZE=page_size,
            KV_FP8=kv_fp8, QK_FP8=qk_fp8, SPLIT=split,
            HAS_KDS=kds is not None, HAS_VDS=vds is not None,
            num_warps=cfg.num_warps, num_stages=cfg.num_stages,
        )


def _resolve_impl(impl, kv_dtype, page_size, device, D) -> str:
    """'auto' -> 'gluon' if it imports and passes its self-test here, else 'triton'."""
    if impl not in ("auto", "gluon", "triton"):
        raise ValueError(f"impl must be auto|gluon|triton, got {impl!r}")
    if impl == "auto" and _AUTO_PREF["fp8" if kv_dtype in _FP8_TYPES else "bf16"] == "triton":
        return "triton"
    if impl == "triton":
        return impl
    if _gluon_module() is None:
        if impl == "gluon":
            raise RuntimeError(f"gluon kernel unavailable: {_GLUON_IMPORT_ERR}")
        return "triton"
    if impl == "gluon" or os.environ.get("ATTN512_SELFTEST", "1") == "0":
        return "gluon"
    # auto: self-test once (needs a non-capturing stream; under capture trust a prior result)
    key = ("gluon", str(kv_dtype), page_size, device.index, D)
    if key not in _SELFTEST:
        if device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            return "triton"
        _SELFTEST[key] = selftest("gluon", kv_dtype, page_size, device, D)
    return "gluon" if _SELFTEST[key] else "triton"


def attn512_paged(q, k_cache, v_cache, block_table, cu_seqlens_q, seqused_k,
                  max_seqlen_q, max_seqlen_k, softmax_scale,
                  k_descale=None, v_descale=None, num_splits=None, out=None,
                  config: Attn512Config | None = None, impl: str | None = None,
                  k_norm_weight=None, cos_sin_cache=None):
    """Causal paged GQA attention; returns out [num_tokens, Hq, D] (q.dtype).

    max_seqlen_q / max_seqlen_k must be python ints (upper bounds are fine);
    num_splits=None -> heuristic, 1 -> no split-KV.  impl: auto (default, env ATTN512_IMPL) |
    gluon | triton.  config: Attn512Config for the chosen impl (default: tuned/default one).

    K-only cache (Gemma 4 attention_k_eq_v global layers): pass k_norm_weight [D] and the
    rotary cos_sin_cache [max_pos, D] (cos | sin halves, proportional RoPE: dims [0,64) and
    [D/2, D/2+64) rotated); k_cache = v_cache = the single cache holding b = V, and
    K = rope(k_norm_weight * b) is rebuilt in the kernel (gluon only).
    """
    assert q.dim() == 3 and k_cache.dim() == 4 and v_cache.dim() == 4
    T, Hq, D = q.shape
    num_pages, page_size, Hkv, Dk = k_cache.shape
    assert Dk == D and v_cache.shape == k_cache.shape, "K/V must be [pages, page, Hkv, D]"
    assert q.stride(-1) == 1 and k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1
    assert Hq % Hkv == 0
    G = Hq // Hkv
    assert k_cache.dtype == v_cache.dtype
    kv_fp8 = k_cache.dtype in _FP8_TYPES
    if not kv_fp8:
        assert k_cache.dtype in (torch.bfloat16, torch.float16)
    assert q.dtype in (torch.bfloat16, torch.float16)
    assert block_table.dtype == torch.int32 and cu_seqlens_q.dtype == torch.int32
    assert seqused_k.dtype == torch.int32
    assert block_table.stride(-1) == 1
    assert 64 % G == 0, "query heads per kv head must divide 64"
    B = cu_seqlens_q.numel() - 1

    k_only = k_norm_weight is not None
    if k_only:
        # cos_sin_cache=None -> sidecar layout: k_cache is the [..., :D] view of rows that hold
        # [b (D) | RoPE'd K dims [0,64) | [D/2, D/2+64)] (row stride >= D + 128)
        impl = "gluon"
        if cos_sin_cache is not None:
            assert cos_sin_cache.stride(-1) == 1 and cos_sin_cache.shape[-1] == D
        else:
            assert k_cache.stride(-2) >= D + 128, "sidecar needs D+128 elements per (token, head)"
        assert k_norm_weight.numel() == D and k_norm_weight.is_contiguous()
    else:
        impl = _resolve_impl(impl or default_impl(), k_cache.dtype, page_size, q.device, D)
    cfg = config if config is not None else get_config(kv_fp8, impl)
    if impl == "gluon" and config is None:
        cfg = fit_gluon_config(cfg, D)
    if k_only and cos_sin_cache is not None and (cfg.lag or 1) < 2:
        cfg = replace(cfg, lag=2)   # the RoPE step reads the previous ring slot
    qk_fp8 = bool(cfg.qk_fp8 and kv_fp8)

    if out is None:
        out = torch.empty_like(q)
    assert out.shape == q.shape and out.stride(-1) == 1
    if T == 0 or B == 0:
        return out

    kds, kds_sb, kds_sh = _as_descale(k_descale, q.device)
    vds, vds_sb, vds_sh = _as_descale(v_descale, q.device)

    # decode variant (few query rows, transposed PV): fp8, no in-kernel RoPE, q*G <= 32
    dec_bm = 0
    if (impl == "gluon" and kv_fp8 and cos_sin_cache is None and int(max_seqlen_q) * G <= 32
            and os.environ.get("ATTN512_DEC", "1") != "0"):
        dec_bm = 16 if int(max_seqlen_q) * G <= 16 else 32
    tpb = (dec_bm or 64) // G
    num_m_blocks = triton.cdiv(max(int(max_seqlen_q), 1), tpb)
    num_ctas_base = B * Hkv * num_m_blocks
    block_n = 128 if impl == "gluon" else cfg.block_n
    if num_splits is None or num_splits <= 0:
        scfg = replace(cfg, block_n=block_n)
        if dec_bm:
            scfg = replace(scfg, split_overhead_tiles=float(os.environ.get("ATTN512_DEC_OVH", "4")),
                           min_tiles_per_split=2)
        num_splits = choose_num_splits(num_ctas_base, int(max_seqlen_k), scfg, _num_sms(q.device))
    num_splits = int(max(1, num_splits))
    split = num_splits > 1

    if split:
        op = torch.empty((num_splits, T, Hq, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((num_splits, T, Hq), dtype=torch.float32, device=q.device)
        o_main = op  # unused placeholder
    else:
        op = out
        lse = out
        o_main = out

    _launch_fwd(impl, cfg, (num_ctas_base, num_splits), q, k_cache, v_cache, o_main, op, lse, block_table, cu_seqlens_q,
                seqused_k, kds, vds, float(softmax_scale) * _LOG2E, out, split,
                kds_sb, kds_sh, vds_sb, vds_sh, Hkv, num_m_blocks, num_splits, G, D,
                page_size, kv_fp8, qk_fp8, k_norm_weight, cos_sin_cache, dec_bm)
    if split:
        block_h = G if (G & (G - 1)) == 0 else triton.next_power_of_2(G)
        _attn512_combine_kernel[(T, triton.cdiv(Hq, block_h))](
            op, lse, out, cu_seqlens_q,
            B, Hq, num_splits,
            op.stride(0), op.stride(1), op.stride(2),
            lse.stride(0), lse.stride(1),
            out.stride(0), out.stride(1),
            HEAD_DIM=D, BLOCK_H=block_h, num_warps=4,
        )
    return out


# --------------------------------------------------------------------------------------
# Reference + self-test
# --------------------------------------------------------------------------------------
def reference_paged(q, k_cache, v_cache, block_table, cu_seqlens_q, seqused_k, softmax_scale,
                    k_descale=None, v_descale=None):
    """Plain PyTorch fp32 reference (slow; for tests).  Same arguments as attn512_paged."""
    T, Hq, D = q.shape
    page_size, Hkv = k_cache.shape[1], k_cache.shape[2]
    G = Hq // Hkv
    out = torch.zeros(T, Hq, D, dtype=torch.float32, device=q.device)
    cu = cu_seqlens_q.tolist()
    lens = seqused_k.tolist()

    def _ds(x, b):
        if x is None:
            return torch.ones(Hkv, device=q.device)
        if not isinstance(x, torch.Tensor):
            return torch.full((Hkv,), float(x), device=q.device)
        if x.numel() == 1:
            return x.reshape(1).float().expand(Hkv)
        return x[b].float()

    for b in range(len(cu) - 1):
        q0, q1, kv = cu[b], cu[b + 1], lens[b]
        ql = q1 - q0
        if ql == 0:
            continue
        npg = (kv + page_size - 1) // page_size
        pages = block_table[b, :npg].long()
        k = k_cache[pages].reshape(-1, Hkv, D)[:kv].float() * _ds(k_descale, b)[None, :, None]
        v = v_cache[pages].reshape(-1, Hkv, D)[:kv].float() * _ds(v_descale, b)[None, :, None]
        k = k.repeat_interleave(G, dim=1)          # [kv, Hq, D]
        v = v.repeat_interleave(G, dim=1)
        s = torch.einsum("thd,khd->htk", q[q0:q1].float(), k) * softmax_scale
        qpos = torch.arange(kv - ql, kv, device=q.device)
        mask = torch.arange(kv, device=q.device)[None, :] <= qpos[:, None]
        s = s.masked_fill(~mask[None], float("-inf"))
        out[q0:q1] = torch.einsum("htk,khd->thd", torch.softmax(s, dim=-1), v)
    return out


def selftest(impl, kv_dtype, page_size, device, D=512, verbose=False) -> bool:
    """Small ragged problem (decode row, partial tiles, diagonal masks, split and no split)
    vs reference_paged.  Returns True iff every check passes; never raises."""
    try:
        g = torch.Generator(device="cpu").manual_seed(1234)
        Hq, Hkv = 16, 2
        q_lens, kv_lens = [1, 16, 37], [700, 1500, 333]
        B = len(q_lens)
        fp8 = kv_dtype in _FP8_TYPES
        cu = torch.tensor([0] + list(itertools.accumulate(q_lens)), dtype=torch.int32)
        T = int(cu[-1])
        q = (torch.randn(T, Hq, D, generator=g) * 0.08).to(torch.bfloat16)
        n_pages = [(kv + page_size - 1) // page_size for kv in kv_lens]
        tot = sum(n_pages) + 3
        perm = torch.randperm(tot, generator=g)
        bt = torch.full((B, max(n_pages) + 2), 10**6, dtype=torch.int32)
        o = 0
        for b, n in enumerate(n_pages):
            bt[b, :n] = perm[o:o + n].int()
            o += n
        kf = torch.randn(tot, page_size, Hkv, D, generator=g)
        vf = torch.randn(tot, page_size, Hkv, D, generator=g)
        if fp8:
            kds, vds = 0.11, 0.07
            kc, vc = kf.to(kv_dtype), vf.to(kv_dtype)
            kd = torch.tensor([kds], dtype=torch.float32, device=device)
            vd = torch.tensor([vds], dtype=torch.float32, device=device)
            kd2 = kd.expand(B, Hkv)
            vd2 = vd.expand(B, Hkv)
        else:
            kc, vc = (kf * 0.11).to(kv_dtype), (vf * 0.07).to(kv_dtype)
            kd2 = vd2 = None
        q, kc, vc, bt, cu = (x.to(device) for x in (q, kc, vc, bt, cu))
        sk = torch.tensor(kv_lens, dtype=torch.int32, device=device)
        ref = reference_paged(q, kc, vc, bt, cu, sk, D ** -0.5, kd2, vd2)
        ok = True
        for ns in (1, 3):
            out = attn512_paged(q, kc, vc, bt, cu, sk, max(q_lens), max(kv_lens), D ** -0.5,
                                kd2, vd2, num_splits=ns, impl=impl)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            err = (out.float() - ref).norm() / ref.norm()
            good = bool(torch.isfinite(out).all()) and float(err) < 2e-2
            if verbose or not good:
                print(f"[attn512] selftest impl={impl} kv={kv_dtype} page={page_size} "
                      f"splits={ns}: rel_l2={float(err):.2e} {'ok' if good else 'FAILED'}",
                      flush=True)
            ok = ok and good
        if not ok:
            _warn_once(f"{impl} kernel failed its self-test for kv={kv_dtype}, page={page_size}"
                       " -> falling back")
        return ok
    except Exception as e:
        _warn_once(f"{impl} kernel self-test raised {type(e).__name__}: {e} -> falling back")
        return False


def flops(num_seqs: int, q_len: int, kv_len: int, num_heads: int = 32, head_dim: int = 512) -> float:
    """Convention used in the task: 4 * B * q_len * kv_len * Hq * D (no causal discount)."""
    return 4.0 * num_seqs * q_len * kv_len * num_heads * head_dim


__all__ = ["attn512_paged", "Attn512Config", "get_config", "choose_num_splits", "flops",
           "replace", "default_impl", "reference_paged", "selftest", "fit_gluon_config"]


def warmup_konly(q_ref, cache, block_table_ref, softmax_scale, descale=None,
                 k_norm_weight=None, cos_sin_cache=None):
    """Compile every kernel variant the K-only path can hit -- decode kernel (16 / 32 rows), main kernel,
    split / no split, combine -- on the real cache layout, so no Triton JIT happens during inference.
    q_ref [T, Hq, D] and block_table_ref only provide dtype / strides; cache is the real K-only cache view
    (only block 0 is read).  Call once per process outside CUDA graph capture."""
    dev, H, D = q_ref.device, q_ref.shape[-2], q_ref.shape[-1]
    # width = the real table's row stride, so stride_bt specializes (%16 / ==1) exactly as at runtime
    page = cache.shape[1]
    width = max(int(block_table_ref.stride(0)), int(block_table_ref.shape[1]), 2)
    for B, mq, kv in ((1, 1, 8192), (64, 1, 128), (1, 2, 8192), (1, 4, 8192), (64, 4, 128),
                      (2, 3, 8192), (1, 16, 8192), (64, 16, 128), (2, 64, 4096)):
        kv = min(kv, page * width)
        bt = torch.zeros(B, width, dtype=block_table_ref.dtype, device=dev)
        cu = torch.arange(0, B + 1, dtype=torch.int32, device=dev) * mq
        sk = torch.full((B,), kv, dtype=torch.int32, device=dev)
        q = torch.zeros(B * mq, H, D, dtype=q_ref.dtype, device=dev)
        attn512_paged(q, cache, cache, bt, cu, sk, mq, kv, softmax_scale, k_descale=descale,
                      v_descale=descale, k_norm_weight=k_norm_weight, cos_sin_cache=cos_sin_cache)
    torch.cuda.synchronize(dev)
