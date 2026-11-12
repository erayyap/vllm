"""attn512 Gluon (explicitly pipelined) kernel for SM90.  Imported lazily by kernel.py.

Same semantics / arguments as kernel._attn512_fwd_kernel (plus RING / LAG).  Why Gluon: Triton's
automatic pipeliner cannot express this schedule for a 512-wide head (it buffers whole tiles per
load op, waits every wgmma immediately and replicates S across warpgroups).

Per CTA: 64 query rows (= 64 // G tokens x G heads of one KV head, GQA packed into M), 8 warps.

  * K and V stream as [BLOCK_N=128 keys, DC=64 dims] chunks (K0..K7 then V0..V7 per tile)
    through a RING-slot cp.async ring (prefetch distance RING-LAG chunks).  Slots are separate
    SMEM allocations indexed statically (RING divides the 16 chunks of a tile), so Triton's
    barrier analysis can see they are disjoint: exactly one CTA barrier per chunk covers both
    RAW (chunk landed) and WAR (slot free) hazards.
  * S^T = K Q^T per tile: warps_per_cta=[8,1] gives each warpgroup its own 64 keys
    (FlashMLA-style KV split, no replicated work), K is the *register* A operand (FP8 is
    dequantised in registers, never written back), Q^T (fp16/bf16) stays resident in SMEM as the
    B operand.  Async wgmma with wait(1): the dequant of chunk c+1 overlaps the wgmma of chunk c.
  * O (64 x 512 fp32) is split by columns across the 2 warpgroups (warps [4,2]) in NDC
    accumulators of [64, DC]; P is exchanged once per tile through SMEM and kept in registers as
    the A operand of every PV wgmma.
  * FP8 V chunks are dequantised into a 3-deep fp16 SMEM buffer, with the PV wgmma lagging one
    chunk behind the dequant (no extra barrier); BF16 V chunks feed the wgmma straight from the
    ring slot.
  * FP8 descales: k_descale folds into the softmax scale, v_descale into the output scale.

Every accumulator of an in-flight wgmma is threaded through the next warpgroup_mma_wait and
re-read from its result, and no wgmma is in flight across the tile-loop back edge: otherwise
ptxas serialises the whole wgmma pipeline (C7514/C7515).
"""

from __future__ import annotations

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    warpgroup_mma,
    warpgroup_mma_wait,
)


@gluon.constexpr_function
def _compute_dtype(kv_dtype, kv_fp8):
    return gl.float16 if kv_fp8 else kv_dtype


@gluon.constexpr_function
def _dord(c, NDC):
    """K_ONLY: head-dim chunk loaded at K-phase position c.  The RoPE'd dims of Gemma 4's
    proportional RoPE ([0, 64) paired with [256, 320) by neox rotate_half) are chunk 0 and
    chunk NDC//2; chunk 0 is issued at position NDC//2 - 1, right before its partner, so both
    sit in the ring at position NDC//2.  Order: 1, 2, 3, 0, 4, 5, 6, 7."""
    c = getattr(c, "value", c)
    NDC = getattr(NDC, "value", NDC)
    h = NDC // 2
    if c < h - 1:
        return c + 1
    if c == h - 1:
        return 0
    return c


@gluon.constexpr_function
def _vec(dtype):
    return 16 // (dtype.primitive_bitwidth // 8)


@gluon.jit
def _fetch_pages(bt_row, tile, last_pg, PAGE_SIZE: gl.constexpr, BLOCK_N: gl.constexpr,
                 row_layout: gl.constexpr):
    """Per-row page ids of key tile `tile` (clamped to the sequence's last page)."""
    offs_r = gl.arange(0, BLOCK_N, layout=row_layout)
    pg = gl.minimum((tile * BLOCK_N + offs_r) // PAGE_SIZE, last_pg)
    return gl.load(bt_row + pg)


@gluon.jit
def _issue_chunk(slot, head_ptr, pages, tile, kv_len, valid, d0, stride_p, stride_t,
                 PAGE_SIZE: gl.constexpr, BLOCK_N: gl.constexpr, DC: gl.constexpr,
                 cp_layout: gl.constexpr):
    offs_r = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, cp_layout))
    offs_d = gl.arange(0, DC, layout=gl.SliceLayout(0, cp_layout))
    n = tile * BLOCK_N + offs_r
    rows = pages.to(gl.int64) * stride_p + (n % PAGE_SIZE).to(gl.int64) * stride_t
    ptrs = head_ptr + rows[:, None] + (d0 + offs_d)[None, :]
    ok = (n < kv_len) & valid
    # masked rows are zero-filled (no global read): keys past kv_len, chunks past the split
    cp.async_load(slot, ptrs, mask=ok[:, None])


@gluon.jit
def _issue(ring, c_abs, tile, pages_t, pages_t1, k_head, v_head, kv_len, tile_hi,
           stride_kp, stride_kt, stride_vp, stride_vt,
           RING: gl.constexpr, NDC: gl.constexpr, PAGE_SIZE: gl.constexpr,
           BLOCK_N: gl.constexpr, DC: gl.constexpr, cp_layout: gl.constexpr,
           K_ONLY: gl.constexpr):
    """Issue chunk `c_abs` (constexpr, counted from chunk 0 of `tile`) and commit its group."""
    # (named constexpr temporaries are demoted to tensors by the frontend: keep them inline)
    if c_abs // (2 * NDC) == 0:
        pg = pages_t
    else:
        pg = pages_t1
    itile = tile + c_abs // (2 * NDC)
    if c_abs % (2 * NDC) < NDC:
        if K_ONLY == 2:
            # sidecar layout per (token, head): [b (D) | rope(w*b) dims [0,DC) | dims [D/2, D/2+DC)]
            if (c_abs % (2 * NDC)) % (NDC // 2) == 0:
                _issue_chunk(ring[c_abs % RING], k_head, pg, itile, kv_len, itile < tile_hi,
                             NDC * DC + ((c_abs % (2 * NDC)) // (NDC // 2)) * DC, stride_kp,
                             stride_kt, PAGE_SIZE, BLOCK_N, DC, cp_layout)
            else:
                _issue_chunk(ring[c_abs % RING], k_head, pg, itile, kv_len, itile < tile_hi,
                             (c_abs % (2 * NDC)) * DC, stride_kp, stride_kt, PAGE_SIZE, BLOCK_N,
                             DC, cp_layout)
        elif K_ONLY:
            _issue_chunk(ring[c_abs % RING], k_head, pg, itile, kv_len, itile < tile_hi,
                         _dord(c_abs % (2 * NDC), NDC) * DC, stride_kp, stride_kt, PAGE_SIZE,
                         BLOCK_N, DC, cp_layout)
        else:
            _issue_chunk(ring[c_abs % RING], k_head, pg, itile, kv_len, itile < tile_hi,
                         (c_abs % (2 * NDC)) * DC, stride_kp, stride_kt, PAGE_SIZE, BLOCK_N,
                         DC, cp_layout)
    else:
        _issue_chunk(ring[c_abs % RING], v_head, pg, itile, kv_len, itile < tile_hi,
                     (c_abs % (2 * NDC) - NDC) * DC, stride_vp, stride_vt, PAGE_SIZE, BLOCK_N,
                     DC, cp_layout)
    cp.commit_group()


# runtime-varying ints: no ==1 / %16 specialization (no mid-inference recompiles)
@gluon.jit(do_not_specialize=["num_m_blocks", "num_splits"])
def _attn512_gluon_kernel(
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
    KW, CS, stride_cs,
    G: gl.constexpr, HEAD_DIM: gl.constexpr, DC: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, PAGE_SIZE: gl.constexpr,
    RING: gl.constexpr, LAG: gl.constexpr,
    KV_FP8: gl.constexpr, SPLIT: gl.constexpr,
    HAS_KDS: gl.constexpr, HAS_VDS: gl.constexpr,
    K_ONLY: gl.constexpr = 0, KO_NOROPE: gl.constexpr = 0,
):
    NDC: gl.constexpr = HEAD_DIM // DC
    NCH: gl.constexpr = 2 * NDC          # K chunks then V chunks, per tile
    PF: gl.constexpr = RING - LAG        # prefetch distance in chunks
    TPB: gl.constexpr = BLOCK_M // G
    NVB: gl.constexpr = 3                # fp16 V staging buffers (FP8 path)
    gl.static_assert(BLOCK_M == 64, "BLOCK_M must be 64")
    gl.static_assert(BLOCK_N == 128, "BLOCK_N must be 128 (2 warpgroups x 64 keys)")
    gl.static_assert(gl.num_warps() == 8, "num_warps must be 8")
    gl.static_assert(NDC >= 2, "HEAD_DIM / DC must be >= 2")
    gl.static_assert(NCH % RING == 0, "RING must divide 2 * HEAD_DIM / DC")
    gl.static_assert(PF >= 1, "RING must exceed LAG")
    # FP8: ring slots are only read synchronously (ld.shared) -> LAG 1 is safe.
    # BF16: the PV wgmma reads V straight from the ring, retired one step later -> LAG >= 2.
    gl.static_assert(LAG >= 2 - KV_FP8, "LAG must be >= 2 for bf16/fp16 KV, >= 1 for fp8")
    # K_ONLY (Gemma 4 k_eq_v global layers): the cache holds only b = v_norm(k_proj x) = V and
    # K = rope(KW * b) is rebuilt per tile (KW = k_norm weight, CS = vLLM cos_sin_cache rows
    # [cos(256) | sin(256)], RoPE'd dims [0,64) <-> [256,320)).  The RoPE step reads the ring
    # slot of the previous chunk, which LAG >= 2 keeps intact.
    # K_ONLY == 2 (sidecar): the cache row also holds the 2*DC RoPE'd K dims after b -> no RoPE math.
    gl.static_assert(LAG >= 1 + (K_ONLY == 1), "K_ONLY=1 needs LAG >= 2")
    gl.static_assert(HEAD_DIM * (K_ONLY > 0) == 512 * (K_ONLY > 0), "K_ONLY: HEAD_DIM must be 512")
    gl.static_assert(DC * (K_ONLY > 0) == 64 * (K_ONLY > 0), "K_ONLY: DC must be 64 (= RoPE'd dims per half)")

    KV_T: gl.constexpr = K.dtype.element_ty
    CDT: gl.constexpr = _compute_dtype(KV_T, KV_FP8)
    VEC: gl.constexpr = _vec(KV_T)

    # ---- layouts ---------------------------------------------------------------------
    s_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[8, 1], instr_shape=[16, BLOCK_M, 16])
    k_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=s_layout, k_width=2)
    o_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 2], instr_shape=[16, DC // 2, 16])
    p_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=o_layout, k_width=2)
    row_o: gl.constexpr = gl.SliceLayout(1, o_layout)
    cp_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, VEC], threads_per_warp=[32 // (DC // VEC), DC // VEC],
        warps_per_cta=[8, 1], order=[1, 0])
    v_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16], threads_per_warp=[32 // (DC // 16), DC // 16],
        warps_per_cta=[8, 1], order=[1, 0])
    q_ld_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[32 // (DC // 8), DC // 8],
        warps_per_cta=[8, 1], order=[1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, cp_layout)

    pid = gl.program_id(0)
    split = gl.program_id(1)
    m_block = pid % num_m_blocks
    bh = pid // num_m_blocks
    h = bh % num_kv_heads
    b = bh // num_kv_heads

    q_start = gl.load(CU_Q + b)
    q_len = gl.load(CU_Q + b + 1) - q_start
    t0 = m_block * TPB
    if t0 < q_len:
        kv_len = gl.load(SEQ_K + b)
        ctx = kv_len - q_len
        t_last = gl.minimum(t0 + TPB, q_len) - 1
        kv_end = ctx + t_last + 1
        n_tiles = gl.cdiv(kv_end, BLOCK_N)
        if SPLIT:
            tiles_per = gl.cdiv(n_tiles, num_splits)
            tile_lo = gl.minimum(split * tiles_per, n_tiles)
            tile_hi = gl.minimum(tile_lo + tiles_per, n_tiles)
        else:
            tile_lo = n_tiles * 0
            tile_hi = n_tiles
        last_pg = (kv_len - 1) // PAGE_SIZE

        # ---- shared memory ---------------------------------------------------------------
        q_sl: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, DC], CDT)
        ring_sl: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, DC], KV_T)
        v16_sl: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, DC], CDT)
        p_sl: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, BLOCK_M], CDT)
        q_smem = gl.allocate_shared_memory(CDT, [NDC, BLOCK_M, DC], q_sl)
        ring = ()
        for i in gl.static_range(RING):
            ring = ring + (gl.allocate_shared_memory(KV_T, [BLOCK_N, DC], ring_sl),)
        if KV_FP8:
            v16 = ()
            for i in gl.static_range(NVB):
                v16 = v16 + (gl.allocate_shared_memory(CDT, [BLOCK_N, DC], v16_sl),)
        p_smem = gl.allocate_shared_memory(CDT, [BLOCK_N, BLOCK_M], p_sl)

        bt_row = BT + b.to(gl.int64) * stride_bt
        k_head = K + h * stride_kh
        v_head = V + h * stride_vh

        # ---- prologue: block-table rows of the first two tiles, first PF chunks --------
        pages_t = _fetch_pages(bt_row, tile_lo, last_pg, PAGE_SIZE, BLOCK_N, row_layout)
        pages_t1 = _fetch_pages(bt_row, tile_lo + 1, last_pg, PAGE_SIZE, BLOCK_N, row_layout)
        for cc in gl.static_range(PF):
            _issue(ring, cc, tile_lo, pages_t, pages_t1, k_head, v_head, kv_len, tile_hi,
                   stride_kp, stride_kt, stride_vp, stride_vt,
                   RING, NDC, PAGE_SIZE, BLOCK_N, DC, cp_layout, K_ONLY)

        # ---- Q: [64 rows, 512] bf16 -> CDT, resident in SMEM (overlaps the prologue) ----
        offs_qr = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, q_ld_layout))
        offs_qd = gl.arange(0, DC, layout=gl.SliceLayout(0, q_ld_layout))
        q_tok = (q_start + t0 + offs_qr // G).to(gl.int64)
        q_rows = Q + q_tok * stride_qt + (h * G + offs_qr % G) * stride_qh
        q_ok = (t0 + offs_qr // G) < q_len
        for c in gl.static_range(NDC):
            qc = gl.load(q_rows[:, None] + (c * DC + offs_qd)[None, :], mask=q_ok[:, None],
                         other=0.0)
            if K_ONLY:
                if c % (NDC // 2) != 0:
                    # K = k_norm.w * b on the identity-RoPE dims: fold w into Q once per CTA
                    wq = gl.load(KW + c * DC + offs_qd).to(gl.float32)
                    qc = qc.to(gl.float32) * wq[None, :]
            q_smem.index(c).store(qc.to(CDT))
        fence_async_shared()
        gl.barrier()

        qk_scale = sm_scale_log2
        if HAS_KDS:
            qk_scale = qk_scale * gl.load(KDS + b * stride_kds_b + h * stride_kds_h)
        o_scale = 1.0
        if HAS_VDS:
            o_scale = gl.load(VDS + b * stride_vds_b + h * stride_vds_h)

        # ---- running state ---------------------------------------------------------------
        # m_i lives in the S^T column slice (16 values / thread, where exp2 needs it); l_i and
        # alpha live in the compact O row slice (2 values / thread).
        m_i = gl.full([BLOCK_M], float("-inf"), gl.float32, layout=gl.SliceLayout(0, s_layout))
        l_i = gl.zeros([BLOCK_M], gl.float32, layout=row_o)
        # tiles below tile_mid are visible to every row of this block: no mask needed there
        tile_mid = (ctx + t0 + 1) // BLOCK_N
        accs = ()
        for c in gl.static_range(NDC):
            accs = accs + (gl.zeros([BLOCK_M, DC], gl.float32, layout=o_layout),)

        for tile in range(tile_lo, tile_hi):
            pages_t2 = _fetch_pages(bt_row, tile + 2, last_pg, PAGE_SIZE, BLOCK_N, row_layout)
            sT = gl.zeros([BLOCK_N, BLOCK_M], gl.float32, layout=s_layout)
            for c in gl.static_range(NCH):
                # Chunk c landed (own copies) and is visible to all threads (barrier); every
                # thread finished reading the slot that the next copy overwrites.
                cp.wait_group(PF - 1)
                gl.barrier()
                _issue(ring, c + PF, tile, pages_t, pages_t1, k_head, v_head, kv_len, tile_hi,
                       stride_kp, stride_kt, stride_vp, stride_vt,
                       RING, NDC, PAGE_SIZE, BLOCK_N, DC, cp_layout, K_ONLY)
                rslot = ring[c % RING]

                if c < NDC:
                    # ---------------- S^T += K_c Q_c^T ----------------------------------
                    if K_ONLY == 1:
                        if c != NDC // 2 - 1:   # position NDC//2-1 holds dim-chunk 0: deferred
                            ko_d = gl.arange(0, DC, layout=gl.SliceLayout(0, k_layout))
                            if c == NDC // 2 and KO_NOROPE == 0:
                                # RoPE pair: dims [0,DC) (previous slot) & [D/2, D/2+DC) (this)
                                offs_kr = gl.arange(0, BLOCK_N,
                                                    layout=gl.SliceLayout(1, k_layout))
                                kpos = tile * BLOCK_N + offs_kr
                                cs_row = CS + gl.minimum(kpos, kv_len).to(gl.int64) * stride_cs
                                cosv = gl.load(cs_row[:, None] + ko_d[None, :]).to(gl.float32)
                                sinv = gl.load(cs_row[:, None] + (HEAD_DIM // 2 + ko_d)[None, :]
                                               ).to(gl.float32)
                                w_lo = gl.load(KW + ko_d).to(gl.float32)
                                w_hi = gl.load(KW + HEAD_DIM // 2 + ko_d).to(gl.float32)
                                b_lo = ring[(c - 1) % RING].load(k_layout).to(gl.float32) \
                                    * w_lo[None, :]
                                b_hi = rslot.load(k_layout).to(gl.float32) * w_hi[None, :]
                                k_lo = (b_lo * cosv - b_hi * sinv).to(CDT)
                                kc = (b_hi * cosv + b_lo * sinv).to(CDT)
                                sT = warpgroup_mma(k_lo, q_smem.index(0).permute((1, 0)), sT,
                                                   is_async=True)
                                res = warpgroup_mma_wait(1, deps=(sT, k_prev))
                                sT = res[0]
                                k_prev = k_lo
                                sT = warpgroup_mma(kc, q_smem.index(NDC // 2).permute((1, 0)),
                                                   sT, is_async=True)
                            else:
                                kc = rslot.load(k_layout).to(CDT)   # w folded into Q
                                sT = warpgroup_mma(kc, q_smem.index(_dord(c, NDC)).permute((1, 0)),
                                                   sT, use_acc=c > 0, is_async=True)
                            if c > 0:
                                res = warpgroup_mma_wait(1, deps=(sT, k_prev))
                                sT = res[0]
                            k_prev = kc
                    else:
                        kc = rslot.load(k_layout)
                        if KV_FP8:
                            kc = kc.to(CDT)
                        sT = warpgroup_mma(kc, q_smem.index(c).permute((1, 0)), sT,
                                           use_acc=c > 0, is_async=True)
                        if c > 0:
                            # retires chunk c-1's wgmma, so its register operand may be reused
                            res = warpgroup_mma_wait(1, deps=(sT, k_prev))
                            sT = res[0]
                        k_prev = kc
                    if c == NDC - 1:
                        sT = warpgroup_mma_wait(0, deps=(sT, k_prev))[0]
                        # ---------------- online softmax over the 128 keys --------------
                        sT = sT * qk_scale
                        if tile >= tile_mid:
                            # causal (bottom-right) mask; keys >= kv_len only reach padded rows
                            # (t >= q_len, never stored) and were zero-filled by the copy.
                            offs_kn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, s_layout))
                            offs_sm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, s_layout))
                            rel = ctx + t0 - tile * BLOCK_N
                            ok = offs_kn[:, None] <= (rel + offs_sm // G)[None, :]
                            sT = gl.where(ok, sT, float("-inf"))
                        m_new = gl.maximum(m_i, gl.max(sT, axis=0))
                        m_use = gl.where(m_new == float("-inf"), 0.0, m_new)
                        pT = gl.exp2(sT - m_use[None, :])
                        alpha_o = gl.convert_layout(gl.exp2(m_i - m_use), row_o)
                        m_i = m_new
                        p_smem.store(pT.to(CDT))
                        gl.barrier()
                        p = p_smem.permute((1, 0)).load(p_layout)
                        # row sums of the (rounded) P that is actually multiplied into O
                        l_i = l_i * alpha_o + gl.convert_layout(
                            gl.sum(p.to(gl.float32), axis=1), row_o)
                        scaled = ()
                        for i in gl.static_range(NDC):
                            scaled = scaled + (accs[i] * alpha_o[:, None],)
                        accs = scaled
                elif KV_FP8:
                    # ---------------- FP8 V: dequant chunk v, PV wgmma of chunk v-1 --------
                    # (inline `c - NDC` = v: a named temporary would stop being constexpr)
                    v8 = rslot.load(v_layout)
                    v16[(c - NDC) % NVB].store(v8.to(CDT))
                    fence_async_shared()         # generic-proxy writes -> wgmma (async proxy)
                    if c > NDC:
                        # v16[v-1] was written + fenced last step; this step's barrier made
                        # it visible to both warpgroups.
                        acc_new = warpgroup_mma(p, v16[(c - NDC - 1) % NVB], accs[c - NDC - 1],
                                                is_async=True)
                        if c == NDC + 1:
                            res = warpgroup_mma_wait(1, deps=(acc_new, p))
                            accs = (res[0],) + accs[1:]
                        else:
                            # keeps <= 1 PV wgmma in flight: with 3 staging buffers the one
                            # rewritten next step is no longer read.
                            res = warpgroup_mma_wait(1, deps=(acc_new, accs[c - NDC - 2], p))
                            accs = (accs[:c - NDC - 2] + (res[1], res[0]) + accs[c - NDC:])
                        p = res[len(res) - 1]
                    if c == NCH - 1:
                        gl.barrier()
                        acc_new = warpgroup_mma(p, v16[(c - NDC) % NVB], accs[NDC - 1],
                                                is_async=True)
                        res = warpgroup_mma_wait(0, deps=(acc_new, accs[NDC - 2], p))
                        accs = accs[:NDC - 2] + (res[1], res[0])
                else:
                    # ---------------- BF16/FP16 V: PV wgmma straight from the ring slot ----
                    fence_async_shared()         # cp.async (generic proxy) -> wgmma
                    acc_new = warpgroup_mma(p, rslot, accs[c - NDC], is_async=True)
                    if c == NDC:
                        res = warpgroup_mma_wait(1, deps=(acc_new, p))
                        accs = (res[0],) + accs[1:]
                    else:
                        res = warpgroup_mma_wait(1, deps=(acc_new, accs[c - NDC - 1], p))
                        accs = (accs[:c - NDC - 1] + (res[1], res[0]) + accs[c - NDC + 1:])
                    p = res[len(res) - 1]
                    if c == NCH - 1:
                        res = warpgroup_mma_wait(0, deps=(accs[NDC - 1], p))
                        accs = accs[:NDC - 1] + (res[0],)
            pages_t = pages_t1
            pages_t1 = pages_t2

        cp.wait_group(0)

        # ---- epilogue ------------------------------------------------------------------------
        l_ok = l_i > 0
        inv_o = gl.where(l_ok, o_scale / gl.where(l_ok, l_i, 1.0), 0.0)
        offs_or = gl.arange(0, BLOCK_M, layout=row_o)
        offs_od = gl.arange(0, DC, layout=gl.SliceLayout(0, o_layout))
        o_tok = (q_start + t0 + offs_or // G).to(gl.int64)
        o_head = h * G + offs_or % G
        o_ok = (t0 + offs_or // G) < q_len
        if SPLIT:
            m_o = gl.convert_layout(m_i, row_o)
            lse = gl.where(l_ok, m_o + gl.log2(gl.where(l_ok, l_i, 1.0)), float("-inf"))
            # both warpgroups hold identical row statistics; the duplicate stores are benign
            gl.store(LSE + split.to(gl.int64) * stride_ls + o_tok * stride_lt + o_head, lse,
                     mask=o_ok)
            o_rows = (OP + split.to(gl.int64) * stride_ops + o_tok * stride_opt
                      + o_head * stride_oph)
            for c in gl.static_range(NDC):
                gl.store(o_rows[:, None] + (c * DC + offs_od)[None, :],
                         accs[c] * inv_o[:, None], mask=o_ok[:, None])
        else:
            o_rows = O + o_tok * stride_ot + o_head * stride_oh
            for c in gl.static_range(NDC):
                gl.store(o_rows[:, None] + (c * DC + offs_od)[None, :],
                         (accs[c] * inv_o[:, None]).to(O.dtype.element_ty), mask=o_ok[:, None])


# ============================================================================================
# Decode variant: few query rows (BLOCK_M = 16 or 32 = q tokens x G heads, padded).
# S^T = K Q^T exactly as above (N = BLOCK_M).  PV is computed transposed, O^T += V^T P^T, so the
# wgmma M dimension is the head dim (64 per chunk) instead of padded query rows: the O
# accumulator is 512 x BLOCK_M (2*BLOCK_M regs/thread instead of 128) and nothing spills.
# FP8 KV only; K_ONLY 0 (regular K/V caches) or 2 (K-only sidecar rows).
# ============================================================================================
# runtime-varying ints: no ==1 / %16 specialization (no mid-inference recompiles)
@gluon.jit(do_not_specialize=["num_m_blocks", "num_splits"])
def _attn512_gluon_dec_kernel(
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
    KW, CS, stride_cs,
    G: gl.constexpr, HEAD_DIM: gl.constexpr, DC: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, PAGE_SIZE: gl.constexpr,
    RING: gl.constexpr, LAG: gl.constexpr,
    KV_FP8: gl.constexpr, SPLIT: gl.constexpr,
    HAS_KDS: gl.constexpr, HAS_VDS: gl.constexpr,
    K_ONLY: gl.constexpr = 0, PAIR: gl.constexpr = 0,
):
    # PAIR=1: one cp.async wait + CTA barrier per 2 chunks (8 per tile instead of 16); V is staged in pairs.
    NDC: gl.constexpr = HEAD_DIM // DC
    NCH: gl.constexpr = 2 * NDC
    PF: gl.constexpr = RING - LAG
    TPB: gl.constexpr = BLOCK_M // G
    NVB: gl.constexpr = 3 + PAIR
    gl.static_assert(PAIR == 0 or LAG >= 2, "PAIR needs LAG >= 2 (two slots refilled per barrier)")
    gl.static_assert(BLOCK_N == 128, "BLOCK_N must be 128 (2 warpgroups x 64 keys)")
    gl.static_assert(gl.num_warps() == 8, "num_warps must be 8")
    gl.static_assert(NCH % RING == 0, "RING must divide 2 * HEAD_DIM / DC")
    gl.static_assert(PF >= 1, "RING must exceed LAG")
    # LAG 0: chunk c+RING would be cp.async'd into the slot this step is about to ld.shared (WAR race)
    gl.static_assert(LAG >= 1, "decode kernel: LAG must be >= 1")
    gl.static_assert(PAIR == 0 or PF >= 2, "PAIR needs RING - LAG >= 2 (wait_group(PF - 2))")
    gl.static_assert(KV_FP8 == 1, "decode kernel: fp8 KV only")
    gl.static_assert(K_ONLY != 1, "decode kernel: K_ONLY must be 0 or 2")

    KV_T: gl.constexpr = K.dtype.element_ty
    CDT: gl.constexpr = gl.float16
    VEC: gl.constexpr = _vec(KV_T)

    s_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[8, 1], instr_shape=[16, BLOCK_M, 16])
    k_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=s_layout, k_width=2)
    o_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 2], instr_shape=[16, BLOCK_M // 2, 16])
    col_o: gl.constexpr = gl.SliceLayout(0, o_layout)
    cp_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, VEC], threads_per_warp=[32 // (DC // VEC), DC // VEC],
        warps_per_cta=[8, 1], order=[1, 0])
    v_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16], threads_per_warp=[32 // (DC // 16), DC // 16],
        warps_per_cta=[8, 1], order=[1, 0])
    q_ld_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[32 // (DC // 8), DC // 8],
        warps_per_cta=[8, 1], order=[1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, cp_layout)

    pid = gl.program_id(0)
    split = gl.program_id(1)
    m_block = pid % num_m_blocks
    bh = pid // num_m_blocks
    h = bh % num_kv_heads
    b = bh // num_kv_heads

    q_start = gl.load(CU_Q + b)
    q_len = gl.load(CU_Q + b + 1) - q_start
    t0 = m_block * TPB
    if t0 < q_len:
        kv_len = gl.load(SEQ_K + b)
        ctx = kv_len - q_len
        t_last = gl.minimum(t0 + TPB, q_len) - 1
        kv_end = ctx + t_last + 1
        n_tiles = gl.cdiv(kv_end, BLOCK_N)
        if SPLIT:
            tiles_per = gl.cdiv(n_tiles, num_splits)
            tile_lo = gl.minimum(split * tiles_per, n_tiles)
            tile_hi = gl.minimum(tile_lo + tiles_per, n_tiles)
        else:
            tile_lo = n_tiles * 0
            tile_hi = n_tiles
        last_pg = (kv_len - 1) // PAGE_SIZE

        q_sl: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, DC], CDT)
        ring_sl: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, DC], KV_T)
        v16_sl: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, DC], CDT)
        p_sl: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_N, BLOCK_M], CDT)
        q_smem = gl.allocate_shared_memory(CDT, [NDC, BLOCK_M, DC], q_sl)
        ring = ()
        for i in gl.static_range(RING):
            ring = ring + (gl.allocate_shared_memory(KV_T, [BLOCK_N, DC], ring_sl),)
        v16 = ()
        for i in gl.static_range(NVB):
            v16 = v16 + (gl.allocate_shared_memory(CDT, [BLOCK_N, DC], v16_sl),)
        p_smem = gl.allocate_shared_memory(CDT, [BLOCK_N, BLOCK_M], p_sl)

        bt_row = BT + b.to(gl.int64) * stride_bt
        k_head = K + h * stride_kh
        v_head = V + h * stride_vh

        pages_t = _fetch_pages(bt_row, tile_lo, last_pg, PAGE_SIZE, BLOCK_N, row_layout)
        pages_t1 = _fetch_pages(bt_row, tile_lo + 1, last_pg, PAGE_SIZE, BLOCK_N, row_layout)
        for cc in gl.static_range(PF):
            _issue(ring, cc, tile_lo, pages_t, pages_t1, k_head, v_head, kv_len, tile_hi,
                   stride_kp, stride_kt, stride_vp, stride_vt,
                   RING, NDC, PAGE_SIZE, BLOCK_N, DC, cp_layout, K_ONLY)

        offs_qr = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, q_ld_layout))
        offs_qd = gl.arange(0, DC, layout=gl.SliceLayout(0, q_ld_layout))
        q_tok = (q_start + t0 + offs_qr // G).to(gl.int64)
        q_rows = Q + q_tok * stride_qt + (h * G + offs_qr % G) * stride_qh
        q_ok = (t0 + offs_qr // G) < q_len
        for c in gl.static_range(NDC):
            qc = gl.load(q_rows[:, None] + (c * DC + offs_qd)[None, :], mask=q_ok[:, None],
                         other=0.0)
            if K_ONLY:
                if c % (NDC // 2) != 0:
                    wq = gl.load(KW + c * DC + offs_qd).to(gl.float32)
                    qc = qc.to(gl.float32) * wq[None, :]
            q_smem.index(c).store(qc.to(CDT))
        fence_async_shared()
        gl.barrier()

        qk_scale = sm_scale_log2
        if HAS_KDS:
            qk_scale = qk_scale * gl.load(KDS + b * stride_kds_b + h * stride_kds_h)
        o_scale = 1.0
        if HAS_VDS:
            o_scale = gl.load(VDS + b * stride_vds_b + h * stride_vds_h)

        m_i = gl.full([BLOCK_M], float("-inf"), gl.float32, layout=gl.SliceLayout(0, s_layout))
        l_i = gl.zeros([BLOCK_M], gl.float32, layout=col_o)
        tile_mid = (ctx + t0 + 1) // BLOCK_N
        accs = ()
        for c in gl.static_range(NDC):
            accs = accs + (gl.zeros([DC, BLOCK_M], gl.float32, layout=o_layout),)

        for tile in range(tile_lo, tile_hi):
            pages_t2 = _fetch_pages(bt_row, tile + 2, last_pg, PAGE_SIZE, BLOCK_N, row_layout)
            sT = gl.zeros([BLOCK_N, BLOCK_M], gl.float32, layout=s_layout)
            for c in gl.static_range(NCH):
                if PAIR:
                    if c % 2 == 0:
                        cp.wait_group(PF - 2)   # chunks <= c + 1 landed
                        gl.barrier()
                        _issue(ring, c + PF, tile, pages_t, pages_t1, k_head, v_head, kv_len, tile_hi,
                               stride_kp, stride_kt, stride_vp, stride_vt,
                               RING, NDC, PAGE_SIZE, BLOCK_N, DC, cp_layout, K_ONLY)
                        _issue(ring, c + PF + 1, tile, pages_t, pages_t1, k_head, v_head, kv_len, tile_hi,
                               stride_kp, stride_kt, stride_vp, stride_vt,
                               RING, NDC, PAGE_SIZE, BLOCK_N, DC, cp_layout, K_ONLY)
                else:
                    cp.wait_group(PF - 1)
                    gl.barrier()
                    _issue(ring, c + PF, tile, pages_t, pages_t1, k_head, v_head, kv_len, tile_hi,
                           stride_kp, stride_kt, stride_vp, stride_vt,
                           RING, NDC, PAGE_SIZE, BLOCK_N, DC, cp_layout, K_ONLY)
                rslot = ring[c % RING]
                if c < NDC:
                    kc = rslot.load(k_layout).to(CDT)
                    sT = warpgroup_mma(kc, q_smem.index(c).permute((1, 0)), sT,
                                       use_acc=c > 0, is_async=True)
                    if c > 0:
                        res = warpgroup_mma_wait(1, deps=(sT, k_prev))
                        sT = res[0]
                    k_prev = kc
                    if c == NDC - 1:
                        sT = warpgroup_mma_wait(0, deps=(sT, k_prev))[0]
                        sT = sT * qk_scale
                        if tile >= tile_mid:
                            offs_kn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, s_layout))
                            offs_sm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, s_layout))
                            rel = ctx + t0 - tile * BLOCK_N
                            ok = offs_kn[:, None] <= (rel + offs_sm // G)[None, :]
                            sT = gl.where(ok, sT, float("-inf"))
                        m_new = gl.maximum(m_i, gl.max(sT, axis=0))
                        m_use = gl.where(m_new == float("-inf"), 0.0, m_new)
                        pT = gl.exp2(sT - m_use[None, :]).to(CDT)
                        alpha_o = gl.convert_layout(gl.exp2(m_i - m_use), col_o)
                        m_i = m_new
                        p_smem.store(pT)
                        fence_async_shared()   # P^T (generic proxy) -> PV wgmma (async proxy)
                        l_i = l_i * alpha_o + gl.convert_layout(
                            gl.sum(pT.to(gl.float32), axis=0), col_o)
                        scaled = ()
                        for i in gl.static_range(NDC):
                            scaled = scaled + (accs[i] * alpha_o[None, :],)
                        accs = scaled
                elif PAIR:
                    # pairs (c, c+1) at even c, right after the barrier: O^T += V^T P^T for the previous
                    # pair (staged before the barrier), then stage this pair into the other 2 buffers.
                    if c % 2 == 0:
                        if c >= NDC + 2:
                            a0 = warpgroup_mma(v16[(c - NDC - 2) % NVB].permute((1, 0)), p_smem,
                                               accs[c - NDC - 2], is_async=True)
                            a1 = warpgroup_mma(v16[(c - NDC - 1) % NVB].permute((1, 0)), p_smem,
                                               accs[c - NDC - 1], is_async=True)
                        va = rslot.load(v_layout)
                        v16[(c - NDC) % NVB].store(va.to(CDT))
                        vb = ring[(c + 1) % RING].load(v_layout)
                        v16[(c - NDC + 1) % NVB].store(vb.to(CDT))
                        fence_async_shared()
                        if c >= NDC + 2:
                            res = warpgroup_mma_wait(0, deps=(a0, a1))
                            accs = accs[:c - NDC - 2] + (res[0], res[1]) + accs[c - NDC:]
                    if c == NCH - 1:
                        gl.barrier()
                        a0 = warpgroup_mma(v16[(NDC - 2) % NVB].permute((1, 0)), p_smem,
                                           accs[NDC - 2], is_async=True)
                        a1 = warpgroup_mma(v16[(NDC - 1) % NVB].permute((1, 0)), p_smem,
                                           accs[NDC - 1], is_async=True)
                        res = warpgroup_mma_wait(0, deps=(a0, a1))
                        accs = accs[:NDC - 2] + (res[0], res[1])
                else:
                    # FP8 V chunk v: dequant into v16[v % NVB]; O^T_{v-1} += V_{v-1}^T P^T
                    v8 = rslot.load(v_layout)
                    v16[(c - NDC) % NVB].store(v8.to(CDT))
                    fence_async_shared()
                    if c > NDC:
                        acc_new = warpgroup_mma(v16[(c - NDC - 1) % NVB].permute((1, 0)), p_smem,
                                                accs[c - NDC - 1], is_async=True)
                        if c == NDC + 1:
                            res = warpgroup_mma_wait(1, deps=(acc_new, accs[1]))
                            accs = (res[0], res[1]) + accs[2:]
                        else:
                            res = warpgroup_mma_wait(1, deps=(acc_new, accs[c - NDC - 2]))
                            accs = (accs[:c - NDC - 2] + (res[1], res[0]) + accs[c - NDC:])
                    if c == NCH - 1:
                        gl.barrier()
                        acc_new = warpgroup_mma(v16[(c - NDC) % NVB].permute((1, 0)), p_smem,
                                                accs[NDC - 1], is_async=True)
                        res = warpgroup_mma_wait(0, deps=(acc_new, accs[NDC - 2]))
                        accs = accs[:NDC - 2] + (res[1], res[0])
            pages_t = pages_t1
            pages_t1 = pages_t2

        cp.wait_group(0)

        l_ok = l_i > 0
        inv_o = gl.where(l_ok, o_scale / gl.where(l_ok, l_i, 1.0), 0.0)
        offs_od = gl.arange(0, DC, layout=gl.SliceLayout(1, o_layout))
        offs_om = gl.arange(0, BLOCK_M, layout=col_o)
        o_tok = (q_start + t0 + offs_om // G).to(gl.int64)
        o_head = h * G + offs_om % G
        o_ok = (t0 + offs_om // G) < q_len
        if SPLIT:
            m_o = gl.convert_layout(m_i, col_o)
            lse = gl.where(l_ok, m_o + gl.log2(gl.where(l_ok, l_i, 1.0)), float("-inf"))
            gl.store(LSE + split.to(gl.int64) * stride_ls + o_tok * stride_lt + o_head, lse,
                     mask=o_ok)
            o_cols = (OP + split.to(gl.int64) * stride_ops + o_tok * stride_opt
                      + o_head * stride_oph)
            for c in gl.static_range(NDC):
                gl.store(o_cols[None, :] + (c * DC + offs_od)[:, None],
                         accs[c] * inv_o[None, :], mask=o_ok[None, :])
        else:
            o_cols = O + o_tok * stride_ot + o_head * stride_oh
            for c in gl.static_range(NDC):
                gl.store(o_cols[None, :] + (c * DC + offs_od)[:, None],
                         (accs[c] * inv_o[None, :]).to(O.dtype.element_ty), mask=o_ok[None, :])
