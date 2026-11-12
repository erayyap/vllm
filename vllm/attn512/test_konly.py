#!/usr/bin/env python3
"""K-only cache (Gemma 4 attention_k_eq_v global layers) for attn512: correctness + speed.

The cache holds only b = v_norm(k_proj x) (= V, fp8); K = rope(k_norm_weight * b) is rebuilt
inside the Gluon kernel.  Reference: explicit fp32 K cache built in PyTorch, reference_paged.

  python3 test_konly.py            # correctness (ragged, pages 64/256, splits 1/3/auto) + bench
  python3 test_konly.py --no-bench
"""
import argparse, math, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel as K  # noqa: E402
from bench import time_fn  # noqa: E402

FP8 = torch.float8_e4m3fn
D, HQ, HKV = 512, 32, 4


def rope_cache(max_pos, base=1e6, D=D, rope_angles=64, dtype=torch.float32, device="cuda"):
    """vLLM Gemma4RotaryEmbedding (proportional) cos_sin_cache: [max_pos, D] = cos(D/2) | sin(D/2)."""
    inv = 1.0 / (base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / D))
    inv = torch.cat([inv, torch.zeros(D // 2 - rope_angles)])
    t = torch.arange(max_pos, dtype=torch.float)
    f = torch.einsum("i,j->ij", t, inv)
    return torch.cat([f.cos(), f.sin()], dim=-1).to(dtype).to(device)


def neox_rope(x, cs, pos):
    """x [..., T, H, D] fp32, pos [T] -> rotate_half over the full head (vLLM forward_static)."""
    c = cs[pos].float()
    cos, sin = c[:, : D // 2][:, None, :], c[:, D // 2:][:, None, :]
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def make(q_lens, kv_lens, page, seed=0, device="cuda", cs_dtype=torch.float32):
    g = torch.Generator(device=device).manual_seed(seed)
    B = len(q_lens)
    npg = [math.ceil(k / page) for k in kv_lens]
    tot = sum(npg) + 5
    perm = torch.randperm(tot, generator=g, device=device).cpu()
    bt = torch.zeros(B, max(npg) + 2, dtype=torch.int32)
    o = 0
    for b, n in enumerate(npg):
        bt[b, :n] = perm[o:o + n].int()
        bt[b, n:] = 10 ** 6 + b
        o += n
    cache = torch.empty(tot, page, HKV, D, dtype=FP8, device=device)
    for i in range(0, tot, 64):                                   # chunked: B=20 x 30k is GBs
        x = torch.randn(min(64, tot - i), page, HKV, D, generator=g, device=device)
        cache[i:i + 64] = (x / x.pow(2).mean(-1, keepdim=True).sqrt()).to(FP8)  # v_norm, scale 1
    w = (1.0 + 0.4 * torch.randn(D, generator=g, device=device)).to(torch.bfloat16)  # k_norm weight
    q = torch.randn(sum(q_lens), HQ, D, generator=g, device=device)
    q = (q / q.pow(2).mean(-1, keepdim=True).sqrt() * 0.12).to(torch.bfloat16)
    cu = torch.tensor([0] + list(__import__("itertools").accumulate(q_lens)), dtype=torch.int32)
    sk = torch.tensor(kv_lens, dtype=torch.int32)
    cs = rope_cache(max(kv_lens) + 256, dtype=cs_dtype, device=device)
    return dict(q=q, cache=cache, w=w, cs=cs, bt=bt.to(device), cu=cu.to(device), sk=sk.to(device),
                max_q=max(q_lens), max_k=max(kv_lens), page=page, npg=npg, kv_lens=kv_lens)


def explicit_k(p):
    """fp32 K cache: K[page, slot] = rope(w * b, position of that slot in its sequence)."""
    bdeq = p["cache"].float()
    kc = torch.zeros_like(bdeq)
    w = p["w"].float()
    for b, kv in enumerate(p["kv_lens"]):
        pages = p["bt"][b, : p["npg"][b]].long()
        rows = bdeq[pages].reshape(-1, HKV, D)[:kv] * w
        pos = torch.arange(kv, device=rows.device)
        kr = neox_rope(rows, p["cs"], pos)
        full = torch.zeros(len(pages) * p["page"], HKV, D, device=rows.device)
        full[:kv] = kr
        kc[pages] = full.reshape(len(pages), p["page"], HKV, D)
    return kc, bdeq


def run_konly(p, ns=None, out=None):
    one = torch.ones(1, device="cuda")
    return K.attn512_paged(p["q"], p["cache"], p["cache"], p["bt"], p["cu"], p["sk"], p["max_q"],
                           p["max_k"], 1.0, k_descale=one, v_descale=one, num_splits=ns, out=out,
                           k_norm_weight=p["w"], cos_sin_cache=p["cs"])


def check(name, q_lens, kv_lens, page, cs_dtype=torch.float32):
    p = make(q_lens, kv_lens, page, cs_dtype=cs_dtype)
    kc, vc = explicit_k(p)
    ref = K.reference_paged(p["q"], kc, vc, p["bt"], p["cu"], p["sk"], 1.0)
    ok = True
    for ns in (1, 3, None):
        out = run_konly(p, ns)
        torch.cuda.synchronize()
        err = float((out.float() - ref).norm() / ref.norm())
        good = bool(torch.isfinite(out).all()) and err < 2e-2
        ok &= good
        print(f"  {name:28s} page={page:3d} splits={ns}: rel_l2={err:.2e} {'ok' if good else 'FAIL'}",
              flush=True)
    return ok


def make_sidecar(p, vllm_pad=None):
    """Sidecar layout: rows [b (512) | rope(w*b) dims 0..63 | dims 256..319] (640 fp8). Returns (p2, kref, vref).
    vllm_pad=N: vLLM's physical layout instead -- [blocks, Hkv, page, 640] (head-major), each block padded
    by N extra token rows (page_size_padded), viewed as [blocks, page, Hkv, 640] via transpose."""
    kc, vc = explicit_k(p)
    side = torch.cat([kc[..., 0:64], kc[..., 256:320]], -1).to(FP8)
    rows = torch.cat([p["cache"].view(torch.uint8), side.view(torch.uint8)], -1).view(FP8)
    if vllm_pad is not None:  # vllm_pad = bytes of padding after each block's real page
        nb, pg = rows.shape[0], rows.shape[1]
        real = HKV * pg * (D + 128)
        buf = torch.zeros(nb, real + vllm_pad, dtype=torch.uint8, device=rows.device)
        blk = buf[:, :real].view(nb, HKV, pg, D + 128)
        blk.copy_(rows.view(torch.uint8).transpose(1, 2))
        rows = blk.view(FP8).transpose(1, 2)
    kref = kc.clone()
    kref[..., 0:64] = side[..., :64].float()
    kref[..., 256:320] = side[..., 64:].float()
    p2 = dict(p); p2["side_rows"] = rows; p2["side_view"] = rows[..., :512]
    return p2, kref, vc


def run_sidecar(p, ns=None, out=None):
    one = torch.ones(1, device="cuda")
    return K.attn512_paged(p["q"], p["side_view"], p["side_view"], p["bt"], p["cu"], p["sk"], p["max_q"],
                           p["max_k"], 1.0, k_descale=one, v_descale=one, num_splits=ns, out=out,
                           k_norm_weight=p["w"], cos_sin_cache=None)


def check_sidecar(name, q_lens, kv_lens, page, vllm_pad=None):
    p, kref, vref = make_sidecar(make(q_lens, kv_lens, page), vllm_pad)
    ref = K.reference_paged(p["q"], kref, vref, p["bt"], p["cu"], p["sk"], 1.0)
    ok = True
    for ns in (1, 3, None):
        out = run_sidecar(p, ns)
        torch.cuda.synchronize()
        err = float((out.float() - ref).norm() / ref.norm())
        good = bool(torch.isfinite(out).all()) and err < 2e-2
        ok &= good
        print(f"  sidecar {name:20s} page={page:3d} splits={ns}: rel_l2={err:.2e} {'ok' if good else 'FAIL'}",
              flush=True)
    return ok


def bench(Bn=20, kv=30080):
    import importlib
    fa = None
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func as fa
    except Exception as e:
        print("no FA4:", e)
    for q in (1, 4, 16, 32):
        res = {}
        # K-only, page 256 (vLLM picks a larger global block when the page halves)
        for page in (64, 256):
            p = make([q] * Bn, [kv] * Bn, page)
            out = torch.empty_like(p["q"])
            res[f"konly_p{page}"] = time_fn(lambda: run_konly(p, None, out), 3, 20)
            del p
        # regular K+V gluon + FA4 on the interleaved [pages, page, Hkv, 2D] layout
        p = make([q] * Bn, [kv] * Bn, 64)
        kv2 = torch.cat([p["cache"].view(torch.uint8), p["cache"].view(torch.uint8)], -1).view(FP8)
        kcache, vcache = kv2.split(D, -1)
        one = torch.ones(1, device="cuda")
        out = torch.empty_like(p["q"])
        res["gluon_kv"] = time_fn(lambda: K.attn512_paged(
            p["q"], kcache, vcache, p["bt"], p["cu"], p["sk"], q, kv, 1.0, k_descale=one.expand(Bn, HKV),
            v_descale=one.expand(Bn, HKV), out=out), 3, 20)
        if fa is not None:
            best = None
            for ns in (0, 1, 2, 4, 8):
                try:
                    t = time_fn(lambda: fa(p["q"], kcache, vcache, q, p["cu"], kv, seqused_k=p["sk"],
                                           softmax_scale=1.0, causal=True, block_table=p["bt"],
                                           k_descale=one.expand(Bn, HKV), v_descale=one.expand(Bn, HKV),
                                           num_splits=ns, fa_version=4), 3, 20)
                    res[f"fa4_s{ns}"] = t
                    best = t if best is None else min(best, t)
                except Exception as e:
                    print("fa4 err", ns, e)
            res["fa4_best"] = best
        del p, kv2, kcache, vcache
        torch.cuda.empty_cache()
        print(f"B={Bn} q={q:2d} kv={kv}: " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--bench-only", action="store_true")
    a = ap.parse_args()
    if not a.bench_only:
        ok = True
        print("regular K+V path self-test:", K.selftest("gluon", FP8, 64, torch.device("cuda"), verbose=True))
        for page in (64, 256):
            ok &= check("ragged", [1, 16, 37, 4, 64], [700, 1500, 333, 4100, 2048], page)
        ok &= check("bf16 cos/sin table", [1, 8, 23], [900, 130, 5000], 256, cs_dtype=torch.bfloat16)
        ok &= check("long decode q=4", [4, 4], [8000, 12345], 256)
        for page in (64, 256):
            ok &= check_sidecar("ragged", [1, 16, 37, 4, 64], [700, 1500, 333, 4100, 2048], page)
        ok &= check_sidecar("long decode q=4", [4, 4], [8000, 12345], 256)
        for ql in (1, 2, 4):
            ok &= check_sidecar(f"dec q={ql}", [ql] * 5, [333, 8000, 16000, 129, 4097], 256)
        ok &= check_sidecar("dec mixed q", [1, 4, 2, 3], [5000, 64, 16000, 777], 64)
        # vLLM sidecar pages: 192 tokens, head-major, block padded 480 -> 512 KB
        ok &= check_sidecar("vllm192 dec", [1, 4, 2, 3], [5000, 191, 16000, 777], 192, vllm_pad=32768)
        ok &= check_sidecar("vllm192 prefill", [1, 16, 37, 64], [700, 1500, 333, 4100], 192, vllm_pad=32768)
        ok &= check("dec regular via konly1 path q=1", [1, 1], [3000, 900], 256)
        print("ALL PASSED" if ok else "SOME FAILED", flush=True)
    if not a.no_bench:
        bench()
