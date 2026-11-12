"""Quick K-only kernel timing: K-only (RoPE / no-RoPE timing variant) vs the K+V gluon kernel. B=20, kv=30080."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_konly as T, kernel as K
from bench import time_fn
FP8 = torch.float8_e4m3fn
Bn, kv = 20, int(os.environ.get("KV", "30080"))
for q in [int(x) for x in os.environ.get("QS", "1,4,16,32").split(",")]:
    p = T.make([q] * Bn, [kv] * Bn, 256)
    out = torch.empty_like(p["q"])
    r = {}
    for nr in (0, 1):
        os.environ["ATTN512_KO_NOROPE"] = str(nr)
        r[f"konly{'_norope' if nr else ''}"] = time_fn(lambda: T.run_konly(p, None, out), 3, 20)
    os.environ["ATTN512_KO_NOROPE"] = "0"
    p2, _, _ = T.make_sidecar(p) if q <= 0 else (None, None, None)
    del p
    p = T.make([q] * Bn, [kv] * Bn, 256)
    rows = torch.cat([p["cache"].view(torch.uint8), p["cache"][..., :128].view(torch.uint8)], -1).view(FP8)
    p["side_view"] = rows[..., :512]
    os.environ["ATTN512_DEC"] = "0"
    r["sidecar"] = time_fn(lambda: T.run_sidecar(p, None, out), 3, 20)
    os.environ["ATTN512_DEC"] = "1"
    if q * 8 <= 32:
        r["sidecar_dec"] = time_fn(lambda: T.run_sidecar(p, None, out), 3, 20)
        for ns in (2, 3, 4, 6, 8, 12):
            r[f"dec_s{ns}"] = time_fn(lambda: T.run_sidecar(p, ns, out), 3, 20)
    del p, rows
    p = T.make([q] * Bn, [kv] * Bn, 64)
    kv2 = torch.cat([p["cache"].view(torch.uint8)] * 2, -1).view(FP8)
    kc, vc = kv2.split(512, -1)
    one = torch.ones(1, device="cuda")
    r["gluon_kv"] = time_fn(lambda: K.attn512_paged(p["q"], kc, vc, p["bt"], p["cu"], p["sk"], q, kv, 1.0,
                                                    k_descale=one, v_descale=one, out=out), 3, 20)
    del p, kv2, kc, vc; torch.cuda.empty_cache()
    print(f"q={q:2d}: " + "  ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
