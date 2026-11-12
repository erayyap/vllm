# attn512: head_dim 512 paged attention for Gemma 4 global layers (H100)

Causal, paged, GQA attention (32 query heads, 4 KV heads, D=512, scale 1.0) for vLLM 0.33.0's
FLASH_ATTN backend on SM90.  K/V cache is FP8 e4m3 with per-tensor descales (page 64) or bf16.
It covers the "20 sessions × ~30k cached tokens, 16–64 new tokens per turn" prefill case,
where FA4 (CuTe DSL, SM90) reaches about 250–275 TFLOPS (5.1 ms/layer, or 4.6 ms with splits).

**Status: tested on the H100.** Correctness (`test_correctness.py`, `test_konly.py`), tuning, and the K-only
path end to end in vLLM (quality, burst, voice and capacity runs, see below).

| file | purpose |
|---|---|
| `kernel.py` | `attn512_paged(...)` public op, Triton kernel, split-KV combine kernel, config/tuning, PyTorch reference, self-test |
| `kernel_gluon.py` | fast path: Gluon (explicit-layout Triton) kernel |
| `test_correctness.py` | fp32 PyTorch reference tests (GPU, or the CPU interpreter for tiny shapes) |
| `bench.py` | ms/layer and TFLOPS vs vLLM FA4; `--tune` writes `attn512_tuned.json` |
| `attn512_vllm.py` | routing hook called by the patched vLLM backend |
| `vllm_patch/apply.py`, `revert.py`, `attn512.diff` | vLLM 0.33.0 patch (idempotent, `.orig` backup) |

## K-only cache for Gemma 4 global layers (`VLLM_KONLY=1|2`)

Gemma 4's 10 global layers use `attention_k_eq_v`: there is no `v_proj`, and K and V both come from `k_proj x`:

    b = rmsnorm(k_proj x)            # v_norm has no weight
    V = b
    K = rope(k_norm.weight * b)      # proportional RoPE: dims [0,64) paired with [256,320), others identity

So the cache only needs `b` (= V), plus at most the 128 RoPE'd dims of K. Both modes are exact up to rounding.

| mode | cache row per (token, KV head) | kernel | speed |
|---|---|---|---|
| `VLLM_KONLY=1` (pure) | `b`: 512 B fp8 | `K_ONLY=1`: rebuilds K in registers (weight multiply + RoPE of chunks 0/4 from `cos_sin_cache`; chunk order 1,2,3,0,4,..., lag >= 2) | ~1.6x slower than FA4 (RoPE on the critical path, register spills) |
| `VLLM_KONLY=2` (sidecar, **default**) | `b` (512) + `rope(w*b)` dims 0..63 + dims 256..319: 640 B | `K_ONLY=2`: the RoPE chunks are read from the sidecar, `w` is folded into Q for the 6 identity chunks: no RoPE math | FA4 speed (prefill), faster than FA4 for decode (below) |

The V phase re-reads the same `b` tile, which hits L2.

**Decode kernel** (`_attn512_gluon_dec_kernel`, used automatically when `max_query_len * 8 <= 32`, i.e. decode and
MTP verify with q <= 4; `ATTN512_DEC=0` disables it). The main kernel always computes 64 query rows (8 tokens x 8
heads), so q=1 wastes 7/8 of each tile and the 64x512 fp32 O accumulator pins it at 255 registers. The decode kernel
uses 16 or 32 rows and computes O transposed (O^T += V^T P^T, V^T from the fp16 staging buffer as the MN-major A
operand), so the accumulator is 8 x [64, R]. ms per layer, B=20, kv=30080, fp8 (H100, page 256):

| q per seq | dec kernel (sidecar) | main kernel (sidecar) | K+V gluon | FA4 best (K+V) |
|---|---|---|---|---|
| 1 | **0.88** | 1.27 | 1.09 | 1.30 |
| 4 (MTP verify) | **1.12** | 1.42 | 1.14 | 1.34 |

(Memory floor for q=1 is ~0.45 ms: 1.54 GB at 3.35 TB/s.) Knobs: `ATTN512_DEC_RING` (8), `ATTN512_DEC_LAG` (1),
`ATTN512_DEC_PAIR` (0; 1 = one cp.async wait + barrier per two chunks), `ATTN512_DEC_OVH` (split cost model).

- `kernel.attn512_paged(..., k_norm_weight=w, cos_sin_cache=cs)` (mode 1) or `cos_sin_cache=None` (mode 2; the
  cache view is `rows[..., :512]` of the 640-byte rows) with `k_cache = v_cache = the single cache`.
- `kernel.warmup_konly(...)`: compiles every variant the K-only path can hit (14) on the real cache layout. The
  vLLM patch calls it once before CUDA graph capture. `num_splits` / `num_m_blocks` / `num_seqs` are excluded from
  Triton's int specialization. Without both, a mid-call compile stalled all live sessions for 2.6 s.
- `test_konly.py`: correctness against an explicit fp32 K cache. It covers ragged batches, pages 64/192/256
  (192 with vLLM's padded head-major layout), splits 1/3/auto, bf16 or fp32 cos/sin, and decode q 1/2/4/mixed.
  It also benchmarks against the regular K+V kernel and FA4.
  - Measured on H100: rel L2 1.7e-3 everywhere, the same as the K+V path (bf16 output rounding).
  - Negative controls: without RoPE 1.06, without the k_norm weight 0.59.
- `kbench.py`, `../results/patches/decsweep.py`: kernel timing and the dec-kernel knob sweep.
- vLLM integration: `../results/patches/fixkonly.py`, `fixkonly_bs.py`, `fixkonly_warm.py` (all included in
  `gemma4_voice_vllm0.33.0.patch`).
  - `gemma4.py`: marks the k_eq_v layers (mode 1 passes V as both key and value).
  - `attention.py`: `FullAttentionSpec(head_size_v=0|128)`. Mode 2 uses 192-token global blocks: 2560 B/token
    never divides the 512 KB SWA page, and with 64-token blocks vLLM padded each 160 KB page to 512 KB (3.2x).
    That collapsed capacity and hung a 20 x 30k burst. 192 x 2560 = 480 KB, padded to 512 KB, is 6.25% waste.
  - `flash_attn.py`: writes V (and the sidecar) into the single cache and routes attention to the kernel, also
    under full CUDA graphs. K-only specs are exempt from the SM90 FA4-fp8 64-token kernel-block rule.
  - `spec_decode/gemma4.py`, `worker/gpu/spec_decode/gemma4/speculator.py`: MTP draft layers that share a K-only
    target layer use the same path.

## Commands on the H100 box

The commands below assume this directory was copied to `/root/attn512`. Any path works.

```bash
cd /root/attn512
python3 -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name())"
#   expect triton 3.8.x (the Gluon kernel is written against 3.8.0) and an H100

# 1) correctness (gluon + triton kernels, bf16 and fp8 KV, num_splits auto/1/3)
python3 test_correctness.py --quick          # a few min (JIT compiles dominate): small varlen, 2k prefill, mixed
python3 test_correctness.py                  # + B=20 x {q=1,16,32,64} x kv=30080 and a ragged B=20 batch
python3 test_correctness.py --vs-fa4 --quick # also prints FA4's own error vs the same reference
#   options: --impl gluon|triton, --kv fp8|bf16, --splits auto,1,3, --qk-fp8

# 2) performance vs FA4 (B=20, q in {16,32,64}, kv=30080; fp8/page 64 and bf16/page 128)
python3 bench.py
python3 bench.py --kv fp8 --q 32 --impl gluon   # single case

# 3) optional: tune (about 20-40 min, mostly JIT compiles). Writes attn512_tuned.json, which kernel.py loads automatically.
python3 bench.py --tune
python3 bench.py                                 # re-run with the tuned configs

# 4) vLLM integration
python3 vllm_patch/apply.py      # patches /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flash_attn.py
python3 vllm_patch/apply.py --check
VLLM_ATTN512=1 VLLM_ATTN512_VERBOSE=1 vllm serve <gemma-4-31b> --kv-cache-dtype fp8 ...
python3 vllm_patch/revert.py     # restores flash_attn.py from flash_attn.py.orig
```

`apply.py` writes the absolute path of this directory into the patched file. If you move
the directory afterwards, set `ATTN512_PATH=/new/path`. `attn512.diff` is the same change
for review. It assumes `/root/attn512`, or set `ATTN512_PATH`. If an anchor is not found
exactly once, `apply.py` stops without changing anything.

### Expected output

- **test_correctness.py** prints one line per case and ends with `ALL PASSED` (exit code 0):
  ```
  PASS B20-q32-kv30080   kv=fp8  impl=gluon  qk_fp8=0 splits=None max_abs=... max_rel=... rel_l2=...
  ```
  Typical errors: `rel_l2` about 3e-3 for fp8 KV (fp16 compute path) and about 4–6e-3 for bf16.
  The tolerances are `rel_l2` < 5e-3 for fp8 and < 1e-2 for bf16, and `max_rel` < 1e-2 for fp8
  and < 2e-2 for bf16 (`max_rel` is max abs error / max |ref|).
- **bench.py** prints a table with one row per kernel and shape. The note column shows each
  kernel's `rel_l2` vs FA4's output, which should be about 1e-3–1e-2:
  ```
  kv    page   q  kernel                  ms/layer  TFLOPS  note
  fp8     64  32  FA4(splits=..)             ~4.6    ~275
  fp8     64  32  attn512-gluon               ?       ?     vsFA4 rel_l2=...
  fp8     64  32  attn512-triton              ?       ?     vsFA4 rel_l2=...
  ```
  TFLOPS = 4·B·q·kv·32·512 / time.
- **vLLM with `VLLM_ATTN512_VERBOSE=1`** logs `attn512: head_size=512 prefill/mixed attention from ...`
  once per process, then `[attn512] routed: kv=torch.float8_e4m3fn page=64 max_q=...` for the
  first routed batch. The first routed batch in each process JIT-compiles the kernels (tens
  of seconds; the result is cached in `~/.triton/cache`). With `ATTN512_IMPL=auto`, that batch
  also runs a self-test of about 1 s.

## What gets routed (vLLM)

When `VLLM_ATTN512=1`, `FlashAttentionImpl.forward` sends a batch to attn512 only if every
condition holds:

- `head_size == 512`, and DCP is 1 (vLLM handles the DCP path before this point).
- `causal is True`, with no dynamic causal and no sliding window.
- No `mask_mod` (no multimodal-prefix or R-SWA mask), no sinks, no alibi and no softcap.
- `max_query_len >= VLLM_ATTN512_MIN_Q` (default 2).
- The SM90 device is not capturing a CUDA graph.

Everything else stays on FA4, including pure decode and every CUDA-graph-captured batch. If
attn512 raises, it disables itself for the rest of the process and FA4 takes over.

| env | default | meaning |
|---|---|---|
| `VLLM_ATTN512` | `0` | `1` enables the routing (patched vLLM only) |
| `VLLM_ATTN512_MIN_Q` | `2` | route batches with `max_query_len` ≥ this; `1` also routes decode |
| `ATTN512_IMPL` | `auto` | `auto`: Gluon if its one-off self-test passes, else Triton (`bench.py --tune` may record a per-dtype preference); `gluon`/`triton` force one |
| `ATTN512_SELFTEST` | `1` | `0` skips the auto-mode self-test |
| `ATTN512_CONFIG` | – | JSON overrides for every config, e.g. `'{"ring": 4}'` |
| `ATTN512_TUNED_JSON` | `./attn512_tuned.json` | tuned-config file |
| `ATTN512_PATH` | apply-time dir | where the patched vLLM finds `attn512_vllm.py` |

## Design

**One CTA per (sequence, KV head, 64 query rows).** The 64 rows are 64/G tokens × the G=8
query heads of one KV head (GQA packed into M, as in FlashMLA). Each K/V byte feeds 64 rows.
The M-blocks of a (sequence, head) pair are adjacent in the grid, so they share pages through L2.

### Gluon kernel (`kernel_gluon.py`, default)

8 warps (2 warpgroups) process 128-key tiles.

- **Scores.** It computes S^T = K·Q^T. With `warps_per_cta=[8,1]`, each warpgroup owns a
  different block of 64 keys, which is FlashMLA's per-warpgroup KV split with no duplicated
  work. K is the *register* A operand, so FP8 is dequantised to fp16 in registers. Q^T stays
  resident in SMEM.
- **Output.** The 64×512 fp32 O is split by columns over the two warpgroups, in eight [64,64]
  accumulators (128 registers per thread). P goes through SMEM once per tile and is then the
  register A operand of every PV wgmma.
- **Loads.** K and V stream as [128 keys × 64 dims] chunks through an 8-slot cp.async ring.
  The slots are separate SMEM allocations indexed statically, so there is exactly one CTA
  barrier per chunk. Page ids are per row, so any page size works; they are prefetched two
  tiles ahead.
- **Overlap.** The wgmmas are async with `wait(1)`: dequant of chunk c+1 overlaps the wgmma of
  chunk c. FP8 V is staged in a 3-deep fp16 buffer, with the PV wgmma one chunk behind.
- **FP8 descales.** k_descale folds into the softmax scale and v_descale into the output scale,
  as in FA4.
- **Softmax.** Online softmax in the exp2 domain. The causal mask is applied only on diagonal tiles.
- **Split-KV.** The causal tile range of each sequence is split evenly, followed by an LSE
  combine kernel. A wave-quantisation model picks the split count.

Offline sm_90a compile (ptxas):

| KV | SMEM | registers | spill | per-tile loop |
|---|---|---|---|---|
| fp8 | 197 KB | 255 | ~300 B | 96 HGMMA, 27 CTA barriers, ~1.9k SASS instructions |
| bf16 | 213 KB | 255 | ~150 B | — |

ptxas does not serialise the wgmmas (no C7514/C7515 warnings).

### Triton kernel (`kernel.py`, fallback)

The Triton kernel computes the same math with Triton's automatic pipeliner. It passes the
CPU-interpreter tests. Offline compiles show that Triton waits every wgmma immediately and
has many barriers per tile, so expect FA4-level speed or slower.

## Performance expectations (not measured)

For B=20, q=32, kv=30080, one layer is 1.26 TFLOP (task convention), so the 500 TFLOPS target
means ≤ 2.5 ms/layer.

- **Gluon, fp8.** At 128 keys × 64 rows per tile, the tensor-core floor is about 4.1k clocks per
  tile. SMEM traffic is about 5k clocks per tile. The softmax (exp2 on the SFU) is not
  overlapped with the GEMMs, which adds about 0.7k clocks. With 128 FLOP per L2 byte for fp8,
  L2→SM bandwidth (about 5–6 TB/s) caps the kernel near 600–700 TFLOPS. A realistic hope is
  **400–600 TFLOPS (2.1–3.2 ms/layer)**, against FA4's 4.6–5.1 ms.
- **Gluon, bf16.** At 64 FLOP per L2 byte it is L2-bound near 300–350 TFLOPS.
- **Triton fallback.** Probably 200–300 TFLOPS.

Run `bench.py` and `--tune` before relying on any of these numbers.

## Limitations and risks

- **The Gluon kernel has never run.** Its synchronisation was reasoned through and the compiled
  SASS was inspected: one cp.async group per chunk, `wait_group(PF-1)` followed by one barrier,
  wgmma waits, and proxy fences. A hang, wrong results or an illegal address on real hardware
  is possible.
  - `ATTN512_IMPL=auto` runs a reference self-test first and falls back to Triton. A crash
    from an illegal address cannot be caught; use `ATTN512_IMPL=triton` if that happens.
  - Gluon is experimental Triton API. A Triton version other than 3.8 may fail to import or
    compile it, and `auto` then falls back.
- **Gluon fp8 spills.** The kernel spills about 300 B per thread (mostly loop-invariant
  addresses, reloaded from L1).
- **Gluon config limits.** `block_n=128`, `dc=64`, 8 warps, and `ring` must divide 16.
- **No pure-decode speedup.** Decode stays on FA4 by default. With q_len=1, 56 of each
  CTA's 64 rows are padding.
- **Feature coverage.** No support for sinks, softcap, sliding windows, alibi, multimodal
  bidirectional masks, DCP, or `return_lse`. Those batches stay on FA4.
- **Memory.** With split-KV, fp32 partials of `[splits, tokens, 32, 512]` are allocated per call
  (about 84 MB at 640 tokens × 2 splits). This is one reason captured CUDA graphs are excluded.
- **QK_FP8.** The opt-in approximate mode (Triton kernel only) quantises Q per row to e4m3, and
  is off by default.
