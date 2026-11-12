# Gemma 4 31B voice sessions on one H100 (vLLM 0.33.0)

Goal: 20 concurrent call-center sessions on a single H100 80GB, with TTFT < 100 ms per caller turn.
- Each session keeps a unique context resident on the GPU: typically ~30k tokens, up to 64k (`max_model_len` 65536).
- KV cache stays FP8 or better.
- All numbers use Gemma's recommended sampling (temperature 1.0, top_k 64, top_p 0.95); nothing here is greedy.

**Status: met only for resident sessions in steady state; not met while calls join.**
- Cold-prefilling 20 × 30k fresh tokens within 100 ms is physically impossible (~600k tokens). Everything here
  assumes the sessions' contexts are already on the GPU.
- With all contexts loaded and no call joining, p50 and p95 stay under 100 ms at 28k context. At ~59k, p95 is
  105 ms (max 121).
- While a new call loads its context, live turns' p95 is 110–128 ms at 28k context and 210 ms at ~59k context:
  the target is missed.
- The candidate fix, load-split scheduling, is on by default in `server.py` but has never run on a GPU. All
  numbers in this README were measured without it (`VLLM_LOAD_SPLIT=0`).

Engine-level benchmark: with 20 live sessions, both presets keep voice-turn TTFT p95 at 62–78 ms and p50 at
44–52 ms.
- The worst single turn measured was 102 ms (QAT, stagger); every other max was 81–85 ms.
- In a burst, 20 simultaneous turns of up to 8 new tokens each also stay under 100 ms, at 30k or 63k context.
- Through the WebSocket server with continuous traffic (all 20 calls always talking), p50 is 63–67 ms and p95
  is 83–99 ms at 28k context. At ~59k context (QAT), p50 is 74 ms and p95 105 ms.
- While a new call joins, live turns' p95 rises to 110–128 ms (see Call churn).
- `server.py` refuses turns that would overflow a session's 64k, and refuses new calls whose context would not fit
  in the KV cache. Either overflow otherwise crashes vLLM 0.33 for all calls or stalls the joining call
  (see Overload).

The two presets differ in weights:

| preset | weights | MTP drafter | fits | voice TTFT p50 / p95 (stagger, partial) | decode tok/s per call |
|---|---|---|---|---|---|
| **fp8** (default; fidelity first) | `RedHatAI/gemma-4-31B-it-FP8-block` | `google/gemma-4-31B-it-assistant` | 16 × 60k (20 × ~48k) | 52 / 71, 51 / 76 ms | 99–107 |
| **qat** (capacity first) | `google/gemma-4-31B-it-qat-w4a16-ct` | `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant` | **20 × 63k** | 49 / 78, **44 / 62** ms | 112 |

"fits" is the engine's KV capacity. The requirement is 20 × 30k resident (64k is the per-call maximum), which both
presets meet, so `server.py` defaults to the fp8 preset. By default `server.py` admits new calls only up to 90% estimated usage
(`--kv-admit`), keeping the rest as headroom for live calls to grow.

The QAT checkpoint is Google's quantization-aware-trained int4 release. Its next-token distribution is
measurably further from the original BF16 model than the FP8-block checkpoint's (see Quality below).

## What is in here

| file | what |
|---|---|
| `server.py` | WebSocket session server. One resumable vLLM streaming-input request per call; STT partials are prefilled while the caller speaks; the final chunk triggers the reply. Env defaults = the best config. |
| `client_bench.py` | Load generator for `server.py`: N calls load their context, then talk (STT partials + final, random think time). `--joiners` adds calls that join while the others talk. `--seed` changes the contexts, which defeats the prefix cache. |
| `lsim.py` | Step-timing model behind the load-split estimate. |
| `../vllm/` (this fork) | All engine patches: `git log v0.33.0..` (or `4ba787e..`) lists them, and `git diff 4ba787e -- vllm/` is the combined patch. |
| `../vllm/attn512/` | The head_dim-512 Gluon attention kernel used for the K-only global layers, its tuned config and its tests (`test_konly.py`, `kbench.py`). `ATTN512_PATH` points vLLM at another copy. |

## Engine patches

| patch | env (server default) | effect |
|---|---|---|
| pause SWA fix (`scheduler._handle_stopped_request`) | always on | **Bug fix.** Paused sessions pinned stale sliding-window blocks; with 20×30k sessions the 7th session deadlocked waiting for KV. |
| detokenizer prime (`detokenizer.py`) | always on | Prime the incremental detokenizer with the last 32 prompt tokens instead of the whole 30k prompt. |
| stream-delta-resume (scheduler/output/states/model_runner) | `VLLM_STREAM_DELTA_RESUME=1`, `VLLM_SDR_ALLOW_SPEC=1` | A resumed session ships only its new tokens and KV blocks to the worker, not the full 30k history and ~10k block ids. |
| FA4 wave-aware splits (`flash_attn.py`) | `VLLM_FA4_WAVE_SPLITS=32` | Split-KV count for FA4 from a tail-aware wave model (K+V global layers; unused with K-only). |
| burst coalescing (`core.py`) | `VLLM_COALESCE_GAP_MS=3`, `VLLM_COALESCE_MAX_MS=30` | When idle, wait for the rest of a burst so simultaneous turns share one forward pass. |
| GIL switch interval (`core.py`) | `VLLM_SWITCH_INTERVAL_US=200` | Lower hand-off latency between the engine's input thread and the busy loop. |
| JIT lookahead (`core.py`) | `VLLM_JITLA=1` | Turns that arrive while a step is running join the very next step. Voice TTFT p50 68.6 → 51.0 ms. |
| pace (`scheduler.py`) | `VLLM_PACE_TPS=15` | Sessions already far ahead of TTS playback yield to sessions still racing to their first chunk. T25 369 → 301 ms with long replies. |
| **K-only global KV** (`gemma4.py`, `attention.py`, `flash_attn.py`, MTP speculator) | `VLLM_KONLY=2` | Gemma 4's global layers have K = rope(k_norm.w · V). The cache stores V plus the 128 RoPE'd K dims (640 B instead of 1024 B per token and head). Attention runs on `attn512` (decode kernel for q ≤ 4). |
| K-only pages (`attention.py`, `flash_attn.py`) | with `VLLM_KONLY=2` | 192-token global blocks. With 64-token blocks vLLM padded every page 3.2×, which hung a 20 × 30k burst. |
| K-only warmup (`flash_attn.py`) | with `VLLM_KONLY` | Compiles all attn512 variants before CUDA-graph capture. One mid-call Triton compile had stalled all sessions 2.6 s. |
| context-load chunk cap (engine arg) | `--long-prefill-cap 256` | A joining call's ~30k-token context is prefilled at most 256 tokens per step while other calls are active (vLLM `long_prefill_token_threshold`). Inert in steady state. |
| turn-first (`scheduler.py`) | `VLLM_TURN_FIRST_LONG=1024` | When a caller turn or STT partial (≤ 1024 tokens) is pending, context-load chunks sit that step out, so the turn's step is as fast as in steady state. Inert without joiners. |
| load-split (`scheduler.py`), **untested** | `VLLM_LOAD_SPLIT=1` | Context-load chunks never share a step with live calls' work: load-only and normal steps alternate while no turn is pending. Written after the GPU was returned; never run (see below). `VLLM_LOAD_SPLIT=0` restores the measured setup. |
| spec-plain (`scheduler.py`) | `VLLM_SPEC_PLAIN_STEPS=n` (off) | Run a newly admitted turn's step without draft verification. |
| timeline (`core.py`) | `VLLM_TL=/path/log` (off) | Per-step timing log (debug). |

Engine args (server.py): `kv_cache_dtype=fp8`, `max_model_len=65536`, `max_num_batched_tokens=8192`,
`gpu_memory_utilization=0.95`, `prefix_caching_hash_algo=xxhash`, `language_model_only=True`, CUDA graph capture
sizes 1..1280, MTP with 3 speculative tokens.

## Run

```bash
# install: see "Install" in the fork's top-level README.md; then, from gemma4_voice/:
# default: fp8 preset (RedHatAI/gemma-4-31B-it-FP8-block + google/gemma-4-31B-it-assistant, MTP 3)
python3 server.py --warmup
# qat preset: a bit faster (server p95 83 vs 93-99 ms), lower fidelity (KL 0.09 vs 0.015 to BF16)
python3 server.py --model google/gemma-4-31B-it-qat-w4a16-ct \
                  --spec-model google/gemma-4-31B-it-qat-q4_0-unquantized-assistant --warmup
python3 client_bench.py --calls 20 --turns 3                                   # steady state
python3 client_bench.py --calls 20 --turns 34 --joiners 6 --joiner-gap 15 --seed 1   # + call churn
python3 client_bench.py --calls 20 --turns 4 --context-chars 230000 --seed 2         # 20 x ~60k contexts
```
`--long-prefill-cap 512` loads a joining call's 28k context in ~8 s instead of ~12 s (default 256), at a higher
live-turn TTFT p95 during the load: 139 ms instead of 110 ms (see Call churn below). `--long-prefill-cap 0` restores vLLM's default chunking.

**QAT preset restarts:** vLLM 0.33 + `humming` crash in `profile_run` whenever the compiled graph comes from a
cache, i.e. on server starts after the first. vLLM's AOT cache gives `'str' object has no attribute 'num_bits'`;
Inductor's FX-graph cache gives `use_tma_as requires m-major input scales`. For int4 compressed-tensors
checkpoints, `server.py` sets `VLLM_DISABLE_COMPILE_CACHE=1`, `TORCHINDUCTOR_FX_GRAPH_CACHE=0` and
`TORCHINDUCTOR_AUTOGRAD_CACHE=0`, which forces a fresh compile: ~120 s to ready instead of ~45 s. The FP8 preset
is not affected and keeps its caches.

## Results (H100 80GB SXM, vLLM 0.33.0, recommended sampling)

### Voice traffic, 20 sessions × 30k resident, MTP k=3

`stagger`: caller turns arrive at random times. `partial`: the caller's words are streamed in as STT partials
while they speak. TTFT = end of the caller's turn to the first reply token. T5 / T25 = time to 5 / 25 tokens.

| config | mode | TTFT p50 / p95 / max (ms) | T5 p50 | T25 p50 | tok/s per call |
|---|---|---|---|---|---|
| FP8 weights, FP8 K+V (previous best) | stagger | 51.0 / 78.9 / — | — | 308 | 99 |
| | partial | 54.0 / 78.3 / — | — | 317 | — |
| **FP8 weights, K-only sidecar** | stagger | 52.1 / 70.9 / 82.8 | 110 | 294 | 107 |
| | partial | 50.6 / 76.0 / 81.5 | 113 | 310 | 99 |
| **QAT W4A16, K-only sidecar** | stagger | 49.3 / 77.5 / 101.7 | 96 | 282 | 112 |
| | partial | 44.2 / 62.0 / 85.0 | 93 | 279 | 112 |

### Burst: all 20 sessions send a turn at the same instant (TTFT ms, p50)

| config | context | 1 new token | 8 | 16 | 32 |
|---|---|---|---|---|---|
| FP8 weights, FP8 K+V | 30k | 60 | 68 | 90 | 126 |
| FP8 weights, K-only sidecar | 30k | 57 | 72 | 93 | 131 |
| QAT W4A16, K-only sidecar | 30k | 54 | 71 | 93 | 146 |
| QAT W4A16, K-only sidecar | **63k** | 62 | 84 | 118 | 192 |

QAT int4 GEMM: vLLM picks `HummingLinearKernel`. Forcing Marlin (`VLLM_DISABLED_KERNELS=HummingLinearKernel`)
is slower: 54 / 85 / 117 / 193 ms. Machete does not support the checkpoint's group size 32.

Cold prefill of 20 × 30k fresh tokens within 100 ms is physically impossible (~600k tokens). These numbers assume
resident sessions, which is what the server keeps.

### Capacity (20 sessions, how much resident context fits)

| global-layer KV | bytes / token / KV head | FP8 weights | QAT W4A16 weights |
|---|---|---|---|
| FP8 K+V | 1024 | 13 × 60k | — |
| K-only sidecar (`VLLM_KONLY=2`) | 640 (+6.25% page padding) | 16 × 60k | **20 × 63k** |
| pure K-only (`VLLM_KONLY=1`) | 512 | 20 × 60k, but ~1.6× slower attention (burst 83/93/133/224 ms) | — |

### Quality

KV precision: next-token KL and top-1 agreement are measured against BF16 KV with the same FP8-block weights.
They are teacher-forced on the reference reply over six ~30k-token Turkish contexts (576 positions). A
needle-in-haystack test runs at 30k and 60k (36 sampled retrievals).

| config | KL mean | top-1 agree | needles |
|---|---|---|---|
| FP8 K+V | 0.0149 | 97.4% | 36/36 |
| FP8 global + BF16 sliding-window KV | 0.0124 | 97.9% | 36/36 |
| **K-only sidecar, FP8 (default)** | 0.0153 | 97.2% | 36/36 |
| BF16 original weights + sidecar FP8 KV | 0.0229 | 96.5% | 36/36 |
| QAT W4A16 weights + sidecar FP8 KV | 0.0905 | 92.4% | 36/36 |

The FP8-block checkpoint stays very close to Google's BF16 original. The QAT model is about 4× further away:
it is a separately trained model, so this measures difference, not necessarily worse answers. Check it on your
own call transcripts before choosing the `qat` preset.

### Call churn: calls joining while others talk

In production, calls start and end all the time, and each new call loads a ~28k-token unique context: roughly
2.5 s of full-GPU prefill. Uncapped, vLLM prefills it in 8192-token chunks. Every step that carries a chunk takes
~0.5 s, so live callers' turns behind it wait seconds.

Test (`client_bench.py --calls 20 --turns 34 --joiners 6 --joiner-gap 15`): 20 live calls keep talking while a new
call joins every 15 s. Each joiner loads a fresh context, takes one turn and hangs up. That is about the arrival
rate of a 20-seat center with ~5-minute calls. Results are for the live calls' turns, FP8 preset, through
`server.py`:

| scheduling | TTFT p50 / p95 / max (ms) | T5 p95 | T25 p95 | joiner context load |
|---|---|---|---|---|
| default vLLM (8192-token chunks) | 63 / 118 / 2954 | 2233 | 2543 | 3.1 s |
| chunk cap 512 | 59 / 198 / 353 | 341 | 1025 | 5.2 s |
| chunk cap 1024 | 57 / 324 / 497 | 541 | 1574 | 4.2 s |
| turn-first + cap 512 | 60 / 139 / 240 | 202 | 587 | 8.1 s |
| **turn-first + cap 256 (server default)** | 64 / 110 / 207 | 190 | 589 | 11.8 s |
| turn-first + cap 128 | 66 / 105 / 278 | 158 | 483 | 21.4 s |

A turn cannot jump into a step that is already running. So during a load, TTFT is the rest of the current
step plus the turn's own step, and only short chunk steps keep that low. The cost is a slower context load for
the joining call; load it while the call rings. Cap 128 tightens the live tail a little more, but a joiner's 21 s load overlaps the next joiner's, and a new caller may finish their first sentence before the context is ready. The final default run (End-to-end server, below) measured 67 / 128 / 180 with cap 256. At a much higher join rate (one every 5 s, about 50% of the GPU
spent on context loads), no setting kept p95 under 200 ms.

### End-to-end server (`server.py` + `client_bench.py`)

These are the final numbers with the server defaults: 20 calls over WebSocket, 3 STT partials per turn, 80-token
replies, random think time of up to 2 s. TTFT here is server-side: from the final chunk arriving at the server to
the first reply token leaving the engine. It includes the server's asyncio and vLLM front-end time, which the
engine-level voice table above does not.

| preset | load | TTFT p50 / p95 / max (ms) | T5 p50 | T25 p50 / p95 | joiner context load |
|---|---|---|---|---|---|
| fp8 | 20 × 28k, steady | 65.4 / 93.2 / 128.6 | 96 | 337 / 423 | — |
| fp8 | 20 × 28k + a joiner every 15 s | 67.2 / 127.7 / 179.6 | 101 | 302 / 544 | 12.4 s |
| qat | 20 × 28k, steady | 63.0 / 83.1 / 94.1 | 99 | 328 / 401 | — |
| qat | 20 × 28k + a joiner every 15 s | 66.8 / 119.3 / 203.5 | 103 | 309 / 602 | 11.9 s |
| qat | **20 × ~59k**, steady | 73.6 / 104.9 / 121.1 | 113 | 358 / 556 | — |
| fp8 | 20 × 28k + a joiner every 15 s, final `server.py` (admission control on, same seed) | 64.2 / 110.6 / 159.0 | 99 | 306 / 554 | 12.7 s |
| fp8 | 15-min soak: 20 × 28k + 40 joiners, one every 15 s | 70.1 / 119.7 / 196.5 | 105 | 297 / 451 | 12.6 s (p95 14.2) |

No runtime JIT compiles and no errors occurred in any of these runs. The soak loaded and freed ~1.1M context tokens
(4,400 live turns), more than the KV cache holds at once, with no preemptions. Run-to-run spread is a few ms at p50 and
10–20 ms at p95: 60 turns is a small sample. Another FP8 steady run measured 67.2 / 98.6 / 109.6 ms.

JIT lookahead lead (`VLLM_JITLA_LEAD_MS`) A/B, FP8 steady, TTFT p50 / p95: lead 2 gave 67.2 / 98.6 (seed 65) and
62.0 / 86.0 (seed 67); lead 5 gave 62.4 / 91.7 and 68.3 / 83.9; lead 8 gave 64.3 / 85.3 (seed 65). No setting
is better beyond run-to-run noise, so the default stays at 2.

`VOICE_TRACE=1` prints a per-turn breakdown. The server's own share, from the WebSocket message to the engine
input, is 0.4 ms; everything else is vLLM (front end, waiting for the step in flight, the step itself). These
numbers sit above the engine-level voice table most likely because of the load shape (not verified):
`client_bench.py` keeps all 20 calls continuously busy, while `voice.py` runs rounds that end with a lull.

### Overload and long calls (server guards)

A QAT soak with 19 calls at ~59k context crashed the engine after ~4 minutes. The sessions had grown past 65,536
tokens, and an `AssertionError` in vLLM's `block_pool.cache_full_blocks` took down all calls at once. vLLM 0.33's
resumable streaming sessions are not length-checked when a chunk is appended. `server.py` now guards this:

| guard | default | behaviour |
|---|---|---|
| session length | `--max-model-len` | A turn that would push the session past `max_model_len` (prompt + `max_tokens`) is refused with `session context full`. Tested: 6 sessions grown from 63k to 65.5k, every overflowing turn was refused, no engine error. |
| new-call admission | `--kv-admit 0.90` | A call's context is admitted only if the KV cache usage after loading it is estimated below 90%. The estimate scales current usage by the tokens of fully loaded sessions and counts contexts still loading as reserved. |
| turn refusal | `--kv-turn-max 0.97` | Above 97% KV usage, turns are refused (`KV cache full`) and the caller's words are kept for the next turn. |

Test with a deliberately small KV cache (`--gpu-util 0.60`, 13 GiB): 10 live calls × 28k plus 12 joiners every 10 s.
- Without admission, vLLM does not crash: joiners wait 145–195 s for free blocks while live turns stay at
  51 / 79 ms (p50 / p95).
- With admission: the 10 calls loaded to 85%, all 12 joiners were refused immediately, and usage peaked at 96%.
  Live turns ran at 53 / 81 / 163 ms with no errors.

The QAT scenario that crashed, rerun with the guards: 19 calls × ~59k plus 10 joiners × ~59k, one every 30 s, 8 min.
- All 19 sessions grew to ~65.5k and got `session context full`.
- All joiners were admitted, and nothing crashed.
- Live TTFT was **103 / 210 / 312 ms** (p50 / p95 / max): a 59k context load (~42 s each with cap 256) was
  running almost continuously. At 64k contexts, churn this heavy does not meet 100 ms: lower the join rate, or
  use shorter contexts for most calls.

An earlier variant that loaded contexts one at a time stalled the engine with 7 of 10 sessions loaded (GPU idle,
with and without turn-first); the cause was not found. Contexts are loaded concurrently, as in every other test.
`/health` reports `kv_usage` and `session_tokens`.

### Load-split scheduling for call churn (on by default, untested)

`server.py` now sets `VLLM_LOAD_SPLIT=1`; the patch and the fork include it
(commit "Scheduler: load-split"). Every measurement above was taken
without it; run with `VLLM_LOAD_SPLIT=0` to reproduce them. Load-split never lets
a context-load chunk share a step with live calls' work: while no turn is pending, load-only steps (~15–20 ms)
alternate with normal steps. A turn arriving mid-step then waits at most one short load-only step instead of a
~60 ms mixed one.

It was written after the GPU was returned. It applies cleanly to the patched scheduler and compiles, but has never
run. A step-timing model (`lsim.py`, fitted to the measured turn-first churn p95 of 111 ms)
estimates churn TTFT p95 111 → 95 ms with about the same context-load time. That is an estimate, not a
measurement. Test plan: the churn run (`client_bench.py --calls 20 --turns 34 --joiners 6 --joiner-gap 15 --seed 62`) with `VLLM_LOAD_SPLIT=0` vs `1`, then the 59k QAT churn soak.
Once a load has 1024 tokens or fewer left (`VLLM_TURN_FIRST_LONG`), its tail runs in normal steps, as before.
Expected cost: live replies decode somewhat slower while a context loads, since every load chunk gets its own
weight pass.

### Where a decode step's time goes (FP8 preset, 20 × 30k, MTP 3, torch profiler over a stagger run)

| share of GPU time | what |
|---|---|
| 52% | FP8 weight GEMMs (DeepGEMM, M ≈ 80): bound by reading ~30 GB of weights each step; int4 QAT weights halve this |
| ~12% | sliding-window attention (FA3), 50 layers: reads 8 KB of FP8 K+V per token (16 KV heads × 256), memory-bound |
| ~9% | global layers (attn512 decode kernel), ~0.17 ms × 10 per step |
| ~18% | norms, activations, quantization of activations, sampling |

A steady decode step is ~19.6 ms of kernels, with ~2.4 ms of GPU idle between steps. JIT lookahead
(`VLLM_JITLA_LEAD_MS`, default 2) deliberately launches the next step late so that turns arriving meanwhile can
join it. Per session, a step reads ~420 MB of sliding-window KV (50 × 1024 tokens × 8 KB) and ~770 MB of global
K-only KV at 30k (10 × 30k × 4 heads × 640 B), next to the shared weights. The remaining lever is the GEMM
half: int4 weights (the qat preset) halve it.
