"""Voice call-center session server: Gemma 4 31B on one H100 via vLLM 0.33 resumable streaming-input sessions.

One WebSocket per call. The call's KV cache (system prompt + knowledge base + history) stays resident on the
GPU between turns, so each caller turn only prefills the new words.

Protocol (JSON messages, client -> server):
  {"type": "context", "text": "...system prompt + documents..."}   first message (or "messages": [chat msgs])
  {"type": "partial", "text": " more words"}   STT partial while the caller is still speaking. Appended to the
                                               session and prefilled immediately (max_tokens=1, output discarded),
                                               so the end-of-turn step only has a few tokens left to prefill.
  {"type": "final", "text": " last words", "max_tokens": 160}
                                               end of the caller's turn: closes the user turn, opens the model
                                               turn and streams the reply.
  {"type": "close"}

Server -> client:
  {"type": "ready", "context_tokens": n, "ms": ...}
  {"type": "delta", "text": "...", "n": tokens_so_far}       streamed reply
  {"type": "done", "ttft_ms": ..., "tokens": n, "finish_reason": "stop"|"length"}
  {"type": "error", "message": "..."}
      "server full: ..."            the call is not admitted: no free seat, or its context would not fit in the KV
                                    cache (--kv-admit); the WebSocket is closed
      "session context full ..."    this turn would push the session past max_model_len; end the call
      "KV cache full ..."           KV usage is above --kv-turn-max; the turn is refused, its words are kept for
                                    the next turn

Partial text is appended verbatim; send the STT *increment* (new stable words), not the whole hypothesis.
If the STT revises already-sent words, the simplest correct handling is to only forward words once stable.

Run:  python3 server.py --model /workspace/models/fp8block --port 8100   (see README.md for env/patches)
"""
import argparse, asyncio, json, os, sys, time, uuid

# Defaults for the patched vLLM (gemma4_voice_vllm0.33.0.patch); every knob can be overridden from the env.
for _k, _v in {
    "VLLM_STREAM_DELTA_RESUME": "1",   # resumed sessions only send their new tokens to the runner
    "VLLM_SDR_ALLOW_SPEC": "1",        # ... also with MTP
    "VLLM_FA4_WAVE_SPLITS": "32",      # wave-aware FA4 split heuristic (global layers)
    "VLLM_SWITCH_INTERVAL_US": "200",  # GIL switch interval (engine core / API threads)
    "VLLM_COALESCE_GAP_MS": "3",       # burst coalescing: wait up to 3 ms for more turns ...
    "VLLM_COALESCE_MAX_MS": "30",      # ... at most 30 ms in total
    "VLLM_JITLA": "1",                 # just-in-time lookahead: late arrivals join the next step
    "VLLM_PACE_TPS": "15",             # sessions ahead of TTS playback yield to sessions racing to 1st chunk
    "VLLM_KONLY": "2",                 # K-only global KV + RoPE sidecar (640 B/token/head, attn512 kernel)
    "VLLM_TURN_FIRST_LONG": "1024",    # a pending caller turn / partial (<= 1024 tokens) keeps context loads out of its step
    # UNTESTED on a GPU (written after the H100 was returned): context-load chunks never share a step with live
    # calls' work; load-only and normal steps alternate. Targets TTFT while calls join (modelled churn p95
    # 111 -> 95 ms at 28k). Needs the load-split scheduler patch; VLLM_LOAD_SPLIT=0 restores the measured setup.
    "VLLM_LOAD_SPLIT": "1",
}.items():
    os.environ.setdefault(_k, _v)
# attn512 kernel dir for the K-only global layers. The fork ships it in vllm/attn512/ (used when ATTN512_PATH is
# unset); a ../attn512 checkout next to this file takes precedence, as on the measurement box.
_attn512 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "attn512")
if os.path.isfile(os.path.join(_attn512, "kernel.py")):
    os.environ.setdefault("ATTN512_PATH", os.path.normpath(_attn512))

if os.environ.get("VLLM_SWITCH_INTERVAL_US"):
    sys.setswitchinterval(float(os.environ["VLLM_SWITCH_INTERVAL_US"]) / 1e6)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.protocol import StreamingInput
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind
from vllm.v1.metrics.loggers import StatLoggerBase

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/workspace/models/fp8block")
ap.add_argument("--port", type=int, default=8100)
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--max-sessions", type=int, default=24)
ap.add_argument("--max-model-len", type=int, default=65536)
ap.add_argument("--spec-model", default=None, help="e.g. /workspace/models/mtp (google/gemma-4-31B-it-assistant)")
ap.add_argument("--spec-k", type=int, default=3)
# Gemma 4 generation_config.json defaults
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-k", type=int, default=64)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--warmup", action="store_true", help="run a JIT warmup over burst shapes at startup")
ap.add_argument("--gpu-util", type=float, default=0.95, help="vLLM gpu_memory_utilization")
ap.add_argument("--long-prefill-cap", type=int, default=256,
                help="max prefill tokens per step for one request while others are active (vLLM "
                     "long_prefill_token_threshold). Keeps a joining call's context load from stalling live turns.")
ap.add_argument("--kv-admit", type=float, default=0.90,
                help="admit a new call only if KV cache usage after loading its context is estimated to stay below "
                     "this fraction (the rest is headroom for live calls to grow)")
ap.add_argument("--kv-turn-max", type=float, default=0.97,
                help="refuse a caller turn when KV cache usage is above this fraction. Paused sessions cannot be "
                     "preempted, and vLLM 0.33 crashes the engine when the KV cache runs out under this workload.")
a = ap.parse_args()


def _is_int4_ct(model_dir):
    """True for compressed-tensors int4 (W4A16) checkpoints, which vLLM 0.33 serves with HummingLinearKernel."""
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            q = json.load(f).get("quantization_config") or {}
        return q.get("quant_method") == "compressed-tensors" and '"num_bits": 4' in json.dumps(q)
    except OSError:
        return False


# vLLM 0.33 + humming bug: a compiled graph restored from a cache (vLLM's AOT cache or Inductor's FX-graph cache,
# i.e. any restart after the first) passes humming its layer config as JSON, and kernel construction crashes in
# profile_run ("'str' object has no attribute 'num_bits'" or "use_tma_as requires m-major input scales").
# A fresh compile works (~75 s extra startup), so int4 checkpoints run with every compile cache off.
if _is_int4_ct(a.model):
    for _k in ("VLLM_DISABLE_COMPILE_CACHE", "TORCHINDUCTOR_FX_GRAPH_CACHE", "TORCHINDUCTOR_AUTOGRAD_CACHE"):
        os.environ.setdefault(_k, "1" if _k.startswith("VLLM") else "0")

CAPTURE = [1, 2, 4, 8, 16, 20, 24, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 1280]
engine_kw = dict(
    model=a.model, kv_cache_dtype="fp8", max_model_len=a.max_model_len,
    max_num_seqs=max(32, a.max_sessions + 4), max_num_batched_tokens=8192, gpu_memory_utilization=a.gpu_util,
    prefix_caching_hash_algo="xxhash", language_model_only=True, enable_log_requests=False,
    compilation_config={"cudagraph_capture_sizes": CAPTURE},
    attention_config={"flash_attn_max_num_splits_for_cuda_graph": 8},
)
if a.long_prefill_cap > 0:
    engine_kw["long_prefill_token_threshold"] = a.long_prefill_cap
if a.spec_model:
    engine_kw["speculative_config"] = {"method": "mtp", "model": a.spec_model, "num_speculative_tokens": a.spec_k}

app = FastAPI()
ENG: AsyncLLM | None = None
TOK = None
TURN_HEAD = TURN_GEN = None
SEM = asyncio.Semaphore(a.max_sessions)
TRACE = os.environ.get("VOICE_TRACE") == "1"   # per-turn latency breakdown on stdout (debug)
KV = {"usage": 0.0}      # engine KV cache usage (0..1), updated every step by _KVUsage
SESSIONS: set = set()    # live sessions, for token accounting


class _KVUsage(StatLoggerBase):
    """Receives the engine's scheduler stats; keeps the current KV cache usage for admission control."""

    def __init__(self, vllm_config, engine_index=0):
        pass

    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
        if scheduler_stats is not None:
            KV["usage"] = scheduler_stats.kv_cache_usage

    def log_engine_initialized(self):
        pass


def _turn_template(tok):
    full = tok.apply_chat_template([{"role": "user", "content": "AAAA"}, {"role": "assistant", "content": "BBBB"},
                                    {"role": "user", "content": "CCCC"}], tokenize=False, add_generation_prompt=True)
    tail = full[full.index("BBBB") + 4:]
    head, gen = tail.split("CCCC")
    return head, gen


class Session:
    """A resident vLLM streaming-input request. Each queue item becomes one StreamingInput chunk."""

    def __init__(self, sid: str):
        self.sid = sid
        self.q: asyncio.Queue = asyncio.Queue()
        self.listener = None  # coroutine callback for the active chunk
        self.done: asyncio.Future | None = None
        self.t_yield = self.t_first = None
        self.tokens = 0          # tokens resident in this session (context + turns + replies)
        self.loaded = False      # context load finished
        self.pending = ""        # STT partial text held back while the KV cache is nearly full
        self.task = asyncio.create_task(self._run())
        self.in_user_turn = False

    async def _inputs(self):
        while True:
            item = await self.q.get()
            if item is None:
                return
            toks, max_tokens = item
            self.t_yield = time.perf_counter()
            yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=toks),
                                 sampling_params=SamplingParams(max_tokens=max_tokens, temperature=a.temperature,
                                                                top_k=a.top_k, top_p=a.top_p,
                                                                output_kind=RequestOutputKind.DELTA))

    async def _run(self):
        sp = SamplingParams(max_tokens=1, temperature=a.temperature, top_k=a.top_k, top_p=a.top_p,
                            output_kind=RequestOutputKind.DELTA)
        try:
            async for out in ENG.generate(prompt=self._inputs(), sampling_params=sp, request_id=self.sid):
                o = out.outputs[0]
                if self.t_first is None and o.token_ids:
                    self.t_first = time.perf_counter()
                self.tokens += len(o.token_ids)
                if self.listener is not None:
                    await self.listener(o)
                if o.finish_reason is not None and self.done is not None and not self.done.done():
                    self.done.set_result(o.finish_reason)
        except Exception as e:  # surface engine errors to the waiting chunk
            if self.done is not None and not self.done.done():
                self.done.set_exception(e)

    async def chunk(self, toks, max_tokens, listener=None):
        self.t_yield = self.t_first = None
        self.listener = listener
        self.done = asyncio.get_running_loop().create_future()
        self.tokens += len(toks)
        await self.q.put((toks, max_tokens))
        return await self.done

    async def close(self):
        await self.q.put(None)
        try:
            await asyncio.wait_for(self.task, 5)
        except Exception:
            self.task.cancel()


def _enc(text):
    return TOK.encode(text, add_special_tokens=False)


@app.websocket("/call")
async def call(ws: WebSocket):
    await ws.accept()
    if SEM.locked():
        await ws.send_json({"type": "error", "message": "server full"})
        await ws.close()
        return
    async with SEM:
        s = Session(f"call-{uuid.uuid4().hex[:12]}")
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                t = msg.get("type")
                if t == "context":
                    t0 = time.perf_counter()
                    if "messages" in msg:
                        text = TOK.apply_chat_template(msg["messages"], tokenize=False, add_generation_prompt=False)
                    else:
                        text = TOK.apply_chat_template([{"role": "system", "content": msg["text"]}], tokenize=False,
                                                       add_generation_prompt=False)
                    toks = _enc(text)
                    # Admission: estimate this context's KV share (plus contexts still loading) from the current
                    # usage per resident token of the sessions that have finished loading.
                    u = KV["usage"]
                    tot = sum(x.tokens for x in SESSIONS if x.loaded)
                    pending = sum(x.tokens for x in SESSIONS if not x.loaded)
                    est = u * (len(toks) + pending) / tot if tot > 0 else 0.0
                    if u + est > a.kv_admit:
                        await ws.send_json({"type": "error", "message": f"server full: KV cache {100 * u:.0f}% "
                                            f"used, this context needs ~{100 * est:.0f}%"})
                        break
                    SESSIONS.add(s)
                    await s.chunk(toks, 1)  # prefill; the sampled token is discarded by the next chunk
                    s.loaded = True
                    s.in_user_turn = False
                    await ws.send_json({"type": "ready", "context_tokens": len(toks),
                                        "ms": round(1e3 * (time.perf_counter() - t0), 1)})
                elif t == "partial":
                    text = msg.get("text", "")
                    if not s.in_user_turn:
                        text = TURN_HEAD + text
                        s.in_user_turn = True
                    if KV["usage"] > a.kv_turn_max:
                        s.pending += text  # hold the words back; they go out with the next chunk
                        continue
                    text, s.pending = s.pending + text, ""
                    await s.chunk(_enc(text), 1)
                elif t == "final":
                    text = msg.get("text", "")
                    if not s.in_user_turn:
                        text = TURN_HEAD + text
                        s.in_user_turn = True
                    max_tokens = int(msg.get("max_tokens", 160))
                    if s.tokens + len(_enc(s.pending + text)) + max_tokens + 16 > a.max_model_len:
                        await ws.send_json({"type": "error", "message": f"session context full "
                                            f"({s.tokens} of {a.max_model_len} tokens)"})
                        continue
                    if KV["usage"] > a.kv_turn_max:
                        # The engine would run out of KV blocks. Keep the caller's words for the next turn.
                        s.pending += text
                        await ws.send_json({"type": "error", "message": f"KV cache full ({100 * KV['usage']:.0f}%), "
                                            "turn refused; end a call to free memory"})
                        continue
                    text = s.pending + text + TURN_GEN
                    s.pending = ""
                    s.in_user_turn = False
                    t0 = time.perf_counter()
                    state = {"n": 0, "ttft": None, "ids": []}

                    async def on_out(o, state=state):
                        if o.token_ids:
                            if state["ttft"] is None:
                                state["ttft"] = 1e3 * (time.perf_counter() - t0)
                            state["n"] += len(o.token_ids)
                            await ws.send_json({"type": "delta", "text": o.text, "n": state["n"]})

                    fr = await s.chunk(_enc(text), max_tokens, on_out)
                    if TRACE and s.t_yield and s.t_first:
                        print(f"TRACE yield {1e3 * (s.t_yield - t0):.2f} engine {1e3 * (s.t_first - s.t_yield):.2f} "
                              f"ttft {state['ttft'] or -1:.2f}", flush=True)
                    await ws.send_json({"type": "done", "ttft_ms": round(state["ttft"] or -1, 1),
                                        "tokens": state["n"], "finish_reason": fr})
                    # The model's end-of-turn token is the last sampled token and is not in the KV cache;
                    # TURN_HEAD of the next user turn starts with it, so history stays well-formed.
                elif t == "close":
                    break
                else:
                    await ws.send_json({"type": "error", "message": f"unknown type {t!r}"})
        except WebSocketDisconnect:
            pass
        finally:
            SESSIONS.discard(s)
            await s.close()


@app.get("/health")
async def health():
    return {"ok": ENG is not None, "free_slots": SEM._value, "kv_usage": round(KV["usage"], 4),
            "session_tokens": sum(x.tokens for x in SESSIONS)}


async def _warmup():
    """Compile/JIT every burst shape once so the first real calls don't hit multi-second CuTe-DSL compiles."""
    import random
    rng = random.Random(0)
    n = min(8, a.max_sessions)
    ss = [Session(f"warm-{i}") for i in range(n)]
    await asyncio.gather(*[x.chunk([2] + [rng.randrange(1000, 200000) for _ in range(4096)], 1) for x in ss])
    for sl in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128):
        await asyncio.gather(*[x.chunk([rng.randrange(1000, 200000) for _ in range(sl)], 2) for x in ss])
    for x in ss:
        await x.close()


@app.on_event("startup")
async def _startup():
    global ENG, TOK, TURN_HEAD, TURN_GEN
    ENG = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kw), stat_loggers=[_KVUsage])
    TOK = ENG.tokenizer if getattr(ENG, "tokenizer", None) is not None else None
    if TOK is None:
        from transformers import AutoTokenizer
        TOK = AutoTokenizer.from_pretrained(a.model)
    TURN_HEAD, TURN_GEN = _turn_template(TOK)
    if a.warmup:
        t = time.perf_counter()
        await _warmup()
        print(f"warmup done in {time.perf_counter() - t:.1f}s", flush=True)


if __name__ == "__main__":
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", ws_ping_interval=20)
