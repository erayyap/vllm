"""Load test for server.py: N concurrent calls, each with a long context, then caller turns streamed as STT
partials followed by a final chunk. Reports TTFT (from the final chunk) and time to the 5th / 25th token.

python3 client_bench.py --url ws://127.0.0.1:8100/call --calls 20 --context-chars 110000 --turns 3
"""
import argparse, asyncio, json, random, time, statistics
import websockets

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="ws://127.0.0.1:8100/call")
ap.add_argument("--calls", type=int, default=20)
ap.add_argument("--context-chars", type=int, default=110000)
ap.add_argument("--turns", type=int, default=3)
ap.add_argument("--partials", type=int, default=3)
ap.add_argument("--partial-gap", type=float, default=0.3)
ap.add_argument("--think", type=float, default=2.0, help="max random pause before each caller turn (s)")
ap.add_argument("--max-tokens", type=int, default=80)
ap.add_argument("--joiners", type=int, default=0,
                help="after all --calls are loaded and talking, this many extra calls join one by one "
                     "(fresh context, one turn, hang up): measures how a joining call's context load "
                     "disturbs the live calls' turns")
ap.add_argument("--joiner-gap", type=float, default=6.0, help="seconds between joiners")
ap.add_argument("--seed", type=int, default=0, help="context seed; change it to defeat the server's prefix cache")
ap.add_argument("--join-gap", type=float, default=0.0,
                help="0 (default): all calls load their context, wait until every call is ready, then talk "
                     "(steady state). >0: call i connects at i*gap s and talks right away, so turns overlap "
                     "other calls' context loads (call-center churn).")
a = ap.parse_args()

QUESTIONS = [
    "Merhaba, faturamda bu ay fazladan bir ücret görüyorum, bunun sebebi nedir acaba?",
    "Anladım, peki bu ücreti iade alabilmem için ne yapmam gerekiyor?",
    "Tamam, son olarak tarifemi daha uygun bir pakete geçirebilir miyiz?",
    "Bir de internet hızım akşamları çok düşüyor, bununla ilgili ne önerirsiniz?",
]
WORDS = ("müşteri hizmetleri tarife fatura ödeme iade paket internet hız modem kurulum abonelik kampanya indirim "
         "taahhüt sözleşme numara hat fatura tarihi son ödeme gecikme bedeli otomatik ödeme talimatı").split()


def make_context(rng):
    doc = []
    n = 0
    while n < a.context_chars:
        sent = " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 20))).capitalize() + ". "
        doc.append(sent)
        n += len(sent)
    return ("Sen bir çağrı merkezi asistanısın. Aşağıdaki müşteri kayıtlarını ve bilgi bankasını kullanarak kısa, "
            "net ve nazik Türkçe yanıt ver.\n\n" + "".join(doc))


READY = {"n": 0, "ev": None}


async def _mark_ready():
    READY["n"] += 1
    if READY["n"] >= a.calls:
        READY["ev"].set()


async def call(i, results):
    joiner = i >= a.calls
    rng = random.Random(a.seed * 100003 + i)
    if joiner:
        await READY["ev"].wait()
        await asyncio.sleep(1.0 + (i - a.calls) * a.joiner_gap)
    elif a.join_gap > 0:
        await asyncio.sleep(i * a.join_gap)
    async with websockets.connect(a.url, max_size=None, ping_interval=None) as ws:
        t_ctx = time.perf_counter()
        await ws.send(json.dumps({"type": "context", "text": make_context(rng)}))
        r = json.loads(await ws.recv())
        if r["type"] != "ready":
            print("call", i, r); await _mark_ready(); return
        results.append({"ctx_load_ms": 1e3 * (time.perf_counter() - t_ctx), "joiner": joiner})
        if a.join_gap <= 0 and not joiner:
            await _mark_ready()
            await READY["ev"].wait()   # steady state: nobody talks while contexts are still loading
        for t in range(1 if joiner else a.turns):
            await asyncio.sleep(rng.random() * a.think)
            words = QUESTIONS[(i + t) % len(QUESTIONS)].split()
            k = max(1, len(words) // (a.partials + 1))
            chunks = [" ".join(words[j:j + k]) for j in range(0, len(words), k)]
            for c in chunks[:-1]:
                await ws.send(json.dumps({"type": "partial", "text": (" " if c is not chunks[0] else "") + c}))
                await asyncio.sleep(a.partial_gap)
            t0 = time.perf_counter()
            await ws.send(json.dumps({"type": "final", "text": " " + chunks[-1], "max_tokens": a.max_tokens}))
            t5 = t25 = None
            while True:
                m = json.loads(await ws.recv())
                if m["type"] == "delta":
                    if m["n"] >= 5 and t5 is None: t5 = time.perf_counter() - t0
                    if m["n"] >= 25 and t25 is None: t25 = time.perf_counter() - t0
                elif m["type"] == "done":
                    results.append({"joiner": joiner, "ttft_ms": m["ttft_ms"], "client_total_ms": 1e3 * (time.perf_counter() - t0),
                                    "t5_ms": t5 and 1e3 * t5, "t25_ms": t25 and 1e3 * t25, "tokens": m["tokens"]})
                    break
                elif m["type"] == "error":
                    print("error", m); return
        await ws.send(json.dumps({"type": "close"}))


async def main():
    results = []
    READY["ev"] = asyncio.Event()
    t = time.perf_counter()
    await asyncio.gather(*[call(i, results) for i in range(a.calls + a.joiners)])
    turns = [r for r in results if "ttft_ms" in r and not r["joiner"]]
    mode = f"join every {a.join_gap:g}s" if a.join_gap > 0 else "steady state (all contexts loaded first)"
    if a.joiners:
        mode += f" + {a.joiners} joiners every {a.joiner_gap:g}s"
    print(f"{len(turns)} live-call turns in {time.perf_counter() - t:.1f}s, {mode}")

    def stats(label, rows, key):
        xs = sorted(r[key] for r in rows if r.get(key))
        if xs:
            print(f"{label:14s} p50={xs[len(xs) // 2]:7.1f} p95={xs[int(0.95 * (len(xs) - 1))]:7.1f} max={xs[-1]:7.1f}")
    live = [r for r in results if not r["joiner"]]
    stats("ctx_load_ms", live, "ctx_load_ms")
    for key in ("ttft_ms", "t5_ms", "t25_ms"):
        stats(key, live, key)
    if a.joiners:
        j = [r for r in results if r["joiner"]]
        stats("joiner ctx_ms", j, "ctx_load_ms")
        stats("joiner ttft", j, "ttft_ms")


asyncio.run(main())
