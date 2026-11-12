"""Step-timing model of live-turn TTFT while a 28k context loads: turn-first (current) vs load-split.
Model only: step durations and overhead are fitted to measured runs, not measured for load-split."""
import random, sys
def run(policy, chunk_ms, seed, T=600.0, overhead=(8, 22), n_step=(30, 48), l_extra=6.0):
    rng = random.Random(seed)
    finals, shorts = [], []          # arrival times
    t = 0.0
    while t < T:                      # 20 callers: a final every ~3 s each, 3 partials before it
        t += rng.expovariate(20 / 3.0)
        finals.append(t)
        shorts += [t - 0.3 * k for k in (1, 2, 3)]
    events = sorted([(x, 1) for x in finals] + [(x, 0) for x in shorts if x > 0])
    loads = [15.0 * k for k in range(1, int(T / 15))]   # a 28k context joins every 15 s
    CH = 28000 / 256
    now, i, pend, ttft, prev_load = 0.0, 0, [], [], False
    left, active, ldone, lstart = 0, False, [], 0.0
    while now < T:
        while i < len(events) and events[i][0] <= now:
            pend.append(events[i]); i += 1
        if not active and loads and loads[0] <= now:
            loads.pop(0); active, left, lstart = True, CH, now
        n = rng.uniform(*n_step) / 1e3
        turn = bool(pend)
        if policy == "turnfirst":
            if turn or not active: dur, chunk = n, False
            else: dur, chunk = n + chunk_ms / 1e3, True
        else:  # loadsplit
            if turn or not active or prev_load: dur, chunk = n, False
            else: dur, chunk = (chunk_ms + l_extra) / 1e3, True
            prev_load = chunk
        served = [e for e in pend]
        if policy == "loadsplit" and chunk: served = []   # load step: turns wait
        end = now + dur
        for e in served:
            if e[1] == 1: ttft.append(1e3 * (end - e[0]) + rng.uniform(*overhead))
        if served: pend = []
        if chunk:
            left -= 1
            if left <= 0: active = False; ldone.append(end - lstart)
        now = end
    ttft.sort()
    q = lambda p: ttft[int(p * (len(ttft) - 1))]
    return q(.5), q(.95), max(ttft), sum(ldone) / max(1, len(ldone))
for pol in ("turnfirst", "loadsplit"):
    r = [run(pol, 25.0, s) for s in range(5)]
    avg = [sum(x[k] for x in r) / len(r) for k in range(4)]
    print(f"{pol:10s} TTFT p50 {avg[0]:5.1f}  p95 {avg[1]:5.1f}  max {avg[2]:5.1f} ms   context load {avg[3]:5.1f} s")
