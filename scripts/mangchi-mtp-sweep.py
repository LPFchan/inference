#!/usr/bin/env python3
"""Fixed prompt suite for comparing MTP k values on the Flash-Next endpoint.

Usage: mangchi-mtp-sweep.py <label>    # writes /tmp/mtp-sweep-<label>.json
Records per-prompt throughput, spec-decode counters, and the exact greedy
output so runs at different k can be compared for speed and for identity.
"""
import json, sys, time, urllib.request, hashlib, subprocess, statistics

BASE = "http://127.0.0.1:8002"
MODEL = "/models/qwen3.8-flash-next-abliterated-w4a4"
PROMPTS = {
    "code": ("Write a Python function that merges two sorted lists into one sorted "
             "list, with a docstring and type hints. Then write three unit tests "
             "for it. Code only, no explanation.", 400),
    "prose": ("Write a short reflective essay about why people find old lighthouses "
              "haunting. No lists, just flowing prose.", 400),
    "reasoning": ("A train leaves at 14:20 travelling 84 km/h. A second leaves the "
                  "same station at 15:05 travelling 112 km/h on the same track. At "
                  "what clock time does the second catch the first? Show your work.", 400),
    "arith": ("What is 19 * 23? Reply with only the number.", 64),
}
REPEATS = {"arith": 4}


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return r.read().decode()


def counters():
    out = {}
    for line in get("/metrics").splitlines():
        if line.startswith("vllm:spec_decode_num_"):
            name, _, val = line.rpartition(" ")
            out[name] = float(val)
    return out


def delta(before, after, prefix, extra=None):
    return sum(after[k] - before.get(k, 0.0) for k in after
               if k.startswith(prefix) and (extra is None or extra in k))


def chat(prompt, max_tokens):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode())
    return d, time.time() - t0


def run(name, prompt, max_tokens):
    b = counters()
    d, wall = chat(prompt, max_tokens)
    a = counters()
    msg = d["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    n = d["usage"]["completion_tokens"]
    drafts = delta(b, a, "vllm:spec_decode_num_drafts_total")
    drafted = delta(b, a, "vllm:spec_decode_num_draft_tokens_total")
    accepted = delta(b, a, "vllm:spec_decode_num_accepted_tokens_total")
    pos = []
    i = 0
    while True:
        v = delta(b, a, "vllm:spec_decode_num_accepted_tokens_per_pos_total", f'position="{i}"')
        if v == 0 and i > 0 and not any(f'position="{i}"' in k for k in a):
            break
        if not any(f'position="{i}"' in k for k in a):
            break
        pos.append(v)
        i += 1
    return {
        "tokens": n, "wall_s": round(wall, 2), "tok_s": round(n / wall, 2),
        "drafts": drafts, "drafted": drafted, "accepted": accepted,
        "accept_rate": round(100 * accepted / drafted, 1) if drafted else None,
        "mean_accept_len": round((accepted + drafts) / drafts, 3) if drafts else None,
        "steps_per_s": round(drafts / wall, 2) if drafts else None,
        "per_pos_pct": [round(100 * p / drafts, 1) for p in pos] if drafts else [],
        "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
        "text": text,
    }


def main():
    label = sys.argv[1]
    res = {"label": label, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    run("warmup", "hi", 8)  # avoid first-request effects
    for name, (prompt, mt) in PROMPTS.items():
        runs = [run(name, prompt, mt) for _ in range(REPEATS.get(name, 1))]
        res[name] = runs[0] if len(runs) == 1 else {
            "runs": runs,
            "shas": sorted({r["sha"] for r in runs}),
            "tok_s": round(statistics.mean(r["tok_s"] for r in runs), 2),
        }
        r = res[name]
        print(f'{name:10s} {r.get("tok_s")} tok/s  accept_len={runs[0].get("mean_accept_len")} '
              f'per_pos={runs[0].get("per_pos_pct")} sha={runs[0]["sha"]}')
    out = f"/tmp/mtp-sweep-{label}.json"
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
