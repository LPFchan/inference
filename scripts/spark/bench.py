#!/usr/bin/env python3
"""Host-agnostic decode + prefill probe for Qwen3.8 Flash-Next NVFP4.

The same file must run against Mangchi (Thor) and a DGX Spark so the only
variable is the hardware and its engine config. Prompts, sampling, repeat
count and metric definitions are fixed here; everything host-specific comes
from the environment.

    BENCH_BASE=http://127.0.0.1:8002 BENCH_MODEL=/models/... ./bench.py thor-k4
    BENCH_BASE=http://127.0.0.1:8000 BENCH_MODEL=qwen3.8-flash-next ./bench.py spark-k4

Decode prompts and counter handling follow scripts/mangchi-mtp-sweep.py.
Prefill method follows RSH-20260910-001: one long prompt with a leading nonce
so prefix caching cannot serve it, timed to first streamed token.

Writes /tmp/bench-<label>.json.
"""
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request
import uuid

BASE = os.environ.get("BENCH_BASE", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("BENCH_MODEL", "qwen3.8-flash-next")
KEY = os.environ.get("BENCH_API_KEY", "")
# Prefill prompt sizes in tokens. 8192 matches the served chunk budget; 36864
# reproduces the RSH-20260910-001 sweep prompt.
PREFILL_TOKENS = [int(x) for x in os.environ.get("BENCH_PREFILL", "8192,36864").split(",")]
PASSES = int(os.environ.get("BENCH_PASSES", "3"))  # pass 1 is discarded as cold

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

# One filler line is close enough to 16 tokens that a token target can be hit
# by repetition and then reported exactly from the server's usage block.
FILLER = ("The harbour light turned once again across the water and the gulls "
          "answered it from the rocks below.\n")


def _headers():
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = f"Bearer {KEY}"
    return h


def get(path):
    req = urllib.request.Request(BASE + path, headers=_headers())
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


def counters():
    out = {}
    try:
        body = get("/metrics")
    except Exception:
        return out
    for line in body.splitlines():
        if line.startswith("vllm:spec_decode_num_"):
            name, _, val = line.rpartition(" ")
            out[name] = float(val)
    return out


def delta(before, after, prefix, extra=None):
    return sum(after[k] - before.get(k, 0.0) for k in after
               if k.startswith(prefix) and (extra is None or extra in k))


def stream_chat(prompt, max_tokens):
    """Return (text, usage, ttft_s, total_s). Streaming so TTFT is real."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers=_headers())
    parts, usage, ttft = [], {}, None
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                d = choice.get("delta") or {}
                piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    parts.append(piece)
    return "".join(parts), usage, ttft, time.time() - t0


def decode_run(prompt, max_tokens):
    before = counters()
    text, usage, ttft, total = stream_chat(prompt, max_tokens)
    after = counters()
    n = usage.get("completion_tokens", 0)
    ttft = ttft if ttft else 0.0
    # Decode rate excludes the prefill: (tokens - 1) / (total - ttft).
    decode_s = max(total - ttft, 1e-6)
    drafts = delta(before, after, "vllm:spec_decode_num_drafts_total")
    drafted = delta(before, after, "vllm:spec_decode_num_draft_tokens_total")
    accepted = delta(before, after, "vllm:spec_decode_num_accepted_tokens_total")
    pos = []
    i = 0
    while any(f'position="{i}"' in k for k in after):
        pos.append(delta(before, after,
                         "vllm:spec_decode_num_accepted_tokens_per_pos_total",
                         f'position="{i}"'))
        i += 1
    return {
        "tokens": n,
        "ttft_s": round(ttft, 3),
        "wall_s": round(total, 2),
        "decode_tok_s": round((n - 1) / decode_s, 2) if n > 1 else None,
        "e2e_tok_s": round(n / total, 2) if total else None,
        "drafts": drafts, "drafted": drafted, "accepted": accepted,
        "accept_rate": round(100 * accepted / drafted, 1) if drafted else None,
        "mean_accept_len": round((accepted + drafts) / drafts, 3) if drafts else None,
        "steps_per_s": round(drafts / decode_s, 2) if drafts else None,
        "per_pos_pct": [round(100 * p / drafts, 1) for p in pos] if drafts else [],
        "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
    }


def prefill_run(target_tokens):
    """Time a cold prefill. The nonce defeats prefix-cache reuse."""
    reps = max(1, target_tokens // 16)
    prompt = (f"Session {uuid.uuid4()}. Read the log below and reply with the "
              f"single word OK.\n\n" + FILLER * reps)
    _, usage, ttft, total = stream_chat(prompt, 1)
    p = usage.get("prompt_tokens", 0)
    return {
        "prompt_tokens": p,
        "ttft_s": round(ttft or total, 3),
        "prefill_tok_s": round(p / (ttft or total), 1) if p else None,
    }


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    res = {"label": label, "base": BASE, "model": MODEL,
           "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "passes": PASSES}
    decode_run("hi", 8)  # discard first-request effects

    res["prefill"] = {}
    for target in PREFILL_TOKENS:
        runs = [prefill_run(target) for _ in range(PASSES)]
        warm = runs[1:] or runs
        res["prefill"][str(target)] = {
            "runs": runs,
            "warm_median_tok_s": round(statistics.median(
                r["prefill_tok_s"] for r in warm if r["prefill_tok_s"]), 1),
            "prompt_tokens": runs[0]["prompt_tokens"],
        }
        m = res["prefill"][str(target)]
        print(f'prefill {m["prompt_tokens"]:>6} tok  '
              f'{m["warm_median_tok_s"]:>8} tok/s (warm median of {len(warm)})')

    res["decode"] = {}
    for name, (prompt, mt) in PROMPTS.items():
        runs = [decode_run(prompt, mt) for _ in range(PASSES)]
        warm = runs[1:] or runs
        rates = [r["decode_tok_s"] for r in warm if r["decode_tok_s"]]
        res["decode"][name] = {
            "runs": runs,
            "warm_mean_decode_tok_s": round(statistics.mean(rates), 2) if rates else None,
            "warm_mean_accept_len": round(statistics.mean(
                [r["mean_accept_len"] for r in warm if r["mean_accept_len"]] or [0]), 3),
            "shas": sorted({r["sha"] for r in runs}),
        }
        d = res["decode"][name]
        print(f'decode  {name:<10} {d["warm_mean_decode_tok_s"]:>8} tok/s  '
              f'accept_len={d["warm_mean_accept_len"]}  '
              f'per_pos={runs[-1]["per_pos_pct"]}')

    out = f"/tmp/bench-{label}.json"
    with open(out, "w") as handle:
        json.dump(res, handle, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
