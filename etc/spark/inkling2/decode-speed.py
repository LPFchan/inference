#!/usr/bin/env python3
"""Decode speed without the prompt: stream greedy-probe's prompts, time first to last content token (stdlib only)."""
import json, os, time, urllib.request
PROMPTS = ["What is the capital of Australia, and why isn't it Sydney? Answer in three sentences.",
 "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
 "Explain how a transformer's attention mechanism works to a high-school student.",
 "List the first ten prime numbers, then give their sum.",
 "Translate into Korean: 'The weather is lovely today, so let's go for a walk by the river.'",
 "A train leaves at 14:35 and arrives at 18:10. How long is the trip? Show your reasoning briefly.",
 "Write a haiku about a fox in the snow.",
 "What are the main differences between TCP and UDP?"]
for p in PROMPTS:
    body = json.dumps({"model": "inkling-2-small", "messages": [{"role": "user", "content": p}], "temperature": 0,
        "max_tokens": 256, "stream": True, "stream_options": {"include_usage": True}}).encode()
    t0 = time.time(); first = last = None; n = 0
    with urllib.request.urlopen(urllib.request.Request(os.environ.get("BASE", "http://127.0.0.1:8000") + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): n = d["usage"]["completion_tokens"]
            if d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                now = time.time(); first = first or now; last = now
    print(f"{n:4d} tok  ttft {first-t0:5.2f}s  decode {(n-1)/(last-first) if last>first else 0:5.1f} tok/s  overall {n/(last-t0):5.1f}  {p[:40]}")
