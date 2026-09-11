#!/usr/bin/env python3
"""Varied-length prefix-cache parity test for the mangchi vLLM remote path.

Sends the same base prompt at several distinct lengths twice each (second
request is a full prefix hit), plus a multi-turn continuation (partial hit),
and compares greedy output between cold and hit runs. A same-length repeat
cannot catch a misaligned Mamba state restore (blazux root cause: state
seeded from the wrong block size), so every length differs and none is a
multiple of the 1600-token Mamba block.

Usage: python3 scripts/mangchi-prefix-cache-parity.py [--base-url URL] [--model ID]
"""

import argparse
import json
import sys
import time
import urllib.request

LENGTHS = [1523, 2417, 3301, 4757, 6311]  # all distinct, none 1600-aligned

BASE_TEXT = (
    "The history of the Jetson Thor platform spans several NVIDIA design "
    "generations, from early Tegra automotive chips through the Orin and "
    "Thor families, each bringing changes to memory architecture, tensor "
    "core design, and software tooling. "
)


def chat(base_url, model, messages, max_tokens=48, logprobs=False):
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 5
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as resp:
        payload = json.loads(resp.read())
    dt = time.monotonic() - t0
    choice = payload["choices"][0]
    usage = payload.get("usage", {})
    top = None
    lp = choice.get("logprobs")
    if lp and lp.get("content"):
        top = [t.get("token") for t in lp["content"][0].get("top_logprobs", [])]
    msg = choice["message"]
    text_out = msg.get("content") or msg.get("reasoning") or ""
    return text_out, usage, dt, top


def make_prompt(n_tokens_hint):
    # ~0.75 tokens/word for this prose; approximate length is fine -- what
    # matters is distinct, unaligned lengths across cases
    words = n_tokens_hint * 4 // 3
    text = (BASE_TEXT * (words // len(BASE_TEXT.split()) + 2))
    text = " ".join(text.split()[:words])
    return text + "\n\nSummarize the above in one sentence."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://mangchi.lost.plus:8002")
    ap.add_argument("--model", default="/models/qwen3.8-flash-next-abliterated-w4a4")
    args = ap.parse_args()

    failures = 0
    print(f"target: {args.base_url} model={args.model}")
    for n in LENGTHS:
        prompt = make_prompt(n)
        cold, u1, dt1, top1 = chat(args.base_url, args.model,
                                   [{"role": "user", "content": prompt}], logprobs=True)
        hit, u2, dt2, top2 = chat(args.base_url, args.model,
                                  [{"role": "user", "content": prompt}], logprobs=True)
        same = cold.strip() == hit.strip()
        top_same = top1 == top2
        status = "OK  " if (same and top_same) else "FAIL"
        if not (same and top_same):
            failures += 1
        print(
            f"{status} len~{n}: cold {dt1:6.1f}s / hit {dt2:6.1f}s | "
            f"prompt_tokens={u2.get('prompt_tokens')} | text_match={same} "
            f"first_token_top5_match={top_same}"
        )
        if not same:
            print(f"  cold: {cold[:120]!r}")
            print(f"  hit : {hit[:120]!r}")

    # multi-turn: second turn reuses the turn-1 prefix (partial hit)
    p1 = make_prompt(2000)
    a1, _, _, _ = chat(args.base_url, args.model, [{"role": "user", "content": p1}])
    _, usage, dt, _ = chat(args.base_url, args.model, [
        {"role": "user", "content": p1},
        {"role": "assistant", "content": a1},
        {"role": "user", "content": "Now make it one word shorter."},
    ])
    print(f"multi-turn continuation: {dt:.1f}s, usage={usage}")

    print("RESULT:", "PASS" if failures == 0 else f"{failures} FAILURES")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
