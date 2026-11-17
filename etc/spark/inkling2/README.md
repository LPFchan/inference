# Inkling 2 Small on spark1+spark2: kortexa's TensorFold plus our patch

[thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) (594B MoE, 17B active) at
TP 2 on the two Sparks, from [kortexa-ai/Inkling-2-Small-EXL3-2.59bpw](https://huggingface.co/kortexa-ai/Inkling-2-Small-EXL3-2.59bpw)
through kortexa's TensorFold fork (branch `kortexa-ai/inkling2-draft`, `e2614feb`). This folder holds what we add.

| File | What it is |
| --- | --- |
| `0001-inkling2-spark.patch` | Our 18 engine commits on `e2614feb` (`git am`): speed, the refusal edit per request, greedy fallback, Inkling's reply messages (thinking, tool calls). |
| `inkling2.jinja` | The chat template `serve-rank.sh` passes: kortexa's, without the forced text opener, so the model writes its own message headers. |
| `serve-rank.sh` | One rank, natively (no Docker). Rank 0 on spark2 serves the API; rank 1 on spark1 reads spark2's models over NFS. |
| `pair.sh` | Both ranks from spark2: ships the binary to spark1, starts rank 1 there, runs rank 0 in the foreground. |
| `inkling2-tensorfold.service` | The systemd user unit on spark2 that runs `pair.sh` on port 8889. |
| `split-experts.py` | Cuts each rank's half of every expert once, so a load reads 97 GiB a rank instead of the whole pack. |
| `export_edit.py` | Turns OpenUncensored's `edit.pt` into the folder `TF_INKLING_EDIT` reads (`edit.json`, `q.f32`). |
| `edit-gate.py`, `decode-speed.py` | Checks against a live server (`BASE=http://127.0.0.1:8889`). |

## Names

Both names run on one set of weights; the edit is chosen per request.

- `Inkling-2-Small`: with [OpenUncensored's refusal edit](https://huggingface.co/OpenUncensored/Inkling-2-Small-Uncensored)
  (`TENSORFOLD_KEYS_MODEL`). Grimoire lists this one.
- `Inkling-2-Small-Stock`: the model as released (the served `--name`). Reachable on spark2 only.

The edit is `W' = W - s·QQᵀW` on `wo` and every expert's `w2` in layers 9-47. TensorFold applies the same thing at run
time as `y -= s·Q(Qᵀy)` on each layer's summed attention and MoE outputs. Edited requests replay their own CUDA graphs,
so the stock model's kernels and outputs are unchanged.

## Layout on the Sparks

- spark2 `~/inkling-2-small`: `scripts/` from this folder; `ref/kortexa-tf` is the engine checkout, built with
  `zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121` (zig 0.17.0 in `~/.local/opt`).
- spark2 `~/models/inkling2`: `kx/rank{0,1}` (kortexa's per-rank packs), `exl3` (the full pack), `split/rank{0,1}`
  (`split-experts.py`), `uncensored-edit/tf` (`export_edit.py`).
- spark1 `~/inkling-2-small/{scripts,bin,logs}`; it reads `/mnt/spark2-models` (spark2's `~/models`, read-only NFS).

## Install

1. On spark2: check out kortexa's TensorFold at `e2614feb`, `git am 0001-inkling2-spark.patch`, build as above.
2. Copy the scripts to `~/inkling-2-small/scripts` and `inkling2.jinja` to `~/inkling-2-small/templates` on both Sparks.
3. `cp inkling2-tensorfold.service ~/.config/systemd/user/ && systemctl --user daemon-reload &&
   systemctl --user enable --now inkling2-tensorfold`. It conflicts with `dsv41-tensorfold.service`: each pair needs
   nearly all of both Sparks' memory.
4. A load takes about 2.5 minutes. `journalctl --user -u inkling2-tensorfold` shows rank 0; rank 1 logs to spark1
   `~/inkling-2-small/logs/serve-r1.log`.

## What it does today

- Decode, greedy, 131,072-token window: prose 38-42 tok/s, code 69, 8k context 37. About 0.2 s to the first token on
  short prompts. The uncensored name runs 36-39 on prose.
- The engine decodes greedily only. `TENSORFOLD_GREEDY_FALLBACK=1` makes a request with a temperature decode greedily
  instead of failing.
- Tool calls come back as OpenAI `tool_calls` (parallel calls too, streamed whole once each call's message closes).
  Thinking comes back as `reasoning_content`. The server reads Inkling's messages by token id
  (`zig/src/server/inkling_reply.zig`), and only `<|content_model_end_sampling|>` ends a reply.
- Thinking is off by default (`--no-thinking`). A request turns it on with `reasoning_effort` (`low`, `medium`,
  `high`, ...) or `chat_template_kwargs: {"enable_thinking": true}` (effort 0.9). With tools offered the model often
  writes a one-line thought before a call even with thinking off.
- Text only: no images.
