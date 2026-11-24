# Inkling 2 Small on spark1+spark2: kortexa's TensorFold plus our patch

[thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) (594B MoE, 17B active) at
TP 2 on the two Sparks, from [kortexa-ai/Inkling-2-Small-EXL3-2.59bpw](https://huggingface.co/kortexa-ai/Inkling-2-Small-EXL3-2.59bpw)
through kortexa's TensorFold fork (branch `kortexa-ai/inkling2-draft`, `e2614feb`). This folder holds what we add.

| File | What it is |
| --- | --- |
| `0001-inkling2-spark.patch` | Our 46 engine commits on `e2614feb` (`git am`): speed, the refusal edit per request, Inkling's reply messages (thinking, tool calls), NVFP4 KV caches, up to 4 streams at once, images, temperature sampling, exact-length turn reuse (`len - 1`) and in-place idle pool adoption. |
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
- Both Sparks: `~/inkling-2-small/lib` holds libjpeg-turbo 3.1.4.1's `libturbojpeg.so.0` and libwebp 1.6.0's
  `libwebpdecoder.so.3`, built from the release tarballs (`~/src/imglibs` on spark2). The engine takes JPEG and WebP
  only from those exact releases; `serve-rank.sh` puts `lib` on the loader path.

## Install

1. On spark2: check out kortexa's TensorFold at `e2614feb`, `git am 0001-inkling2-spark.patch`, build as above.
2. Copy the scripts to `~/inkling-2-small/scripts` and `inkling2.jinja` to `~/inkling-2-small/templates` on both Sparks.
3. `cp inkling2-tensorfold.service ~/.config/systemd/user/ && systemctl --user daemon-reload &&
   systemctl --user enable --now inkling2-tensorfold`. It conflicts with `dsv41-tensorfold.service`: each pair needs
   nearly all of both Sparks' memory.
4. A load takes about 2.5 minutes. `journalctl --user -u inkling2-tensorfold` shows rank 0; rank 1 logs to spark1
   `~/inkling-2-small/logs/serve-r1.log`.

## What it does today

- Decode, greedy, 524,288-token window: prose 38-44 tok/s, code 63-69, 8k context 37, 32k 67, 115k 53, 250k 34, 481k 27.
  About 0.2 s to the first token on short prompts. The uncensored name runs 36-39 on prose.
- Prompt reading: about 2000 tok/s at 10.9k tokens (1975-2020 from one restart to the next), 1900 at 43k. A pass
  takes up to 11,264 tokens (`PROMPT_ROWS`, about 0.4 MB of scratch a token), so a prompt that fits streams the expert
  weights once. A pass of 512 tokens or more runs in three row chunks: while one chunk's sums cross to the other Spark,
  the GPU already works on the next chunk. The exchange runs over both RoCE twins of the cable's port (one carries only
  ~100 Gb/s) with 4 NCCL channels.
- Long context: the 8 global layers and the drafter's global layer keep their KV caches as NVFP4 (`KV4=1`, the
  default): E2M1 values with an E4M3 scale per 16, 72 bytes a token and head instead of 256, so 524,288 tokens take
  2.99 GiB a Spark. The attention kernels widen them back to BF16 tiles, so the math is unchanged. `KV4=0` keeps BF16
  caches, which fit up to 131,072. Against BF16: 12 of 12 math problems both ways, three hidden facts found at 32k
  and 115k (and at 250k and 481k, past BF16's reach), drafter acceptance 58% against 57%, prompts within 3%. Replies
  are no longer byte-identical to BF16's, but they stay correct. A 481k prompt takes about 7.5 minutes (1100 tok/s).
- Memory: the engine takes about 108 GiB of each Spark with the shared pool, so about 6 GB stays free after the load. The expert matmuls
  reuse two buffers attention is done with, each of the 40 sliding-window layers keeps 1,280 positions (a longer pass
  borrows one shared ring and copies the window in and back out), and the drafter keeps only the tap rows it reads.
- Streams: prod serves 2 requests at once (`STREAMS`, default 2, up to 4). Their long-history caches share one pool
  that takes the memory left after the load past a 5 GiB reserve (`RESERVE_GIB`), about 700k positions, so one request
  can still use the whole 524,288-token window. Streams verify their drafts in one shared pass, and each reply stays
  the same as when it runs alone. All streams together: 44 tok/s with one, 50 with two, 56 with four (4 streams cap a
  request near 384k). Most of a step reads the experts' weights, and each chat's tokens need other experts, so more
  streams add little. A long prompt still reads one stream at a time, so it stalls the other while it reads.
- Turn reuse: a prompt that starts with the previous prompt, or shares its system prompt, resumes there. A second
  turn of a 39k-token conversation starts answering in 0.25 s (75 s for the first), and a new session on the same
  system prompt in 0.18 s. The engine keeps four turns: the last prompt's end and three cuts near the system
  prompt's end, about 91 MB each per Spark. They all read one live copy of the global layers' cache, so any request
  that starts elsewhere (even a short unrelated one) drops the turns past where it started.
- Replies are sampled at temperature 1 by default (`TEMPERATURE`; 0 is greedy), as the model was trained. A request's
  own `temperature` and `seed` apply; the same seed gives the same text, with or without drafts or another request
  running. `top_k`, `top_p` and `min_p` are ignored. Sampling costs about 18% of decode speed (44.8 vs 54.5 tok/s on
  short replies): the drafter still guesses the most likely token, so fewer guesses match.
- Tool calls come back as OpenAI `tool_calls` (parallel calls too, streamed whole once each call's message closes).
  Thinking comes back as `reasoning_content`. The server reads Inkling's messages by token id
  (`zig/src/server/inkling_reply.zig`), and only `<|content_model_end_sampling|>` ends a reply.
- Thinking is on at effort high by default (`--reasoning-effort high`). A request sets `reasoning_effort` (`low`,
  `medium`, ...; `none` turns it off) or `chat_template_kwargs: {"enable_thinking": false}`.
- Images (`VISION`, default 1): PNG, JPEG and WebP `image_url` parts, up to 16 a request and 16,384 tokens in all (one
  token a 40x40 tile; a bigger image shrinks to fit). The engine's own defaults are 4 and 4,096, so `serve-rank.sh`
  passes `--vision-max-images` (`VISION_IMAGES`, up to 64) and `--vision-image-tokens` (`VISION_IMAGE_TOKENS`, up to
  262,144) instead. The count spans the whole conversation, not the last message: an agent that re-sends its
  screenshots each turn is refused with HTTP 400 `a request may hold at most N images (--vision-max-images)` once its
  history passes the count, in about 1 s and before any token. The tokens are shared, so each image shrinks to the
  budget divided by the images present (about 1,024 tokens an image at 16, where the engine's 4,096 would leave 256).
  The rows are bf16 [tokens, 4608] on both ranks, so the two ranks need the same numbers (`pair.sh` passes `VISION*`
  through) and rank 1 holds 0.14 GiB for them instead of 0.04. Rank 0 runs the checkpoint's own image tower
  (`dense/vision.safetensors`) and sends the rows to rank 1. Its rows match a torch run of SGLang's tower (as close to
  float32 as torch's own bf16). A 300-token image adds about 0.1 s, a 4,070-token one about 2.5 s, and the other chat
  pauses that long. Follow-up turns resume past the image; another image never resumes it. The tower and the rows take
  1.40 GiB on rank 0 (0.83 at the engine's 4,096 tokens: the tower's scratch grows with the budget too), so the KV pool
  is 395,776 positions against 436,224 before. The pool has been under the 524,288 context since well before this
  raise, and the server says so at startup: a request longer than the pool is refused, so lowering
  `VISION_IMAGE_TOKENS` is the knob that buys those positions back.
