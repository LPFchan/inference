# Current Status

**Snapshot:** 2026-10-29
**Posture:** The Grimoire gateway serves local llama.cpp models and admits remote vLLM models through Mangchi's residency agent. The local engine remains TheTom's turboquant fork with turbo4 KV. Mangchi model artifacts are managed separately from the gateway.
**Focus:** Serve Qwen3.8 Flash-Next and the grugthink 27B as Mangchi's registered remote models (one resident at a time) and continue Flash-Next's its decode, prefix-cache, and CUDA-graph optimization program.

## Autoresearch — Qwen3-ASR

- Charter: [DEC-20261029-001](decisions/DEC-20261029-001-qwen-asr-autoresearch.md).
  Single worker; no Grimoire operations or automatic checkpoint deployment.
- Nemotron backend is unloaded for ongoing training at the operator's request.
  Gateway/tunnel and retained weights are preserved. Do not restore it automatically.
- Selection set: frozen 1,295 FLEURS validation clips, manifest
  `c75a4b05069eba7d11a6b50efd7bb11d43800fc66e49b761cd08fb86354acf01`.
  The original 2,624-clip test and nine predictions stay sealed.
- Incumbent: original MLX 8-bit anchor. All three larger-set calibrations
  completed; both starter adapters fail the selection gates. EN/KO WER and
  JA/ZH CER are 4.25/10.85/4.62/6.61% for the anchor,
  6.10/18.80/6.76/7.86% for SFT-200 and 6.15/18.94/6.76/7.86% for GRPO-50.
  The small starter sample's gains do not generalize to this development set.
- Closed trial `sft-1716b8bc1e5c73e3`: 5,000 updates from the base on
  `data-v001-16h` (9,852 clips / 16.5054 hours), LR 3e-5, seed 21774173.
  Full validation primary errors are EN 4.69%, KO 19.21%, JA 6.80%, ZH 7.75%;
  Korean CER also rises from 4.33% to 6.33%, so spacing alone does not explain
  the regression. Score 1.4724 against base 1.0; paired delta CI95
  [0.3733, 0.5746]. Rejected. All 5,000 updates are finite; optimization takes
  84.90 minutes, the complete run 100.62 minutes, peak MLX allocation 4.15 GB.
- Closed trial `grpo-103d0a8dd807ff06`: 1,000 updates from SFT-200 on the
  same snapshot, LR 1e-6, Korean WER weight 0.5. Full validation primary errors
  are EN 6.09%, KO 18.77%, JA 6.59%, ZH 7.87%; score 1.4493 against base 1.0,
  paired delta CI95 [0.3601, 0.5485]. Rejected. The change from its SFT parent
  is also unconfirmed: paired score delta CI95 [-0.0309, 0.0076]. All 1,000
  updates are finite, with 533 variable-reward and 467 equal-reward groups;
  51.09 optimization minutes, 66.14 total minutes, 4.09 GB peak MLX allocation.
- Closed trial `sft-b508876066da8567`: 5,000 updates from the base on
  `data-v001-16h`, LR 1e-5, seed 32125228. Full validation primary errors are
  EN 4.93%, KO 20.07%, JA 6.59%, ZH 7.91%; CER all is
  2.89/6.96/6.59/7.91%. Score 1.4915, paired delta CI95 [0.3940, 0.6074].
  Rejected, including two empty Korean outputs. All updates are finite;
  84.87 optimization minutes, 100.53 total minutes, 4.15 GB peak MLX allocation.
  Its sampling seed differs from the 3e-5 run, so these two recipes do not
  isolate learning rate.
- Closed trial `grpo-2e5b6950468db2d0`: 1,000 updates directly from the
  untouched base with that policy as the fixed KL reference, LR 1e-6,
  seed 21325231, Korean WER weight 0.5 on `data-v001-16h`.
  Full validation primary errors are EN 4.27%, KO 10.85%, JA 4.70%, ZH 6.67%;
  CER all is 2.13/4.32/4.70/6.67%. Score 1.0091 against base 1.0;
  paired delta CI95 [-0.0034, 0.0233]. Rejected: neither the required gain nor
  a bootstrap interval below zero is achieved. Regression/empty/truncation
  limits pass; this run does not establish a statistically reliable change.
  All 1,000 updates are finite, with 423 variable-reward and 577 equal-reward
  groups; 51.10 optimization minutes, 66.19 total minutes, 4.18 GB peak MLX
  allocation. All four scored hill-climb trials are rejected; base remains best.
- Live on Mac `main:qwen-batch32`: `sft-c3422927b2181bc0`, effective
  batch 32 × 625 updates from the untouched base, on `data-v001-16h`.
  It uses exactly the same ordered 20,000 example IDs, seed 21774173,
  LR 3e-5 and frozen training source `code/655b245` as the original batch-4
  SFT trial. Sequence SHA256 is
  `1705d82249f480b3a68807122f9784c12dadc37f8960170007a4d97cfa840d7b`.
  `/tmp/stt-qwen-batch32-full-20261029.log` and `loop-state.json` identify
  the single driver/worker; wins require independent-seed confirmation.
  The new worker starts only after the completed GRPO driver releases the lock.
- PR19 is merged as `dca3bab`. The completed direct-base GRPO keeps its
  immutable actual-run source `code/3ee30f6`, native acceptance evidence,
  model/checkpoint hashes and full validation outputs.
- The original controller was drained after the third rejection. Its next
  higher-rate SFT-parent GRPO run `grpo-daafe1970256e5e9` is superseded after
  223 observed updates, with checkpoint 200 retained; it is not a scored trial.
  The handoff receipt preserves the prior controller state.
- Pinned intake completed all 157 files without final errors. The next immutable
  snapshot, `data-v002-clean`, is verified: 67,794 clips / 108.5695 hours,
  including 48.95 h Korean, 13.12 h Japanese, 45.00 h Mandarin and 1.50 h English.
  Its train manifest is `180554e0676f47a46ba99f8e5a2b9996ad1288bc5554b2e1492879010116273d`;
  development bytes remain identical. `next-data-ready.json` retains the queued
  snapshot; it does not change the active trial's inputs.
  YODAS remains quarantined pending audio-language/alignment checks because
  caption inspection found translated lecture material in the Japanese pool.
- Root: `~/models/stt/qwen3-asr-hillclimb-20261029`. `loop-state.json` and
  per-trial proposals/verdicts hold runtime evidence. A 10-minute heartbeat
  wakes this thread with `continue`; its handle is retained in `monitor.json`.
  Check existing processes/saved files before launching replacements.
  All four completed trials' final adapters/optimizers, full outputs and
  verified summaries are mirrored on Dumpling and Mangchi. The latest summary
  is `base-reference-grpo-summary.json`; its archive receipt covers the actual
  runner, checkpoint, full outputs and saved verdict. Earlier native preflight
  and superseded partial-run evidence remain retained.

## Migration Summary

Bee (`Anbeeld/beellama.cpp`) is the canonical engine. Single binary serves DFlash (`--spec-type dflash`), PFlash (via `pflash_daemon`), and normal traffic. Content-hash KV caching provides cross-conversation sysprompt reuse (1.93x verified). Legacy `backend:dflash` daemon, Lucebox code, and `/opt/dflash` fully retired. For completed phase details see git history (Phases 1-7, commits `f40874c` through `ef4fcb1`).

## Atomic CUDA FA V2 Track

Patch chain in `patches/atomic-llama-cpp/`, applied in order by `Dockerfile`:

| Default-served | File | Concern |
| --- | --- | --- |
| ✓ | `0002-cuda-fa-v2-scratch-owner.patch` | V2 graph-safe FA scratch owner with recoverable failure |
| ✓ | `0004-pool-flush-on-oom.patch` | Legacy pool flush-and-retry on OOM (backport of upstream PR #22155) |
|  | `0001-cuda-fa-temp-buffers-bypass-vmm-pool.patch` | V1, rollback only |
|  | `0003-cuda-fa-view_src-sizing.patch` | view_src sizing (closed upstream PR #23620), kept for slow-VRAM-creep regime if it appears |

`GRIMOIRE_LLAMA_CPP_CUDA_GRAPHS=ON` is the Dockerfile default. The trail of investigation and the decision rationale are in `records/research/RSH-20260523-001`, `RSH-20260526-001` (validation matrix), `RSH-20260526-002` (V3 design, deferred), and `RSH-20260526-003` (V2+ON shipping decision with the bench data and upstream survey).

## Recent Changes

- 2026-10-29: **Qwen3-ASR Mac training is operational** — a separate MLX
  workflow prepares speaker/text/audio-disjoint adaptation data with frozen
  FLEURS exclusions, trains decoder LoRA, resumes checkpoints and refines with
  audio-conditioned GRPO. Actual starter runs complete 200 SFT and 50 GRPO
  updates on the 24 GB M4 Pro, peaking at 4.02 GB MLX allocation. On 105 small
  development clips, SFT improves primary errors; this GRPO run trails SFT.
  No adapted Qwen checkpoint is deployed. Nemotron stays unloaded for the
  ongoing training program. Procedure:
  [`training/qwen_asr`](../training/qwen_asr/README.md); measured evidence:
  [`RSH-20261029-002`](research/RSH-20261029-002-qwen-asr-sft-rl.md).

- 2026-10-29: **Mac transcription is live** — `transcribe.lost.plus` serves
  Nemotron 3.5 Transcribe 30B-A4B as a 19.26 GB MLX INT4 export on yeowoolmac's
  The Rest partition (M4 Pro, 24 GB). The inference Common Auth gateway and a
  dedicated Mac tunnel protect the loopback backend. All four language HTTPS
  smokes pass; missing/invalid credentials return 401 and unsupported language
  returns 400. Twenty preflight clips, including each language's longest
  frozen FLEURS file, pass with measured MLX peak memory of 20.18 GB. oMLX is
  stopped and disabled while transcription uses this Mac. All nine pipelines
  now have 2,624 validated predictions; the original eight files and accuracy
  counts are unchanged. The new INT4 model's primary errors are EN 5.35% WER,
  KO 13.77% WER, JA 4.13% CER and ZH 8.52% CER. It improves Japanese over Qwen,
  while MAI remains best in all four languages. Full methods/results:
  [`RSH-20261029-001`](research/RSH-20261029-001-nemotron-transcribe-mac.md).
  Installation and operation: [`deploy/yeowoolmac`](../deploy/yeowoolmac/README.md).

- 2026-10-28: **Silent streams stay open through Cloudflare** — a cold
  prefill of a long prompt on DeepSeek-V4.1-Flash-1017 sends `message_start`
  and then nothing until the first token, and Cloudflare dropped it after
  ~120 seconds (524), which Claude Code retried for up to half an hour. The
  public proxy now pings any event stream that has been silent for 15 seconds
  (Anthropic's ping event on `/v1/messages`, an SSE comment elsewhere). A cold
  random-text prompt with 190 seconds of prefill answered 200 through
  chat.lost.plus.
- 2026-10-26: **qwen3.6-35B-A3B served from yeowoolmac's The Rest
  partition** — rayray916's oQ3 MLX quant (~3.6 bpw, 15.8 GiB with the
  vision tower, no MTP head) of the Heretic-abliterated Qwen3.6-35B-A3B runs
  in oMLX 0.7.2 (release wheel with its prebuilt Metal kernels, venv
  `~/omlx/.venv`) under the user LaunchAgent `plus.lost.omlx` on
  `100.122.130.86:8000`; Grimoire's entry is a `vllm-remote` without a
  residency agent. oMLX allows keyless inference; its admin API needs
  `~/.omlx/api_key` on that Mac. The root LaunchDaemon
  `plus.lost.iogpu-wired-limit` raises `iogpu.wired_limit_mb` to 23040 at
  boot and oMLX's custom memory guard sits at its 22 GiB maximum (RAM minus
  a fixed 2 GiB), leaving a 20.9 GiB prefill limit. With BF16 KV the window
  is capped at 212,992 tokens: a 210k-token needle prompt answers correctly
  (122 tok/s prefill, 29 tok/s decode), short prompts decode at 82-86 tok/s,
  and 228k is rejected by the guard. dopaemon's oQ4e did not fit usefully:
  20.1 GiB of weights left 32k text-only or 27k with vision, and Affine4 KV
  cut long-context decode to 10 tok/s because its fast kernel needs M5
  TensorOps. Chat, streaming, image input and thinking off
  (`enable_thinking: false`) pass through chat.lost.plus. It serves only
  while The Rest is booted.
- 2026-10-25: **Only administrators push other models out** — any lost.plus account admitted to chat can use chat.lost.plus with its global token, so a request-driven load that would stop another model now needs the Common Auth administrator role, on Grimoire's GPUs and through the Mangchi/spark agents (`load?evict=false`). Everyone else loads cold models only into free room and otherwise gets a 409 naming the model it would unload.
- 2026-10-25: **Restarts keep remote models loaded** — `POST /restart` on the gateway (administrator) and on a residency agent restart them without unloading: the gateway leaves remote models resident and an agent leaves its model containers running for the next agent to adopt. A plain stop (`docker restart`, `systemctl`, reboot) is still a shutdown and unloads. Verified live: gateway and Mangchi agent both restarted while Flash-Next stayed up, and both remote models answered within 24 seconds.
- 2026-10-25: **DeepSeek-V4.1-Flash-1017 served from spark1+spark2** — the
  [MiaAI-Lab two-Spark TensorFold recipe](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks)
  at `27741e1` serves the EXL3 EAQ v1 2.9 bpw checkpoint at TP=2 with the full
  1,048,576-token window, image input and DSpark drafting, with drowzeys' Keys
  abliteration overlay (`ABLIT=1`, `0bdc2036`, 104 attention tensors in layers
  10-35) applied; the stock weight files were removed after it passed refusal,
  arithmetic, tool-call and image checks at 53 (prose) and 72 (code) tok/s
  through chat.lost.plus. spark2 (4 TB) is the
  head and API (`100.120.192.122:8888`); spark1 is the worker over the CX7 link
  (`192.168.100.0/24` and `192.168.101.0/24`, both rails, MTU 9000; NetworkManager
  `cx7-a`/`cx7-b`). spark2 exports its Hugging Face cache read-only to spark1 over
  NFS; Engram's tables and rank 1's weight file sit on spark1's own disk. The
  recipe's systemd user unit `dsv41-tensorfold.service` on spark2 keeps both
  ranks up, so Grimoire's `DeepSeek-V4.1-Flash-1017` is a `vllm-remote` entry
  without a residency agent: unloading it in Grimoire does not stop the pair.
  It needs nearly all memory on both Sparks, so spark1's `spark-agent` is
  stopped and the grugthink 27B entries cannot load while it runs. Chat,
  streaming, thinking off and image input pass through chat.lost.plus.

- 2026-10-20: **Speech benchmark complete** — eight pipelines each have 2,624
  FLEURS test predictions across English, Korean, Japanese and Mandarin.
  Native/local provenance is in
  [`RSH-20261020-001`](research/RSH-20261020-001-eg2-multilingual-stt.md);
  [`RSH-20261020-002`](research/RSH-20261020-002-openrouter-stt.md) adds
  MAI-Transcribe 2 and Scribe v2, costing $2.77 in total.
  [`RSH-20261020-003`](research/RSH-20261020-003-nemotron-stt.md) adds Nemotron
  3.5 ASR, which trails Qwen on all four primary error rates. Data, model assets
  and measured results are retained on Dumpling and Mangchi.

- 2026-10-07: **Presets no longer lock remote models** — a preset governs only Grimoire's own GPUs. Models served by another machine's residency agent (`vllm-remote`, today Flash-Next on Mangchi and grugthink on spark1) can be started and stopped while a locked preset such as `training` is active, and activating a preset no longer stops them. Previously `training` returned `409 Preset 'training' is active` for every cold load, including models that never touch Grimoire's 3090.
- 2026-10-07: **Grugthink 27B served from spark1** — `qwen3.8-27b-grugthink` and `qwen3.8-27b-vanilla` moved from Mangchi to spark1 (ASUS Ascent GX10), so loading them no longer evicts Flash-Next from Mangchi's memory budget. spark1 runs the same residency agent (`docker-compose.spark.yml`, `spark-agent`, port 9700) with `etc/spark-agent.json`, launching the identical spec on spark1's `spark-vllm:pin-5fd5dd5` build (same vLLM commit as Thor). Grimoire reaches it at spark1's Tailscale address `100.94.227.60`, since spark1's LAN address is DHCP. Sibling vLLM containers now name `vllm` as the Docker entrypoint, because spark1's image already starts `vllm serve`. The grugthink D6 sampling server that occupied spark1 was stopped; training is stopped.
- 2026-10-07: **Grugthink 27B moved to the D6 run-3 iteration-4 adapter** — `qwen3.8-27b-grugthink` now applies the grugthink D6 run-3 iteration-4 LoRA (`/home/yeowool/models/grugthink/dense-27b-d6-run3-iter-0004`) instead of D6 iteration 15, at the operator's request. In grugthink's all-checkpoint sweep of run 3 (five harness contexts, 50 math and 24 coding prompts, two samples each), iteration 4 lost at most 4.7 accuracy points in any context and thought 0.37 to 0.57 as much as the untouched model, the closest of the ten checkpoints to the teacher-relative target; it still failed that target in two math contexts and about a fifth of its math runs thought more than 0.77 as much. It has not passed a final exam; `qwen3.8-27b-vanilla` stays beside it.
- 2026-10-03: **Grugthink 27B moved to the D6 iteration-15 adapter** — `qwen3.8-27b-grugthink` now applies the grugthink D6 iteration-15 LoRA (`/home/yeowool/models/grugthink/dense-27b-d6-iter-0015`) instead of D3 round 14. D6 trained with varied system prompts and harness contexts, so its shorter thinking carries over to harnesses it never saw: at its iteration-15 check it thought 0.38 to 0.65 as much as the untouched model under held-out Claude Code, Codex, and Korean prompts. Round 14 compressed mainly under its own training prompt. Iteration 15 has not passed a final exam, and its math accuracy check stopped training 4.2 points down; `qwen3.8-27b-vanilla` stays beside it for comparison.
- 2026-09-30: **Grugthink 27B moved to the round-14 adapter, with the base model beside it** — `qwen3.8-27b-grugthink` now applies the round-14 LoRA (`/home/yeowool/models/grugthink/dense-27b-d3-iter-0014`) instead of round 11. Round 14 thought less on the 50-prompt math probe (think ratio 0.40 against 0.48) but has no coding check or final exam. The same port 8003 container also answers as `qwen3.8-27b-vanilla`, the checkpoint without the adapter, so both can be compared on identical weights; Grimoire lists both, and unloading either stops both.
- 2026-09-30: **Grugthink 27B served from Mangchi** — `qwen3.8-27b-grugthink` runs the vanilla Qwen3.8-27B NVFP4 checkpoint with the grugthink round-11 LoRA (`/home/yeowool/models/grugthink/dense-27b-d4-iter-0011`) applied at runtime, vision enabled, and the shipped MTP head drafting three tokens, on port 8003. Its 62 GiB reservation cannot share Mangchi's budget with Flash-Next's 101 GiB, so loading one evicts the other. Cold start took about 3 minutes; direct text, image, and MTP smokes passed (about 1.7 accepted draft tokens per step). Grimoire's live registry and seed carry the remote entry; the gateway container reaches port 8003. Both Qwen3.8 services now load `templates/qwen3.8-chat_template.jinja`, which renders a later system message in place instead of raising `System message must be at the beginning.`; Codex through OpenCodex sends its later instructions that way (lidge-jun/opencodex#6242).
- 2026-09-15: **Flash-Next warm restart reduced from about 753 to 154-168 seconds** — a prepared vLLM sharded state avoids repeating Thor ModelOpt/CuTe DSL weight conversion, while a persistent cache retains compiled Triton kernels. Direct NVMe reads reached 3.32 GiB/s at queue depth 1 and 5.22 GiB/s at queue depth 16; the old target loader achieved only about 0.19 GiB/s, confirming conversion rather than storage as the bottleneck. Target loading fell from 387.91 to 65.81-68.11 seconds. A truly cold Thor compiler start took about 379 seconds and remains below the gateway's 600-second deadline. The explicit 13.49 GiB FP8 KV reservation provides 844,883 tokens (3.22 native contexts), and text, vision, and four simultaneous request smokes passed. (`RSH-20260915-001`)
- 2026-09-14: **Grimoire now uses common auth for browser and API access** — `auth.lost.plus` validates the shared `lp_auth` browser session with `chat` visibility and validates bearer or `X-API-Key` credentials against `chat-v1`. The public proxy strips client identity headers and passes immutable `sub` plus the shared role to the loopback manager. Chat settings manages common auth's global/per-service token mode directly, while account-specific browser keys and expected-`sub` checks prevent stale tabs from crossing accounts. Shared model, registry, ingest, and plugin mutations require the common-auth administrator role. The browser MCP proxy admits only public targets, pins validated DNS results, isolates target credentials, rejects redirects, and returns an allowlisted response surface. History tree references and imports are account-scoped and cycle-safe. The existing bearer was imported without exposing it, and 104 conversations, 10 settings, 71,132 usage events, and the dashboard layout moved to `sub=1`. Eleven independent GPT-6 Astra high review rounds converged. Production bearer inference, model listing, unauthenticated rejection, browser login redirect, token-management rejection for bearer clients, private MCP rejection, and an exact `AUTH_OK` Flash-Next completion pass after deployment. (`DEC-20260914-001`)
- 2026-09-13: **Flash-Next uses vLLM's default 999-image prompt ceiling** — native vision on the sole Mangchi remote model retains a 2,097,152-pixel cap per image, 262,144-token context, FP8 KV, reduced-vocabulary MTP k=4, and eager mode. Production startup allocated 16.89 GiB / 975,077 KV tokens (3.72 full-context requests), and host availability stayed above 12.44 GiB. Direct five-image inference identified red, green, blue, yellow, and magenta in order; both authenticated Grimoire vision smokes passed, and the live registry matches the tracked multimodal seed.
- 2026-09-13: **Mangchi serving is Flash-Next-only with a 3x KV allocation** — retired the dense Qwen3.8-27B launch spec and its matching Grimoire remote-registry entry while preserving its model artifact, image, native dense optimizations, patches, tests, and research. Flash-Next now reserves 101 GiB at `gpu-memory-utilization=0.82`; production measured 18.8 GiB of KV cache and 1,085,325 cache tokens against 6.13 GiB at 0.72, a 3.07x increase. Host availability remained above 13.89 GiB during startup, and an authenticated Grimoire smoke returned the expected answer.
- 2026-09-12: **Reduced-vocabulary MTP drafting ships; decode +9-15% on top of k=4** — image `mangchi-vllm:thor-dense-candidate-v8-draft-vocab` adds an opt-in FR-Spec draft head (`QWEN4EXP_DRAFT_VOCAB`), now serving at 98,304. `lm_head` is the one unquantized BF16 tensor left (248320x2560 = 1.27 GB) and ~98% of a draft pass's weight traffic, so restricting the draft to frequent token ids removes cost without touching the target, which still verifies over the full vocabulary. Measured at k=4: code 36.30 (+14.7%), arithmetic 35.78 (+11.6%), prose 20.74 (+13.0%), reasoning 32.84 (+8.9%) tok/s, with prose accepted length unchanged at 2.229. 98,304 beats the upstream 65,536 default here: 99.96% of generated tokens fall below it against 97.89%, and it keeps more acceptance at equal speed. Combined with k=4 this is +38% code and +28% reasoning against the previously shipped k=2 full-vocabulary config. Piecewise CUDA graphs remain blocked: the PLE disk gather stays inside capture even as a registered splitting op, so graphs need the staged-gather rework rather than an annotation. (`RSH-20260912-003`)
- 2026-09-12: **Flash-Next MTP depth raised to k=4; measured optimum, not assumed** — a warm four-workload sweep of `num_speculative_tokens` 2/3/4/5 puts the turnover at k=4: code 31.6, reasoning 30.1, arithmetic 32.3, prose 18.5 tok/s, against 25.0/23.9/24.4/17.7 at the shipped k=2 (+21-32% on everything but prose). Mean accepted length rises monotonically through k=5, so deeper speculation never guesses worse; wall-clock stops following it because each extra draft pass costs a fixed increment while the marginal accepted token shrinks (k=5 buys 3% more accepted tokens on arithmetic for ~13% more step cost). k=3 is marginally better on prose alone. Output varies run-to-run at fixed k, so the reported MTP=3 quality cost does not reproduce here. k>=5 cannot boot at the derived block size: the QSA ring capacity (8 for k<=4, 12 for k>=5) never joins vLLM's block-size alignment, so 3184 fails to divide — workaround is `--block-size` at a multiple of 48. The residency agent now captures failed containers' stderr, without which that assertion was invisible. (`RSH-20260912-002`)
- 2026-09-12: **Flash-Next v7 ships MTP=2 and the prefix-caching mamba fix; decode 1.6x on prose, 2.5x on code** — new image `mangchi-vllm:thor-dense-candidate-v7-prefix-mtp` ports blazux/qwen3.8-Flash-DGX's two-line mamba block_size fix and enables MTP from the in-checkpoint draft head. MTP required raising Flash-Next to util 0.72 / 88 GiB (vLLM measured -1.96 GiB KV at 0.68) and disabling the unused multimodal warmup. Measured end-to-end against the 10.56 tok/s pre-MTP baseline: code 26.0 tok/s (2.46x, mean accepted length 2.82/3.00, per-position 95.1/86.6%), prose 16.5-17.5 tok/s (1.56-1.65x, 1.97-2.00/3.00, 61-64/36%), arithmetic 2.95/3.00. Acceptance matches or beats the tonyd2wild Spark reference at every comparable position (93/78% code, 65/39% prose), so the abliteration shift did not hurt the draft head. Step rate holds at ~9/s regardless of workload against 10.56/s pre-MTP, so MTP costs ~15% per step and returns the accepted length. Arithmetic determinism holds (437 x4). Prefix caching is enabled but inert on this vLLM pin — the unannotated MTP draft flags Mamba groups as draft groups, which vLLM documents as disabling cross-request reuse silently, so cached_tokens stays 0 and there is no wrong-restore risk. blazux serves real hits because their image runs the qwen38-flash-next release line, a different codebase from our PR #55557 pin; draft-group annotation is the follow-up that unlocks it. (`RSH-20260912-001`)
- 2026-09-11: **Mangchi co-residency withdrawn; models sized for standalone residency** — after guarded tests at host-memory floors of 6, 4, and 3.5 GiB, Flash-Next's MoE startup spike drove host available memory to ~3.3 GiB regardless of reservation, below any safe floor. 27B is now at util 0.22 / 27 GiB (0.20 left only ~1.07 GiB KV against the 3.35 GiB needed for 100K FP8) and Flash-Next stays at util 0.68 / 83 GiB (its 74.33 GiB weights + ~5.7 GiB non-torch overhead leave no KV room below 0.68). The two models are served one at a time via the agent's LRU eviction; re-enabling co-residency requires shrinking Flash-Next's PLE memory window, not a lower host floor. (`DEC-20260910-001`)
- 2026-09-10: **Remote vLLM startup now remains visibly loading until health passes** — Mangchi reports each resident model's explicit `loading` or `loaded` lifecycle state. Grimoire propagates that state during remote health refreshes instead of preserving a stale loaded state while the model container is alive but its vLLM endpoint is not ready.
- 2026-09-10: **Qwen3.8 reasoning and tool output are parsed separately** — both Mangchi launch specs use vLLM's `qwen3` reasoning parser alongside the `qwen3_xml` tool parser, preventing the tool parser from consuming `</think>` and folding hidden reasoning into visible content.
- 2026-09-10: **The custom Thor vLLM build now keeps vision while dropping unsupported native targets** — pinned and reduced vLLM FlashAttention to 14 BF16 FA2 sources needed by the accepted Qwen3.8 text/vision paths; FA3 and external targets whose own configure checks reject SM110 are no longer built. The retained FA2 library is 52.8 MB, down from 307.8 MB of FA2 plus 39.8 MB of FA3. The native wheel completed in about 21 minutes and now uses persistent BuildKit caches. The residency agent uses the accepted W4A4 checkpoints and v6 image: 27B keeps its measured 2,048-token prefill batch and vision path, while Flash-Next uses an 8,192-token prefill batch at 0.68 GPU-memory utilization. Both retain a 262,144-token request limit, FP8 KV, and Qwen XML automatic tool-call parsing. The prior v5 canary container remains stopped under a rollback name.
- 2026-09-10: **Remote model indicators now refresh when the selector opens** — Grimoire asks Mangchi for residency and process health, then confirms the vLLM `/health` endpoint before showing online. A timeout or ambiguous response keeps the previous indicator; only confirmed health or confirmed absence/death replaces it. Healthy remote processes are rediscovered after a gateway restart, and stale Mangchi residency records are reaped before relaunch. Deployed on Grimoire and Mangchi; the authenticated refresh endpoint reports both registered remote models offline while the separate native vLLM canary remains outside residency-agent control.
- 2026-09-10: **Mangchi remote-backend gateway support deployed** — added `vllm-remote` lifecycle and request routing, registered both Mangchi NVFP4 services in tracked and live Grimoire state, excluded remote models from local GPU allocation and llama.cpp KV slots, and made the registry HTTP admission validator backend-aware. Production health and authenticated management checks pass. A cold-load request reached Mangchi and failed safely because the registered 27B model directory is not present; Mangchi released the reservation and has no resident model. A separate Flash-Next artifact download is still active, so an end-to-end remote chat smoke remains pending. (`DEC-20260909-002`)
- 2026-09-10: **Qwen3.8 Flash-Next now serves native 262K context with FP8 QSA KV on Jetson Thor** — pinned vLLM PR #55557 at `5fd5dd5`, retained the SSD-backed 95.4 GiB PLE table and SM110 top-k fallback, and raised the model allocation to 0.67. Production allocated 5.88 GiB for 434,087 cache tokens (1.66x native context) and passed a deterministic chat smoke. The canary completed an exact 262,144-token request in 1,642.64 seconds with zero preemptions or request errors. Attention KV is FP8; the GDN/Mamba recurrent cache remains float32. (`DEC-20260909-003`, `RSH-20260909-002`)
- 2026-09-09: **Qwen3.8 Flash-Next ported to Jetson Thor with SSD-backed PLE** — ported Blazux's PLE mmap approach to vLLM v0.29's `Qwen4Exp` implementation while retaining the Thor-native `sm_110a` image. The 95.4 GiB BF16 PLE table remains on NVMe; the rest of the model uses 71.11 GiB. Added Thor fallbacks for incompatible FlashInfer autotuning and QSA cooperative top-k kernels. (`RSH-20260909-001`)
- 2026-07-19: **Eastself serving registry restored from retained intake provenance** — reconciled the tracked seed and persistent runtime registry to one byte-identical document, restored all 19 valid manifest-backed Eastself completion aliases in ascending intake `created_at` order, and preserved the fixed embedder/reranker assignments. The failed `eastself-gemma4-12b-dec20260718` alias and pin were removed; its unshared PEFT directory, LoRA GGUF, tokenizer-aligned GGUF, and intake manifest were permanently deleted while the shared 12B base and healthy v2 artifacts were preserved. All 19 restored entries pass real-file registry validation. Live sequential smokes passed for accepted 31B generations (`ck17972-final`, `ck3749-120k-final`) and healthy 12B v2; historical probes remain available as rollback/evidence and are not implied to be quality-accepted (for example, `ck250-dec4` loaded but produced an empty visible response).
- 2026-06-22: **Multi-process gateway + data-parallel GPU replicas** — split the gateway into a stateful manager (internal :9000, owns lifecycle + chat) and N stateless proxy workers (:9001) that round-robin encoder endpoints across per-GPU replicas. Embedder + reranker each get an always-on GPU-0 replica; one model name fans out across both GPUs. Production embeddings 115 → **224 req/s** (full 2x). Cost: ~2.9 GiB/GPU headroom (both GPUs ~21 GiB free), capping large-ctx chat models. (DEC-20260622-002, RSH-20260622-001, commit `8031d6e`)
- 2026-06-22: **Gateway proxy throughput fix** — every proxy path created a fresh per-request `httpx.AsyncClient` (no keepalive), capping high-RPS endpoints at ~30 req/s. Switched to a shared connection-pooled client (`proxy/client.py`); gateway rerank 29.2 → 64.4 req/s. Also confirmed reranker `--parallel` does not help (prefill-only encoder; parallel=1 optimal). Corrects the chat-only conclusion of RSH-20260518-006. (RSH-20260622-001, commit `4c37298`)
- 2026-06-22: **GPU co-location for always-on small models** — replaced strict one-model-per-GPU with a per-model `vram-budget-mib` + live `nvidia-smi` free-VRAM check. Both 0.6B models (embedder + reranker) now co-locate on GPU 1 (~2.9 GiB), freeing GPU 0 for chat. (DEC-20260622-001, commit `b616a99`)
- 2026-06-20/21: **llama.cpp engine + model tuning** — switched to TheTom `llama-cpp-turboquant` fork; turbo4 KV cache with ctx-size tuned to verified maximums across chat models; auto-MTP for nextn models; embeddinggemma-300m registered/tuned. (commits `3ca14cd`, `c59a4ba`, `f2edaac`, `3b16be0`, `e7e8720`)
- 2026-05-30: **vLLM migration prototype — llmcompressor quantization pipeline validated** (RSH-20260529-006, DEC-20260528-001). Driver upgraded to 580.159.04 for CUDA 13. vLLM 0.21.0 serving commercial AWQ model `Qwen3.6-27B-AWQ-INT4` on GPU 1. llmcompressor quantizes BF16 → AWQ (19.17 GB from 52 GB). vLLM loading blocked on post-quantization config patching (Marlin kernel shape requirements, missing multimodal weights).
- 2026-05-26: **Prod Dockerfile default flipped to V2+ON + 0004** (RSH-20260526-003). Pending: rebuild + recreate `grimoire:local` container.


- 2026-05-18: **DFlash MAX_VERIFY_TOKENS cap fixed** — `LLAMA_DFLASH_MAX_VERIFY_TOKENS=25` was silently breaking DDTree tree-mode (3-5% acceptance). Patched with env-var `GGML_DFLASH_MAX_VERIFY_TOKENS` following existing `GGML_DFLASH_MAX_CTX` pattern. Default stays 25. Auto-derived in `model_manager.py` from n_max + branch_budget. (RSH-20260518-005, commit `2f6a345`)
- 2026-05-18: **DDTree tree-mode benchmarked** — Tree-mode works correctly after cap fix (85% acceptance) but adds only +1% throughput over flat mode for greedy decode. Draft is already well-matched; tree branches have nothing to rescue. (RSH-20260518-005)
- 2026-05-18: **Gateway overhead profiled** — Instrumented `proxy/llama.py` with `time.perf_counter()`. Gateway adds 0-60ms per request (<1%). The ~1.3s "gateway overhead" from earlier measurements was a measurement error (comparing warm bench vs production through reasoning mode). (RSH-20260518-006)
- 2026-05-18: **DFlash think-split prototyped** — Streaming SSE parser detects think/answer boundary via `reasoning_content` delta fields. Hybrid split (AR for thinking, DFlash for output) is not beneficial because re-prefilling the assistant message costs more than the DFlash speedup saves. DFlash provides consistent ~1.55x speedup regardless of thinking mode. Per-request `speculative.n_max: 0` toggle already works — no patch needed. (RSH-20260518-006)
- 2026-05-18: **DFlash VRAM overhead measured** — +2,188 MiB (+12%) vs AR-only (17.4 GB → 19.6 GB). Draft model weights (~1.84 GB) + ring buffer + tape buffers. Leaves ~4.4 GB free on 24 GB for KV cache. (RSH-20260518-006)
- 2026-05-18: **PFlash stale-thread deadlock fixed** — `PflashDaemon` replaced `loop.run_in_executor` with a dedicated compressor thread + async Queue. Verified: 100/100 consecutive PFlash-compressing iterations, zero failures, zero VRAM drift, zero deadlocks. (DEC-20260518-001, commit `78cd53b`)
- 2026-05-18: **VRAM drift soak** — `soak_vram_drift.py` added, 100-iteration soak confirmed zero VRAM drift across repeated PFlash compression cycles
- 2026-05-18: **PINNED_SHA enforcement** — Docker build now verifies cloned SHA matches `GRIMOIRE_LLAMA_CPP_PINNED_SHA` (commit `c90d2a0`)
- 2026-05-18: Phase 4 — pflash daemon propagation fixed, catastrophic 413 on compression failure, slot-save-mtmd patch regenerated for Bee HEAD 4db14be0
- 2026-05-18: Phase 3 — Docker rebuild + canary verified (1.93x speedup)
- 2026-05-18: Hygiene cleanup — stale patches deleted, SPEC/README updated, dead BACKEND_DFLASH code removed, .dockerignore expanded
- 2026-05-18: Phase 5 — doc cleanup + speedup verification
- 2026-05-18: Phase 7 — lucebox/ deleted, pflash_daemon extracted, dflash model files removed (3.3 GB), 87 GB disk freed
