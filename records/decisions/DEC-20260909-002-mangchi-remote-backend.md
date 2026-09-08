# DEC-20260909-002: Front mangchi as a remote backend ("third GPU") in the grimoire registry

Opened: 2026-09-09 03-20-00 KST
Recorded by agent: codex

## Metadata

- Status: accepted
- Deciders: operator, codex
- Area: grimoire gateway, `src/grimoire/model_manager.py`, `src/grimoire/registry.py`,
  `src/grimoire/proxy/llama.py`, `etc/models.grimoire.json`
- Related: DEC-20260909-001 (mangchi moves to vLLM/NVFP4),
  DEC-20260528-001 (grimoire engine migration llama.cpp -> vLLM, in-progress),
  DEC-20260622-001 (GPU co-location allocator),
  DEC-20260622-002 (multi-process gateway / data-parallel replicas)
- Greenlit by operator 2026-09-09: keep chat.lost.plus + webui on grimoire;
  add mangchi-hosted NVFP4 models to the grimoire registry "as if mangchi is a
  third gpu"

## Decision

chat.lost.plus and the webui **stay on grimoire**; that does not change. What
changes is that the grimoire model registry gains entries whose inference runs
on **mangchi's vLLM server over the network**, surfaced through grimoire exactly
like a locally-served model. Mangchi is treated as a third inference device in
the system — addressable from the same gateway, the same `/v1` API, and the
same webui — but it is a *remote* backend, not a CUDA device grimoire manages.

This requires a new **remote backend type** in the gateway. Today every backend
is local: `ActiveModel.start()` does `subprocess.Popen` of a llama-server and
the proxy hardcodes `http://127.0.0.1:{port}`
(`model_manager.py:383`, `proxy/llama.py:169`). A remote backend inverts the
lifecycle assumption: grimoire does **not** spawn or kill a process for it —
the process lives on mangchi under mangchi's own control — grimoire only holds
a **base URL** and routes/forwards to it.

Concretely:

- A registry model gains a remote marker, e.g. `"backend": "vllm-remote"` plus a
  target such as `"remote-url": "http://mangchi.lost.plus:8000"` (or the
  Tailscale/IP equivalent), instead of a GGUF `file` + GPU placement.
- The model manager treats remote models as always-"running" once healthy: it
  performs health/poll against the remote `/v1` endpoint rather than managing a
  subprocess, and is excluded from the local GPU allocator (it consumes no
  grimoire VRAM, so it neither triggers nor is a victim of local eviction).
- The proxy forwards chat/completion requests for these models to the remote
  base URL instead of `127.0.0.1:{port}`, preserving the OpenAI-compatible
  surface the webui already speaks.

## Context

Mangchi (Jetson AGX Thor, Blackwell `sm_110`, 128 GB unified) is moving to
vLLM serving NVFP4 models that grimoire cannot run: the two gated
`orcarouter` NVFP4 repos (27B-Uncensored ~24.7 GB; Flash-Next-Uncensored
~183.5 GB with its ~102 GB n-gram PLE table offloaded). The operator wants those
models usable from the existing grimoire endpoint and webui without standing up
a second gateway or moving the public entrypoint.

Grimoire's registry/proxy already has the seam for this. `registry.py` carries a
per-model `backend` field (today only `"llama"`), and the proxy already speaks
OpenAI-compatible HTTP to a per-model base URL — it is just hardcoded to
loopback. vLLM serves the same `/v1/chat/completions` and `/v1/models` surface,
so a remote vLLM backend is a routing change, not a protocol change.

This is deliberately narrower than DEC-20260528-001 (which replaces grimoire's
*local* engine with vLLM). Here grimoire's local engine is untouched; we only
add the ability to *front* a remote vLLM that happens to live on mangchi. The
two efforts share the vLLM adapter work but are independently shippable.

Operator-resolved design points (2026-09-09):

- **Network / reachability.** Grimoire reaches mangchi at the Tailscale name
  `mangchi.lost.plus`. Not the LAN IP, not a Cloudflare tunnel.
- **Lifecycle (load/unload).** Grimoire actively **signals mangchi to load and
  unload** models, and it should behave the way grimoire's own model
  load/unload behaves today. Because vLLM does not load/unload arbitrary
  checkpoints by name the way grimoire spawns a subprocess per model, this
  needs a **thin control agent on mangchi**: the gateway's
  `start_model`/`stop_model` for a remote backend translate to HTTP calls to
  that agent, which launches / tears down the matching vLLM process
  (`vllm serve <model>`) for the requested checkpoint. From the webui and
  `/v1`, a remote model loads and unloads indistinguishably from a local one;
  only the mechanism (remote process control instead of local subprocess)
  differs. See "Control agent" below.
- **Outage behavior.** A mangchi model **stays listed** in the registry when
  mangchi is down/unreachable, and **errors on request** — it is not hidden from
  `/v1/models`, and it is not shown as a distinct "offline" state. The model is
  present; requests to it fail while the backend is unreachable.
- **Auth.** No API key on mangchi's vLLM (or its control agent). The private
  tailnet is the trust boundary.

### Control agent on mangchi

To make remote load/unload behave like local load/unload, mangchi runs a small
always-on control service (separate from any single vLLM process) that owns
vLLM process lifecycle:

- `POST /models/<id>/load` -> start `vllm serve` for that model's checkpoint
  (with the NVFP4 + PLE-offload flags), return once `/v1` is healthy.
- `POST /models/<id>/unload` -> stop that vLLM process gracefully.
- `GET /models/<id>/status` (or `/healthz`) -> whether the model's vLLM is up
  and serving, for grimoire's health/error reporting.

Grimoire's remote backend holds the mangchi base URL and the model id, and its
`start_model`/`stop_model`/health paths map onto these agent endpoints. Since
mangchi is single-GPU, loading one model implies unloading whatever is resident
(the agent enforces single-residency), mirroring how grimoire evicts to make
room.

Remaining implementation detail (not blocking the decision): whether SSE
streaming and any KV-cache persistence passthrough work unchanged against the
remote `/v1`, or need a remote-aware path in `proxy/llama.py` / `proxy/sse.py`.

## Options Considered

### Move chat.lost.plus / webui to mangchi

- Upside: single box serving the NVFP4 models directly.
- Downside: operator explicitly wants the endpoint and webui to stay on
  grimoire; mangchi is a Jetson that gets rebooted and is the wrong place for
  the public entrypoint.

### Run a second, independent gateway on mangchi

- Upside: no grimoire code change.
- Downside: two endpoints, two registries, two webuis; the operator's whole
  point is one registry where mangchi models appear alongside grimoire's.

### Remote backend type in the existing gateway (chosen)

- Upside: one registry, one `/v1`, one webui; mangchi models are
  indistinguishable from local ones to clients; grimoire keeps its engine and
  allocator for local models.
- Downside: the gateway must learn a lifecycle it does not own (remote
  health/reachability instead of subprocess management) and a non-loopback
  proxy path.

## Rationale

The seam is small and already present: a per-model `backend` field exists, and
the proxy already speaks the same OpenAI HTTP dialect vLLM serves. Treating
mangchi as a remote backend keeps the single-source-of-truth registry the
operator values, keeps the public endpoint stable, and does not entangle the
change with the larger local-engine migration (DEC-20260528-001). The remote
backend consumes no grimoire VRAM, so it slots cleanly outside the existing GPU
allocator rather than forcing a rework of it.

## Consequences

- `registry.py` gains a remote backend type; `backend` is no longer implicitly
  `"llama"`-only.
- `model_manager.py` gains a remote-lifecycle path (health poll, no subprocess,
  excluded from the GPU allocator / eviction).
- `proxy/llama.py` (and possibly `proxy/sse.py`) gains a remote-base-URL path
  instead of hardcoded `127.0.0.1:{port}`.
- `etc/models.grimoire.json` can register mangchi NVFP4 models (27B-Uncensored,
  Flash-Next-Uncensored) pointing at mangchi's vLLM endpoint.
- The webui needs no changes if the `/v1` surface is preserved (per the API
  surface note in DEC-20260528-001).
- Mangchi must run a reachable vLLM server (see DEC-20260909-001); this DEC
  depends on that endpoint existing but not on its internal PLE-offload tuning.
- A follow-up may unify this remote-vLLM adapter with the local-vLLM adapter
  from DEC-20260528-001 so "vLLM, near or far" is one code path.
