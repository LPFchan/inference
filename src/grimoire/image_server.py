"""MiniMax-H3-Image served as an OpenAI-compatible chat model.

Runs on Mangchi next to ComfyUI and is registered in Grimoire as an external
`vllm-remote` model. One chat turn produces one image:

1. Every image in the conversation (user attachments and earlier outputs) is
   numbered <Picture N> in order.
2. A vision chat model (the writer) turns the conversation into an H3 caption
   with MiniMax's official prompt-expansion prompts
   (templates/minimax-h3-image-pe/). This replaces ComfyUI's built-in enhancer.
3. ComfyUI renders the caption at about 4 MP, conditioned on the pictures the
   caption names.
4. The reply streams the writer's reasoning and the final caption as
   reasoning_content, then the image as a Markdown JPEG data URL. Because the
   image lives in the conversation, the next turn can refer back to it.
"""
import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

MODEL_ID = "MiniMax-H3-Image"
COMFY_URL = os.environ.get("H3_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
WRITER_URL = os.environ.get("H3_WRITER_URL", "http://spark2.tail.lost.plus:8888").rstrip("/")
WRITER_MODEL = os.environ.get("H3_WRITER_MODEL", "DeepSeek-V4.1-Flash-1017")
PROMPT_DIR = Path(os.environ.get(
    "H3_PROMPT_DIR", Path(__file__).resolve().parents[2] / "templates" / "minimax-h3-image-pe"))

OUTPUT_PIXELS = 4 * 1024 * 1024    # ComfyUI's "4 megapixels"
MAX_PICTURES = 16                   # H3's reference limit
WRITER_PREVIEW_PIXELS = 1024 * 1024  # the writer only needs to see what is in a picture
PING_SECONDS = 10

DATA_IMAGE = re.compile(r"!\[[^\]]*\]\((data:image/[^)\s]+)\)")
PICTURE_LABEL = re.compile(r"<Picture (\d+)>")

app = FastAPI(title="MiniMax-H3-Image chat", docs_url=None, redoc_url=None, openapi_url=None)


class GenerationError(Exception):
    pass


# --- conversation -> pictures + instruction ---------------------------------

def _decode_data_url(url):
    try:
        header, data = url.split(",", 1)
        if not header.startswith("data:image/") or ";base64" not in header:
            return None
        return base64.b64decode(data)
    except (ValueError, TypeError):
        return None


def _parts(content):
    """Yield ("text", str) and ("image", bytes) from an OpenAI message content."""
    if isinstance(content, str):
        pos = 0
        for m in DATA_IMAGE.finditer(content):
            yield "text", content[pos:m.start()]
            raw = _decode_data_url(m.group(1))
            if raw:
                yield "image", raw
            pos = m.end()
        yield "text", content[pos:]
        return
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            yield from _parts(part.get("text") or "")
        elif part.get("type") == "image_url":
            raw = _decode_data_url((part.get("image_url") or {}).get("url", ""))
            if raw is None:
                # Fetching links would let any user make Mangchi request
                # arbitrary addresses on the tailnet; ask for an upload instead.
                raise GenerationError("Attach images as uploaded files; image links are not supported.")
            yield "image", raw


def read_conversation(messages):
    """Number every picture and return (pictures, instruction).

    A single request is passed to the writer verbatim, as MiniMax's prompts
    expect. Follow-up turns become a short transcript so "make it darker" knows
    what "it" is.
    """
    pictures = []
    turns = []
    for message in messages:
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        text, labels = [], []
        for kind, value in _parts(message.get("content")):
            if kind == "text":
                text.append(value)
            else:
                pictures.append(value)
                labels.append(f"<Picture {len(pictures)}>")
        turns.append((role, " ".join("".join(text).split()), labels))

    while turns and turns[-1][0] != "user":
        turns.pop()
    if not turns:
        raise GenerationError("Send a message describing the image you want.")

    # Keep the newest pictures within H3's limit and renumber the labels.
    drop = max(0, len(pictures) - MAX_PICTURES)
    pictures = pictures[drop:]

    def relabel(labels):
        kept = [int(PICTURE_LABEL.match(l).group(1)) - drop for l in labels]
        return [f"<Picture {n}>" for n in kept if n > 0]

    turns = [(role, text, relabel(labels)) for role, text, labels in turns]
    if sum(1 for t in turns if t[0] == "user") == 1:
        return pictures, turns[-1][1]

    lines = ["Earlier conversation, oldest first:"]
    for role, text, labels in turns[:-1]:
        if role == "user":
            attached = f" [attached {', '.join(labels)}]" if labels else ""
            lines.append(f"User: {text}{attached}")
        else:
            made = f"generated {', '.join(labels)}" if labels else text
            lines.append(f"Assistant: {made}")
    role, text, labels = turns[-1]
    attached = f" [attached {', '.join(labels)}]" if labels else ""
    lines.append(f"\nCurrent request: {text}{attached}")
    return pictures, "\n".join(lines)


# --- writer -------------------------------------------------------------------

def _ratio(width, height):
    g = math.gcd(width, height)
    return f"{width // g}:{height // g}"


def _preview_url(raw):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    scale = min(1.0, math.sqrt(WRITER_PREVIEW_PIXELS / (img.width * img.height)))
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def writer_messages(pictures, instruction):
    """Assemble the request exactly as MiniMax-H3-Image-PE's README describes."""
    if not pictures:
        system = (PROMPT_DIR / "t2i_system_prompt.txt").read_text(encoding="utf-8").rstrip("\n")
        task = (PROMPT_DIR / "t2i_task_prompt.txt").read_text(encoding="utf-8")
        return [{"role": "system", "content": system},
                {"role": "user", "content": task.format(instruction)}]

    count = len(pictures)
    content = [
        {"type": "text", "text": (PROMPT_DIR / "edit_task_prompt.txt").read_text(encoding="utf-8")},
        {"type": "text", "text": (f"Reference Image Count:\n{count} input image(s), "
                                  f"labeled <Picture 1> through <Picture {count}>.")},
    ]
    for index, raw in enumerate(pictures, start=1):
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
        content += [
            {"type": "text", "text": f"Reference Media:\n- <Picture {index}> [aspect_ratio {_ratio(*img.size)}]"},
            {"type": "image_url", "image_url": {"url": _preview_url(raw)}},
        ]
    content.append({"type": "text", "text": (
        "\nUser Editing Instruction (JSON string):\n"
        f"{json.dumps(instruction, ensure_ascii=False)}\n\n"
        "User Aspect Ratio Selection:\nadaptive\n\n"
        "Apply the instructions above to the supplied inputs and reference images. "
        "Output only the required two-field JSON object.")})
    return [{"role": "user", "content": content}]


def parse_writer_answer(text):
    start, end = text.find("{"), text.rfind("}")
    try:
        answer = json.loads(text[start:end + 1])
        caption = answer["expanded_caption"].strip()
        ratio = answer.get("target_aspect_ratio") or answer.get("aspect_ratio") or "1:1"
        w, h = (int(x) for x in ratio.split(":"))
        if not caption or w <= 0 or h <= 0:
            raise ValueError
    except (ValueError, KeyError, AttributeError, TypeError):
        raise GenerationError("The prompt writer did not return a usable caption.")
    return caption, w / h


def select_pictures(caption, count):
    """H3 sees only the pictures the caption names, renumbered from 1.

    They go in the order the caption first names them. The caption opens with
    the main canvas, and H3 builds on its first image: given an earlier output
    plus the style reference it came from, numeric order made H3 redraw the
    reference instead of editing the output.
    """
    named = list(dict.fromkeys(int(n) for n in PICTURE_LABEL.findall(caption) if 0 < int(n) <= count))

    def relabel(m):
        n = int(m.group(1))
        return f"<Picture {named.index(n) + 1}>" if n in named else m.group(0)
    return PICTURE_LABEL.sub(relabel, caption), named


def output_size(aspect):
    aspect = min(3.0, max(1 / 3, aspect))
    width = math.sqrt(OUTPUT_PIXELS * aspect)
    return round(width / 32) * 32, round(width / aspect / 32) * 32


async def stream_writer(client, messages, on_reasoning):
    """Stream the writer, forwarding its reasoning; return the final content."""
    body = {"model": WRITER_MODEL, "messages": messages, "stream": True,
            "max_tokens": 32768, "temperature": 1.0, "top_p": 0.95}
    content = []
    try:
        async with client.stream("POST", f"{WRITER_URL}/v1/chat/completions", json=body,
                                 timeout=httpx.Timeout(30, read=600)) as response:
            if response.status_code != 200:
                detail = (await response.aread())[:300].decode(errors="replace")
                raise GenerationError(f"The prompt writer failed ({response.status_code}): {detail}")
            async for line in response.aiter_lines():
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                choices = json.loads(line[5:]).get("choices") or [{}]
                delta = choices[0].get("delta") or {}
                thought = delta.get("reasoning_content") or delta.get("reasoning")
                if thought:
                    await on_reasoning(thought)
                if delta.get("content"):
                    content.append(delta["content"])
    except httpx.HTTPError as e:
        raise GenerationError(f"The prompt writer {WRITER_MODEL} is unreachable: {e}")
    return "".join(content)


# --- ComfyUI ------------------------------------------------------------------

def build_workflow(caption, width, height, image_names, seed):
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "minimax_h3_image_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_image_vae_fp16.safetensors"}},
        "4": {"class_type": "MiniMaxH3ImageConditioning", "inputs": {
            "prompt": caption, "width": width, "height": height, "clip": ["2", 0], "vae": ["3", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 6, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["4", 1]}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": "chat/h3", "images": ["7", 0]}},
    }
    # Each reference goes through the same resizing as ComfyUI's edit template.
    for i, name in enumerate(image_names, start=1):
        load, total, multiple = f"{i}01", f"{i}02", f"{i}03"
        g[load] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g[total] = {"class_type": "ResizeImageMaskNode", "inputs": {
            "resize_type": "scale total pixels", "resize_type.megapixels": 4, "scale_method": "area",
            "input": [load, 0]}}
        g[multiple] = {"class_type": "ResizeImageMaskNode", "inputs": {
            "resize_type": "scale to multiple", "resize_type.multiple": 32, "scale_method": "area",
            "input": [total, 0]}}
        g["4"]["inputs"][f"images.image_{i}"] = [multiple, 0]
    return g


async def upload_picture(client, raw):
    name = f"chat-{hashlib.sha256(raw).hexdigest()[:24]}.png"
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    response = await client.post(f"{COMFY_URL}/upload/image", data={"overwrite": "true"},
                                 files={"image": (name, buf.getvalue(), "image/png")})
    response.raise_for_status()
    return response.json()["name"]


async def cancel_prompt(client, prompt_id):
    """Drop our job from ComfyUI's queue, or interrupt it if it is the one running."""
    try:
        queue = (await client.get(f"{COMFY_URL}/queue")).json()
        if any(item[1] == prompt_id for item in queue.get("queue_running", [])):
            await client.post(f"{COMFY_URL}/interrupt", json={"prompt_id": prompt_id})
        else:
            await client.post(f"{COMFY_URL}/queue", json={"delete": [prompt_id]})
    except (httpx.HTTPError, ValueError, IndexError, TypeError):
        logger.warning("could not cancel ComfyUI prompt %s", prompt_id)


async def render(client, workflow):
    """Queue the workflow and return the output JPEG bytes."""
    response = await client.post(f"{COMFY_URL}/prompt", json={"prompt": workflow})
    if response.status_code != 200:
        raise GenerationError(f"ComfyUI rejected the workflow: {response.text[:300]}")
    prompt_id = response.json()["prompt_id"]
    finished = False
    try:
        while True:
            await asyncio.sleep(1)
            history = (await client.get(f"{COMFY_URL}/history/{prompt_id}")).json().get(prompt_id)
            if history and history.get("status", {}).get("completed") is not None:
                break
        finished = True
    finally:
        if not finished:
            await cancel_prompt(client, prompt_id)

    status = history.get("status", {})
    if status.get("status_str") != "success":
        errors = [m[1].get("exception_message", "") for m in status.get("messages", [])
                  if m[0] == "execution_error"]
        raise GenerationError(f"ComfyUI failed: {' '.join(errors) or status.get('status_str')}")
    image = history["outputs"]["8"]["images"][0]
    png = await client.get(f"{COMFY_URL}/view", params={
        "filename": image["filename"], "subfolder": image["subfolder"], "type": image["type"]})
    png.raise_for_status()
    buf = io.BytesIO()
    Image.open(io.BytesIO(png.content)).convert("RGB").save(buf, "JPEG", quality=92)
    return buf.getvalue()


# --- request flow -----------------------------------------------------------

async def generate(messages, emit_reasoning):
    """Run one turn; return the assistant's Markdown content."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(60)) as client:
        pictures, instruction = read_conversation(messages)
        answer = await stream_writer(client, writer_messages(pictures, instruction), emit_reasoning)
        caption, aspect = parse_writer_answer(answer)

        caption, named = select_pictures(caption, len(pictures))
        width, height = output_size(aspect)
        await emit_reasoning(f"\n\n---\n**Prompt:** {caption}\n\nRendering {width}×{height}"
                             f"{f' from {len(named)} reference(s)' if named else ''}…\n")

        names = [await upload_picture(client, pictures[n - 1]) for n in named]
        started = time.monotonic()
        jpeg = await render(client, build_workflow(caption, width, height, names, random.randrange(2**48)))
        await emit_reasoning(f"Done in {time.monotonic() - started:.0f} s.\n")
    return f"![{MODEL_ID}](data:image/jpeg;base64,{base64.b64encode(jpeg).decode()})"


def _chunk(completion_id, created, delta, finish=None):
    return "data: " + json.dumps({
        "id": completion_id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }) + "\n\n"


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            (await client.get(f"{COMFY_URL}/system_stats")).raise_for_status()
    except httpx.HTTPError:
        raise HTTPException(503, "ComfyUI is not reachable")
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "mangchi"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    messages = payload.get("messages") or []
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if not payload.get("stream"):
        thoughts = []

        async def collect(text):
            thoughts.append(text)
        try:
            content = await generate(messages, collect)
        except GenerationError as e:
            content = str(e)
        return JSONResponse({
            "id": completion_id, "object": "chat.completion", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": content, "reasoning_content": "".join(thoughts)}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    async def events():
        queue = asyncio.Queue()

        async def emit(text):
            await queue.put(_chunk(completion_id, created, {"reasoning_content": text}))

        async def run():
            try:
                content = await generate(messages, emit)
            except GenerationError as e:
                content = str(e)
            except Exception:
                logger.exception("image generation failed")
                content = "Image generation failed; see the image service log."
            await queue.put(_chunk(completion_id, created, {"content": content}))
            await queue.put(_chunk(completion_id, created, {}, "stop"))
            await queue.put(None)

        yield _chunk(completion_id, created, {"role": "assistant"})
        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), PING_SECONDS)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                if item is None:
                    break
                yield item
            yield "data: [DONE]\n\n"
        finally:
            # A client that disconnects cancels the writer and the ComfyUI job.
            task.cancel()

    return StreamingResponse(events(), media_type="text/event-stream")
