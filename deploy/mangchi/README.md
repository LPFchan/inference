# Mangchi image generation

`MiniMax-H3-Image` and `MiniMax-H3-Image-PE` on chat.lost.plus are Grimoire
`vllm-remote` entries with no residency agent. Both point at the
`mangchi-image` service in `docker-compose.mangchi.yml` (port 9710), which runs
`src/grimoire/image_server.py` and tells them apart by `model`:

- `MiniMax-H3-Image` (raw): the last user message's text is the H3 caption,
  word for word, and its attachments are the references. An optional `size`
  field (`"1536x1024"` or `"16:9"`) sets the shape; otherwise the main
  reference's shape, or square. Meant for a stronger model or agent that
  writes the caption itself.
- `MiniMax-H3-Image-PE`: the writer model (`H3_WRITER_MODEL`, default
  DeepSeek-V4.1-Flash on spark2) first expands the whole conversation into an
  H3 caption with MiniMax's official prompt-expansion prompts in
  `src/grimoire/minimax_h3_pe/`.

ComfyUI (`comfyui.service`, port 8188) renders at about 4 MP with the pictures
the caption names as `<Picture N>`, in first-mention order; a raw caption that
names none uses every attachment in order.

ComfyUI needs these files under `~/comfyui/models/`:

| Folder | File |
| --- | --- |
| `diffusion_models` | `minimax_h3_image_pruned_int8_convrot.safetensors` |
| `text_encoders` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| `vae` | `minimax_h3_image_vae_fp16.safetensors` |

All three come from Comfy-Org's `MiniMax-H3-Image` and `MiniMax-H3` Hugging
Face repositories. Together they hold about 50 GB of unified memory while
loaded.
