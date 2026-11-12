# Mangchi image generation

`MiniMax-H3-Image` on chat.lost.plus is a Grimoire `vllm-remote` entry with no
residency agent. It points at the `mangchi-image` service in
`docker-compose.mangchi.yml` (port 9710), which runs
`src/grimoire/image_server.py`:

1. The writer model (`H3_WRITER_MODEL`, default DeepSeek-V4.1-Flash on spark2)
   expands the conversation into an H3 caption with MiniMax's official
   prompt-expansion prompts in `templates/minimax-h3-image-pe/`.
2. ComfyUI (`comfyui.service`, port 8188) renders the caption at about 4 MP
   with the conversation's pictures that the caption names.

ComfyUI needs these files under `~/comfyui/models/`:

| Folder | File |
| --- | --- |
| `diffusion_models` | `minimax_h3_image_pruned_int8_convrot.safetensors` |
| `text_encoders` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| `vae` | `minimax_h3_image_vae_fp16.safetensors` |

All three come from Comfy-Org's `MiniMax-H3-Image` and `MiniMax-H3` Hugging
Face repositories. Together they hold about 50 GB of unified memory while
loaded.
