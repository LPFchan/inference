# Training

Reproducible, opt-in model adaptation jobs belong here. Keep weights, audio,
prepared manifests, checkpoints, dependency freezes and run logs outside Git
under `~/models/stt`. Each workflow documents its input provenance, held-out
split, runtime requirements, checkpoint format and acceptance gates. Training
never changes a production model automatically.

- [Qwen3-ASR SFT and GRPO](qwen_asr/README.md): Korean, Japanese and Mandarin
  adaptation with English replay on Apple Silicon.
