"""Recover the Keys overlay's refusal direction for DeepSeek-V4.1-Flash-1017 (RSH-20261113-001).

The overlay edits layers 10-35 `attn.wo_b` as W' = (I - lambda r r^T) W with one 5120-d direction r. This takes the
top left singular vector of (W' - W) at layers 10, 20 and 35 from the FP8 sidecar and stock FP8 shards, checks they
agree, and writes their mean as fp32.

Inputs in DIR: abl.safetensors (drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated@0bdc2036
wo_b_l10_35_1017.safetensors, gated) and base-13/23/38.safetensors (deepseek-ai/DeepSeek-V4.1-Flash-1017
model-000NN-of-00048.safetensors). Needs torch and safetensors (repo .venv).

    .venv/bin/python -I scripts/spark/dsv41_keys_direction.py DIR etc/spark/dsv41/keys_dir_1017.safetensors
"""
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SHARDS = {10: 13, 20: 23, 35: 38}  # layer -> stock shard holding its wo_b


def dequant(w, s):  # FP8 e4m3, UE8M0 32x32 block scales
    return w.float() * s.float().repeat_interleave(32, 0).repeat_interleave(32, 1)


def main(src, out):
    abl = safe_open(f"{src}/abl.safetensors", "pt")
    rs = []
    for layer, shard in SHARDS.items():
        base = safe_open(f"{src}/base-{shard}.safetensors", "pt")
        k = f"layers.{layer}.attn.wo_b"
        w = dequant(base.get_tensor(k + ".weight"), base.get_tensor(k + ".scale"))
        dw = dequant(abl.get_tensor(k + ".weight"), abl.get_tensor(k + ".scale")) - w
        u, s, _ = torch.linalg.svd(dw, full_matrices=False)
        r = u[:, 0]
        if rs and r @ rs[0] < 0:
            r = -r
        rw = r @ w
        lam = -(r @ dw @ rw) / (rw @ rw)
        share = float(s[0] ** 2 / (s ** 2).sum())
        print(f"layer {layer}: top-1 share {share:.3f}, fitted lambda {lam:.2f}")
        rs.append(r)
    r = torch.stack(rs).mean(0)
    r = r / r.norm()
    cos = [float(x @ r) for x in rs]
    print("cos to mean:", [round(c, 5) for c in cos])
    if min(cos) < 0.999:
        sys.exit("layers disagree on the direction; not writing")
    save_file({"direction": r.contiguous()}, out,
              metadata={"lambda": "3.5", "layers": "10-35", "overlay": "drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated@0bdc2036"})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
