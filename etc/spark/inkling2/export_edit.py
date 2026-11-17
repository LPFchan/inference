#!/usr/bin/env python3
"""The refusal edit (edit.pt: Q [hidden][k] orthonormal, s_attn and s_mlp per layer) as TensorFold reads it:
edit.json {"hidden", "rank", "s_attn", "s_mlp"} and q.f32, Q row-major fp32.

Usage: export_edit.py EDIT_PT OUT_DIR
"""
import json
import sys
from pathlib import Path

import torch


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    e = torch.load(src, weights_only=False)
    q = e["Q"].float().contiguous()
    gram = q.double().T @ q.double()
    assert torch.allclose(gram, torch.eye(q.shape[1], dtype=torch.float64), atol=1e-5), "Q is not orthonormal"
    out.mkdir(parents=True, exist_ok=True)
    q.numpy().tofile(out / "q.f32")
    meta = {"hidden": q.shape[0], "rank": q.shape[1], "s_attn": e["s_attn"].tolist(), "s_mlp": e["s_mlp"].tolist(),
            "name": e.get("name", "")}
    (out / "edit.json").write_text(json.dumps(meta))
    print(f"{out}: Q {tuple(q.shape)}, edited layers {sum(s > 0 for s in meta['s_attn'])} attention, "
          f"{sum(s > 0 for s in meta['s_mlp'])} MoE")


if __name__ == "__main__":
    main()
