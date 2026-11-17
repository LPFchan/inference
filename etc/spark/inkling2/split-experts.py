#!/usr/bin/env python3
"""Cut the EXL3 expert files down to one rank's half of every expert (TF_INKLING_EXPERTS reads either form).
w13: this rank's output tile columns (and svh); w2: its input tile rows (and suh). Everything else is copied.
Usage: split-experts.py SRC_EXL3_DIR DST_DIR RANK [WORLD=2]"""
import json, os, re, struct, sys

src, dst, rank = sys.argv[1], sys.argv[2], int(sys.argv[3])
world = int(sys.argv[4]) if len(sys.argv) > 4 else 2
size = {"I16": 2, "F16": 2, "I32": 4, "BF16": 2, "F32": 4}

def cut(name, meta, data):
    m = re.fullmatch(r"experts\.\d+\.(w13|w2)\.(trellis|suh|svh)", name)
    if not m: return meta["shape"], data
    proj, part = m.groups()
    shape = meta["shape"]
    if proj == "w13" and part == "trellis":
        rows, cols, words = shape
        width = cols // world * words * 2
        line = cols * words * 2
        out = b"".join(data[i * line + rank * width:][:width] for i in range(rows))
        return [rows, cols // world, words], out
    if proj == "w13" and part == "svh" or proj == "w2" and part == "suh":
        n = shape[0] // world
        return [n], data[rank * n * 2:][:n * 2]
    if proj == "w2" and part == "trellis":
        rows, cols, words = shape
        piece = rows // world * cols * words * 2
        return [rows // world, cols, words], data[rank * piece:][:piece]
    return shape, data

for layer in sorted(os.listdir(os.path.join(src, "layers"))):
    os.makedirs(os.path.join(dst, "layers", layer), exist_ok=True)
    for file in sorted(os.listdir(os.path.join(src, "layers", layer))):
        if not file.startswith("experts-"): continue
        target = os.path.join(dst, "layers", layer, file)
        if os.path.exists(target): continue
        with open(os.path.join(src, "layers", layer, file), "rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(length))
            body = f.read()
        out_header, chunks, at = {}, [], 0
        for name, meta in header.items():
            if name == "__metadata__": out_header[name] = meta; continue
            b, e = meta["data_offsets"]
            shape, data = cut(name, meta, memoryview(body)[b:e])
            out_header[name] = {"dtype": meta["dtype"], "shape": shape, "data_offsets": [at, at + len(data)]}
            chunks.append(data); at += len(data)
        text = json.dumps(out_header, separators=(",", ":")).encode()
        text += b" " * (-len(text) % 8)
        with open(target + ".part", "wb") as f:
            f.write(struct.pack("<Q", len(text))); f.write(text)
            for c in chunks: f.write(c)
        os.rename(target + ".part", target)
    print(layer, flush=True)
