#!/usr/bin/env python3
"""Expose the pinned original's existing library code to our headless Swift host.

Only Mach-O loader metadata changes. Machine code and model weights are intact.
The source executable is kept unchanged. No original binary is committed.
"""
import argparse
import hashlib
from pathlib import Path
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.app / "Contents/MacOS/EG2Transcribe"
    data = bytearray(source.read_bytes())
    if hashlib.sha256(data).hexdigest() != "7bcdc0085df93f5e5538e54e943d82c9701172361c56ec6a4241e325fa182348":
        raise ValueError("Original app does not match the reverse-engineered 1.6 ABI")
    if struct.unpack_from("<I", data)[0] != 0xFEEDFACF:
        raise ValueError("Expected the pinned arm64 Mach-O executable")
    count, size = struct.unpack_from("<II", data, 16)
    end = 32 + size
    name = str(args.output.resolve()).encode() + b"\0"
    command_size = (24 + len(name) + 7) // 8 * 8
    # Check actual section offsets before using linker header padding.
    offset = 32
    first_section = len(data)
    for _ in range(count):
        command, length = struct.unpack_from("<II", data, offset)
        if command == 0x19:
            sections = struct.unpack_from("<I", data, offset + 64)[0]
            for index in range(sections):
                section_offset = struct.unpack_from("<I", data, offset + 72 + index * 80 + 48)[0]
                if section_offset:
                    first_section = min(first_section, section_offset)
        offset += length
    if end + command_size > first_section or any(data[end:end + command_size]):
        raise ValueError("No safe Mach-O header padding")
    command = struct.pack("<6I", 0xD, command_size, 24, 0, 0, 0) + name
    data[end:end + command_size] = command + bytes(command_size - len(command))
    struct.pack_into("<I", data, 12, 6)  # MH_DYLIB
    struct.pack_into("<II", data, 16, count + 1, size + command_size)
    flags = struct.unpack_from("<I", data, 24)[0]
    struct.pack_into("<I", data, 24, flags & ~0x200000)  # MH_PIE is executable-only
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    sparkle = args.app / "Contents/Frameworks/Sparkle.framework/Versions/B/Sparkle"
    subprocess.run(["install_name_tool", "-change", "@rpath/Sparkle.framework/Versions/B/Sparkle",
                    str(sparkle.resolve()), str(args.output)], check=True)
    subprocess.run(["codesign", "--force", "--sign", "-", str(args.output)], check=True)
    print("Original executable SHA-256:", hashlib.sha256(source.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
