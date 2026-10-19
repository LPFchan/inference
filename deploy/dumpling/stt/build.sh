#!/bin/sh
set -eu
stt_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
stt_build="${GRIMOIRE_STT_BUILD:-$stt_dir/build}"
mkdir -p "$stt_build"
: "${EG2_ORIGINAL_APP:?Set EG2_ORIGINAL_APP to the installed original app}"
python3 "$stt_dir/prepare-original.py" --app "$EG2_ORIGINAL_APP" --output "$stt_build/libEG2Original.dylib"
xcrun swift-frontend -compile-module-from-interface "$stt_dir/EG2Transcriber.swiftinterface" \
    -module-name EG2Transcriber -sdk "$(xcrun --show-sdk-path)" \
    -target arm64-apple-macosx27.0 -o "$stt_build/EG2Transcriber.swiftmodule"
swiftc -O -I "$stt_build" "$stt_dir/OriginalWorker.swift" "$stt_build/libEG2Original.dylib" -o "$stt_build/eg2-original-worker"
swiftc -O -parse-as-library "$stt_dir/OpenWorker.swift" "$stt_dir"/open/*.swift -o "$stt_build/eg2-open-worker"
