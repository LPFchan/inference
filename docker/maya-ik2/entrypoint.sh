#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="/models/inkling-2-mini"
GGUF_PATH="${MODEL_DIR}/Inkling-2-Mini-UD-Q4_K_XL.gguf"
TOKENIZER_DIR="${MODEL_DIR}/tokenizer"

if [ ! -f "${GGUF_PATH}" ]; then
    echo "ERROR: model weights not found at ${GGUF_PATH}" >&2
    exit 1
fi

if [ ! -f "${TOKENIZER_DIR}/chat_template.jinja" ]; then
    echo "Extracting tokenizer to ${TOKENIZER_DIR}..."
    /opt/venv/bin/python /app/tools/strata_tokenizer.py \
        --gguf "${GGUF_PATH}" \
        --out "${MODEL_DIR}" \
        --chat-template /app/tools/ik2/chat_template.jinja \
        --check
fi

cd /app
exec /opt/venv/bin/python serve/server.py \
    --engine maya \
    --config /app/ik2-mini.json \
    --host 0.0.0.0 \
    --port 18000 \
    --fit-max-tokens
