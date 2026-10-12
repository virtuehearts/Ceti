#!/usr/bin/env bash
#
# CETI launcher — starts local services (if installed) and opens the prompt.
#
#   ./run.sh
#
# Services are started only if their binaries/models exist; anything
# missing just degrades the matching tier (the boot screen says what).
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
[ -f "$ROOT/ceti-env.sh" ] && source "$ROOT/ceti-env.sh"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "no venv — run ./install.sh first"; exit 1; }
mkdir -p "$ROOT/logs"

up() { curl -sf --noproxy '*' "$1" >/dev/null 2>&1; }

# --- ollama ---------------------------------------------------------------
if command -v ollama >/dev/null; then
    if ! up http://127.0.0.1:11434/api/tags; then
        echo "starting ollama..."
        OLLAMA_MODELS="${OLLAMA_MODELS:-$ROOT/models/ollama}" \
            nohup ollama serve >"$ROOT/logs/ollama.log" 2>&1 &
        for _ in $(seq 1 30); do up http://127.0.0.1:11434/api/tags && break; sleep 1; done
    fi
fi

# --- embedding server ------------------------------------------------------
SERVER="${CETI_LLAMA_SERVER:-$ROOT/bin/llama-server}"
MODEL="${CETI_EMBEDDING_MODEL:-$ROOT/models/gguf/embeddinggemma-2.gguf}"
if [ -x "$SERVER" ] && [ -f "$MODEL" ]; then
    if ! up http://127.0.0.1:11435/v1/models; then
        echo "starting embedding server..."
        nohup "$SERVER" --embedding -m "$MODEL" \
            -b 4096 -ub 4096 --port 11435 --host 127.0.0.1 \
            >"$ROOT/logs/embeddings.log" 2>&1 &
        for _ in $(seq 1 30); do up http://127.0.0.1:11435/v1/models && break; sleep 1; done
    fi
fi

exec "$PY" -m ceti.cli "$@"
