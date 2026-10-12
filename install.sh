#!/usr/bin/env bash
#
# CETI installer — Python brain framework with a name, tasks, and teeth.
#
#   ./install.sh [--yes] [--no-ollama] [--no-models]
#                [--embedding-model-url=URL] [--research-model-url=URL]
#
# What it does:
#   1. Python 3.10+ venv at ./.venv, pip install -r requirements.txt
#   2. Ollama (unless --no-ollama) + qwen2.5-coder:1.5b (unless --no-models)
#   3. llama-server binary from the latest llama.cpp release -> ./bin/
#   4. embedding GGUF download (only with --embedding-model-url)
#   5. research worker model (only with --research-model-url)
#   6. writes ceti-env.sh (OLLAMA_MODELS, PATH, CETI_* defaults)
#
# The brain degrades gracefully: no Ollama -> TF-IDF memory + AIML only;
# no embedding model -> sqlite memory; no research stack -> research
# tasks skip with a clear message. Run ./run.sh after installing.
#
set -euo pipefail

YES=0; DO_OLLAMA=1; DO_MODELS=1; EMB_URL=""; RES_URL=""
for a in "$@"; do
    case "$a" in
        --yes) YES=1 ;;
        --no-ollama) DO_OLLAMA=0 ;;
        --no-models) DO_MODELS=0 ;;
        --embedding-model-url=*) EMB_URL="${a#*=}" ;;
        --research-model-url=*) RES_URL="${a#*=}" ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown flag: $a (try --help)"; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
BIN="$ROOT/bin"
MODELS="$ROOT/models"
OLLAMA_MODELS="$MODELS/ollama"
GGUF="$MODELS/gguf"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mXX\033[0m %s\n' "$*"; exit 1; }
ask()  { [ "$YES" = 1 ] && return 0; read -r -p "$1 [y/N] " r; [ "$r" = y ] || [ "$r" = Y ]; }

OS="$(uname -s)"; ARCH="$(uname -m)"

# --- 1. python -----------------------------------------------------------
say "checking python"
PY=python3
command -v $PY >/dev/null || die "python3 not found — install Python 3.10+ first"
PYVER="$($PY -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
$PY -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
    || die "python $PYVER too old — need 3.10+"

if [ ! -x "$VENV/bin/python" ]; then
    say "creating venv at $VENV"
    $PY -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
say "installing python deps"
"$VENV/bin/pip" install -q -r "$ROOT/requirements.txt"

mkdir -p "$BIN" "$OLLAMA_MODELS" "$GGUF"

# --- 2. ollama -----------------------------------------------------------
export OLLAMA_MODELS
if [ "$DO_OLLAMA" = 1 ]; then
    if ! command -v ollama >/dev/null; then
        say "ollama not found"
        if [ "$OS" = "Linux" ]; then
            if ask "install ollama via the official script (needs sudo)?"; then
                curl -fsSL https://ollama.com/install.sh | sh
            else
                warn "skipping ollama install — local LLM tiers will be down"
                DO_MODELS=0
            fi
        else
            warn "on $OS install ollama from https://ollama.com/download then re-run"
            DO_MODELS=0
        fi
    fi
    if [ "$DO_MODELS" = 1 ] && command -v ollama >/dev/null; then
        say "pulling qwen2.5-coder:1.5b (the handler model, ~1GB)"
        ollama pull qwen2.5-coder:1.5b
    fi
fi

# --- 3. llama-server (embedding backend) ----------------------------------
SERVER="$BIN/llama-server"
if [ ! -x "$SERVER" ]; then
    say "fetching llama-server"
    case "$OS-$ARCH" in
        Linux-x86_64)  PAT="bin-ubuntu-x64.zip" ;;
        Linux-aarch64) PAT="bin-ubuntu-arm64.zip" ;;
        Darwin-arm64)  PAT="bin-macos-arm64.zip" ;;
        *) warn "no prebuilt llama-server for $OS-$ARCH — embeddings disabled"; PAT="" ;;
    esac
    if [ -n "$PAT" ]; then
        API="https://api.github.com/ggerganov/llama.cpp/releases/latest"
        URL="$(curl -fsSL "$API" | grep '"browser_download_url"' \
               | grep "$PAT" | head -1 | sed 's/.*"browser_download_url": *"//; s/".*//')"
        [ -n "$URL" ] || die "could not find $PAT in the latest llama.cpp release"
        tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
        curl -fsSL -o "$tmp/srv.zip" "$URL"
        unzip -q -o "$tmp/srv.zip" -d "$tmp"
        find "$tmp" -name "llama-server" -type f -exec cp {} "$SERVER" \;
        chmod +x "$SERVER"
        trap - EXIT; rm -rf "$tmp"
        say "llama-server -> $SERVER"
    fi
else
    say "llama-server already present"
fi

# --- 4. embedding model ----------------------------------------------------
EMB="$GGUF/embeddinggemma-2.gguf"
if [ -n "$EMB_URL" ]; then
    say "downloading embedding model"
    curl -fsSL -o "$EMB" "$EMB_URL"
elif [ ! -f "$EMB" ]; then
    warn "no embedding model. The brain will use TF-IDF memory instead of vectors."
    warn "to enable vectors: download a 768-dim embeddinggemma-2 GGUF from"
    warn "Hugging Face and re-run with --embedding-model-url=URL"
fi

# --- 5. research worker model (optional) -----------------------------------
if [ -n "$RES_URL" ] && command -v ollama >/dev/null; then
    say "creating research worker model from $RES_URL"
    ollama create lfm2.5-2.6b-research -f - <<EOF
FROM $RES_URL
PARAMETER num_ctx 8192
PARAMETER temperature 0.3
PARAMETER num_predict 2200
SYSTEM """You are a precise research assistant. Output only the requested markdown brief. No preamble, no meta-commentary, no visible reasoning trace."""
EOF
elif [ "$DO_MODELS" = 1 ]; then
    warn "no research model URL given — deep research cycles stay disabled."
    warn "pass --research-model-url=<GGUF URL> (a ~2.6B instruct GGUF, e.g. LFM2.5)"
fi

# --- 6. env file -----------------------------------------------------------
ENVSH="$ROOT/ceti-env.sh"
cat > "$ENVSH" <<EOF
# source this before running CETI:  source ./ceti-env.sh
export OLLAMA_MODELS="$OLLAMA_MODELS"
export CETI_LLAMA_SERVER="$SERVER"
export CETI_EMBEDDING_MODEL="$EMB"
export PATH="$BIN:\$PATH"
EOF
say "wrote $ENVSH"

# --- 7. smoke test ----------------------------------------------------------
say "running boot self-test"
# shellcheck disable=SC1090
source "$ENVSH"
"$VENV/bin/python" -m ceti.boot || \
    warn "boot degraded — missing pieces are warnings above, tiers fall through"

say "done. Next:"
echo "    source ./ceti-env.sh"
echo "    ./run.sh            # starts services and opens the CETI prompt"
