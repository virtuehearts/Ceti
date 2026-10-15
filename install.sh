#!/usr/bin/env bash
#
# CETI installer — Python brain framework with a name, tasks, and teeth.
#
#   ./install.sh [--yes] [--no-ollama] [--no-models] [--build-llama]
#                [--node-name=NAME]
#                [--embedding-model-url=URL] [--research-model-url=URL]
#
# What it does:
#   1. Python 3.10+ venv at ./.venv, pip install -r requirements.txt
#   2. Ollama (unless --no-ollama) + qwen2.5-coder:1.5b (unless --no-models)
#   3. Embedding GGUF from Hugging Face (unless --embedding-model-url)
#   4. llama-server binary from a pinned llama.cpp build -> ./bin/
#      (falls back to the latest nightly on download failure;
#       --build-llama builds it from source instead)
#   5. research worker model (only with --research-model-url)
#   6. writes ceti-env.sh (OLLAMA_MODELS, PATH, CETI_* defaults)
#   7. asks for a node name (default CETI), generates identity.dat
#   8. boot self-test — proves the stack works before declaring success
#
# No expertise needed: defaults cover everything. Run ./run.sh after.
#
set -euo pipefail

YES=0; DO_OLLAMA=1; DO_MODELS=1; EMB_URL=""; RES_URL=""
BUILD_LLAMA=0; NODE_NAME=""; NODE_NAME_GIVEN=0
for a in "$@"; do
    case "$a" in
        --yes) YES=1 ;;
        --no-ollama) DO_OLLAMA=0 ;;
        --no-models) DO_MODELS=0 ;;
        --build-llama) BUILD_LLAMA=1 ;;
        --node-name=*) NODE_NAME="${a#*=}"; NODE_NAME_GIVEN=1 ;;
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

# Node name: asked once, up front, before the long downloads.
# --yes (or --node-name=) never prompts; default is CETI.
if [ "$NODE_NAME_GIVEN" = 0 ] && [ "$YES" = 0 ]; then
    read -r -p "Node name [CETI]: " NODE_NAME
fi
NODE_NAME="${NODE_NAME:-CETI}"
say "node name: $NODE_NAME"

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

# --- 3. embedding model (Hugging Face default) ------------------------------
EMB="$GGUF/embeddinggemma-2.gguf"
EMB_DEFAULT="https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/main/embeddinggemma-2-Q8_0.gguf"
if [ -n "$EMB_URL" ]; then
    say "downloading embedding model (from --embedding-model-url)"
    curl -fsSL -o "$EMB" "$EMB_URL"
elif [ ! -f "$EMB" ]; then
    say "downloading embedding model (embeddinggemma-2 Q8_0, Hugging Face)"
    curl -fsSL -o "$EMB" "$EMB_DEFAULT"
else
    say "embedding model already present"
fi

# --- 4. llama-server (embedding backend) ----------------------------------
# Pinned to a known-good build: llama.cpp b11630, the first verified build
# with gemma-embedding2 support. The repo moved ggerganov -> ggml-org, and
# the "latest" release no longer carries binaries (only nightly-tag.txt),
# so download URLs are constructed directly. The tarball ships a chain
# of shared libraries next to llama-server (libllama-server-impl,
# libllama-common, libllama, libggml, libmtmd, CPU-dispatch ggml libs) —
# the binary fails to start without them, so copy ALL lib*.so* files
# into $BIN with symlinks preserved (the loader resolves versioned
# SONAMEs like libllama.so.0 through the symlink chain).
#
# Fallbacks, in order:
#   download failure -> latest nightly build (same asset layout)
#   "unknown model architecture" -> re-run with --build-llama
#     (the nightly tag trails the pinned build, so it cannot help
#      a newer architecture — only a source build from main can)
LLAMA_BUILD="b11630"
NIGHTLY_TAG_URL="https://github.com/ggml-org/llama.cpp/releases/download/v0.6.0/nightly-tag.txt"
SERVER="$BIN/llama-server"

fetch_llama_server() {
    # fetch_llama_server <build> — download + install one llama.cpp build
    local build="$1" pat url tmp
    case "$OS-$ARCH" in
        Linux-x86_64)  pat="llama-$build-bin-ubuntu-x64.tar.gz" ;;
        Linux-aarch64) pat="llama-$build-bin-ubuntu-arm64.tar.gz" ;;
        Darwin-arm64)  pat="llama-$build-bin-macos-arm64.tar.gz" ;;
        *) warn "no prebuilt llama-server for $OS-$ARCH — embeddings disabled"; return 1 ;;
    esac
    url="https://github.com/ggml-org/llama.cpp/releases/download/$build/$pat"
    tmp="$(mktemp -d)"
    curl -fsSL -o "$tmp/srv.tar.gz" "$url" || { rm -rf "$tmp"; return 1; }
    tar xzf "$tmp/srv.tar.gz" -C "$tmp" || { rm -rf "$tmp"; return 1; }
    find "$tmp" -name "llama-server" -type f -exec cp {} "$SERVER" \;
    find "$tmp" \( -name "lib*.so*" -o -name "lib*.dylib" \) -exec cp -a {} "$BIN/" \;
    chmod +x "$SERVER"
    rm -rf "$tmp"
    # A partial copy must not report success: the binary and at least
    # one shared library have to be there (.so on Linux, .dylib on macOS).
    # NOTE: never `ls` two globs here — an unmatched glob stays literal
    # and makes ls fail even when the other glob matched.
    [ -x "$SERVER" ] || return 1
    for f in "$BIN"/lib*.so* "$BIN"/lib*.dylib; do
        [ -e "$f" ] && return 0
    done
    return 1
}

build_llama_server() {
    # build_llama_server — build llama-server from latest llama.cpp main.
    # Last resort for architectures no released binary knows yet.
    # Needs git + cmake + a C++ compiler; takes a while on low-spec iron
    # (10+ minutes on 2 cores). The checkout stays at $ROOT/.build/ so a
    # re-run is incremental instead of starting over.
    command -v git >/dev/null || die "git not found — needed for --build-llama"
    command -v cmake >/dev/null || die "cmake not found — needed for --build-llama"
    command -v c++ >/dev/null || command -v g++ >/dev/null \
        || die "no C++ compiler found — needed for --build-llama"
    local src="$ROOT/.build/llama.cpp"
    local jobs
    jobs="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
    say "building llama-server from source (llama.cpp main, this takes a while)"
    if [ -d "$src" ]; then
        git -C "$src" pull --ff-only 2>/dev/null \
            || warn "could not update existing checkout, building as-is"
    else
        git clone --depth 1 https://github.com/ggml-org/llama.cpp "$src" \
            || die "could not clone llama.cpp"
    fi
    cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release \
        || die "cmake configure failed"
    cmake --build "$src/build" --target llama-server -j"$jobs" \
        || die "llama-server build failed"
    cp "$src/build/bin/llama-server" "$SERVER" \
        || die "built llama-server not found at $src/build/bin/llama-server"
    # Copy every built shared library (.so on Linux, .dylib on macOS).
    # Globs that match nothing stay literal, so guard each file.
    for pat in "lib*.so*" "lib*.dylib"; do
        for f in "$src/build/bin"/$pat; do
            [ -e "$f" ] && cp -a "$f" "$BIN/"
        done
    done
    chmod +x "$SERVER"
    [ -x "$SERVER" ] || die "source build did not produce $SERVER"
    # Same guard as the fetch path: at least one shared library must
    # have landed in $BIN (never `ls` two globs — see above).
    for f in "$BIN"/lib*.so* "$BIN"/lib*.dylib; do
        [ -e "$f" ] && { say "llama-server -> $SERVER (built from source)"; return 0; }
    done
    die "source build produced no shared libraries in $BIN"
}

smoke_test_server() {
    # smoke_test_server — start the server on a scratch port, embed one
    # sentence, shut it down. 0 = works, 2 = unknown model architecture,
    # 1 = anything else. Proves the binary can run this model.
    local port=11439 log pid i
    log="$(mktemp)"
    "$SERVER" --embedding -m "$EMB" -b 4096 -ub 4096 \
        --port "$port" --host 127.0.0.1 >"$log" 2>&1 &
    pid=$!
    for i in $(seq 1 20); do
        if curl -sf --noproxy '*' -X POST "http://127.0.0.1:$port/v1/embeddings" \
                -H 'Content-Type: application/json' \
                -d '{"input":"smoke test","model":"x"}' 2>/dev/null \
                | grep -q '"embedding"'; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            rm -f "$log"
            return 0
        fi
        sleep 1
    done
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    if grep -q "unknown model architecture" "$log"; then
        rm -f "$log"
        return 2
    fi
    warn "embedding server smoke test failed (log kept at $log)"
    return 1
}

if [ ! -x "$SERVER" ]; then
    if [ "$BUILD_LLAMA" = 1 ]; then
        build_llama_server
    else
        say "fetching llama-server ($LLAMA_BUILD)"
        if ! fetch_llama_server "$LLAMA_BUILD"; then
            warn "pinned build $LLAMA_BUILD download failed — trying latest nightly"
            NIGHTLY="$(curl -fsSL "$NIGHTLY_TAG_URL" || true)"
            if [ -n "$NIGHTLY" ] && [ "$NIGHTLY" != "$LLAMA_BUILD" ]; then
                fetch_llama_server "$NIGHTLY" \
                    || die "could not fetch llama-server (pinned or nightly)"
                say "llama-server -> $SERVER (nightly $NIGHTLY)"
            else
                die "could not fetch llama-server (pinned build failed, no nightly tag)"
            fi
        else
            say "llama-server -> $SERVER"
        fi
    fi
else
    say "llama-server already present"
fi

# Prove the binary runs this model. Without a model there is nothing to
# prove, so an install with no model skips the smoke test.
if [ -x "$SERVER" ] && [ -f "$EMB" ]; then
    say "smoke-testing embedding server with the model"
    if smoke_test_server; then
        say "embedding server runs this model"
    else
        rc=$?
        if [ "$rc" = 2 ]; then
            die "this llama-server does not know the model architecture. Re-run with --build-llama to build the latest llama.cpp from source."
        else
            die "embedding server smoke test failed — see the warning above"
        fi
    fi
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

# --- 7. identity + node name ------------------------------------------------
# The Brain generates identity.dat on first run anyway; doing it here
# means the user sees their fingerprint at install time, and the node
# has an identity before first boot. An explicit --node-name= always
# wins; otherwise we only fill in the default, never clobbering a name
# the user already set (re-running install.sh keeps their name).
say "generating node identity (data/identity.dat) and setting the node name"
CETI_INSTALL_NODE_NAME="$NODE_NAME" CETI_INSTALL_NODE_NAME_GIVEN="$NODE_NAME_GIVEN" \
    "$VENV/bin/python" - <<'EOF'
import os
from ceti.agent import Agent, DEFAULT_NAME
from ceti.brain import Brain

b = Brain()
a = Agent(b)
given = os.environ.get("CETI_INSTALL_NODE_NAME_GIVEN") == "1"
name = os.environ.get("CETI_INSTALL_NODE_NAME", DEFAULT_NAME)
if given or a.name == DEFAULT_NAME:
    a.rename(name)
print("identity: %s" % b.identity.fingerprint)
print("node name: %s" % a.name)
EOF

# --- 8. smoke test ----------------------------------------------------------
say "running boot self-test"
# shellcheck disable=SC1090
source "$ENVSH"
"$VENV/bin/python" -m ceti.boot || \
    warn "boot degraded — missing pieces are warnings above, tiers fall through"

say "done. Next:"
echo "    source ./ceti-env.sh"
echo "    ./run.sh            # starts services and opens the CETI prompt"
