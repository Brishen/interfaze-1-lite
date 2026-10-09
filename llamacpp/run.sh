#!/usr/bin/env bash
# Interfaze 1 Lite on llama.cpp, without Docker or vLLM.
#
# The two language models run as quantized GGUFs under llama-server: the reasoning core
# (Qwen3.8 27B) and the document reader (Chandra OCR 2). The small perception models --
# speech, segmentation, line geometry and layout, forecasting -- have no GGUF form and
# run under torch/paddle in the perception and diarization sidecars, as in the container.
# Text guardrails are answered by the reasoning core, so there is no guard model.
#
#   uv sync --extra llamacpp          # once (on NixOS: inside nix-shell llamacpp/shell.nix)
#   llamacpp/run.sh                   # downloads missing GGUFs, then starts everything
#
# On NixOS this re-runs itself inside llamacpp/shell.nix.
#
# Every setting below is an environment variable; the defaults fit two 24 GB cards (the
# reasoning core on the first, everything else on the second) and also run on one card
# of 40 GB or more.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# On NixOS the torch and paddle wheels need the libraries llamacpp/shell.nix provides.
if [[ -e /etc/NIXOS && -z "${IN_NIX_SHELL:-}" ]] && command -v nix-shell >/dev/null; then
    exec nix-shell "${ROOT}/llamacpp/shell.nix" --run "$(printf '%q ' "${BASH_SOURCE[0]}" "$@")"
fi

LLAMA_SERVER="${LLAMA_SERVER:-${HOME}/llama.cpp/build-cuda-native/bin/llama-server}"
MODELS_DIR="${MODELS_DIR:-${HOME}/models/interfaze}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Reasoning core. Q4_K_XL is ~17.6 GB; on a card with room, Q5_K_XL or Q6_K is closer to
# the FP8 checkpoint the published scores were measured on.
BRAIN_REPO="${BRAIN_REPO:-unsloth/Qwen3.8-27B-GGUF}"
BRAIN_GGUF="${BRAIN_GGUF:-Qwen3.8-27B-UD-Q4_K_XL.gguf}"
BRAIN_MMPROJ="${BRAIN_MMPROJ:-mmproj-BF16.gguf}"
BRAIN_CTX="${BRAIN_CTX:-65536}"
BRAIN_PARALLEL="${BRAIN_PARALLEL:-4}"
# Grounding sends screenshots of up to ~4.2 MP, and a vision token covers 32x32 pixels.
BRAIN_IMAGE_MAX_TOKENS="${BRAIN_IMAGE_MAX_TOKENS:-4096}"

# Document reader. It is fine-tuned against one prompt and one output format, so the
# quantization is kept high: Q8_0 is 5.2 GB.
OCR_REPO="${OCR_REPO:-prithivMLmods/chandra-ocr-2-GGUF}"
OCR_GGUF="${OCR_GGUF:-chandra-ocr-2.Q8_0.gguf}"
OCR_MMPROJ="${OCR_MMPROJ:-chandra-ocr-2.mmproj-f16.gguf}"
# A dense page is read at up to 8.4 MP (8192 vision tokens) and may write 8192 tokens.
OCR_PARALLEL="${OCR_PARALLEL:-4}"
OCR_CTX="${OCR_CTX:-$(( OCR_PARALLEL * 18432 ))}"
OCR_IMAGE_MAX_TOKENS="${OCR_IMAGE_MAX_TOKENS:-8192}"

# Placement. llama.cpp names devices CUDA0, CUDA1, ...; the torch sidecars take a
# CUDA_VISIBLE_DEVICES index. With one GPU everything shares it.
GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
if (( GPU_COUNT >= 2 )); then
    BRAIN_DEVICE="${BRAIN_DEVICE:-CUDA0}"; OCR_DEVICE="${OCR_DEVICE:-CUDA1}"; SIDECAR_GPU="${SIDECAR_GPU:-1}"
else
    BRAIN_DEVICE="${BRAIN_DEVICE:-CUDA0}"; OCR_DEVICE="${OCR_DEVICE:-CUDA0}"; SIDECAR_GPU="${SIDECAR_GPU:-0}"
fi

# Extra flags, appended last so they win.
BRAIN_ARGS_EXTRA="${BRAIN_ARGS_EXTRA:-}"
OCR_ARGS_EXTRA="${OCR_ARGS_EXTRA:-}"

LOG_DIR="${LOG_DIR:-${ROOT}/llamacpp/logs}"
mkdir -p "${LOG_DIR}"

# On NixOS the driver's libcuda lives here, and llama-server needs it on the loader path.
[[ -d /run/opengl-driver/lib ]] && export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/current-system/sw/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ ! -x "${LLAMA_SERVER}" ]]; then
    echo "FATAL: no llama-server at ${LLAMA_SERVER}; set LLAMA_SERVER to your build's binary" >&2
    exit 1
fi
if [[ ! -x "${ROOT}/.venv/bin/uvicorn" ]]; then
    echo "FATAL: no Python environment; run 'uv sync --extra llamacpp' first" >&2
    exit 1
fi
PY_BIN="${ROOT}/.venv/bin"

# Component mapping for the sidecars, as the container reads it: a COMPONENT_* you pass
# wins, then components.env, then components.env.example.
load_components() {
    local key value
    while IFS='=' read -r key value || [[ -n "${key}" ]]; do
        value="${value%$'\r'}"
        [[ "${key}" =~ ^COMPONENT_[A-Z_]+$ && -n "${value}" ]] || continue
        [[ -n "${!key:-}" ]] || export "${key}=${value}"
    done < "$1"
}
[[ -f components.env ]] && load_components components.env
load_components components.env.example
# The two language models are the GGUFs above, not these repos.
export COMPONENT_BRAIN="${BRAIN_REPO}" COMPONENT_OCR_VLM="${OCR_REPO}"
COMPONENT_NAMES="$(env | grep '^COMPONENT_' | grep -v '^COMPONENT_NAMES=' | cut -d= -f2- | paste -sd, -)"
export COMPONENT_NAMES

export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${hf_token:-}}}"
[[ -n "${HF_TOKEN}" ]] && export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
[[ -n "${HF_TOKEN}" ]] || unset HF_TOKEN

# resolve VAR REPO DIR: an absolute path is used as is; a file name is looked for under
# MODELS_DIR/DIR and downloaded from REPO if it is not there.
resolve() {
    local var="$1" repo="$2" dir="$3" file="${!1}"
    if [[ "${file}" == /* ]]; then
        [[ -f "${file}" ]] || { echo "FATAL: ${var}=${file} does not exist" >&2; exit 1; }
        return
    fi
    if [[ ! -f "${MODELS_DIR}/${dir}/${file}" ]]; then
        echo "      downloading ${repo}/${file}"
        "${PY_BIN}/hf" download "${repo}" "${file}" --local-dir "${MODELS_DIR}/${dir}" >/dev/null
    fi
    printf -v "${var}" '%s' "${MODELS_DIR}/${dir}/${file}"
}

echo "[0/5] weights in ${MODELS_DIR}"
resolve BRAIN_GGUF "${BRAIN_REPO}" brain
resolve BRAIN_MMPROJ "${BRAIN_REPO}" brain
resolve OCR_GGUF "${OCR_REPO}" ocr
resolve OCR_MMPROJ "${OCR_REPO}" ocr

PIDS=()
term() { echo "shutting down"; kill "${PIDS[@]}" 2>/dev/null || true; wait || true; }
trap 'term; exit 0' SIGTERM SIGINT

# wait_for NAME URL PID BUDGET_S LOG
wait_for() {
    local started=$SECONDS
    until curl -fsS "$2" >/dev/null 2>&1; do
        if ! kill -0 "$3" 2>/dev/null; then
            echo "FATAL: $1 exited during startup; last lines of $5:" >&2
            tail -n 20 "$5" >&2
            term; exit 1
        fi
        if (( SECONDS - started > $4 )); then echo "FATAL: $1 not healthy after $4 s (see $5)" >&2; term; exit 1; fi
        sleep 2
    done
    echo "      $1 ready in $(( SECONDS - started )) s"
}

# Boot order matters for the same reason it does in the container: llama-server sizes
# what it can offload from the memory free when it starts, so the language models load
# before the torch sidecars take their share.
echo "[1/5] reasoning core on :8001 (${BRAIN_DEVICE})"
# shellcheck disable=SC2086
"${LLAMA_SERVER}" \
    -m "${BRAIN_GGUF}" --mmproj "${BRAIN_MMPROJ}" \
    --host 127.0.0.1 --port 8001 --alias interfaze-lite \
    --device "${BRAIN_DEVICE}" -ngl all \
    -c "${BRAIN_CTX}" -np "${BRAIN_PARALLEL}" --kv-unified \
    -fa on --jinja --reasoning-format deepseek \
    --image-max-tokens "${BRAIN_IMAGE_MAX_TOKENS}" \
    --no-webui ${BRAIN_ARGS_EXTRA} \
    >"${LOG_DIR}/brain.log" 2>&1 &
PIDS+=($!); wait_for "reasoning core" http://127.0.0.1:8001/health "$!" 1800 "${LOG_DIR}/brain.log"

echo "[2/5] document reader on :8004 (${OCR_DEVICE})"
# Reasoning parsing off: the reader writes HTML straight away, and every token of it
# must reach `content`.
# shellcheck disable=SC2086
"${LLAMA_SERVER}" \
    -m "${OCR_GGUF}" --mmproj "${OCR_MMPROJ}" \
    --host 127.0.0.1 --port 8004 --alias document-reader \
    --device "${OCR_DEVICE}" -ngl all \
    -c "${OCR_CTX}" -np "${OCR_PARALLEL}" --kv-unified \
    -fa on --jinja --reasoning-format none \
    --image-max-tokens "${OCR_IMAGE_MAX_TOKENS}" \
    --no-webui ${OCR_ARGS_EXTRA} \
    >"${LOG_DIR}/ocr.log" 2>&1 &
PIDS+=($!); wait_for "document reader" http://127.0.0.1:8004/health "$!" 1800 "${LOG_DIR}/ocr.log"

export BRAIN_BACKEND=llamacpp \
       BRAIN_URL=http://127.0.0.1:8001 \
       BRAIN_SERVED_NAME=interfaze-lite \
       PERCEPTION_URL=http://127.0.0.1:8002 \
       DIARIZE_URL=http://127.0.0.1:8003 \
       OCR_VLM_URL=http://127.0.0.1:8004 \
       PERCEPTION_PRELOAD="${PERCEPTION_PRELOAD:-asr_fallback,segmenter,forecaster}" \
       MAX_TOOL_STEPS="${MAX_TOOL_STEPS:-8}" \
       ENABLE_FAST_ASR="${ENABLE_FAST_ASR:-0}" \
       TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[3/5] perception on :8002 (GPU ${SIDECAR_GPU})"
CUDA_VISIBLE_DEVICES="${SIDECAR_GPU}" "${PY_BIN}/uvicorn" interfaze_lite.services.perception:app \
    --host 127.0.0.1 --port 8002 >"${LOG_DIR}/perception.log" 2>&1 &
PIDS+=($!); wait_for "perception" http://127.0.0.1:8002/health "$!" 1800 "${LOG_DIR}/perception.log"

echo "[4/5] diarization on :8003 (GPU ${SIDECAR_GPU})"
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "      HF_TOKEN is unset: speaker attribution will fail until it is set (the"
    echo "      diarization model is gated); everything else works without it"
fi
CUDA_VISIBLE_DEVICES="${SIDECAR_GPU}" "${PY_BIN}/uvicorn" interfaze_lite.services.diarize:app \
    --host 127.0.0.1 --port 8003 >"${LOG_DIR}/diarize.log" 2>&1 &
PIDS+=($!); wait_for "diarization" http://127.0.0.1:8003/health "$!" 600 "${LOG_DIR}/diarize.log"

echo "[5/5] interfaze-1-lite on :${PORT}"
"${PY_BIN}/uvicorn" interfaze_lite.app:app --host "${HOST}" --port "${PORT}" \
    >"${LOG_DIR}/orchestrator.log" 2>&1 &
PIDS+=($!); wait_for "orchestrator" "http://127.0.0.1:${PORT}/health" "$!" 300 "${LOG_DIR}/orchestrator.log"
echo "ready: POST http://localhost:${PORT}/v1/chat/completions (logs in ${LOG_DIR})"

# One process dying takes the rest down, rather than serving with a piece missing.
set +e
wait -n "${PIDS[@]}"
echo "a service exited (status $?); stopping" >&2
term
exit 1
