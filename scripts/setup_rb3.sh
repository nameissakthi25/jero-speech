#!/usr/bin/env bash
# setup_rb3.sh — provision a fresh RB3 Gen2 (QCS6490, Ubuntu 24.04 aarch64) for
# jero-speech. Idempotent. Run from the repo root:  bash scripts/setup_rb3.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
PY="${PYTHON:-python3}"

echo "==> jero-speech setup on $(uname -m) ($(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}"))"
if [ "$(uname -m)" != "aarch64" ]; then
  echo "WARN: expected aarch64 (RB3 Gen2); continuing anyway."
fi

# --- system deps -----------------------------------------------------------
echo "==> apt deps (audio + build)"
sudo apt-get update -y
sudo apt-get install -y \
  python3 python3-venv python3-dev python3-pip \
  alsa-utils libsndfile1 libasound2-dev \
  portaudio19-dev ffmpeg git curl bzip2 ca-certificates

# --- python venv -----------------------------------------------------------
echo "==> venv at ${VENV}"
"${PY}" -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install --upgrade pip wheel

echo "==> python packages"
# sherpa-onnx ships prebuilt aarch64 wheels (bundles onnxruntime) -> STT + TTS + VAD.
pip install \
  sherpa-onnx \
  onnxruntime \
  openwakeword \
  silero-vad \
  numpy \
  soundfile \
  sounddevice \
  pyyaml

# --- TTS voice (Piper en_US-amy-medium) ------------------------------------
TTS_DIR="${REPO_ROOT}/models/tts/vits-piper-en_US-amy-medium"
if [ ! -f "${TTS_DIR}/en_US-amy-medium.onnx" ]; then
  echo "==> downloading Piper en_US-amy-medium voice + espeak-ng-data"
  mkdir -p "${REPO_ROOT}/models/tts"
  cd "${REPO_ROOT}/models/tts"
  curl -L -o amy.tar.bz2 \
    https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-medium.tar.bz2
  tar xjf amy.tar.bz2 && rm amy.tar.bz2
  cd "${REPO_ROOT}"
else
  echo "==> Piper voice already present"
fi

# --- model placeholders ----------------------------------------------------
# STT (Zipformer) is synced separately; point config.yaml stt.model_dir at it.
# Wake word + intent ONNX land from the A100 under ~/jero/{wakeword,intent}.
mkdir -p "${REPO_ROOT}/models/wakeword" "${REPO_ROOT}/models/intent"
link_if() { # link_if <src> <dst>
  if [ -f "$1" ] && [ ! -e "$2" ]; then ln -s "$1" "$2" && echo "   linked $2"; fi
}
link_if "${HOME}/jero/wakeword/hey_jero.onnx"   "${REPO_ROOT}/models/wakeword/hey_jero.onnx"
link_if "${HOME}/jero/wakeword/silero_vad.onnx" "${REPO_ROOT}/models/wakeword/silero_vad.onnx"
link_if "${HOME}/jero/intent/intent.onnx"       "${REPO_ROOT}/models/intent/intent.onnx"
link_if "${HOME}/jero/intent/post_process.py"   "${REPO_ROOT}/models/intent/post_process.py"

# --- pre-render canned lines ----------------------------------------------
echo "==> pre-rendering canned lines"
cd "${REPO_ROOT}"
python -m brain.speech.prerender || echo "WARN: prerender failed (check config.yaml tts.model_dir)"

# --- systemd ---------------------------------------------------------------
echo "==> installing systemd unit (edit paths/User in jero-speech.service first)"
sudo cp "${REPO_ROOT}/jero-speech.service" /etc/systemd/system/jero-speech.service
sudo systemctl daemon-reload
echo "   enable with: sudo systemctl enable --now jero-speech"

echo "==> done. Smoke test:  source ${VENV}/bin/activate && python -m brain.speech --print"
