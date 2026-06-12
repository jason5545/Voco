#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/LocalModels/EdgeTTSSmoke"
TEXT="${VOCO_QWEN3_ADAPTER_EDGE_TTS_TEXT:-世紀風電2072 今日股價183元}"
VOICE="${VOCO_QWEN3_ADAPTER_EDGE_TTS_VOICE:-zh-TW-HsiaoChenNeural}"
MP3_PATH="${OUTPUT_DIR}/century-wind-2072-stock-183.mp3"
WAV_PATH="${OUTPUT_DIR}/century-wind-2072-stock-183.wav"

mkdir -p "${OUTPUT_DIR}"

python3 -m edge_tts \
  --text "${TEXT}" \
  --voice "${VOICE}" \
  --write-media "${MP3_PATH}"

ffmpeg -y \
  -i "${MP3_PATH}" \
  -ac 1 \
  -ar 16000 \
  -sample_fmt s16 \
  "${WAV_PATH}"

echo "Wrote ${WAV_PATH}"
