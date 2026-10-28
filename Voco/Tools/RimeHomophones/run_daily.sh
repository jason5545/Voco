#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

VOCO_RIME_PYTHON=${VOCO_RIME_PYTHON:-/Library/Frameworks/Python.framework/Versions/3.14/bin/python3}
VOCO_RIME_RSYNC=${VOCO_RIME_RSYNC:-/usr/bin/rsync}
VOCO_RIME_OPENCC=${VOCO_RIME_OPENCC:-/opt/homebrew/bin/opencc}
VOCO_RIME_SSL_CERT_FILE=${VOCO_RIME_SSL_CERT_FILE:-/etc/ssl/cert.pem}
VOCO_RIME_HOME=${VOCO_RIME_HOME:-$HOME/Library/Rime}
VOCO_RIME_DICT_MANAGER=${VOCO_RIME_DICT_MANAGER:-/Library/Input Methods/Squirrel.app/Contents/MacOS/rime_dict_manager}
VOCO_RIME_ESSAY=${VOCO_RIME_ESSAY:-/Library/Input Methods/Squirrel.app/Contents/SharedSupport/essay.txt}
VOCO_RIME_TW_VARIANTS=${VOCO_RIME_TW_VARIANTS:-$HOME/GitHub/rime-penglai/bench-data/TWVariants.txt}
VOCO_RIME_AUTO_APPLY_MODEL=${VOCO_RIME_AUTO_APPLY_MODEL:-$HOME/Library/Application Support/com.jasonchien.Voco/AutoApplyModels/full-db.auto-apply-model.json}
VOCO_RIME_ACTIVE_MODEL=${VOCO_RIME_ACTIVE_MODEL:-$VOCO_RIME_AUTO_APPLY_MODEL}
VOCO_RIME_OUTPUT_DIR=${VOCO_RIME_OUTPUT_DIR:-$HOME/Library/Application Support/com.jasonchien.Voco/RimeHomophones}
VOCO_RIME_GENERATED_DATA=${VOCO_RIME_GENERATED_DATA:-$VOCO_RIME_OUTPUT_DIR/rime-homophones.json}
VOCO_RIME_APPROVED_EXCEPTIONS=${VOCO_RIME_APPROVED_EXCEPTIONS:-$VOCO_RIME_OUTPUT_DIR/approved-b-subset-exceptions.json}
VOCO_RIME_VOCO_STORE=${VOCO_RIME_VOCO_STORE:-$HOME/Library/Application Support/com.jasonchien.Voco/default.store}
VOCO_RIME_EVIDENCE_JSONL=${VOCO_RIME_EVIDENCE_JSONL:-$HOME/Library/Application Support/com.jasonchien.Voco/AutoApplyControl/evidence.jsonl}
VOCO_RIME_ARTIFACT_DIR=${VOCO_RIME_ARTIFACT_DIR:-$HOME/Library/Application Support/com.jasonchien.Voco/AutoApplyControl/artifacts/rime-homophones}
VOCO_RIME_S2TWP_CACHE=${VOCO_RIME_S2TWP_CACHE:-$VOCO_RIME_OUTPUT_DIR/rows-s2twp-cache.json}
VOCO_RIME_WORDFREQ=${VOCO_RIME_WORDFREQ:-$PROJECT_DIR/Sources/Resources/ChineseCorrection/word_freq.tsv}
VOCO_AUTO_APPLY_CONTROL_CLI=${VOCO_AUTO_APPLY_CONTROL_CLI:-$PROJECT_DIR/Tools/voco_auto_apply_control.py}
export VOCO_RIME_PYTHON VOCO_RIME_RSYNC VOCO_RIME_OPENCC VOCO_RIME_HOME VOCO_RIME_DICT_MANAGER VOCO_RIME_ESSAY
export SSL_CERT_FILE="$VOCO_RIME_SSL_CERT_FILE"
export VOCO_RIME_TW_VARIANTS VOCO_RIME_AUTO_APPLY_MODEL VOCO_RIME_ACTIVE_MODEL
export VOCO_RIME_OUTPUT_DIR VOCO_RIME_GENERATED_DATA VOCO_RIME_APPROVED_EXCEPTIONS VOCO_RIME_VOCO_STORE
export VOCO_RIME_EVIDENCE_JSONL VOCO_RIME_ARTIFACT_DIR VOCO_RIME_S2TWP_CACHE
export VOCO_RIME_WORDFREQ VOCO_AUTO_APPLY_CONTROL_CLI

"$VOCO_RIME_PYTHON" -I -B "$SCRIPT_DIR/generate.py"
if [ "${VOCO_RIME_DRY_RUN:-0}" = "1" ]; then
	"$VOCO_RIME_PYTHON" -I -B "$SCRIPT_DIR/mine.py" --dry-run
else
	"$VOCO_RIME_PYTHON" -I -B "$SCRIPT_DIR/mine.py"
fi
