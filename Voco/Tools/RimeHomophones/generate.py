#!/usr/bin/env python3
"""從 Rime 副本產生 Voco 私有同音字資料檔。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    RimeProfile,
    active_rime_profile,
    configure_variants,
    load_char_readings,
    load_essay,
    required_path,
)

FORMAT = "voco.rime-homophones.v1"
READING_WEIGHT_MIN = 5.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rime-dir")
    parser.add_argument("--dict-manager")
    parser.add_argument("--essay")
    parser.add_argument("--tw-variants")
    parser.add_argument("--auto-apply-model")
    parser.add_argument("--output-dir")
    parser.add_argument("--rsync")
    return parser.parse_args()


def export_learning_records(
    dict_manager: Path,
    staged_rime: Path,
    output: Path,
    profile: RimeProfile,
) -> None:
    subprocess.run(
        [str(dict_manager), "--export", profile.schema, str(output)],
        cwd=staged_rime,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def read_learning_records(path: Path) -> list[dict]:
    entries = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.startswith("#") or "\t" not in line:
                continue
            word, code, raw_commits = line.rstrip("\n").split("\t")
            commits = int(raw_commits)
            if commits <= 0:
                continue
            entries.append({"word": word, "syllables": code.split(), "commits": commits})
    return entries


def filtered_readings(readings: dict[str, dict[str, float]]) -> dict[str, list[dict]]:
    output = {}
    for char, values in readings.items():
        common = [(code, weight) for code, weight in values.items() if weight >= READING_WEIGHT_MIN]
        # Oracle 對沒有任何 ≥5% 讀音的字，回退到該字所有已列讀音。
        selected = common or list(values.items())
        output[char] = [
            {"syllable": code, "weightPercent": weight}
            for code, weight in selected
        ]
    return output


def deferred_patterns(path: Path) -> list[str]:
    model = json.loads(path.read_text(encoding="utf-8"))
    return sorted({
        policy["sourcePattern"]
        for policy in model.get("policies", [])
        if policy.get("autoApplyMode") == "apply"
        and policy.get("policyType") == "scopedReplacement"
        and policy.get("contextRequired") is True
        and policy.get("sourcePattern")
    })


def atomic_json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, separators=(",", ":"))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    rime_source = required_path(args.rime_dir, "VOCO_RIME_HOME", "Rime 原始資料夾")
    dict_manager = required_path(args.dict_manager, "VOCO_RIME_DICT_MANAGER", "rime_dict_manager")
    essay_path = required_path(args.essay, "VOCO_RIME_ESSAY", "essay 詞表")
    variants_path = required_path(args.tw_variants, "VOCO_RIME_TW_VARIANTS", "TWVariants")
    model_path = required_path(args.auto_apply_model, "VOCO_RIME_AUTO_APPLY_MODEL", "auto-apply model")
    output_dir = required_path(args.output_dir, "VOCO_RIME_OUTPUT_DIR", "同音字資料輸出資料夾")
    rsync_raw = args.rsync or os.environ.get("VOCO_RIME_RSYNC") or shutil.which("rsync")
    if not rsync_raw:
        raise SystemExit("請用 --rsync 或 VOCO_RIME_RSYNC 指定 rsync 執行檔")
    rsync_path = Path(rsync_raw).expanduser()

    if not rime_source.is_dir():
        raise SystemExit(f"Rime 原始資料夾不存在：{rime_source}")
    if not dict_manager.is_file() or not os.access(dict_manager, os.X_OK):
        raise SystemExit(f"rime_dict_manager 不存在或不可執行：{dict_manager}")

    # librime/rime_dict_manager 僅在臨時副本執行，絕不碰正在使用的 ~/Library/Rime。
    with tempfile.TemporaryDirectory(prefix="voco-rime-homophones-") as temporary_root:
        staged_rime = Path(temporary_root) / "rime"
        staged_rime.mkdir()
        subprocess.run(
            [
                str(rsync_path), "-a",
                "--exclude", "sync",
                "--exclude", "build",
                "--exclude", "log",
                "--exclude", "*.bak*",
                f"{rime_source}/",
                f"{staged_rime}/",
            ],
            check=True,
        )
        profile = active_rime_profile(staged_rime)
        exported = Path(temporary_root) / "userdb.tsv"
        export_learning_records(dict_manager, staged_rime, exported, profile)
        learned_words = read_learning_records(exported)
        char_readings = filtered_readings(load_char_readings(staged_rime, profile))
        essay = load_essay(essay_path)
        opencc_to_taiwan, taiwan_to_opencc = configure_variants(variants_path)

    payload = {
        "format": FORMAT,
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "readingWeightMinimumPercent": READING_WEIGHT_MIN,
        "learnedWords": learned_words,
        "characterReadings": char_readings,
        "essayFrequencies": essay,
        "taiwanToOpenCC": taiwan_to_opencc,
        "openCCToTaiwan": opencc_to_taiwan,
        "deferredSourcePatterns": deferred_patterns(model_path),
    }
    output_path = output_dir / "rime-homophones.json"
    atomic_json_write(output_path, payload)
    print(
        f"output={output_path} learned_words={len(learned_words)} "
        f"characters={len(char_readings)} essay_words={len(essay)} "
        f"deferred_patterns={len(payload['deferredSourcePatterns'])} "
        f"bytes={output_path.stat().st_size}"
    )


if __name__ == "__main__":
    main()
