#!/usr/bin/env python3
"""
Create analysis-only Voco phonetic shadow JSONL from existing transcriptions.

This is a Phase 1 evidence utility. It copies the SwiftData store to a temporary
directory, reads historical transcription/audio metadata from the copy, and
appends pipelineSnapshot JSONL events to the ShadowLogs directory.

It does not transcribe audio, does not paste text, does not apply candidates,
does not write SwiftData stores, and does not create WordReplacement rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unicodedata
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


DOMAIN = "com.jasonchien.Voco"
SHADOW_LOGGING_KEY = "VocoPhoneticShadowLoggingEnabled"
CANDIDATE_APPLICATION_KEY = "VocoPhoneticCandidateApplicationEnabled"
APP_SUPPORT = Path.home() / "Library/Application Support" / DOMAIN
DEFAULT_STORE = APP_SUPPORT / "default.store"
DEFAULT_LOG_DIR = APP_SUPPORT / "ShadowLogs"

TECHNICAL_TERMS = {
    "api",
    "asr",
    "auto",
    "cloudflare",
    "flight",
    "envelope",
    "github",
    "json",
    "jsonl",
    "llm",
    "markdown",
    "mlx",
    "openai",
    "qwen",
    "session",
    "sqlite",
    "swift",
    "swiftdata",
    "voco",
    "workaround",
    "xcode",
}
COMMAND_TERMS = {
    "copy",
    "delete",
    "open",
    "paste",
    "redo",
    "run",
    "save",
    "select",
    "undo",
    "全部",
    "刪除",
    "複製",
    "貼上",
    "開啟",
    "關閉",
    "儲存",
    "修正",
    "排版",
    "辨識",
}


@dataclass(frozen=True)
class StoreCopy:
    source: Path
    copied: Path


@dataclass(frozen=True)
class BuildResult:
    event: dict[str, Any]
    source_rowid: int
    audio_path: Path | None


def main() -> int:
    args = parse_args()
    store = args.store.expanduser()
    log_dir = args.log_dir.expanduser()

    if not store.exists():
        print(f"Store not found: {store}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="voco-shadow-backfill-", dir=args.tmp_dir) as tmp:
        copied = copy_store(store, Path(tmp))
        rows = load_candidate_rows(copied, max(args.limit * 50, args.limit))
        existing_ids = set() if args.no_dedupe else existing_transcription_ids(log_dir)
        feature_flags = {
            SHADOW_LOGGING_KEY: read_defaults_bool(SHADOW_LOGGING_KEY),
            # Phase 1 hard guarantee: candidate application is off even if defaults drift.
            CANDIDATE_APPLICATION_KEY: False,
        }

        results: list[BuildResult] = []
        skipped = {
            "duplicate": 0,
            "missingAudio": 0,
            "noFinalInserted": 0,
            "scriptFilter": 0,
            "buildError": 0,
        }

        for row in rows:
            transcription_id = historical_id(row["Z_PK"])
            if transcription_id in existing_ids:
                skipped["duplicate"] += 1
                continue

            final_inserted = first_non_empty(row["ZFINALPASTEDTEXT"])
            if not final_inserted:
                skipped["noFinalInserted"] += 1
                continue

            raw_asr = first_non_empty(row["ZRAWTRANSCRIPT"], row["ZTEXT"])
            if not matches_script_filter(raw_asr, final_inserted, args.script_filter):
                skipped["scriptFilter"] += 1
                continue

            audio_path = audio_file_path(row["ZAUDIOFILEURL"])
            if audio_path is None or not audio_path.exists():
                if not args.allow_missing_audio:
                    skipped["missingAudio"] += 1
                    continue

            try:
                event = build_event(row, feature_flags=feature_flags, audio_path=audio_path)
            except Exception as error:  # Keep the utility non-destructive and auditable.
                skipped["buildError"] += 1
                print(f"Skipping row {row['Z_PK']}: {error}", file=sys.stderr)
                continue

            results.append(BuildResult(event=event, source_rowid=int(row["Z_PK"]), audio_path=audio_path))
            existing_ids.add(transcription_id)
            if len(results) >= args.limit:
                break

        output_path = active_log_path(log_dir)
        if results and not args.dry_run:
            log_dir.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8") as handle:
                for result in results:
                    handle.write(json.dumps(result.event, ensure_ascii=False, sort_keys=True))
                    handle.write("\n")

        report = {
            "dryRun": args.dry_run,
            "sourceStore": str(store),
            "copiedStore": str(copied.copied),
            "shadowLogPath": str(output_path),
            "candidateRowsRead": len(rows),
            "eventsWritten": 0 if args.dry_run else len(results),
            "eventsPrepared": len(results),
            "skipped": skipped,
            "featureFlags": feature_flags,
            "scriptFilter": args.script_filter,
            "sourceRows": [
                {
                    "rowid": result.source_rowid,
                    "audioPath": str(result.audio_path) if result.audio_path else None,
                    "transcriptionDbId": result.event["transcriptionDbId"],
                    "finalInserted": result.event["pipeline"]["finalInserted"],
                    "rawASR": result.event["pipeline"]["rawASR"],
                }
                for result in results
            ],
        }

        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print_report(report)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Phase 1 shadow JSONL from existing Voco audio/transcription records."
    )
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE, help="Path to Voco default.store.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="ShadowLogs directory to append to.")
    parser.add_argument("--limit", type=int, default=3, help="Maximum events to append.")
    parser.add_argument("--tmp-dir", type=Path, default=None, help="Temporary directory for store copies.")
    parser.add_argument("--allow-missing-audio", action="store_true", help="Emit events even when the audio file is missing.")
    parser.add_argument("--no-dedupe", action="store_true", help="Do not skip existing historical transcriptionDbId values.")
    parser.add_argument(
        "--script-filter",
        choices=("cjk", "any"),
        default="cjk",
        help="Default cjk prefers Chinese/CJK phonetic evidence; use any to include all scripts.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Prepare events and print report without writing JSONL.")
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    return parser.parse_args()


def copy_store(source: Path, tmp_dir: Path) -> StoreCopy:
    copied = tmp_dir / source.name
    shutil.copy2(source, copied)
    for suffix in ("-wal", "-shm"):
        sidecar = source.with_name(source.name + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, copied.with_name(copied.name + suffix))
    return StoreCopy(source=source, copied=copied)


def load_candidate_rows(store: StoreCopy, limit: int) -> list[sqlite3.Row]:
    uri = f"file:{store.copied}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        return con.execute(
            """
            SELECT
                Z_PK,
                ZAUDIOFILEURL,
                ZRAWTRANSCRIPT,
                ZNORMALIZEDTRANSCRIPT,
                ZTEXT,
                ZENHANCEDTEXT,
                ZFINALPASTEDTEXT,
                ZSELECTEDCANDIDATE,
                ZCONFIDENCEROUTE,
                ZCONFIDENCESCORE,
                ZASRENGINEID,
                ZDURATION,
                ZTRANSCRIPTIONDURATION,
                ZENHANCEMENTDURATION,
                ZTIMESTAMP
            FROM ZTRANSCRIPTION
            WHERE ZAUDIOFILEURL IS NOT NULL
              AND TRIM(COALESCE(ZFINALPASTEDTEXT, '')) != ''
              AND (
                TRIM(COALESCE(ZRAWTRANSCRIPT, '')) != ''
                OR TRIM(COALESCE(ZTEXT, '')) != ''
              )
            ORDER BY ZTIMESTAMP DESC
            LIMIT ?
            """,
            (max(limit, 0),),
        ).fetchall()
    finally:
        con.close()


def build_event(row: sqlite3.Row, feature_flags: dict[str, bool], audio_path: Path | None) -> dict[str, Any]:
    raw_asr = first_non_empty(row["ZRAWTRANSCRIPT"], row["ZTEXT"])
    normalized = first_non_empty(row["ZNORMALIZEDTRANSCRIPT"], row["ZTEXT"], raw_asr)
    final_inserted = first_non_empty(row["ZFINALPASTEDTEXT"])
    if not raw_asr or not final_inserted:
        raise ValueError("row does not contain raw/final text")

    audio = audio_info(audio_path, duration_seconds=row["ZDURATION"])
    pipeline = {
        "asrEngine": asr_engine(row["ZASRENGINEID"]),
        "rawASR": raw_asr,
        "afterOpenCC": normalized,
        "afterPinyinCorrector": None,
        "afterHomophoneCorrection": None,
        "afterNasalCorrection": None,
        "afterPersonalCorrection": normalized,
        "llmEnhanced": first_non_empty(row["ZENHANCEDTEXT"]),
        "finalInserted": final_inserted,
        "route": first_non_empty(row["ZCONFIDENCEROUTE"], "unknown"),
        "confidenceScore": row["ZCONFIDENCESCORE"],
        "avgLogprob": None,
        "noSpeechProb": None,
        "compressionRatio": None,
        "posteriorGap": None,
        "latencyMs": latency_ms(row["ZTRANSCRIPTIONDURATION"], row["ZENHANCEMENTDURATION"]),
    }
    classification, phonetics = classify_pair(raw_asr, final_inserted)
    event_id = str(uuid.uuid4())
    transcription_id = historical_id(row["Z_PK"])

    return {
        "schemaVersion": 1,
        "eventId": event_id,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "appVersion": "historical-backfill",
        "buildGitSha": None,
        "eventType": "pipelineSnapshot",
        "utteranceId": transcription_id,
        "transcriptionDbId": transcription_id,
        "featureFlags": feature_flags,
        "audio": audio,
        "pipeline": pipeline,
        "userAction": {
            "source": "none",
            "targetText": None,
            "selectedCandidateText": None,
            "rejectedCandidateText": None,
            "selectedRangeLength": None,
            "timeSinceUtteranceMs": None,
            "estimatedClickCount": None,
            "repeatRequested": None,
        },
        "uiContext": {
            "activeAppBundleId": None,
            "windowTitleHash": None,
            "focusedElementRole": None,
            "selectionTextBefore": None,
            "selectionTextAfter": None,
            "anchorBeforeHash": None,
            "anchorAfterHash": None,
        },
        "classification": classification,
        "phonetics": phonetics,
        "shadowCandidates": [],
        "safety": {
            "wouldHaveChangedFinalOutput": False,
            "autoApplied": False,
            "blockedBecauseLlmOnly": False,
            "blockedBecauseShortPhraseRisk": False,
            "blockedBecauseNoiseSuspected": False,
            "blockedBecauseNegativeEvidence": False,
        },
    }


def audio_info(path: Path | None, duration_seconds: float | None) -> dict[str, Any]:
    sample_rate = None
    duration_ms = float(duration_seconds) * 1000 if duration_seconds else None
    hash_prefix = None
    audio_asset_id = path.name if path else None

    if path and path.exists():
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            hash_prefix = digest.hexdigest()[:16]
        except OSError:
            hash_prefix = None

        try:
            with wave.open(str(path), "rb") as wav:
                sample_rate = float(wav.getframerate())
                if not duration_ms and wav.getframerate() > 0:
                    duration_ms = (wav.getnframes() / wav.getframerate()) * 1000
        except (OSError, wave.Error):
            sample_rate = None

    return {
        "audioAssetId": audio_asset_id,
        "durationMs": duration_ms,
        "sampleRate": sample_rate,
        "audioHashPrefix": hash_prefix,
    }


def classify_pair(raw: str, target: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_features = text_features(raw)
    target_features = text_features(target)
    language_mode = merged_language_mode(raw_features["languageMode"], target_features["languageMode"])
    if is_cross_script(raw_features["scriptMode"], target_features["scriptMode"]):
        language_mode = "crossScript"

    raw_phones = phones_for(raw_features["normalized"])
    target_phones = phones_for(target_features["normalized"])
    distance = edit_distance(raw_phones, target_phones) if raw_phones or target_phones else None

    classification = {
        "lengthBucket": raw_features["lengthBucket"],
        "scriptMode": raw_features["scriptMode"],
        "languageMode": language_mode,
        "isCommandLike": raw_features["isCommandLike"],
        "isTechnicalTermCandidate": raw_features["isTechnicalTermCandidate"] or target_features["isTechnicalTermCandidate"],
        "evidenceTier": "NONE",
        "noiseFlags": [],
        "isPurePhoneticCandidate": False,
    }
    phonetics = {
        "rawNormalized": raw_features["normalized"],
        "targetNormalized": target_features["normalized"],
        "rawPhones": raw_phones,
        "targetPhones": target_phones,
        "weightedPhoneEditDistance": distance,
        "pinyinSimilarity": None,
        "confusionPairs": confusion_pairs(raw_features["normalized"], target_features["normalized"]),
    }
    return classification, phonetics


def text_features(text: str) -> dict[str, Any]:
    normalized = unicodedata.normalize("NFKC", text.strip())
    script_mode = script_mode_for(normalized)
    language_mode = language_mode_for(script_mode)
    unit_count = unit_count_for(normalized, script_mode)
    technical = is_technical(normalized)
    command_like = is_command_like(normalized, technical=technical, length_bucket=length_bucket(unit_count))
    return {
        "normalized": normalized,
        "scriptMode": script_mode,
        "languageMode": language_mode,
        "lengthBucket": length_bucket(unit_count),
        "isCommandLike": command_like,
        "isTechnicalTermCandidate": technical,
    }


def script_mode_for(text: str) -> str:
    if not text:
        return "unknown"
    has_cjk = any(is_cjk(ch) for ch in text)
    has_latin = any(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in text)
    has_number_or_symbol = any(ch.isdigit() or unicodedata.category(ch).startswith("S") for ch in text)
    if has_cjk and has_latin:
        return "mixedZhEn"
    if has_cjk:
        return "zhOnly"
    if has_latin:
        return "enOnly"
    if has_number_or_symbol:
        return "numericSymbol"
    return "unknown"


def language_mode_for(script_mode: str) -> str:
    return {
        "zhOnly": "mandarin",
        "enOnly": "english",
        "mixedZhEn": "codeSwitch",
    }.get(script_mode, "unknown")


def unit_count_for(text: str, script_mode: str) -> int:
    if script_mode == "enOnly":
        return len(latin_tokens(text))
    if script_mode == "mixedZhEn":
        return max(1, len(latin_tokens(text)) + sum(1 for ch in text if is_cjk(ch)))
    if script_mode == "zhOnly":
        return sum(1 for ch in text if is_cjk(ch) or ch.isdigit())
    if script_mode == "numericSymbol":
        return sum(1 for ch in text if not ch.isspace())
    return len(text)


def length_bucket(count: int) -> str:
    if 1 <= count <= 4:
        return "1_4"
    if 5 <= count <= 15:
        return "5_15"
    if count >= 16:
        return "16_plus"
    return "unknown"


def is_technical(text: str) -> bool:
    lower = text.lower()
    tokens = latin_tokens(lower)
    return any(token in TECHNICAL_TERMS for token in tokens) or any(term in lower for term in TECHNICAL_TERMS)


def is_command_like(text: str, technical: bool, length_bucket: str) -> bool:
    lower = text.lower()
    tokens = latin_tokens(lower)
    contains_command = (
        lower in COMMAND_TERMS
        or any(token in COMMAND_TERMS for token in tokens)
        or any(term in lower for term in COMMAND_TERMS)
    )
    return (contains_command or technical) and length_bucket in {"1_4", "5_15"}


def phones_for(text: str) -> list[str]:
    phones: list[str] = []
    token = ""
    for ch in text:
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ch.isdigit():
            token += ch.lower()
            continue
        if token:
            phones.append(f"latin:{token}")
            token = ""
        if is_cjk(ch):
            phones.append(f"han:{ch}")
        elif not ch.isspace():
            phones.append(f"sym:{ch}")
    if token:
        phones.append(f"latin:{token}")
    return phones


def edit_distance(source: list[str], target: list[str]) -> int:
    if not source:
        return len(target)
    if not target:
        return len(source)
    previous = list(range(len(target) + 1))
    for i, source_item in enumerate(source, start=1):
        current = [i]
        for j, target_item in enumerate(target, start=1):
            cost = 0 if source_item == target_item else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def confusion_pairs(raw: str, target: str) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    max_len = max(len(raw), len(target))
    for index in range(max_len):
        raw_ch = raw[index] if index < len(raw) else ""
        target_ch = target[index] if index < len(target) else ""
        if raw_ch == target_ch:
            continue
        if raw_ch and target_ch:
            operation = "substitution"
        elif raw_ch:
            operation = "deletion"
        else:
            operation = "insertion"
        pairs.append({"raw": raw_ch, "target": target_ch, "operation": operation, "position": index})
        if len(pairs) >= 20:
            break
    return pairs


def is_cjk(ch: str) -> bool:
    value = ord(ch)
    return (
        0x4E00 <= value <= 0x9FFF
        or 0x3400 <= value <= 0x4DBF
        or 0x20000 <= value <= 0x2A6DF
    )


def matches_script_filter(raw: str | None, final: str | None, script_filter: str) -> bool:
    if script_filter == "any":
        return True
    combined = f"{raw or ''}\n{final or ''}"
    return any(is_cjk(ch) for ch in combined)


def latin_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    for ch in text:
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ch.isdigit():
            current += ch.lower()
        elif current:
            tokens.append(current)
            current = ""
    if current:
        tokens.append(current)
    return tokens


def is_cross_script(raw_mode: str, target_mode: str) -> bool:
    cross_pairs = {
        ("zhOnly", "enOnly"),
        ("enOnly", "zhOnly"),
        ("zhOnly", "mixedZhEn"),
        ("mixedZhEn", "zhOnly"),
        ("enOnly", "mixedZhEn"),
        ("mixedZhEn", "enOnly"),
    }
    return (raw_mode, target_mode) in cross_pairs


def merged_language_mode(raw_mode: str, target_mode: str) -> str:
    if raw_mode == target_mode:
        return raw_mode
    if raw_mode == "unknown":
        return target_mode
    if target_mode == "unknown":
        return raw_mode
    if "codeSwitch" in {raw_mode, target_mode}:
        return "codeSwitch"
    return "unknown"


def existing_transcription_ids(log_dir: Path) -> set[str]:
    ids: set[str] = set()
    if not log_dir.exists():
        return ids
    for path in sorted(log_dir.glob("phonetic-shadow-*.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        event = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    value = event.get("transcriptionDbId")
                    if isinstance(value, str) and value:
                        ids.add(value)
        except OSError:
            continue
    return ids


def active_log_path(log_dir: Path) -> Path:
    day = datetime.now().strftime("%Y-%m-%d")
    return log_dir / f"phonetic-shadow-{day}.jsonl"


def historical_id(rowid: Any) -> str:
    return f"historical:{int(rowid)}"


def audio_file_path(value: Any) -> Path | None:
    text = first_non_empty(value)
    if not text:
        return None
    parsed = urlparse(text)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(text).expanduser()


def asr_engine(value: Any) -> str:
    text = str(value or "").lower()
    if "qwen3" in text:
        return "Qwen3-ASR"
    if "whisper" in text:
        return "Whisper"
    return "unknown"


def latency_ms(transcription_duration: Any, enhancement_duration: Any) -> float | None:
    total = 0.0
    seen = False
    for value in (transcription_duration, enhancement_duration):
        if isinstance(value, (int, float)):
            total += float(value)
            seen = True
    return total * 1000 if seen and total > 0 else None


def first_non_empty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def read_defaults_bool(key: str) -> bool:
    try:
        result = subprocess.run(
            ["defaults", "read", DOMAIN, key],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        return False
    value = result.stdout.strip().lower()
    return value in {"1", "true", "yes"}


def print_report(report: dict[str, Any]) -> None:
    print("Voco shadow backfill")
    print(f"  source store: {report['sourceStore']}")
    print(f"  shadow log: {report['shadowLogPath']}")
    print(f"  events prepared: {report['eventsPrepared']}")
    print(f"  events written: {report['eventsWritten']}")
    print(f"  skipped: {report['skipped']}")
    print(f"  feature flags: {report['featureFlags']}")
    if report["sourceRows"]:
        print("  source rows:")
        for row in report["sourceRows"]:
            print(f"    - {row['transcriptionDbId']} audio={row['audioPath']}")


if __name__ == "__main__":
    raise SystemExit(main())
