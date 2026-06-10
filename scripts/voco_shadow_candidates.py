#!/usr/bin/env python3
"""
Generate Phase 2A offline-only shadow candidates for historical Voco JSONL.

The script reads existing shadow JSONL and writes enriched JSONL. It does not
transcribe audio, does not paste text, does not touch SwiftData, and does not
create WordReplacement rows. Candidate application remains a hypothetical
shadow analysis layer only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EVENT_PIPELINE = "pipelineSnapshot"
EVENT_USER_CORRECTION = "userCorrection"
EVENT_REVIEW_SELECTION = "reviewSelection"

RAW = "raw"
CONFIRMED_EXACT = "confirmedExact"
DOMAIN_LEXICON = "domainLexicon"
ZH_PHONETIC = "zhPhonetic"
EN_PHONETIC = "enPhonetic"
CROSS_SCRIPT = "crossScript"
LLM = "llm"
RECENT_CONTEXT = "recentContext"

NEGATIVE_ALLOWLIST = {"69 輪"}
UI_NOISE_FLAGS = {
    "selectedSpanMissing",
    "correctionTooLate",
    "activeAppChanged",
    "stalePendingTranscriptSuspected",
    "fullSentenceRewriteSuspected",
}

TECHNICAL_TERMS = {
    "api": "API",
    "asr": "ASR",
    "github": "GitHub",
    "json": "JSON",
    "jsonl": "JSONL",
    "llm": "LLM",
    "markdown": "Markdown",
    "mlx": "MLX",
    "openai": "OpenAI",
    "qwen": "Qwen",
    "qwen3": "Qwen3",
    "repo": "repo",
    "sqlite": "SQLite",
    "swift": "Swift",
    "swiftdata": "SwiftData",
    "voco": "Voco",
    "voiceink": "VoiceInk",
    "xcode": "Xcode",
}

EN_PHONETIC_REPAIRS = {
    "ripple": "repo",
    "rippo": "repo",
    "repoe": "repo",
    "mark down": "Markdown",
    "mark-down": "Markdown",
    "voice ink": "VoiceInk",
    "q when": "Qwen",
}

ZH_REPAIRS = {
    "内": "內",
    "这样": "這樣",
    "这樣": "這樣",
    "这个": "這個",
    "这個": "這個",
    "这边": "這邊",
    "这邊": "這邊",
    "们": "們",
    "过": "過",
    "来": "來",
    "时": "時",
    "后": "後",
    "机": "機",
    "语音": "語音",
    "识": "識",
    "错": "錯",
    "录": "錄",
}


@dataclass
class CandidateDraft:
    text: str
    source: str
    base_score: float
    reason: str
    evidence_tier: str
    blocked_llm_only: bool = False
    blocked_short_phrase: bool = False
    blocked_noise: bool = False
    blocked_negative: bool = False


def main() -> int:
    args = parse_args()
    input_paths = [path.expanduser() for path in args.paths]
    if not input_paths:
        print("No input JSONL path provided.", file=sys.stderr)
        return 2

    events, warnings = load_events(input_paths)
    enriched = [enrich_event(event) for event in events]

    if args.output:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(output, enriched)
    else:
        for event in enriched:
            print(json.dumps(event, ensure_ascii=False, sort_keys=True))

    report = build_report(enriched, input_paths, warnings, args.output)
    if args.report:
        args.report.expanduser().write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.output:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline-only Phase 2A shadow candidates.")
    parser.add_argument("paths", nargs="*", type=Path, help="Input JSONL file or directory.")
    parser.add_argument("--output", type=Path, help="Write enriched JSONL to this path. Defaults to stdout.")
    parser.add_argument("--report", type=Path, help="Write a JSON summary report.")
    return parser.parse_args()


def load_events(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for path in expand_input_files(paths):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        value = json.loads(stripped)
                    except json.JSONDecodeError as error:
                        warnings.append(
                            {
                                "type": "invalidJSON",
                                "path": str(path),
                                "lineNumber": line_number,
                                "message": str(error),
                            }
                        )
                        continue
                    if isinstance(value, dict):
                        events.append(value)
                    else:
                        warnings.append({"type": "nonObjectJSON", "path": str(path), "lineNumber": line_number})
        except OSError as error:
            warnings.append({"type": "readError", "path": str(path), "message": str(error)})
    return events, warnings


def expand_input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        candidates = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(candidate)
    return files


def enrich_event(event: dict[str, Any]) -> dict[str, Any]:
    enriched = json.loads(json.dumps(event, ensure_ascii=False))
    event_type = str(enriched.get("eventType") or "")
    if event_type not in {EVENT_PIPELINE, EVENT_USER_CORRECTION, EVENT_REVIEW_SELECTION}:
        return enforce_phase2a_safety(enriched)

    drafts = candidate_drafts(enriched)
    candidates = rank_candidates(enriched, drafts)
    enriched["shadowCandidates"] = candidates
    return enforce_phase2a_safety(enriched)


def enforce_phase2a_safety(event: dict[str, Any]) -> dict[str, Any]:
    safety = event.get("safety")
    if not isinstance(safety, dict):
        safety = {}
    safety["autoApplied"] = False
    safety["wouldHaveChangedFinalOutput"] = False
    event["safety"] = safety

    feature_flags = event.get("featureFlags")
    if not isinstance(feature_flags, dict):
        feature_flags = {}
    feature_flags["VocoPhoneticCandidateApplicationEnabled"] = False
    event["featureFlags"] = feature_flags
    return event


def candidate_drafts(event: dict[str, Any]) -> list[CandidateDraft]:
    raw = first_text(nested(event, "pipeline", "rawASR"), nested(event, "pipeline", "finalInserted"))
    expected = expected_output(event)
    classification = event.get("classification") if isinstance(event.get("classification"), dict) else {}
    language_mode = str(classification.get("languageMode") or "unknown")
    evidence_tier = str(classification.get("evidenceTier") or "NONE")
    noise_flags = noise_flags_for(event)
    negative = has_negative_evidence(event, raw, expected)
    short_phrase = length_bucket_for(event, raw) == "1_4"
    noisy = bool(UI_NOISE_FLAGS.intersection(noise_flags))

    drafts: list[CandidateDraft] = []
    if raw:
        drafts.append(
            CandidateDraft(
                text=raw,
                source=RAW,
                base_score=0.92 if negative else 0.56,
                reason="preserve raw ASR as an explicit candidate",
                evidence_tier="NONE",
                blocked_negative=negative,
            )
        )

    if expected and expected != raw:
        drafts.append(
            CandidateDraft(
                text=expected,
                source=CONFIRMED_EXACT,
                base_score=0.98,
                reason="historical finalInserted/user target from read-only evidence",
                evidence_tier="T4_GOLD",
                blocked_short_phrase=short_phrase,
                blocked_noise=noisy,
                blocked_negative=negative,
            )
        )

    for text, source, score, reason in generated_text_candidates(raw):
        drafts.append(
            CandidateDraft(
                text=text,
                source=source,
                base_score=score,
                reason=reason,
                evidence_tier=evidence_tier,
                blocked_short_phrase=short_phrase and normalize_text(text) != normalize_text(raw),
                blocked_noise=noisy,
                blocked_negative=negative,
            )
        )

    llm_text = first_text(nested(event, "pipeline", "llmEnhanced"))
    if llm_text and llm_text != raw:
        drafts.append(
            CandidateDraft(
                text=llm_text,
                source=LLM,
                base_score=0.34,
                reason="LLM-only candidate retained for shadow comparison but not trusted evidence",
                evidence_tier="T0_UNTRUSTED",
                blocked_llm_only=True,
                blocked_short_phrase=short_phrase,
                blocked_noise=noisy,
                blocked_negative=negative,
            )
        )

    context_candidate = recent_context_candidate(event, raw)
    if context_candidate:
        drafts.append(
            CandidateDraft(
                text=context_candidate,
                source=RECENT_CONTEXT,
                base_score=0.50,
                reason="nearby context text already present in shadow UI context",
                evidence_tier="T1_WEAK_INTERACTION",
                blocked_short_phrase=short_phrase,
                blocked_noise=noisy,
                blocked_negative=negative,
            )
        )

    if negative:
        return [draft for draft in drafts if draft.source == RAW or normalize_text(draft.text) == normalize_text(raw)]

    if language_mode == "crossScript":
        for draft in drafts:
            if draft.source in {ZH_PHONETIC, EN_PHONETIC, DOMAIN_LEXICON}:
                draft.base_score += 0.04

    return drafts


def generated_text_candidates(raw: str | None) -> list[tuple[str, str, float, str]]:
    if not raw:
        return []

    candidates: list[tuple[str, str, float, str]] = []
    seen: set[str] = set()

    def append_candidate(text: str, source: str, score: float, reason: str) -> None:
        if text == raw:
            return
        identity = candidate_identity(text)
        if not identity or identity in seen:
            return
        seen.add(identity)
        candidates.append((text, source, score, reason))

    zh = apply_zh_repairs(raw)
    if zh != raw:
        append_candidate(zh, ZH_PHONETIC, 0.68, "traditional Chinese phonetic/script normalization")

    en = apply_en_repairs(raw)
    if en != raw:
        append_candidate(en, EN_PHONETIC, 0.64, "English phonetic/domain repair")

    domain = apply_domain_lexicon(raw)
    if domain != raw:
        append_candidate(domain, DOMAIN_LEXICON, 0.72, "technical/domain lexicon normalization")

    cross = apply_domain_lexicon(apply_en_repairs(zh))
    if zh != raw and cross != zh:
        append_candidate(cross, CROSS_SCRIPT, 0.66, "combined cross-script zh/en normalization")

    return candidates


def rank_candidates(event: dict[str, Any], drafts: list[CandidateDraft]) -> list[dict[str, Any]]:
    pipeline_final = first_text(nested(event, "pipeline", "finalInserted"))
    expected = expected_output(event)
    raw = first_text(nested(event, "pipeline", "rawASR"), pipeline_final)
    technical = truthy(nested(event, "classification", "isTechnicalTermCandidate"))

    deduped: dict[tuple[str, str], CandidateDraft] = {}
    for draft in drafts:
        text = clean_candidate_text(draft.text)
        if not text:
            continue
        key = (normalize_text(text), draft.source)
        draft.text = text
        if key not in deduped or draft.base_score > deduped[key].base_score:
            deduped[key] = draft

    scored: list[tuple[float, CandidateDraft]] = []
    for draft in deduped.values():
        score = draft.base_score
        if expected:
            score += 0.18 * similarity(draft.text, expected)
        if raw:
            score += 0.08 * similarity(draft.text, raw)
        if technical and draft.source in {DOMAIN_LEXICON, CROSS_SCRIPT, EN_PHONETIC}:
            score += 0.06
        if draft.blocked_llm_only:
            score -= 0.32
        if draft.blocked_short_phrase:
            score -= 0.18
        if draft.blocked_noise:
            score -= 0.22
        if draft.blocked_negative:
            score -= 0.35 if draft.source != RAW else 0.0
        scored.append((round(clamp(score, 0.0, 1.0), 4), draft))

    scored.sort(key=lambda item: (-trusted_sort_score(item[0], item[1]), source_priority(item[1].source), normalize_text(item[1].text)))

    result: list[dict[str, Any]] = []
    for rank, (score, draft) in enumerate(scored, start=1):
        would_change_output = bool(pipeline_final and normalize_text(draft.text) != normalize_text(pipeline_final))
        blocked = draft.blocked_llm_only or draft.blocked_short_phrase or draft.blocked_noise or draft.blocked_negative
        requires_review = blocked or would_change_output or draft.source not in {RAW, CONFIRMED_EXACT}
        result.append(
            {
                "text": draft.text,
                "source": draft.source,
                "rank": rank,
                "score": score,
                "wouldChangeOutput": would_change_output,
                "requiresReview": requires_review,
                "reason": draft.reason,
                "evidenceTier": draft.evidence_tier,
                "blockedBecauseLlmOnly": draft.blocked_llm_only,
                "blockedBecauseShortPhraseRisk": draft.blocked_short_phrase,
                "blockedBecauseNoiseSuspected": draft.blocked_noise,
                "blockedBecauseNegativeEvidence": draft.blocked_negative,
            }
        )
    return result


def trusted_sort_score(score: float, draft: CandidateDraft) -> float:
    if draft.blocked_llm_only or draft.blocked_noise or draft.blocked_negative:
        return score - 1.0
    if draft.blocked_short_phrase and draft.source != RAW:
        return score - 0.5
    return score


def source_priority(source: str) -> int:
    return {
        CONFIRMED_EXACT: 0,
        DOMAIN_LEXICON: 1,
        CROSS_SCRIPT: 2,
        ZH_PHONETIC: 3,
        EN_PHONETIC: 4,
        RAW: 5,
        RECENT_CONTEXT: 6,
        LLM: 7,
    }.get(source, 99)


def expected_output(event: dict[str, Any]) -> str | None:
    return first_text(
        nested(event, "userAction", "targetText"),
        nested(event, "userAction", "selectedCandidateText"),
        nested(event, "pipeline", "finalInserted"),
    )


def recent_context_candidate(event: dict[str, Any], raw: str | None) -> str | None:
    before = first_text(nested(event, "uiContext", "selectionTextBefore"))
    after = first_text(nested(event, "uiContext", "selectionTextAfter"))
    if not raw or not before and not after:
        return None
    combined = " ".join(part for part in (before, raw, after) if part)
    return combined if combined != raw else None


def has_negative_evidence(event: dict[str, Any], raw: str | None, expected: str | None) -> bool:
    normalized_values = {normalize_text(value) for value in (raw, expected) if value}
    allowlisted = {normalize_text(value) for value in NEGATIVE_ALLOWLIST}
    if normalized_values.intersection(allowlisted):
        return True
    tier = str(nested(event, "classification", "evidenceTier") or "")
    if tier == "NEGATIVE_EVIDENCE":
        return True
    source = str(nested(event, "userAction", "source") or "")
    return source in {"rollback", "rejectedCandidate"}


def length_bucket_for(event: dict[str, Any], raw: str | None) -> str:
    value = nested(event, "classification", "lengthBucket")
    if isinstance(value, str) and value:
        return value
    count = text_unit_count(raw or "")
    if 1 <= count <= 4:
        return "1_4"
    if 5 <= count <= 15:
        return "5_15"
    if count >= 16:
        return "16_plus"
    return "unknown"


def text_unit_count(text: str) -> int:
    normalized = normalize_text(text)
    if not normalized:
        return 0
    if any(is_cjk(ch) for ch in normalized):
        return sum(1 for ch in normalized if is_cjk(ch) or ch.isdigit())
    return len(latin_tokens(normalized)) or len(normalized)


def apply_zh_repairs(text: str) -> str:
    result = text
    for source, target in ZH_REPAIRS.items():
        result = result.replace(source, target)
    return result


def apply_en_repairs(text: str) -> str:
    result = text
    for source, target in sorted(EN_PHONETIC_REPAIRS.items(), key=lambda item: -len(item[0])):
        result = replace_case_insensitive(result, source, target)
    return result


def apply_domain_lexicon(text: str) -> str:
    tokens = latin_tokens_with_spans(text)
    if not tokens:
        return text
    result: list[str] = []
    last = 0
    for token, start, end in tokens:
        replacement = TECHNICAL_TERMS.get(token.lower())
        if not replacement:
            continue
        result.append(text[last:start])
        result.append(replacement)
        last = end
    result.append(text[last:])
    return "".join(result)


def replace_case_insensitive(text: str, source: str, target: str) -> str:
    lower = text.lower()
    source_lower = source.lower()
    cursor = 0
    pieces: list[str] = []
    while True:
        index = lower.find(source_lower, cursor)
        if index < 0:
            pieces.append(text[cursor:])
            break
        pieces.append(text[cursor:index])
        pieces.append(target)
        cursor = index + len(source)
    return "".join(pieces)


def write_jsonl(path: Path, events: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def build_report(
    events: list[dict[str, Any]],
    input_paths: list[Path],
    warnings: list[dict[str, Any]],
    output: Path | None,
) -> dict[str, Any]:
    candidate_count = sum(len(event.get("shadowCandidates") or []) for event in events)
    blocked = {
        "blockedBecauseLlmOnlyCount": candidate_block_count(events, "blockedBecauseLlmOnly"),
        "blockedBecauseShortPhraseRiskCount": candidate_block_count(events, "blockedBecauseShortPhraseRisk"),
        "blockedBecauseNoiseSuspectedCount": candidate_block_count(events, "blockedBecauseNoiseSuspected"),
        "blockedBecauseNegativeEvidenceCount": candidate_block_count(events, "blockedBecauseNegativeEvidence"),
    }
    return {
        "inputPaths": [str(path) for path in input_paths],
        "outputPath": str(output) if output else None,
        "eventCount": len(events),
        "candidateCount": candidate_count,
        "candidateApplicationEnabled": False,
        "safetyAutoAppliedCount": sum(1 for event in events if truthy(nested(event, "safety", "autoApplied"))),
        "safetyWouldHaveChangedFinalOutputCount": sum(
            1 for event in events if truthy(nested(event, "safety", "wouldHaveChangedFinalOutput"))
        ),
        **blocked,
        "warningCount": len(warnings),
        "warnings": warnings[:20],
    }


def candidate_block_count(events: list[dict[str, Any]], key: str) -> int:
    count = 0
    for event in events:
        for candidate in candidates_for(event):
            if truthy(candidate.get(key)):
                count += 1
    return count


def candidates_for(event: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = event.get("shadowCandidates")
    if not isinstance(candidates, list):
        return []
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def clean_candidate_text(text: str) -> str:
    return " ".join(text.strip().split())


def candidate_identity(text: str) -> str:
    return unicodedata.normalize("NFKC", clean_candidate_text(text))


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return unicodedata.normalize("NFKC", " ".join(value.strip().casefold().split()))


def similarity(lhs: str, rhs: str) -> float:
    left = list(normalize_text(lhs))
    right = list(normalize_text(rhs))
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    distance = edit_distance(left, right)
    return 1.0 - (distance / max(len(left), len(right)))


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


def first_text(*values: Any) -> str | None:
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if text:
            return text
    return None


def nested(source: dict[str, Any], *keys: str) -> Any:
    current: Any = source
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def truthy(value: Any) -> bool:
    return value is True


def noise_flags_for(event: dict[str, Any]) -> set[str]:
    flags = nested(event, "classification", "noiseFlags")
    if isinstance(flags, list):
        return {flag for flag in flags if isinstance(flag, str)}
    return set()


def clamp(value: float, lower: float, upper: float) -> float:
    if not math.isfinite(value):
        return lower
    return max(lower, min(upper, value))


def is_cjk(ch: str) -> bool:
    value = ord(ch)
    return (
        0x4E00 <= value <= 0x9FFF
        or 0x3400 <= value <= 0x4DBF
        or 0x20000 <= value <= 0x2A6DF
    )


def latin_tokens(text: str) -> list[str]:
    return [token for token, _, _ in latin_tokens_with_spans(text)]


def latin_tokens_with_spans(text: str) -> list[tuple[str, int, int]]:
    tokens: list[tuple[str, int, int]] = []
    start: int | None = None
    for index, ch in enumerate(text):
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ch.isdigit():
            if start is None:
                start = index
        elif start is not None:
            tokens.append((text[start:index], start, index))
            start = None
    if start is not None:
        tokens.append((text[start:], start, len(text)))
    return tokens


if __name__ == "__main__":
    raise SystemExit(main())
