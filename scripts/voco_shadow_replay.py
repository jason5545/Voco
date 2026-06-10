#!/usr/bin/env python3
"""
Offline replay metrics for Voco phonetic shadow JSONL logs.

This script is intentionally read-only with respect to Voco app data. It only
reads JSONL files emitted by PhoneticShadowLogger and writes optional report
artifacts requested by CLI flags.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EVENT_PIPELINE = "pipelineSnapshot"
EVENT_USER_CORRECTION = "userCorrection"
EVENT_REVIEW_SELECTION = "reviewSelection"
EVENT_ROLLBACK = "rollback"

ROUTE_DIRECT = "directInsertion"
ROUTE_REVIEW = "reviewSuggested"

LENGTH_BUCKETS = ("1_4", "5_15", "16_plus", "unknown")
SCRIPT_LANGUAGE_BUCKETS = (
    "zhOnly",
    "enOnly",
    "mixedZhEn",
    "crossScript",
    "commandLike",
    "technicalTerm",
)
UI_NOISE_FLAGS = {
    "selectedSpanMissing",
    "correctionTooLate",
    "activeAppChanged",
    "stalePendingTranscriptSuspected",
    "fullSentenceRewriteSuspected",
}

BASELINE = {
    "name": "phase1_analysis_only",
    "candidateApplicationEnabledDefault": False,
    "shadowAutoApplyCount": 0,
    "shadowOutputMutationCount": 0,
    "learnedReplacementWritesFromReplay": 0,
    "notes": [
        "Replay does not mutate Voco output or SwiftData stores.",
        "Phase 1 shadow candidates are evidence-only; auto application remains disabled.",
        "Rollback is reported when present in logs; Phase 1 does not create a rollback UI hook.",
    ],
}


@dataclass(frozen=True)
class LoadedEvent:
    path: Path
    line_number: int
    event: dict[str, Any]


def main() -> int:
    args = parse_args()

    if args.write_sample:
        write_sample_jsonl(args.write_sample)

    input_paths = [path.expanduser() for path in args.paths]
    if not input_paths:
        if args.write_sample:
            input_paths = [args.write_sample.expanduser()]
        else:
            print("No JSONL path provided.", file=sys.stderr)
            return 2

    loaded, warnings = load_events(input_paths)
    report = build_report(loaded, input_paths, warnings)

    if args.markdown_output:
        args.markdown_output.expanduser().write_text(markdown_summary(report), encoding="utf-8")

    output = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.expanduser().write_text(output + "\n", encoding="utf-8")
    else:
        print(output)

    return 1 if args.fail_on_invalid and report["input"]["invalidLineCount"] > 0 else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay Voco phonetic shadow JSONL logs into operation-cost metrics."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="JSONL file or directory. Directories are scanned recursively.")
    parser.add_argument("--json-output", type=Path, help="Write JSON report to this path instead of stdout.")
    parser.add_argument("--markdown-output", type=Path, help="Write a compact Markdown summary to this path.")
    parser.add_argument("--write-sample", type=Path, help="Write a deterministic sample JSONL file before replaying.")
    parser.add_argument("--fail-on-invalid", action="store_true", help="Exit non-zero if invalid JSONL lines are found.")
    return parser.parse_args()


def load_events(paths: list[Path]) -> tuple[list[LoadedEvent], list[dict[str, Any]]]:
    loaded: list[LoadedEvent] = []
    warnings: list[dict[str, Any]] = []

    for path in expand_input_files(paths):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        parsed = json.loads(stripped)
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
                    if not isinstance(parsed, dict):
                        warnings.append(
                            {
                                "type": "nonObjectJSON",
                                "path": str(path),
                                "lineNumber": line_number,
                            }
                        )
                        continue
                    loaded.append(LoadedEvent(path=path, line_number=line_number, event=parsed))
        except OSError as error:
            warnings.append({"type": "readError", "path": str(path), "message": str(error)})

    return loaded, warnings


def expand_input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()

    for path in paths:
        if path.is_dir():
            candidates = sorted(path.rglob("*.jsonl"))
        else:
            candidates = [path]

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(candidate)

    return files


def build_report(
    loaded: list[LoadedEvent],
    input_paths: list[Path],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    events = [item.event for item in loaded]
    pipeline_events = [event for event in events if event_type(event) == EVENT_PIPELINE]
    unique_utterances = {
        value
        for value in (event.get("utteranceId") for event in events)
        if isinstance(value, str) and value
    }
    denominator = len(pipeline_events) or len(unique_utterances) or len(events)

    metrics = metric_counts(events, denominator)
    breakdowns = {
        "byLengthBucket": length_breakdown(events, denominator),
        "byLanguageAndScript": language_script_breakdown(events, denominator),
    }

    auto_apply_count = sum(1 for event in events if truthy(nested(event, "safety", "autoApplied")))
    output_change_count = sum(1 for event in events if would_have_changed_output(event))
    blocked_counts = candidate_block_counts(events)

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "input": {
            "paths": [str(path) for path in input_paths],
            "filesProcessed": sorted({str(item.path) for item in loaded}),
            "eventCount": len(events),
            "invalidLineCount": len([warning for warning in warnings if warning["type"] == "invalidJSON"]),
            "warningCount": len(warnings),
        },
        "denominator": {
            "name": "pipelineSnapshots",
            "pipelineSnapshotCount": len(pipeline_events),
            "uniqueUtteranceCount": len(unique_utterances),
            "per100Base": denominator,
            "fallbackUsed": "pipelineSnapshots" if pipeline_events else ("uniqueUtterances" if unique_utterances else "events"),
        },
        "metrics": metrics,
        "breakdowns": breakdowns,
        "safetyAudit": {
            "autoAppliedCount": auto_apply_count,
            "wouldHaveChangedFinalOutputCount": output_change_count,
            "phase1SafetyPass": auto_apply_count == 0 and output_change_count == 0,
        },
        "candidateSafetyAudit": blocked_counts,
        "baseline": BASELINE,
        "metricDefinitions": metric_definitions(),
        "warnings": warnings[:50],
    }


def metric_counts(events: list[dict[str, Any]], denominator: int) -> dict[str, Any]:
    review_shown = [event for event in events if is_review_shown(event)]
    accepted_candidates = [event for event in events if event_type(event) == EVENT_REVIEW_SELECTION]
    repeat_requested = [event for event in events if truthy(nested(event, "userAction", "repeatRequested"))]
    rollbacks = [event for event in events if is_rollback(event)]
    wrong_insertions = [event for event in events if is_wrong_insertion_signal(event)]
    direct_insertions = [event for event in events if is_direct_insertion(event)]
    correction_feedback = [event for event in events if is_correction_feedback(event)]
    short_feedback = [event for event in correction_feedback if length_bucket(event) == "1_4"]
    short_review = [event for event in review_shown if length_bucket(event) == "1_4"]
    short_substitution = [
        event
        for event in correction_feedback
        if length_bucket(event) == "1_4" and nested(event, "userAction", "source") == "userSubstitution"
    ]
    ui_noise = [event for event in events if has_ui_noise(event)]
    llm_only_rejected = [event for event in events if is_llm_only_rejected(event)]
    latencies = [
        float(value)
        for value in (nested(event, "pipeline", "latencyMs") for event in events)
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    top1 = topk_match(events, max_rank=1)
    top3 = topk_match(events, max_rank=3)
    click_total, click_event_count = estimated_clicks(events)
    blocked_counts = candidate_block_counts(events)
    potential_review_savings = [event for event in events if is_potential_review_saving(event)]
    potential_wrong_candidates = [event for event in events if is_potential_wrong_candidate(event)]

    return {
        "reviewShownPer100": per100(len(review_shown), denominator),
        "acceptedCandidatePer100": per100(len(accepted_candidates), denominator),
        "estimatedClicksPer100": per100(click_total, denominator),
        "estimatedClicksKnownEventCount": click_event_count,
        "repeatRequestedPer100": per100(len(repeat_requested), denominator),
        "wrongInsertionPer100": per100(len(wrong_insertions), denominator),
        "rollbackPer100": per100(len(rollbacks), denominator),
        "directInsertionPer100": per100(len(direct_insertions), denominator),
        "correctionFeedbackPer100": per100(len(correction_feedback), denominator),
        "shortUtteranceFeedbackPer100": per100(len(short_feedback), denominator),
        "shortUtteranceReviewPer100": per100(len(short_review), denominator),
        "shortUtteranceSubstitutionPer100": per100(len(short_substitution), denominator),
        "p50LatencyMs": percentile(latencies, 0.50),
        "p95LatencyMs": percentile(latencies, 0.95),
        "shadowTop1WouldHaveMatchedUserCorrection": top1["rate"],
        "shadowTop1MatchCount": top1["matches"],
        "shadowTop1CandidateEventCount": top1["considered"],
        "shadowTop3WouldHaveMatchedUserCorrection": top3["rate"],
        "shadowTop3MatchCount": top3["matches"],
        "shadowTop3CandidateEventCount": top3["considered"],
        "blockedBecauseLlmOnlyCount": blocked_counts["blockedBecauseLlmOnlyCount"],
        "blockedBecauseShortPhraseRiskCount": blocked_counts["blockedBecauseShortPhraseRiskCount"],
        "blockedBecauseNoiseSuspectedCount": blocked_counts["blockedBecauseNoiseSuspectedCount"],
        "blockedBecauseNegativeEvidenceCount": blocked_counts["blockedBecauseNegativeEvidenceCount"],
        "potentialReviewSavingsPer100": per100(len(potential_review_savings), denominator),
        "potentialWrongCandidatePer100": per100(len(potential_wrong_candidates), denominator),
        "uiNoiseSuspectedCount": len(ui_noise),
        "llmOnlyEvidenceRejectedCount": len(llm_only_rejected),
    }


def length_breakdown(events: list[dict[str, Any]], denominator: int) -> dict[str, Any]:
    result = {
        bucket: empty_breakdown_bucket()
        for bucket in LENGTH_BUCKETS
    }

    for event in events:
        bucket = length_bucket(event)
        if bucket not in result:
            bucket = "unknown"
        bump_breakdown(result[bucket], event)

    add_per100(result.values(), denominator)
    return result


def language_script_breakdown(events: list[dict[str, Any]], denominator: int) -> dict[str, Any]:
    result = {
        bucket: empty_breakdown_bucket()
        for bucket in SCRIPT_LANGUAGE_BUCKETS
    }

    for event in events:
        classification = event.get("classification") if isinstance(event.get("classification"), dict) else {}
        script_mode = classification.get("scriptMode")
        language_mode = classification.get("languageMode")
        is_command_like = truthy(classification.get("isCommandLike"))
        is_technical = truthy(classification.get("isTechnicalTermCandidate"))

        buckets: list[str] = []
        if script_mode in {"zhOnly", "enOnly", "mixedZhEn"}:
            buckets.append(str(script_mode))
        if language_mode == "crossScript":
            buckets.append("crossScript")
        if is_command_like:
            buckets.append("commandLike")
        if is_technical:
            buckets.append("technicalTerm")

        for bucket in buckets:
            bump_breakdown(result[bucket], event)

    add_per100(result.values(), denominator)
    return result


def empty_breakdown_bucket() -> dict[str, Any]:
    return {
        "eventCount": 0,
        "pipelineSnapshotCount": 0,
        "reviewShownCount": 0,
        "directInsertionCount": 0,
        "correctionFeedbackCount": 0,
        "rollbackCount": 0,
        "uiNoiseSuspectedCount": 0,
        "llmOnlyEvidenceRejectedCount": 0,
    }


def bump_breakdown(bucket: dict[str, Any], event: dict[str, Any]) -> None:
    bucket["eventCount"] += 1
    if event_type(event) == EVENT_PIPELINE:
        bucket["pipelineSnapshotCount"] += 1
    if is_review_shown(event):
        bucket["reviewShownCount"] += 1
    if is_direct_insertion(event):
        bucket["directInsertionCount"] += 1
    if is_correction_feedback(event):
        bucket["correctionFeedbackCount"] += 1
    if is_rollback(event):
        bucket["rollbackCount"] += 1
    if has_ui_noise(event):
        bucket["uiNoiseSuspectedCount"] += 1
    if is_llm_only_rejected(event):
        bucket["llmOnlyEvidenceRejectedCount"] += 1


def add_per100(buckets: Iterable[dict[str, Any]], denominator: int) -> None:
    for bucket in buckets:
        bucket["eventPer100"] = per100(bucket["eventCount"], denominator)
        bucket["reviewShownPer100"] = per100(bucket["reviewShownCount"], denominator)
        bucket["correctionFeedbackPer100"] = per100(bucket["correctionFeedbackCount"], denominator)


def is_review_shown(event: dict[str, Any]) -> bool:
    return event_type(event) == EVENT_PIPELINE and nested(event, "pipeline", "route") == ROUTE_REVIEW


def is_direct_insertion(event: dict[str, Any]) -> bool:
    return event_type(event) == EVENT_PIPELINE and nested(event, "pipeline", "route") == ROUTE_DIRECT


def is_correction_feedback(event: dict[str, Any]) -> bool:
    source = nested(event, "userAction", "source")
    return event_type(event) in {EVENT_USER_CORRECTION, EVENT_REVIEW_SELECTION} or source in {
        "editMode",
        "correctionFeedback",
        "reviewCandidate",
        "userSubstitution",
        "manualTranscript",
    }


def is_rollback(event: dict[str, Any]) -> bool:
    return event_type(event) == EVENT_ROLLBACK or nested(event, "userAction", "source") == "rollback"


def is_wrong_insertion_signal(event: dict[str, Any]) -> bool:
    if is_rollback(event):
        return True
    source = nested(event, "userAction", "source")
    tier = nested(event, "classification", "evidenceTier")
    return source in {"rejectedCandidate"} or tier == "NEGATIVE_EVIDENCE"


def has_ui_noise(event: dict[str, Any]) -> bool:
    flags = noise_flags(event)
    return bool(UI_NOISE_FLAGS.intersection(flags))


def is_llm_only_rejected(event: dict[str, Any]) -> bool:
    source = nested(event, "userAction", "source")
    flags = noise_flags(event)
    safety_blocked = truthy(nested(event, "safety", "blockedBecauseLlmOnly"))
    t0_llm = nested(event, "classification", "evidenceTier") == "T0_UNTRUSTED" and source in {
        "llmEnhancement",
        "ztextEnhancedDifference",
    }
    return safety_blocked or "llmOnly" in flags or t0_llm


def would_have_changed_output(event: dict[str, Any]) -> bool:
    return truthy(nested(event, "safety", "wouldHaveChangedFinalOutput"))


def candidate_block_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "blockedBecauseLlmOnly",
        "blockedBecauseShortPhraseRisk",
        "blockedBecauseNoiseSuspected",
        "blockedBecauseNegativeEvidence",
    )
    counts = {f"{key}Count": 0 for key in keys}
    for event in events:
        for candidate in candidate_dicts(event):
            for key in keys:
                if truthy(candidate.get(key)):
                    counts[f"{key}Count"] += 1
        safety = event.get("safety")
        if isinstance(safety, dict):
            for key in keys:
                if truthy(safety.get(key)):
                    counts[f"{key}Count"] += 1
    return counts


def is_potential_review_saving(event: dict[str, Any]) -> bool:
    if event_type(event) != EVENT_PIPELINE or nested(event, "pipeline", "route") != ROUTE_REVIEW:
        return False
    expected = expected_text(event)
    candidate = top_nonreview_candidate(event)
    return bool(expected and candidate and normalize_text(candidate.get("text")) == normalize_text(expected))


def is_potential_wrong_candidate(event: dict[str, Any]) -> bool:
    if event_type(event) != EVENT_PIPELINE:
        return False
    expected = expected_text(event)
    candidate = top_nonreview_candidate(event)
    if not expected or not candidate:
        return False
    candidate_text = normalize_text(candidate.get("text"))
    return bool(candidate_text and candidate_text != normalize_text(expected))


def expected_text(event: dict[str, Any]) -> str:
    for value in (
        nested(event, "userAction", "targetText"),
        nested(event, "userAction", "selectedCandidateText"),
        nested(event, "pipeline", "finalInserted"),
    ):
        normalized = normalize_text(value)
        if normalized:
            return str(value).strip()
    return ""


def top_trusted_candidate(event: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [candidate for candidate in candidate_dicts(event) if is_trusted_candidate(candidate)]
    candidates.sort(key=lambda candidate: candidate_rank(candidate))
    return candidates[0] if candidates else None


def top_nonreview_candidate(event: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        candidate
        for candidate in candidate_dicts(event)
        if is_trusted_candidate(candidate) and not truthy(candidate.get("requiresReview"))
    ]
    candidates.sort(key=lambda candidate: candidate_rank(candidate))
    return candidates[0] if candidates else None


def is_trusted_candidate(candidate: dict[str, Any]) -> bool:
    if candidate.get("source") == "llm" or truthy(candidate.get("blockedBecauseLlmOnly")):
        return False
    return not any(
        truthy(candidate.get(key))
        for key in (
            "blockedBecauseNoiseSuspected",
            "blockedBecauseNegativeEvidence",
            "blockedBecauseShortPhraseRisk",
        )
    )


def candidate_dicts(event: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = event.get("shadowCandidates")
    if not isinstance(candidates, list):
        return []
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def estimated_clicks(events: list[dict[str, Any]]) -> tuple[int, int]:
    total = 0
    known = 0
    for event in events:
        value = nested(event, "userAction", "estimatedClickCount")
        if isinstance(value, int):
            total += value
            known += 1
    return total, known


def topk_match(events: list[dict[str, Any]], max_rank: int) -> dict[str, Any]:
    considered = 0
    matches = 0

    for event in events:
        if not is_correction_feedback(event):
            continue

        target = normalize_text(nested(event, "userAction", "targetText"))
        if not target:
            continue

        candidates = ranked_candidates(event, max_rank=max_rank)
        if not candidates:
            continue

        considered += 1
        if any(normalize_text(candidate.get("text")) == target for candidate in candidates):
            matches += 1

    return {
        "matches": matches,
        "considered": considered,
        "rate": round(matches / considered, 4) if considered else 0.0,
    }


def ranked_candidates(event: dict[str, Any], max_rank: int) -> list[dict[str, Any]]:
    candidates = [candidate for candidate in candidate_dicts(event) if is_trusted_candidate(candidate)]
    candidates.sort(key=lambda candidate: candidate_rank(candidate))
    return [candidate for candidate in candidates if candidate_rank(candidate) <= max_rank]


def candidate_rank(candidate: dict[str, Any]) -> int:
    rank = candidate.get("rank")
    if isinstance(rank, int):
        return rank
    return 9999


def event_type(event: dict[str, Any]) -> str:
    value = event.get("eventType")
    return value if isinstance(value, str) else "unknown"


def length_bucket(event: dict[str, Any]) -> str:
    value = nested(event, "classification", "lengthBucket")
    if isinstance(value, str) and value:
        return value
    return "unknown"


def noise_flags(event: dict[str, Any]) -> set[str]:
    flags = nested(event, "classification", "noiseFlags")
    if not isinstance(flags, list):
        return set()
    return {flag for flag in flags if isinstance(flag, str)}


def nested(source: dict[str, Any], *keys: str) -> Any:
    current: Any = source
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def truthy(value: Any) -> bool:
    return value is True


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().casefold().split())


def per100(count: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((count / denominator) * 100.0, 2)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)

    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return round(ordered[int(index)], 2)

    weight = index - lower
    interpolated = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return round(interpolated, 2)


def metric_definitions() -> dict[str, str]:
    return {
        "reviewShownPer100": "pipelineSnapshot events whose pipeline.route is reviewSuggested per 100 pipeline snapshots.",
        "acceptedCandidatePer100": "reviewSelection events per 100 pipeline snapshots.",
        "estimatedClicksPer100": "Sum of logged userAction.estimatedClickCount per 100 pipeline snapshots; missing values are not guessed.",
        "repeatRequestedPer100": "Events with userAction.repeatRequested=true per 100 pipeline snapshots.",
        "wrongInsertionPer100": "Rollback, rejectedCandidate, or NEGATIVE_EVIDENCE events per 100 pipeline snapshots.",
        "rollbackPer100": "rollback events or userAction.source=rollback per 100 pipeline snapshots.",
        "directInsertionPer100": "pipelineSnapshot events whose pipeline.route is directInsertion per 100 pipeline snapshots.",
        "correctionFeedbackPer100": "userCorrection/reviewSelection and trusted correction-action events per 100 pipeline snapshots.",
        "shortUtteranceFeedbackPer100": "Correction feedback events in length bucket 1_4 per 100 pipeline snapshots.",
        "shortUtteranceReviewPer100": "reviewSuggested pipeline snapshots in length bucket 1_4 per 100 pipeline snapshots.",
        "shortUtteranceSubstitutionPer100": "userSubstitution correction events in length bucket 1_4 per 100 pipeline snapshots.",
        "shadowTop1WouldHaveMatchedUserCorrection": "Among correction events that include shadowCandidates, fraction whose rank-1 candidate equals userAction.targetText.",
        "shadowTop3WouldHaveMatchedUserCorrection": "Among correction events that include shadowCandidates, fraction whose top-3 candidates include userAction.targetText.",
        "blockedBecauseLlmOnlyCount": "Candidate and event safety blocks where evidence came only from LLM/enhanced differences.",
        "blockedBecauseShortPhraseRiskCount": "Candidate and event safety blocks caused by risky short phrases.",
        "blockedBecauseNoiseSuspectedCount": "Candidate and event safety blocks caused by UI/timing/stale-text noise.",
        "blockedBecauseNegativeEvidenceCount": "Candidate and event safety blocks caused by rollback/rejected/allowlisted negative evidence.",
        "potentialReviewSavingsPer100": "Review-suggested pipeline snapshots whose top non-review shadow candidate already matches finalInserted per 100 pipeline snapshots.",
        "potentialWrongCandidatePer100": "Pipeline snapshots whose top non-review shadow candidate differs from finalInserted per 100 pipeline snapshots.",
        "uiNoiseSuspectedCount": "Events containing UI/timing/span noise flags.",
        "llmOnlyEvidenceRejectedCount": "Events blocked or classified as untrusted because evidence came only from LLM/enhanced differences.",
    }


def markdown_summary(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    denominator = report["denominator"]
    safety = report["safetyAudit"]
    lines = [
        "# Voco Phonetic Shadow Replay",
        "",
        f"- Events: {report['input']['eventCount']}",
        f"- Per-100 base: {denominator['per100Base']} ({denominator['fallbackUsed']})",
        f"- Invalid JSONL lines: {report['input']['invalidLineCount']}",
        f"- Phase 1 safety pass: {safety['phase1SafetyPass']}",
        "",
        "## Operation Cost",
        "",
        f"- reviewShownPer100: {metrics['reviewShownPer100']}",
        f"- acceptedCandidatePer100: {metrics['acceptedCandidatePer100']}",
        f"- estimatedClicksPer100: {metrics['estimatedClicksPer100']}",
        f"- correctionFeedbackPer100: {metrics['correctionFeedbackPer100']}",
        f"- rollbackPer100: {metrics['rollbackPer100']}",
        f"- potentialReviewSavingsPer100: {metrics['potentialReviewSavingsPer100']}",
        f"- potentialWrongCandidatePer100: {metrics['potentialWrongCandidatePer100']}",
        "",
        "## Evidence",
        "",
        f"- shadowTop1WouldHaveMatchedUserCorrection: {metrics['shadowTop1WouldHaveMatchedUserCorrection']}",
        f"- shadowTop3WouldHaveMatchedUserCorrection: {metrics['shadowTop3WouldHaveMatchedUserCorrection']}",
        f"- blockedBecauseLlmOnlyCount: {metrics['blockedBecauseLlmOnlyCount']}",
        f"- blockedBecauseShortPhraseRiskCount: {metrics['blockedBecauseShortPhraseRiskCount']}",
        f"- blockedBecauseNoiseSuspectedCount: {metrics['blockedBecauseNoiseSuspectedCount']}",
        f"- blockedBecauseNegativeEvidenceCount: {metrics['blockedBecauseNegativeEvidenceCount']}",
        f"- uiNoiseSuspectedCount: {metrics['uiNoiseSuspectedCount']}",
        f"- llmOnlyEvidenceRejectedCount: {metrics['llmOnlyEvidenceRejectedCount']}",
        "",
    ]
    return "\n".join(lines)


def write_sample_jsonl(path: Path) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    events = sample_events()
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def sample_events() -> list[dict[str, Any]]:
    base = {
        "schemaVersion": 1,
        "appVersion": "sample",
        "buildGitSha": None,
        "featureFlags": {
            "VocoPhoneticShadowLoggingEnabled": True,
            "VocoPhoneticCandidateApplicationEnabled": False,
        },
        "audio": {
            "audioAssetId": None,
            "durationMs": None,
            "sampleRate": None,
            "audioHashPrefix": None,
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
        "phonetics": {
            "rawNormalized": None,
            "targetNormalized": None,
            "rawPhones": [],
            "targetPhones": [],
            "weightedPhoneEditDistance": None,
            "pinyinSimilarity": None,
            "confusionPairs": [],
        },
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

    def event(
        event_id: str,
        event_type_value: str,
        utterance_id: str,
        route: str,
        raw: str,
        final: str,
        bucket: str,
        script: str,
        language: str,
        **updates: Any,
    ) -> dict[str, Any]:
        value = copy.deepcopy(base)
        value.update(
            {
                "eventId": event_id,
                "createdAt": "2026-06-10T00:00:00.000Z",
                "eventType": event_type_value,
                "utteranceId": utterance_id,
                "transcriptionDbId": utterance_id,
                "pipeline": {
                    "asrEngine": "sample",
                    "rawASR": raw,
                    "afterOpenCC": raw,
                    "afterPinyinCorrector": raw,
                    "afterHomophoneCorrection": raw,
                    "afterNasalCorrection": raw,
                    "afterPersonalCorrection": raw,
                    "llmEnhanced": None,
                    "finalInserted": final,
                    "route": route,
                    "confidenceScore": 0.9,
                    "avgLogprob": None,
                    "noSpeechProb": None,
                    "compressionRatio": None,
                    "posteriorGap": None,
                    "latencyMs": 120.0,
                },
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
                "classification": {
                    "lengthBucket": bucket,
                    "scriptMode": script,
                    "languageMode": language,
                    "isCommandLike": False,
                    "isTechnicalTermCandidate": False,
                    "evidenceTier": "NONE",
                    "noiseFlags": [],
                    "isPurePhoneticCandidate": False,
                },
            }
        )
        deep_update(value, updates)
        return value

    return [
        event("sample-1", EVENT_PIPELINE, "utt-1", ROUTE_DIRECT, "修正", "修正", "1_4", "zhOnly", "mandarin"),
        event("sample-2", EVENT_PIPELINE, "utt-2", ROUTE_REVIEW, "load fail", "Load Fail", "5_15", "enOnly", "english"),
        event(
            "sample-3",
            EVENT_REVIEW_SELECTION,
            "utt-2",
            ROUTE_REVIEW,
            "load fail",
            "Load Fail",
            "5_15",
            "enOnly",
            "english",
            userAction={
                "source": "reviewCandidate",
                "targetText": "Load Fail",
                "selectedCandidateText": "Load Fail",
                "rejectedCandidateText": None,
                "selectedRangeLength": 9,
                "timeSinceUtteranceMs": 1500,
                "estimatedClickCount": 1,
                "repeatRequested": False,
            },
            shadowCandidates=[
                {
                    "text": "Load Fail",
                    "source": "sample",
                    "rank": 1,
                    "score": 0.9,
                    "wouldChangeOutput": False,
                    "requiresReview": True,
                    "reason": "sample",
                }
            ],
            classification={
                "lengthBucket": "5_15",
                "scriptMode": "enOnly",
                "languageMode": "english",
                "isCommandLike": False,
                "isTechnicalTermCandidate": True,
                "evidenceTier": "T2_CONFIRMED_SPAN",
                "noiseFlags": [],
                "isPurePhoneticCandidate": True,
            },
        ),
        event(
            "sample-4",
            EVENT_USER_CORRECTION,
            "utt-3",
            ROUTE_DIRECT,
            "六九輪",
            "69 輪",
            "1_4",
            "zhOnly",
            "mandarin",
            userAction={
                "source": "userSubstitution",
                "targetText": "69 輪",
                "selectedCandidateText": None,
                "rejectedCandidateText": None,
                "selectedRangeLength": None,
                "timeSinceUtteranceMs": None,
                "estimatedClickCount": 2,
                "repeatRequested": False,
            },
            classification={
                "lengthBucket": "1_4",
                "scriptMode": "zhOnly",
                "languageMode": "mandarin",
                "isCommandLike": False,
                "isTechnicalTermCandidate": False,
                "evidenceTier": "T1_WEAK_INTERACTION",
                "noiseFlags": ["selectedSpanMissing", "correctionTooLate"],
                "isPurePhoneticCandidate": False,
            },
            safety={
                "wouldHaveChangedFinalOutput": False,
                "autoApplied": False,
                "blockedBecauseLlmOnly": False,
                "blockedBecauseShortPhraseRisk": True,
                "blockedBecauseNoiseSuspected": True,
                "blockedBecauseNegativeEvidence": False,
            },
        ),
    ]


def deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
