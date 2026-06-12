#!/usr/bin/env python3
"""
Local control plane for Voco auto-apply correction rules.

This CLI is intentionally boring: every user-confirmed correction, context
lock, tombstone, activation, and rollback is appended to an evidence JSONL
store. The active model is never edited directly; a compiler patches a baseline
model, a validator checks examples/sentinels/replay, then the installer copies
the validated artifact into Voco's Application Support directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import re
import shutil
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


APP_SUPPORT = Path.home() / "Library/Application Support/com.jasonchien.Voco"
DEFAULT_DB = APP_SUPPORT / "default.store"
DEFAULT_ACTIVE_MODEL = APP_SUPPORT / "AutoApplyModels/full-db.auto-apply-model.json"
DEFAULT_CONTROL_DIR = APP_SUPPORT / "AutoApplyControl"
DEFAULT_EVIDENCE_STORE = DEFAULT_CONTROL_DIR / "evidence.jsonl"
DEFAULT_OUTPUT_ROOT = DEFAULT_CONTROL_DIR / "artifacts"
DEFAULT_REPLAYLAB_ROOT = Path.home() / "GitHub/VocoReplayLab"
DEFAULT_CURRENT_CORPUS_DIR = DEFAULT_REPLAYLAB_ROOT / "artifacts/full-db-raw-cleaned-20260611-093103-context10"
DEFAULT_RERAW_CORPUS_DIR = DEFAULT_REPLAYLAB_ROOT / "artifacts/full-db-reraw-cleaned-20260611-pre12022-context10"
CONTROL_SCHEMA_VERSION = 1
STRICT_SPACE_RE = re.compile(r"\s+")
ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+.#/-]*")
MANUAL_CORPUS_ACCEPTANCE_MAX = 25
DEFAULT_BACKUP_RETENTION = 3
PROTECTED_TERM_GUARD_REASON = "auto-apply-model-protected-term-guard"
PROTECTED_TERM_GUARD_KEYS = ("protectedTermAllowlistGuards", "protectedTermAllowlist")
BASELINE_DRIFT_RISK_FLAGS = {
    "storedOutputDisagreesWithRawDerivedCleaned",
    "rerawStoredBaselineDrift",
    "rerawDriftUncertainShortOrFiller",
}


def main() -> int:
    args = parse_args()
    result = run_command(args)
    if result is None:
        return 0
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_human(result)
    return 0 if not result.get("failed") else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control Voco auto-apply correction evidence and model artifacts.")
    parser.add_argument("--evidence-store", type=Path, default=DEFAULT_EVIDENCE_STORE)
    parser.add_argument("--replaylab-root", type=Path, default=DEFAULT_REPLAYLAB_ROOT)
    parser.add_argument("--actor", default="codex")
    parser.add_argument("--json", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_recent = subparsers.add_parser("listRecentTranscriptions")
    list_recent.add_argument("--store", type=Path, default=DEFAULT_DB)
    list_recent.add_argument("--limit", type=int, default=20)
    list_recent.add_argument("--min-pk", type=int)

    correction = subparsers.add_parser("addCorrection")
    correction.add_argument("--source-text", required=True)
    correction.add_argument("--target-text", required=True)
    correction.add_argument("--row-pk", type=int)
    correction.add_argument("--context", default="")
    correction.add_argument("--note")

    hallucination = subparsers.add_parser("addHallucination")
    hallucination.add_argument("--source-text", required=True)
    hallucination.add_argument("--forbidden-target")
    hallucination.add_argument("--policy-id")
    hallucination.add_argument("--context", default="")
    hallucination.add_argument("--note")

    context_rule = subparsers.add_parser("addContextLockedRule")
    context_rule.add_argument("--source-pattern", required=True)
    context_rule.add_argument("--target-text", required=True)
    context_rule.add_argument("--source-text")
    context_rule.add_argument("--row-pk", type=int)
    context_rule.add_argument("--lock-name")
    context_rule.add_argument("--context-token", action="append", default=[])
    context_rule.add_argument("--context-alias", action="append", default=[])
    context_rule.add_argument("--context-from-context-only", action="store_true")
    context_rule.add_argument("--require-alias", action="store_true")
    context_rule.add_argument("--positive", action="append", default=[], help="TEXT||CONTEXT||EXPECTED")
    context_rule.add_argument("--negative", action="append", default=[], help="TEXT||CONTEXT")
    context_rule.add_argument("--positive-text")
    context_rule.add_argument("--positive-context", default="")
    context_rule.add_argument("--expected-text")
    context_rule.add_argument("--negative-text")
    context_rule.add_argument("--negative-context", default="")
    context_rule.add_argument("--note")

    disable = subparsers.add_parser("disableRule")
    disable.add_argument("--policy-id")
    disable.add_argument("--source-pattern")
    disable.add_argument("--target-text")
    disable.add_argument("--reason", required=True)
    disable.add_argument("--disposition", choices=["blocked", "replaced"])

    list_evidence = subparsers.add_parser("listEvidence")
    list_evidence.add_argument("--limit", type=int, default=20)

    compile_model = subparsers.add_parser("compileModel")
    add_model_io_args(compile_model)
    add_validation_args(compile_model)

    validate = subparsers.add_parser("validateModel")
    validate.add_argument("--model", type=Path, required=True)
    validate.add_argument("--base-model", type=Path, default=DEFAULT_ACTIVE_MODEL)
    validate.add_argument("--report", type=Path)
    validate.add_argument("--write-readiness", action=argparse.BooleanOptionalAction, default=True)
    add_validation_args(validate)

    activate = subparsers.add_parser("activateModel")
    activate.add_argument("--model", type=Path, required=True)
    activate.add_argument("--active-model", type=Path, default=DEFAULT_ACTIVE_MODEL)
    activate.add_argument("--base-model", type=Path, default=DEFAULT_ACTIVE_MODEL)
    activate.add_argument("--backup-suffix", default="control")
    activate.add_argument("--backup-dir", type=Path, help="Optional backup directory; default is no backup.")
    activate.add_argument("--backup-retention", type=int, default=DEFAULT_BACKUP_RETENTION)
    add_validation_args(activate)

    rollback = subparsers.add_parser("rollbackModel")
    rollback.add_argument("--active-model", type=Path, default=DEFAULT_ACTIVE_MODEL)
    rollback.add_argument("--backup", type=Path)
    rollback.add_argument("--backup-dir", type=Path, help="Optional backup directory to list or use for newest backup lookup.")
    rollback.add_argument("--list", action="store_true")
    rollback.add_argument("--reason", default="manual rollback")
    rollback.add_argument("--pre-rollback-backup-dir", type=Path, help="Optional directory for backing up the current active model before rollback.")
    rollback.add_argument("--pre-rollback-backup-retention", type=int, default=DEFAULT_BACKUP_RETENTION)

    protected_guard = subparsers.add_parser("upsertProtectedTermAllowlistGuard")
    protected_guard.add_argument("--model", type=Path, default=DEFAULT_ACTIVE_MODEL)
    protected_guard.add_argument("--guard-id", required=True)
    protected_guard.add_argument("--term", required=True)
    protected_guard.add_argument("--allowed-phrase", action="append", required=True)
    protected_guard.add_argument("--reason", default=PROTECTED_TERM_GUARD_REASON)
    protected_guard.add_argument("--backup-suffix", default="protected-term-guard")
    protected_guard.add_argument("--backup-dir", type=Path, help="Optional backup directory; default is no backup.")
    protected_guard.add_argument("--backup-retention", type=int, default=DEFAULT_BACKUP_RETENTION)

    explain = subparsers.add_parser("explainRuleMatch")
    explain.add_argument("--model", type=Path, default=DEFAULT_ACTIVE_MODEL)
    explain.add_argument("--text", required=True)
    explain.add_argument("--context", default="")

    return parser.parse_args()


def add_model_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", type=Path, default=DEFAULT_ACTIVE_MODEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-model", type=Path)


def add_validation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--current-corpus-dir", type=Path, default=DEFAULT_CURRENT_CORPUS_DIR)
    parser.add_argument("--reraw-corpus-dir", type=Path, default=DEFAULT_RERAW_CORPUS_DIR)
    parser.add_argument("--skip-corpus-replay", action="store_true")
    parser.add_argument("--skip-raw-input-replay", action="store_true")


def run_command(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.command == "listRecentTranscriptions":
        return list_recent_transcriptions(args.store.expanduser(), args.limit, args.min_pk)
    if args.command == "addCorrection":
        event = correction_event(args)
        append_event(args.evidence_store.expanduser(), event)
        return {"event": event, "evidenceStore": str(args.evidence_store.expanduser())}
    if args.command == "addHallucination":
        event = hallucination_event(args)
        append_event(args.evidence_store.expanduser(), event)
        return {"event": event, "evidenceStore": str(args.evidence_store.expanduser())}
    if args.command == "addContextLockedRule":
        event = context_locked_rule_event(args)
        append_event(args.evidence_store.expanduser(), event)
        return {"event": event, "evidenceStore": str(args.evidence_store.expanduser())}
    if args.command == "disableRule":
        event = disable_rule_event(args)
        append_event(args.evidence_store.expanduser(), event)
        return {"event": event, "evidenceStore": str(args.evidence_store.expanduser())}
    if args.command == "listEvidence":
        events = load_events(args.evidence_store.expanduser())
        return {
            "evidenceStore": str(args.evidence_store.expanduser()),
            "eventCount": len(events),
            "events": events[-args.limit :],
        }
    if args.command == "compileModel":
        return compile_model_command(args)
    if args.command == "validateModel":
        return validate_model_command(args)
    if args.command == "activateModel":
        return activate_model_command(args)
    if args.command == "rollbackModel":
        return rollback_model_command(args)
    if args.command == "upsertProtectedTermAllowlistGuard":
        return upsert_protected_term_allowlist_guard_command(args)
    if args.command == "explainRuleMatch":
        return explain_rule_match(args.model.expanduser(), args.text, args.context)
    raise AssertionError(f"Unhandled command: {args.command}")


def correction_event(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "ruleType": "exactTrainablePair",
        "rowPk": args.row_pk,
        "sourceText": args.source_text,
        "targetText": args.target_text,
        "context": args.context or "",
        "examples": {
            "positive": [
                {
                    "text": args.source_text,
                    "context": args.context or "",
                    "expectedText": args.target_text,
                }
            ],
            "negative": [],
        },
        "provenance": {
            "manualLabel": "confirmed-correction",
            "evidenceTier": "T4_GOLD",
            "note": args.note,
        },
    }
    return make_event(args.actor, "addCorrection", payload)


def hallucination_event(args: argparse.Namespace) -> dict[str, Any]:
    negative: dict[str, Any] = {
        "text": args.source_text,
        "context": args.context or "",
        "expectedText": args.source_text,
    }
    if args.forbidden_target:
        negative["forbiddenText"] = args.forbidden_target
    payload = {
        "rowPk": None,
        "sourceText": args.source_text,
        "forbiddenTarget": args.forbidden_target,
        "policyId": args.policy_id,
        "context": args.context or "",
        "examples": {"positive": [], "negative": [negative]},
        "tombstone": {
            "policyId": args.policy_id,
            "sourcePattern": args.source_text,
            "targetText": args.forbidden_target,
            "reason": args.note or "negative hallucination evidence",
            "blockedBecauseNegativeEvidence": True,
            "disposition": "blocked",
        },
        "provenance": {
            "manualLabel": "hallucination",
            "evidenceTier": "NEGATIVE_EVIDENCE",
            "note": args.note,
        },
    }
    return make_event(args.actor, "addHallucination", payload)


def context_locked_rule_event(args: argparse.Namespace) -> dict[str, Any]:
    tokens = compact_strings(args.context_token)
    aliases = compact_strings(args.context_alias)
    if not tokens and not aliases:
        raise SystemExit("addContextLockedRule requires at least one --context-token or --context-alias")

    source_text = args.source_text or args.source_pattern
    positive_examples = parse_positive_examples(args.positive)
    negative_examples = parse_negative_examples(args.negative)
    if args.positive_text:
        positive_examples.append(
            {
                "text": args.positive_text,
                "context": args.positive_context or "",
                "expectedText": args.expected_text or replace_text(args.positive_text, args.source_pattern, args.target_text),
            }
        )
    if not positive_examples:
        positive_examples.append(
            {
                "text": source_text,
                "context": " ".join(tokens + aliases),
                "expectedText": replace_text(source_text, args.source_pattern, args.target_text),
            }
        )
    if args.negative_text:
        negative_examples.append(
            {
                "text": args.negative_text,
                "context": args.negative_context or "",
                "expectedText": args.negative_text,
                "forbiddenText": args.target_text,
            }
        )

    payload = {
        "ruleType": "scopedReplacement",
        "rowPk": args.row_pk,
        "sourceText": source_text,
        "sourcePattern": args.source_pattern,
        "targetText": args.target_text,
        "lockName": args.lock_name or f"manual-context-lock:{short_digest(args.source_pattern + '->' + args.target_text)}",
        "contextTokensAny": tokens,
        "contextAliasesAny": aliases,
        "contextFromContextOnly": bool(args.context_from_context_only),
        "requireAlias": bool(args.require_alias),
        "examples": {
            "positive": positive_examples,
            "negative": negative_examples,
        },
        "provenance": {
            "manualLabel": "confirmed-context-locked-rule",
            "evidenceTier": "T4_GOLD",
            "note": args.note,
        },
    }
    return make_event(args.actor, "addContextLockedRule", payload)


def disable_rule_event(args: argparse.Namespace) -> dict[str, Any]:
    if not args.policy_id and not (args.source_pattern and args.target_text):
        raise SystemExit("disableRule requires --policy-id or both --source-pattern and --target-text")
    payload = {
        "tombstone": {
            "policyId": args.policy_id,
            "sourcePattern": args.source_pattern,
            "targetText": args.target_text,
            "reason": args.reason,
            "blockedBecauseNegativeEvidence": False,
            "disposition": args.disposition,
        }
    }
    return make_event(args.actor, "disableRule", payload)


def make_event(actor: str, action: str, payload: dict[str, Any]) -> dict[str, Any]:
    created_at = now_iso()
    digest = short_digest(json.dumps(payload, ensure_ascii=False, sort_keys=True) + created_at)
    return {
        "schemaVersion": CONTROL_SCHEMA_VERSION,
        "eventId": f"evt-{created_at.replace(':', '').replace('-', '').replace('+0000', 'Z')}-{digest}",
        "createdAt": created_at,
        "actor": actor,
        "source": "voco-auto-apply-control",
        "action": action,
        "payload": payload,
    }


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            events.append(event)
    return events


def compile_model_command(args: argparse.Namespace) -> dict[str, Any]:
    evidence_store = args.evidence_store.expanduser()
    base_model_path = args.base_model.expanduser()
    output_model = output_model_path(args)
    events = load_events(evidence_store)
    base_model = load_model(base_model_path)
    model, compile_report = compile_model(base_model, events, base_model_path=base_model_path, evidence_store=evidence_store)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    write_model(output_model, model)

    validation = validate_model(
        model,
        events,
        model_path=output_model,
        base_model=base_model,
        replaylab_root=args.replaylab_root.expanduser(),
        current_corpus_dir=args.current_corpus_dir.expanduser(),
        reraw_corpus_dir=args.reraw_corpus_dir.expanduser(),
        skip_corpus_replay=args.skip_corpus_replay,
        skip_raw_input_replay=args.skip_raw_input_replay,
    )
    apply_readiness(model, validation)
    write_model(output_model, model)
    report_path = output_model.with_suffix(".validation.json")
    report_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "compiledModel": str(output_model),
        "validationReport": str(report_path),
        "compile": compile_report,
        "validation": validation_summary(validation),
        "failed": not validation["ready"],
    }


def output_model_path(args: argparse.Namespace) -> Path:
    if args.output_model:
        return args.output_model.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir else DEFAULT_OUTPUT_ROOT / timestamp_for_path()
    return output_dir / "full-db.auto-apply-model.json"


def compile_model(
    base_model: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    base_model_path: Path,
    evidence_store: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = copy.deepcopy(base_model)
    model["generatedAt"] = now_iso()
    model["modelType"] = "control_plane_patched_auto_apply_model"
    policies = [copy.deepcopy(policy) for policy in model.get("policies") or []]
    overlay_policy_count = 0
    tombstone_count = 0

    for event in events:
        action = str(event.get("action") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if action == "addCorrection":
            policy = exact_policy_from_event(event)
            overlay_policy_count += upsert_policy(policies, policy, event)
        elif action == "addContextLockedRule":
            policy = context_policy_from_event(event)
            overlay_policy_count += upsert_policy(policies, policy, event)
        elif action in {"disableRule", "addHallucination"}:
            tombstone = payload.get("tombstone") if isinstance(payload.get("tombstone"), dict) else {}
            tombstone_count += tombstone_matching_policies(policies, tombstone, event)

    model["policies"] = policies
    model["policyCounts"] = dict(Counter(str(policy.get("autoApplyMode") or "unknown") for policy in policies))
    model["policyTypeCounts"] = dict(Counter(str(policy.get("policyType") or "unknown") for policy in policies))
    tombstone_disposition_counts = dict(
        Counter(
            str((policy.get("tombstone") or {}).get("disposition") or "unknown")
            for policy in policies
            if isinstance(policy.get("tombstone"), dict)
        )
    )
    append_safety_contract(model)
    model["controlPlane"] = {
        "schemaVersion": CONTROL_SCHEMA_VERSION,
        "compiledAt": now_iso(),
        "baseModel": str(base_model_path),
        "baseModelSha256": sha256_file(base_model_path) if base_model_path.exists() else None,
        "evidenceStore": str(evidence_store),
        "evidenceStoreSha256": sha256_file(evidence_store) if evidence_store.exists() else None,
        "eventCount": len(events),
        "overlayPolicyCount": overlay_policy_count,
        "tombstoneCount": tombstone_count,
        "tombstoneDispositionCounts": tombstone_disposition_counts,
    }
    report = {
        "basePolicyCounts": base_model.get("policyCounts") or {},
        "newPolicyCounts": model["policyCounts"],
        "basePolicyTypeCounts": base_model.get("policyTypeCounts") or {},
        "newPolicyTypeCounts": model["policyTypeCounts"],
        "eventCount": len(events),
        "overlayPolicyCount": overlay_policy_count,
        "tombstoneCount": tombstone_count,
        "tombstoneDispositionCounts": tombstone_disposition_counts,
    }
    return model, report


def exact_policy_from_event(event: dict[str, Any]) -> dict[str, Any]:
    payload = event["payload"]
    source_text = str(payload["sourceText"])
    target_text = str(payload["targetText"])
    input_key = strict_text_key(source_text)
    target_key = strict_text_key(target_text)
    row_pk = payload.get("rowPk")
    evidence_rows = [int(row_pk)] if row_pk else []
    return {
        "policyId": f"manual-exact-{short_digest(input_key + '->' + target_key, length=16)}",
        "policyType": "exactTrainablePair",
        "autoApplyMode": "apply",
        "decisionReason": "manual confirmed correction from control-plane evidence; normalized whole-utterance exact match only",
        "source": source_text,
        "target": target_text,
        "sourcePattern": source_text,
        "targetText": target_text,
        "inputText": source_text,
        "inputStrictKey": input_key,
        "targetStrictKey": target_key,
        "lockName": "manual-exact-trainable-pair",
        "contextRequired": False,
        "contextTokensAny": [],
        "contextAliasesAny": [],
        "contextFromContextOnly": False,
        "requireAlias": False,
        "scopedSourcePhrase": None,
        "scopeWindow": "normalized whole-utterance exact match only",
        "evidenceRows": evidence_rows,
        "trainableRows": evidence_rows,
        "reviewRows": [],
        "evidenceCount": len(evidence_rows) or 1,
        "trainableEvidenceCount": len(evidence_rows) or 1,
        "reviewEvidenceCount": 0,
        "riskFlagCounts": {},
        "labelTierCounts": {"T4_GOLD": 1},
        "cleanedSourceCounts": {"manualControlPlane": 1},
        "pairContextRequiredRows": [],
        "storedOutputDisagreesRows": [],
        "reviewGateConflictRows": [],
        "manualOverrideRows": [],
        "exactInputRequired": True,
        "exactInputResolution": None,
        "sourceSlices": ["manualControlPlane"],
        "sourcePolicies": [],
        "controlEvidenceEventIds": [event["eventId"]],
    }


def context_policy_from_event(event: dict[str, Any]) -> dict[str, Any]:
    payload = event["payload"]
    source_pattern = str(payload["sourcePattern"])
    target_text = str(payload["targetText"])
    source_text = str(payload.get("sourceText") or source_pattern)
    tokens = compact_strings(payload.get("contextTokensAny") or [])
    aliases = compact_strings(payload.get("contextAliasesAny") or [])
    row_pk = payload.get("rowPk")
    evidence_rows = [int(row_pk)] if row_pk else []
    policy_id_key = json.dumps(
        {
            "sourcePattern": source_pattern,
            "targetText": target_text,
            "tokens": tokens,
            "aliases": aliases,
            "lockName": payload.get("lockName"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "policyId": f"manual-context-{short_digest(policy_id_key, length=16)}",
        "policyType": "scopedReplacement",
        "autoApplyMode": "apply",
        "decisionReason": "manual context-locked scoped replacement from control-plane evidence",
        "source": source_text,
        "target": target_text,
        "sourcePattern": source_pattern,
        "targetText": target_text,
        "lockName": payload.get("lockName") or "manual-context-lock",
        "contextRequired": True,
        "contextTokensAny": tokens,
        "contextAliasesAny": aliases,
        "contextFromContextOnly": bool(payload.get("contextFromContextOnly")),
        "requireAlias": bool(payload.get("requireAlias")),
        "scopedSourcePhrase": source_pattern,
        "scopeWindow": "manual context lock; current text plus Voco context hints unless contextFromContextOnly is true",
        "evidenceRows": evidence_rows,
        "trainableRows": evidence_rows,
        "reviewRows": [],
        "evidenceCount": len(evidence_rows) or 1,
        "trainableEvidenceCount": len(evidence_rows) or 1,
        "reviewEvidenceCount": 0,
        "riskFlagCounts": {"manualContextLock": 1},
        "labelTierCounts": {"T4_GOLD": 1},
        "cleanedSourceCounts": {"manualControlPlane": 1},
        "pairContextRequiredRows": evidence_rows,
        "storedOutputDisagreesRows": [],
        "reviewGateConflictRows": [],
        "manualOverrideRows": [],
        "exactInputRequired": False,
        "inputText": None,
        "inputStrictKey": None,
        "targetStrictKey": strict_text_key(target_text),
        "exactInputResolution": None,
        "sourceSlices": ["manualControlPlane"],
        "sourcePolicies": [],
        "controlEvidenceEventIds": [event["eventId"]],
    }


def upsert_policy(policies: list[dict[str, Any]], new_policy: dict[str, Any], event: dict[str, Any]) -> int:
    for policy in policies:
        if policy_identity(policy) == policy_identity(new_policy):
            ids = list(policy.get("controlEvidenceEventIds") or [])
            if event["eventId"] not in ids:
                ids.append(event["eventId"])
            policy["controlEvidenceEventIds"] = ids
            policy["decisionReason"] = str(policy.get("decisionReason") or "") + "; reinforced by manual control-plane evidence"
            return 0
    policies.append(new_policy)
    return 1


def policy_identity(policy: dict[str, Any]) -> tuple[Any, ...]:
    if policy.get("policyType") == "exactTrainablePair":
        return (
            "exactTrainablePair",
            policy.get("inputStrictKey"),
            policy.get("targetStrictKey"),
        )
    return (
        policy.get("policyType"),
        policy.get("sourcePattern"),
        policy.get("targetText"),
        tuple(policy.get("contextTokensAny") or []),
        tuple(policy.get("contextAliasesAny") or []),
    )


def tombstone_matching_policies(policies: list[dict[str, Any]], tombstone: dict[str, Any], event: dict[str, Any]) -> int:
    count = 0
    disposition = tombstone_disposition(tombstone)
    for policy in policies:
        if not tombstone_matches_policy(tombstone, policy):
            continue
        policy["autoApplyMode"] = disposition
        policy["decisionReason"] = f"{disposition} by control-plane evidence: {tombstone.get('reason') or event.get('action')}"
        policy["tombstone"] = {
            "eventId": event["eventId"],
            "createdAt": event["createdAt"],
            "reason": tombstone.get("reason"),
            "blockedBecauseNegativeEvidence": bool(tombstone.get("blockedBecauseNegativeEvidence")),
            "disposition": disposition,
        }
        ids = list(policy.get("controlEvidenceEventIds") or [])
        ids.append(event["eventId"])
        policy["controlEvidenceEventIds"] = sorted(set(ids))
        count += 1
    return count


def tombstone_disposition(tombstone: dict[str, Any]) -> str:
    disposition = tombstone.get("disposition")
    if disposition in {"blocked", "replaced"}:
        return str(disposition)
    if bool(tombstone.get("blockedBecauseNegativeEvidence")):
        return "blocked"

    reason = str(tombstone.get("reason") or "")
    if reason.startswith("Promote suggest-only"):
        return "replaced"
    return "blocked"


def tombstone_matches_policy(tombstone: dict[str, Any], policy: dict[str, Any]) -> bool:
    policy_id = tombstone.get("policyId")
    if policy_id and policy.get("policyId") == policy_id:
        return True
    source_pattern = tombstone.get("sourcePattern")
    target_text = tombstone.get("targetText")
    if source_pattern and target_text:
        source_key = strict_text_key(str(source_pattern))
        target_key = strict_text_key(str(target_text))
        return (
            strict_text_key(str(policy.get("sourcePattern") or policy.get("source") or "")) == source_key
            and strict_text_key(str(policy.get("targetText") or policy.get("target") or "")) == target_key
        )
    return False


def append_safety_contract(model: dict[str, Any]) -> None:
    existing = list(model.get("safetyContract") or [])
    additions = [
        "control-plane rule changes must originate from append-only evidence JSONL events",
        "manual context-locked scoped replacements require explicit context tokens or aliases",
        "manual tombstones preserve provenance by marking policies blocked or replaced instead of deleting evidence",
        "activation requires positive examples, negative examples, sentinel replay, and corpus replay when available",
        "protected term allowlist guards must be declared in the model artifact, not hard-coded in runtime services",
    ]
    for item in additions:
        if item not in existing:
            existing.append(item)
    model["safetyContract"] = existing


def validate_model_command(args: argparse.Namespace) -> dict[str, Any]:
    model_path = args.model.expanduser()
    evidence_store = args.evidence_store.expanduser()
    base_model_path = args.base_model.expanduser()
    model = load_model(model_path)
    base_model = load_model(base_model_path) if base_model_path.exists() else None
    events = load_events(evidence_store)
    report = validate_model(
        model,
        events,
        model_path=model_path,
        base_model=base_model,
        replaylab_root=args.replaylab_root.expanduser(),
        current_corpus_dir=args.current_corpus_dir.expanduser(),
        reraw_corpus_dir=args.reraw_corpus_dir.expanduser(),
        skip_corpus_replay=args.skip_corpus_replay,
        skip_raw_input_replay=args.skip_raw_input_replay,
    )
    if args.write_readiness:
        apply_readiness(model, report)
        write_model(model_path, model)
    report_path = args.report.expanduser() if args.report else model_path.with_suffix(".validation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "model": str(model_path),
        "report": str(report_path),
        "validation": validation_summary(report),
        "failed": not report["ready"],
    }


def validate_model(
    model: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    model_path: Path,
    base_model: dict[str, Any] | None,
    replaylab_root: Path,
    current_corpus_dir: Path,
    reraw_corpus_dir: Path,
    skip_corpus_replay: bool,
    skip_raw_input_replay: bool,
) -> dict[str, Any]:
    apply_policies = [policy for policy in model.get("policies") or [] if policy.get("autoApplyMode") == "apply"]
    protected_guards = protected_term_allowlist_guards(model)
    failures: list[dict[str, Any]] = []
    positive_results = validate_positive_examples(events, apply_policies, protected_guards)
    negative_results = validate_negative_examples(events, apply_policies, protected_guards)
    failures.extend(item for item in positive_results if not item["passed"])
    failures.extend(item for item in negative_results if not item["passed"])
    exact_conflicts = exact_apply_conflicts(apply_policies)
    failures.extend(exact_conflicts)
    manual_context_failures = manual_context_lock_failures(apply_policies)
    failures.extend(manual_context_failures)
    count_report = policy_count_report(model, base_model)
    failures.extend(count_report["failures"])
    corpus_reports = []
    if not skip_corpus_replay:
        for name, corpus_dir in [("currentRaw", current_corpus_dir), ("rerawPre12022", reraw_corpus_dir)]:
            report = corpus_replay_report(name, corpus_dir, model, model_path, replaylab_root, skip_raw_input_replay)
            corpus_reports.append(report)
            failures.extend(report.get("failures") or [])
    ready = not failures
    return {
        "generatedAt": now_iso(),
        "model": str(model_path),
        "ready": ready,
        "reason": "control-plane validation passed" if ready else "control-plane validation failed",
        "positiveExamples": positive_results,
        "negativeExamples": negative_results,
        "exactApplyConflicts": exact_conflicts,
        "manualContextLockFailures": manual_context_failures,
        "policyCounts": model.get("policyCounts") or {},
        "policyTypeCounts": model.get("policyTypeCounts") or {},
        "policyCountReport": count_report,
        "corpusReplay": corpus_reports,
        "failures": failures,
    }


def validate_positive_examples(
    events: list[dict[str, Any]],
    apply_policies: list[dict[str, Any]],
    protected_guards: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        examples = payload.get("examples") if isinstance(payload.get("examples"), dict) else {}
        for example in examples.get("positive") or []:
            text = str(example.get("text") or "")
            context = str(example.get("context") or "")
            expected = str(example.get("expectedText") or "")
            after, fires = replay_apply_policies(text, context, apply_policies, protected_guards)
            passed = strict_text_key(after) == strict_text_key(expected)
            results.append(
                {
                    "eventId": event.get("eventId"),
                    "kind": "positiveExample",
                    "text": text,
                    "context": context,
                    "expectedText": expected,
                    "actualText": after,
                    "fires": fires,
                    "passed": passed,
                }
            )
    return results


def validate_negative_examples(
    events: list[dict[str, Any]],
    apply_policies: list[dict[str, Any]],
    protected_guards: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        examples = payload.get("examples") if isinstance(payload.get("examples"), dict) else {}
        for example in examples.get("negative") or []:
            text = str(example.get("text") or "")
            context = str(example.get("context") or "")
            expected = str(example.get("expectedText") or text)
            forbidden = str(example.get("forbiddenText") or "")
            after, fires = replay_apply_policies(text, context, apply_policies, protected_guards)
            expected_ok = strict_text_key(after) == strict_text_key(expected)
            forbidden_ok = not forbidden or forbidden not in after
            results.append(
                {
                    "eventId": event.get("eventId"),
                    "kind": "negativeExample",
                    "text": text,
                    "context": context,
                    "expectedText": expected,
                    "forbiddenText": forbidden or None,
                    "actualText": after,
                    "fires": fires,
                    "passed": expected_ok and forbidden_ok,
                }
            )
    return results


def exact_apply_conflicts(apply_policies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    policy_ids: dict[str, list[str]] = defaultdict(list)
    for policy in apply_policies:
        if policy.get("policyType") != "exactTrainablePair":
            continue
        input_key = str(policy.get("inputStrictKey") or "")
        target_key = str(policy.get("targetStrictKey") or strict_text_key(str(policy.get("targetText") or "")))
        if not input_key or not target_key:
            continue
        grouped[input_key].add(target_key)
        policy_ids[input_key].append(str(policy.get("policyId")))
    return [
        {
            "kind": "exactApplyConflict",
            "inputStrictKey": input_key,
            "targetStrictKeys": sorted(target_keys),
            "policyIds": sorted(policy_ids[input_key]),
            "passed": False,
        }
        for input_key, target_keys in grouped.items()
        if len(target_keys) > 1
    ]


def manual_context_lock_failures(apply_policies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for policy in apply_policies:
        if not str(policy.get("policyId") or "").startswith("manual-context-"):
            continue
        tokens = compact_strings(policy.get("contextTokensAny") or [])
        aliases = compact_strings(policy.get("contextAliasesAny") or [])
        if not policy.get("contextRequired") or not (tokens or aliases):
            failures.append(
                {
                    "kind": "manualContextLockMissingContext",
                    "policyId": policy.get("policyId"),
                    "passed": False,
                }
            )
    return failures


def policy_count_report(model: dict[str, Any], base_model: dict[str, Any] | None) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    if not base_model:
        return {"baseTotalPolicies": None, "newTotalPolicies": len(model.get("policies") or []), "failures": failures}
    base_total = len(base_model.get("policies") or [])
    new_total = len(model.get("policies") or [])
    tombstones = int(((model.get("controlPlane") or {}).get("tombstoneCount") or 0))
    if new_total < base_total and tombstones == 0:
        failures.append(
            {
                "kind": "policyCountRegression",
                "baseTotalPolicies": base_total,
                "newTotalPolicies": new_total,
                "passed": False,
            }
        )
    return {
        "baseTotalPolicies": base_total,
        "newTotalPolicies": new_total,
        "tombstoneCount": tombstones,
        "failures": failures,
    }


def corpus_replay_report(
    name: str,
    corpus_dir: Path,
    model: dict[str, Any],
    model_path: Path,
    replaylab_root: Path,
    skip_raw_input_replay: bool,
) -> dict[str, Any]:
    cleaned_path = corpus_dir / "full-db.cleaned.jsonl"
    raw_path = corpus_dir / "full-db.raw.jsonl"
    trainable_path = corpus_dir / "full-db.trainable-pairs.jsonl"
    if not cleaned_path.exists():
        return {
            "sourceSlice": name,
            "corpusDir": str(corpus_dir),
            "skipped": True,
            "reason": "cleaned corpus not found",
            "failures": [],
        }
    backend = load_replaylab_backend(replaylab_root)
    records = load_jsonl(cleaned_path)
    protected_guards = protected_term_allowlist_guards(model)
    cleaned_report = (
        backend["auto_apply"].replay_model(records, model)
        if backend and not protected_guards
        else local_corpus_replay(records, model)
    )
    filter_accepted_manual_corpus_changes(cleaned_report, model)
    failures: list[dict[str, Any]] = []
    if cleaned_report["sentinelFailures"]:
        failures.append(
            {
                "kind": "sentinelFailures",
                "sourceSlice": name,
                "count": len(cleaned_report["sentinelFailures"]),
                "items": cleaned_report["sentinelFailures"][:10],
                "passed": False,
            }
        )
    if cleaned_report["unexpectedChanges"]:
        failures.append(
            {
                "kind": "unexpectedCorpusChanges",
                "sourceSlice": name,
                "count": len(cleaned_report["unexpectedChanges"]),
                "items": cleaned_report["unexpectedChanges"][:10],
                "passed": False,
            }
        )
    raw_input_compact: dict[str, Any] | None = None
    if backend and not skip_raw_input_replay and raw_path.exists() and trainable_path.exists() and model_path.exists():
        raw_report = backend["raw_eval"].evaluate_raw_input(raw_path, cleaned_path, trainable_path, model_path)
        filter_accepted_manual_corpus_changes(raw_report, model)
        raw_input_compact = compact_replay_report(raw_report)
        if not raw_report["readiness"]["rawInputReplayPass"]:
            failures.append(
                {
                    "kind": "rawInputReplayFailure",
                    "sourceSlice": name,
                    "readiness": raw_report["readiness"],
                    "sentinelFailures": len(raw_report.get("sentinelFailures") or []),
                    "unexpectedChanges": len(raw_report.get("unexpectedChanges") or []),
                    "passed": False,
                }
            )
    elif not skip_raw_input_replay:
        raw_input_compact = {"skipped": True, "reason": "ReplayLab raw-input backend unavailable or corpus files missing"}
    return {
        "sourceSlice": name,
        "corpusDir": str(corpus_dir),
        "skipped": False,
        "cleanedReplay": compact_replay_report(cleaned_report),
        "rawInputReplay": raw_input_compact,
        "failures": failures,
    }


def filter_accepted_manual_corpus_changes(report: dict[str, Any], model: dict[str, Any]) -> None:
    unexpected = list(report.get("unexpectedChanges") or [])
    if not unexpected:
        return

    policies_by_id = {
        str(policy.get("policyId")): policy
        for policy in model.get("policies") or []
        if policy.get("policyId")
    }
    accepted: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for item in unexpected:
        if is_accepted_manual_corpus_change(item, policies_by_id):
            accepted.append(item)
        else:
            remaining.append(item)

    report["originalUnexpectedChanges"] = len(unexpected)
    if len(accepted) > MANUAL_CORPUS_ACCEPTANCE_MAX:
        report["manualCorpusAcceptanceExceeded"] = {
            "acceptedCandidateCount": len(accepted),
            "maxAccepted": MANUAL_CORPUS_ACCEPTANCE_MAX,
        }
        return

    report["acceptedManualCorpusChanges"] = accepted
    report["unexpectedChanges"] = remaining
    if report.get("sentinelFailures") or remaining:
        return

    readiness = report.get("readiness") if isinstance(report.get("readiness"), dict) else {}
    if "rawInputReplayPass" in readiness:
        readiness["rawInputReplayPass"] = True
        readiness["reason"] = "raw input replay passed after accepted manual corpus changes"
    elif "autoApplyModelReady" in readiness:
        readiness["autoApplyModelReady"] = True
        readiness["reason"] = "replay passed after accepted manual corpus changes"


def is_accepted_manual_corpus_change(item: dict[str, Any], policies_by_id: dict[str, dict[str, Any]]) -> bool:
    fires = item.get("fires") if isinstance(item.get("fires"), list) else []
    if not fires:
        return False

    row_pk = int_or_none(item.get("rowPk"))
    fired_policies: list[tuple[str, dict[str, Any]]] = []
    for fire in fires:
        policy_id = str((fire if isinstance(fire, dict) else {}).get("policyId") or "")
        policy = policies_by_id.get(policy_id)
        if not is_manual_control_policy(policy_id, policy):
            return False
        fired_policies.append((policy_id, policy))

    if row_pk is not None:
        for policy_id, policy in fired_policies:
            if manual_exact_policy_accepts_change(policy_id, policy, row_pk, item):
                return True

    return is_accepted_manual_baseline_drift(item, fired_policies)


def is_accepted_manual_baseline_drift(
    item: dict[str, Any],
    fired_policies: list[tuple[str, dict[str, Any]]],
) -> bool:
    risk_flags = {str(flag) for flag in item.get("riskFlags") or []}
    if not item.get("requiresReview") and not risk_flags.intersection(BASELINE_DRIFT_RISK_FLAGS):
        return False

    for policy_id, policy in fired_policies:
        if not policy_id.startswith(("manual-context-", "manual-exact-")):
            return False
        if policy.get("policyType") == "scopedReplacement":
            tokens = compact_strings(policy.get("contextTokensAny") or [])
            aliases = compact_strings(policy.get("contextAliasesAny") or [])
            if not policy.get("contextRequired") or not (tokens or aliases):
                return False
    return True


def is_manual_control_policy(policy_id: str, policy: dict[str, Any] | None) -> bool:
    return bool(policy) and policy_id.startswith("manual-") and bool(policy.get("controlEvidenceEventIds"))


def manual_exact_policy_accepts_change(
    policy_id: str,
    policy: dict[str, Any],
    row_pk: int,
    item: dict[str, Any],
) -> bool:
    if not policy_id.startswith("manual-exact-"):
        return False
    if policy.get("policyType") != "exactTrainablePair" or policy.get("exactInputRequired") is not True:
        return False
    if not policy_contains_row(policy, "evidenceRows", row_pk):
        return False
    if not policy_contains_row(policy, "trainableRows", row_pk):
        return False

    before = str(item.get("before") or "")
    after = str(item.get("after") or "")
    cleaned = str(item.get("cleanedText") or "")
    if not before or not after or not cleaned:
        return False

    input_key = str(policy.get("inputStrictKey") or strict_text_key(str(policy.get("sourcePattern") or "")))
    target_key = str(policy.get("targetStrictKey") or strict_text_key(str(policy.get("targetText") or "")))
    if strict_text_key(before) != input_key:
        return False
    if strict_text_key(after) != target_key:
        return False
    return strict_text_key(after) != strict_text_key(cleaned)


def policy_contains_row(policy: dict[str, Any], field: str, row_pk: int) -> bool:
    return row_pk in {row for row in (int_or_none(value) for value in policy.get(field) or []) if row is not None}


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compact_replay_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "rowCount": report.get("rowCount"),
        "rawAsrRowCount": report.get("rawAsrRowCount"),
        "applyPolicyCount": report.get("applyPolicyCount"),
        "candidateFireCount": report.get("candidateFireCount"),
        "rowFireCount": report.get("rowFireCount"),
        "changedRows": report.get("changedRows"),
        "rowsMatchingCleanedText": report.get("rowsMatchingCleanedText"),
        "sentinelFailures": len(report.get("sentinelFailures") or []),
        "unexpectedChanges": len(report.get("unexpectedChanges") or []),
        "originalUnexpectedChanges": report.get("originalUnexpectedChanges"),
        "acceptedManualCorpusChanges": len(report.get("acceptedManualCorpusChanges") or []),
        "manualCorpusAcceptanceExceeded": report.get("manualCorpusAcceptanceExceeded"),
        "readiness": report.get("readiness"),
    }


def apply_readiness(model: dict[str, Any], report: dict[str, Any]) -> None:
    failures = report.get("failures") or []
    model["mergedReplayReadiness"] = {
        "mergedAutoApplyModelReady": bool(report.get("ready")),
        "reason": report.get("reason"),
        "failures": failures,
    }


def activate_model_command(args: argparse.Namespace) -> dict[str, Any]:
    model_path = args.model.expanduser()
    active_model = args.active_model.expanduser()
    evidence_store = args.evidence_store.expanduser()
    backup_dir = expanded_optional_path(getattr(args, "backup_dir", None))
    backup_retention = backup_retention_from_args(args)
    model = load_model(model_path)
    base_model = load_model(args.base_model.expanduser()) if args.base_model.expanduser().exists() else None
    events = load_events(evidence_store)
    validation = validate_model(
        model,
        events,
        model_path=model_path,
        base_model=base_model,
        replaylab_root=args.replaylab_root.expanduser(),
        current_corpus_dir=args.current_corpus_dir.expanduser(),
        reraw_corpus_dir=args.reraw_corpus_dir.expanduser(),
        skip_corpus_replay=args.skip_corpus_replay,
        skip_raw_input_replay=args.skip_raw_input_replay,
    )
    if not validation["ready"]:
        return {"model": str(model_path), "validation": validation_summary(validation), "failed": True}
    apply_readiness(model, validation)
    write_model(model_path, model)
    active_model.parent.mkdir(parents=True, exist_ok=True)
    backup_path = create_model_backup(
        active_model,
        backup_dir=backup_dir,
        suffix=getattr(args, "backup_suffix", "control"),
        retention=backup_retention,
    )
    event = make_event(
        args.actor,
        "activateModel",
        {
            "model": str(model_path),
            "modelSha256": sha256_file(model_path),
            "activeModel": str(active_model),
            "previousActiveModelSha256": sha256_file(active_model) if active_model.exists() else None,
            "backup": str(backup_path) if backup_path else None,
            "backupMode": "directory" if backup_dir else "none",
            "backupDirectory": str(backup_dir) if backup_dir else None,
            "backupRetention": backup_retention if backup_dir else None,
            "validationReady": True,
        },
    )
    append_event(evidence_store, event)
    shutil.copy2(model_path, active_model)
    return {
        "activatedModel": str(model_path),
        "activeModel": str(active_model),
        "backup": str(backup_path) if backup_path else None,
        "backupMode": "directory" if backup_dir else "none",
        "backupDirectory": str(backup_dir) if backup_dir else None,
        "backupRetention": backup_retention if backup_dir else None,
        "event": event,
        "validation": validation_summary(validation),
    }


def rollback_model_command(args: argparse.Namespace) -> dict[str, Any]:
    active_model = args.active_model.expanduser()
    backup_dir = expanded_optional_path(getattr(args, "backup_dir", None))
    if args.list:
        backups = list_backups(active_model, backup_dir=backup_dir) if backup_dir else []
        return {
            "activeModel": str(active_model),
            "backupDirectory": str(backup_dir) if backup_dir else None,
            "backups": [str(path) for path in backups],
            "reason": None if backup_dir else "automatic App Support backup lookup is disabled; pass --backup-dir to list managed backups",
        }
    backup = args.backup.expanduser() if args.backup else newest_backup(active_model, backup_dir=backup_dir) if backup_dir else None
    if not backup or not backup.exists():
        return {
            "activeModel": str(active_model),
            "failed": True,
            "reason": "explicit --backup path or --backup-dir is required; automatic App Support .bak lookup is disabled",
        }
    pre_rollback_backup_dir = expanded_optional_path(getattr(args, "pre_rollback_backup_dir", None))
    pre_rollback_backup = create_model_backup(
        active_model,
        backup_dir=pre_rollback_backup_dir,
        suffix="pre-rollback",
        retention=backup_retention_from_args(args, attr="pre_rollback_backup_retention"),
    )
    event = make_event(
        args.actor,
        "rollbackModel",
        {
            "activeModel": str(active_model),
            "activeModelSha256": sha256_file(active_model) if active_model.exists() else None,
            "rollbackSource": str(backup),
            "rollbackSourceSha256": sha256_file(backup),
            "preRollbackBackup": str(pre_rollback_backup) if pre_rollback_backup else None,
            "preRollbackBackupMode": "directory" if pre_rollback_backup_dir else "none",
            "preRollbackBackupDirectory": str(pre_rollback_backup_dir) if pre_rollback_backup_dir else None,
            "reason": args.reason,
        },
    )
    append_event(args.evidence_store.expanduser(), event)
    active_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup, active_model)
    return {
        "activeModel": str(active_model),
        "rollbackSource": str(backup),
        "preRollbackBackup": str(pre_rollback_backup) if pre_rollback_backup else None,
        "event": event,
    }


def upsert_protected_term_allowlist_guard_command(args: argparse.Namespace) -> dict[str, Any]:
    model_path = args.model.expanduser()
    evidence_store = args.evidence_store.expanduser()
    backup_dir = expanded_optional_path(getattr(args, "backup_dir", None))
    backup_retention = backup_retention_from_args(args)
    model = load_model(model_path)
    allowed_phrases = compact_strings(args.allowed_phrase)
    if not allowed_phrases:
        return {"model": str(model_path), "failed": True, "reason": "at least one allowed phrase is required"}

    guard = {
        "guardId": str(args.guard_id),
        "reason": str(args.reason or PROTECTED_TERM_GUARD_REASON),
        "term": str(args.term),
        "allowedPhrases": allowed_phrases,
    }
    guards = [
        guard
        for guard in model.get("protectedTermAllowlistGuards") or []
        if isinstance(guard, dict)
    ]
    replaced = False
    for index, existing in enumerate(guards):
        if existing.get("guardId") == guard["guardId"] or existing.get("term") == guard["term"]:
            guards[index] = guard
            replaced = True
            break
    if not replaced:
        guards.append(guard)

    model["protectedTermAllowlistGuards"] = guards
    append_safety_contract(model)

    backup_path = create_model_backup(
        model_path,
        backup_dir=backup_dir,
        suffix=getattr(args, "backup_suffix", "protected-term-guard"),
        retention=backup_retention,
    )
    write_model(model_path, model)

    event = make_event(
        args.actor,
        "upsertProtectedTermAllowlistGuard",
        {
            "model": str(model_path),
            "modelSha256": sha256_file(model_path),
            "backup": str(backup_path) if backup_path else None,
            "backupMode": "directory" if backup_dir else "none",
            "backupDirectory": str(backup_dir) if backup_dir else None,
            "backupRetention": backup_retention if backup_dir else None,
            "guard": guard,
            "replacedExisting": replaced,
        },
    )
    append_event(evidence_store, event)
    return {
        "model": str(model_path),
        "backup": str(backup_path) if backup_path else None,
        "backupMode": "directory" if backup_dir else "none",
        "backupDirectory": str(backup_dir) if backup_dir else None,
        "backupRetention": backup_retention if backup_dir else None,
        "guard": guard,
        "guardCount": len(guards),
        "event": event,
    }


def explain_rule_match(model_path: Path, text: str, context: str) -> dict[str, Any]:
    model = load_model(model_path)
    apply_policies = [policy for policy in model.get("policies") or [] if policy.get("autoApplyMode") == "apply"]
    suggest_policies = [policy for policy in model.get("policies") or [] if policy.get("autoApplyMode") == "suggest"]
    replay = replay_apply_policies_with_guards(
        text,
        context,
        apply_policies,
        protected_term_allowlist_guards(model),
    )
    after = replay["outputText"]
    suggestions = [
        {
            "policyId": policy.get("policyId"),
            "policyType": policy.get("policyType"),
            "sourcePattern": policy.get("sourcePattern"),
            "targetText": policy.get("targetText"),
        }
        for policy in suggest_policies
        if policy_fires(policy, after, context)
    ]
    return {
        "model": str(model_path),
        "inputText": text,
        "context": context,
        "outputText": after,
        "changed": strict_text_key(text) != strict_text_key(after),
        "applied": replay["applied"],
        "suggestions": suggestions,
        "blocked": replay["blocked"],
        "guardBlocks": replay["guardBlocks"],
        "proposedOutputText": replay.get("proposedOutputText"),
        "blockedApplied": replay.get("blockedApplied") or [],
    }


def list_recent_transcriptions(store: Path, limit: int, min_pk: int | None) -> dict[str, Any]:
    query = """
        select
          Z_PK,
          datetime(ZTIMESTAMP + 978307200, 'unixepoch', 'localtime') as local_time,
          ZRAWTRANSCRIPT,
          ZTEXT,
          ZENHANCEDTEXT,
          ZSELECTEDCANDIDATE,
          ZFINALPASTEDTEXT
        from ZTRANSCRIPTION
        where (? is null or Z_PK >= ?)
        order by ZTIMESTAMP desc
        limit ?
    """
    rows: list[dict[str, Any]] = []
    with sqlite3.connect(f"file:{store}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute(query, (min_pk, min_pk, limit)):
            rows.append(dict(row))
    return {"store": str(store), "limit": limit, "minPk": min_pk, "rows": rows}


def strict_text_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").strip().casefold()
    return STRICT_SPACE_RE.sub(" ", normalized)


def replay_apply_policies(
    text: str,
    context: str,
    apply_policies: list[dict[str, Any]],
    protected_guards: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    replay = replay_apply_policies_with_guards(text, context, apply_policies, protected_guards)
    return str(replay["outputText"]), list(replay["applied"])


def replay_apply_policies_with_guards(
    text: str,
    context: str,
    apply_policies: list[dict[str, Any]],
    protected_guards: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    after, fires = replay_apply_policies_unchecked(text, context, apply_policies)
    guard_blocks = protected_term_guard_blocks(after, fires, protected_guards or [])
    if guard_blocks:
        return {
            "inputText": text,
            "outputText": text,
            "proposedOutputText": after,
            "applied": [],
            "blockedApplied": fires,
            "blocked": True,
            "guardBlocks": guard_blocks,
        }
    return {
        "inputText": text,
        "outputText": after,
        "proposedOutputText": after,
        "applied": fires,
        "blockedApplied": [],
        "blocked": False,
        "guardBlocks": [],
    }


def replay_apply_policies_unchecked(
    text: str,
    context: str,
    apply_policies: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    exact_policies = [policy for policy in apply_policies if policy.get("policyType") == "exactTrainablePair"]
    replacement_policies = [policy for policy in apply_policies if policy.get("policyType") != "exactTrainablePair"]
    exact_policy = first_exact_policy(exact_policies, text)
    if exact_policy:
        return str(exact_policy.get("targetText") or text), [
            {
                "policyId": exact_policy.get("policyId"),
                "policyType": exact_policy.get("policyType"),
                "sourcePattern": exact_policy.get("sourcePattern"),
                "targetText": exact_policy.get("targetText"),
            }
        ]

    after = text
    fires: list[dict[str, Any]] = []
    for policy in replacement_policies:
        if not policy_fires(policy, after, context):
            continue
        source = str(policy.get("sourcePattern") or "")
        target = str(policy.get("targetText") or "")
        updated = replace_policy_source(after, source, target)
        if updated == after:
            continue
        after = updated
        fires.append(
            {
                "policyId": policy.get("policyId"),
                "policyType": policy.get("policyType"),
                "sourcePattern": source,
                "targetText": target,
            }
        )
    return after, fires


def protected_term_allowlist_guards(model: dict[str, Any]) -> list[dict[str, Any]]:
    guards: list[dict[str, Any]] = []
    for key in PROTECTED_TERM_GUARD_KEYS:
        raw_guards = model.get(key)
        if not isinstance(raw_guards, list):
            continue
        for raw_guard in raw_guards:
            if not isinstance(raw_guard, dict):
                continue
            term = str(raw_guard.get("term") or "").strip()
            allowed_phrases = [
                str(value).strip()
                for value in raw_guard.get("allowedPhrases") or []
                if str(value).strip()
            ]
            guard_id = str(raw_guard.get("guardId") or f"protected-term-allowlist.{term}")
            if not term or not allowed_phrases:
                continue
            guards.append(
                {
                    "guardId": guard_id,
                    "reason": str(raw_guard.get("reason") or PROTECTED_TERM_GUARD_REASON),
                    "term": term,
                    "allowedPhrases": allowed_phrases,
                }
            )
    return guards


def protected_term_guard_blocks(
    text: str,
    applied: list[dict[str, Any]],
    protected_guards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for guard in protected_guards:
        term = str(guard["term"])
        allowed_phrases = [str(value) for value in guard["allowedPhrases"]]
        if (
            term in text
            and not all_protected_term_occurrences_are_allowed(text, term, allowed_phrases)
            and not applied_policy_supports_protected_term(applied, term)
        ):
            blocks.append(
                {
                    "guardId": guard["guardId"],
                    "reason": str(guard.get("reason") or PROTECTED_TERM_GUARD_REASON),
                    "term": term,
                    "blockedText": text,
                    "allowedPhrases": allowed_phrases,
                }
            )
    return blocks


def all_protected_term_occurrences_are_allowed(text: str, term: str, allowed_phrases: list[str]) -> bool:
    start = 0
    while True:
        index = text.find(term, start)
        if index < 0:
            return True
        end = index + len(term)
        if not any(allowed_phrase_contains_range(text, phrase, index, end) for phrase in allowed_phrases):
            return False
        start = end


def allowed_phrase_contains_range(text: str, phrase: str, start: int, end: int) -> bool:
    if not phrase:
        return False
    phrase_start = 0
    while True:
        index = text.find(phrase, phrase_start)
        if index < 0:
            return False
        phrase_end = index + len(phrase)
        if index <= start and phrase_end >= end:
            return True
        phrase_start = phrase_end


def applied_policy_supports_protected_term(applied: list[dict[str, Any]], term: str) -> bool:
    return any(
        term in str(fire.get("sourcePattern") or "") or term in str(fire.get("targetText") or "")
        for fire in applied
    )


def first_exact_policy(exact_policies: list[dict[str, Any]], text: str) -> dict[str, Any] | None:
    text_key = strict_text_key(text)
    for policy in exact_policies:
        if text_key == policy.get("inputStrictKey") and policy.get("exactInputRequired") is True:
            return policy
    return None


def policy_fires(policy: dict[str, Any], text: str, context: str) -> bool:
    source = str(policy.get("sourcePattern") or "")
    if not replacement_matches(text, source):
        return False
    trusted = context if policy.get("contextFromContextOnly") else "\n".join([text, context])
    alias_hits = token_hits(trusted, policy.get("contextAliasesAny") or [])
    context_hits = token_hits(trusted, policy.get("contextTokensAny") or [])
    if policy.get("requireAlias"):
        return bool(alias_hits)
    if policy.get("contextRequired"):
        return bool(alias_hits or context_hits)
    return True


def replacement_matches(text: str, source: str) -> bool:
    if not source:
        return False
    if contains_ascii_token(source):
        return range_for_ascii_bounded_source(source, text) is not None
    return source in text


def replace_policy_source(text: str, source: str, target: str) -> str:
    if contains_ascii_token(source):
        result = text
        while True:
            match = range_for_ascii_bounded_source(source, result)
            if not match:
                return result
            start, end = match
            result = result[:start] + target + result[end:]
    return text.replace(source, target)


def range_for_ascii_bounded_source(source: str, text: str) -> tuple[int, int] | None:
    start = 0
    while True:
        index = text.find(source, start)
        if index < 0:
            return None
        end = index + len(source)
        before_ok = index == 0 or not is_ascii_word_character(text[index - 1])
        after_ok = end == len(text) or not is_ascii_word_character(text[end])
        if before_ok and after_ok:
            return index, end
        start = end


def contains_ascii_token(text: str) -> bool:
    return ASCII_TOKEN_RE.search(text) is not None


def is_ascii_word_character(value: str) -> bool:
    return value == "_" or value.isascii() and value.isalnum()


def token_hits(text: str, tokens: Iterable[str]) -> list[str]:
    folded_text = unicodedata.normalize("NFKC", text or "").casefold()
    return [token for token in tokens if unicodedata.normalize("NFKC", str(token)).casefold() in folded_text]


def local_corpus_replay(records: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, Any]:
    apply_policies = [policy for policy in model.get("policies") or [] if policy.get("autoApplyMode") == "apply"]
    protected_guards = protected_term_allowlist_guards(model)
    row_results: list[dict[str, Any]] = []
    unexpected_changes: list[dict[str, Any]] = []
    changed_rows = 0
    matches_cleaned = 0
    fire_count = 0
    blocked_rows = 0
    for record in records:
        before = str(record.get("rawOpenCC") or record.get("rawASR") or "")
        if not before:
            continue
        context = local_context(record)
        replay = replay_apply_policies_with_guards(before, context, apply_policies, protected_guards)
        after = str(replay["outputText"])
        fires = list(replay["applied"])
        guard_blocks = list(replay["guardBlocks"])
        if guard_blocks:
            blocked_rows += 1
        if not fires and not guard_blocks:
            continue
        fire_count += len(fires)
        changed = strict_text_key(before) != strict_text_key(after)
        if changed:
            changed_rows += 1
        cleaned = str(record.get("cleanedText") or "")
        matches = bool(cleaned) and strict_text_key(after) == strict_text_key(cleaned)
        if matches:
            matches_cleaned += 1
        elif changed:
            unexpected_changes.append(
                {
                    "rowPk": record.get("rowPk"),
                    "before": before,
                    "after": after,
                    "cleanedText": cleaned,
                    "fires": fires,
                    "guardBlocks": guard_blocks,
                    "requiresReview": bool(record.get("requiresReview")),
                    "riskFlags": list(record.get("riskFlags") or []),
                }
            )
        row_results.append(
            {
                "rowPk": record.get("rowPk"),
                "before": before,
                "after": after,
                "matchesCleaned": matches,
                "fires": fires,
                "guardBlocks": guard_blocks,
                "blocked": bool(guard_blocks),
            }
        )
    readiness = {
        "autoApplyModelReady": not unexpected_changes,
        "reason": "local cleaned corpus replay had no unexpected changes" if not unexpected_changes else "local cleaned corpus replay found unexpected changes",
    }
    return {
        "rowCount": len(records),
        "applyPolicyCount": len(apply_policies),
        "candidateFireCount": fire_count,
        "guardBlockedRows": blocked_rows,
        "changedRows": changed_rows,
        "rowsMatchingCleanedText": matches_cleaned,
        "partialAcceptedChanges": [],
        "sentinelFailures": [],
        "unexpectedChanges": unexpected_changes,
        "rowResults": row_results,
        "readiness": readiness,
    }


def local_context(record: dict[str, Any]) -> str:
    pieces: list[str] = []
    context = record.get("context") if isinstance(record.get("context"), dict) else {}
    for side in ("before", "after"):
        for item in context.get(side) or []:
            if isinstance(item, dict) and isinstance(item.get("rawOpenCC"), str):
                pieces.append(item["rawOpenCC"])
    return "\n".join(pieces)


def load_replaylab_backend(replaylab_root: Path) -> dict[str, Any] | None:
    tools_dir = replaylab_root / "tools"
    if not tools_dir.exists():
        return None
    import sys

    tools_dir_str = str(tools_dir)
    if tools_dir_str not in sys.path:
        sys.path.insert(0, tools_dir_str)
    try:
        auto_apply_module = importlib.import_module("voco_train_auto_apply_model")
        raw_eval_module = importlib.import_module("voco_eval_auto_apply_raw_input")
    except ImportError:
        return None
    return {"auto_apply": auto_apply_module, "raw_eval": raw_eval_module}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def load_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_model(path: Path, model: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model["policyCounts"] = dict(Counter(str(policy.get("autoApplyMode") or "unknown") for policy in model.get("policies") or []))
    model["policyTypeCounts"] = dict(Counter(str(policy.get("policyType") or "unknown") for policy in model.get("policies") or []))
    path.write_text(json.dumps(model, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_positive_examples(values: Iterable[str]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for value in values:
        parts = value.split("||")
        if len(parts) != 3:
            raise SystemExit("--positive must use TEXT||CONTEXT||EXPECTED")
        examples.append({"text": parts[0], "context": parts[1], "expectedText": parts[2]})
    return examples


def parse_negative_examples(values: Iterable[str]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for value in values:
        parts = value.split("||")
        if len(parts) not in {1, 2}:
            raise SystemExit("--negative must use TEXT or TEXT||CONTEXT")
        examples.append({"text": parts[0], "context": parts[1] if len(parts) == 2 else "", "expectedText": parts[0]})
    return examples


def replace_text(text: str, source: str, target: str) -> str:
    return text.replace(source, target)


def compact_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        key = unicodedata.normalize("NFKC", item).casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_digest(value: str, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def timestamp_for_path() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def expanded_optional_path(path: Path | None) -> Path | None:
    return path.expanduser() if path else None


def backup_retention_from_args(args: argparse.Namespace, attr: str = "backup_retention") -> int:
    try:
        return max(0, int(getattr(args, attr, DEFAULT_BACKUP_RETENTION)))
    except (TypeError, ValueError):
        return DEFAULT_BACKUP_RETENTION


def create_model_backup(
    model_path: Path,
    *,
    backup_dir: Path | None,
    suffix: str,
    retention: int,
) -> Path | None:
    if not backup_dir or retention <= 0 or not model_path.exists():
        return None
    backup_path = backup_model_path(model_path, suffix, backup_dir=backup_dir)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, backup_path)
    prune_backups(model_path, backup_dir=backup_dir, retention=retention)
    return backup_path if backup_path.exists() else None


def backup_model_path(model_path: Path, suffix: str, *, backup_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return backup_dir / f"{model_path.name}.bak-{timestamp}-{suffix}"


def list_backups(active_model: Path, *, backup_dir: Path) -> list[Path]:
    if not backup_dir.exists():
        return []
    return sorted(backup_dir.glob(f"{active_model.name}.bak-*"), key=lambda path: path.stat().st_mtime, reverse=True)


def prune_backups(active_model: Path, *, backup_dir: Path, retention: int) -> None:
    if retention < 0:
        return
    for stale in list_backups(active_model, backup_dir=backup_dir)[retention:]:
        stale.unlink()


def newest_backup(active_model: Path, *, backup_dir: Path) -> Path | None:
    backups = list_backups(active_model, backup_dir=backup_dir)
    return backups[0] if backups else None


def validation_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ready": report["ready"],
        "reason": report["reason"],
        "positiveFailures": sum(1 for item in report.get("positiveExamples") or [] if not item.get("passed")),
        "negativeFailures": sum(1 for item in report.get("negativeExamples") or [] if not item.get("passed")),
        "exactApplyConflicts": len(report.get("exactApplyConflicts") or []),
        "manualContextLockFailures": len(report.get("manualContextLockFailures") or []),
        "corpusFailures": sum(len(item.get("failures") or []) for item in report.get("corpusReplay") or []),
        "failureCount": len(report.get("failures") or []),
    }


def print_human(result: dict[str, Any]) -> None:
    for key, value in result.items():
        if isinstance(value, (dict, list)):
            print(f"{key}: {json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
