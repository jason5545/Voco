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
DEFAULT_REPLAYLAB_ROOT = Path.home() / "VocoReplayLab"
DEFAULT_CURRENT_CORPUS_DIR = DEFAULT_REPLAYLAB_ROOT / "artifacts/full-db-raw-cleaned-20260611-093103-context10"
DEFAULT_RERAW_CORPUS_DIR = DEFAULT_REPLAYLAB_ROOT / "artifacts/full-db-reraw-cleaned-20260611-pre12022-context10"
CONTROL_SCHEMA_VERSION = 1
EVALUATION_CONTRACT_SCHEMA_VERSION = 1
DEFAULT_ACTION_COMMAND_GUARDS = [
    {"surface": "全部刪除"},
    {"surface": "全部删除"},
]
STRICT_SPACE_RE = re.compile(r"\s+")
ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+.#/-]*")
FAMILY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{1,96}$")
MANUAL_CORPUS_ACCEPTANCE_MAX = 25
DEFAULT_BACKUP_RETENTION = 3
PROTECTED_TERM_GUARD_REASON = "auto-apply-model-protected-term-guard"
PROTECTED_TERM_GUARD_KEYS = ("protectedTermAllowlistGuards", "protectedTermAllowlist")
BASELINE_DRIFT_RISK_FLAGS = {
    "storedOutputDisagreesWithRawDerivedCleaned",
    "rerawStoredBaselineDrift",
    "rerawDriftUncertainShortOrFiller",
}
POLICY_PROPOSAL_MODEL_FILE = "proposal-ranker-model.joblib"
POLICY_PROPOSAL_MANIFEST_FILE = "dataset-manifest.json"
POLICY_PROPOSAL_REPORT_FILE = "proposal-ranker-report.json"
POLICY_PROPOSAL_RELEASE_GATE_DIR = "proposal-release-gate-dry-run"
POLICY_PROPOSAL_REPLACEMENT_GATE_DIR = "proposal-replacement-gate-dry-run"
POLICY_PROPOSAL_SAFETY_GATE_DIR = "proposal-safety-gate-dry-run"
POLICY_PROPOSAL_SAFETY_GATE_REPORT_FILE = "proposal-safety-gate.report.json"
POLICY_PROPOSAL_REPLACEMENT_GATE_REPORT_FILE = "proposal-replacement-gate.report.json"


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

    replacement_rule = subparsers.add_parser("addReplacementRule")
    replacement_rule.add_argument("--source-pattern", required=True)
    replacement_rule.add_argument("--target-text", required=True)
    replacement_rule.add_argument("--source-text")
    replacement_rule.add_argument("--row-pk", type=int)
    replacement_rule.add_argument("--rule-name")
    replacement_rule.add_argument("--positive", action="append", default=[], help="TEXT||CONTEXT||EXPECTED")
    replacement_rule.add_argument("--negative", action="append", default=[], help="TEXT or TEXT||CONTEXT")
    replacement_rule.add_argument("--positive-text")
    replacement_rule.add_argument("--positive-context", default="")
    replacement_rule.add_argument("--expected-text")
    replacement_rule.add_argument("--negative-text")
    replacement_rule.add_argument("--negative-context", default="")
    replacement_rule.add_argument("--family-id")
    replacement_rule.add_argument("--family-role", default="alias")
    replacement_rule.add_argument("--family-reason")
    replacement_rule.add_argument("--note")

    replacement_family = subparsers.add_parser("addReplacementFamily")
    replacement_family.add_argument("--family-id", required=True)
    replacement_family.add_argument("--target-text", required=True)
    replacement_family.add_argument("--alias", action="append", required=True, default=[])
    replacement_family.add_argument("--rule-name-prefix")
    replacement_family.add_argument("--allow-strict-equivalent-alias", action="store_true")
    replacement_family.add_argument("--row-pk", type=int)
    replacement_family.add_argument("--positive", action="append", default=[], help="TEXT||CONTEXT||EXPECTED")
    replacement_family.add_argument("--negative", action="append", default=[], help="TEXT or TEXT||CONTEXT")
    replacement_family.add_argument("--note")

    family_tag = subparsers.add_parser("tagPolicyFamily")
    family_tag.add_argument("--policy-id", action="append", default=[])
    family_tag.add_argument("--source-pattern")
    family_tag.add_argument("--target-text")
    family_tag.add_argument("--family-id", required=True)
    family_tag.add_argument("--family-role", default="alias")
    family_tag.add_argument("--reason", required=True)
    family_tag.add_argument("--note")

    list_families = subparsers.add_parser("listPolicyFamilies")
    list_families.add_argument("--model", type=Path, default=DEFAULT_ACTIVE_MODEL)
    list_families.add_argument("--limit", type=int, default=50)

    inspect_family = subparsers.add_parser("inspectPolicyFamily")
    inspect_family.add_argument("--model", type=Path, default=DEFAULT_ACTIVE_MODEL)
    inspect_family.add_argument("--family-id", required=True)

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
    activate.add_argument("--activation-manifest", type=Path, help="Required for ReplayLab proposal candidates.")
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

    proposal = subparsers.add_parser("inspectProposalArtifact")
    proposal.add_argument("--artifact-dir", type=Path, required=True)

    replacement = subparsers.add_parser("evalProposalReplacementGate")
    replacement.add_argument("--artifact-dir", type=Path, required=True)
    replacement.add_argument("--output-dir", type=Path)
    replacement.add_argument("--active-model", type=Path)
    replacement.add_argument("--skip-raw-input-replay", action="store_true")

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
    if args.command == "addReplacementRule":
        event = replacement_rule_event(args)
        append_event(args.evidence_store.expanduser(), event)
        return {"event": event, "evidenceStore": str(args.evidence_store.expanduser())}
    if args.command == "addReplacementFamily":
        event = replacement_family_event(args)
        append_event(args.evidence_store.expanduser(), event)
        return {"event": event, "evidenceStore": str(args.evidence_store.expanduser())}
    if args.command == "tagPolicyFamily":
        event = tag_policy_family_event(args)
        append_event(args.evidence_store.expanduser(), event)
        return {"event": event, "evidenceStore": str(args.evidence_store.expanduser())}
    if args.command == "listPolicyFamilies":
        return list_policy_families(args.model.expanduser(), args.limit)
    if args.command == "inspectPolicyFamily":
        return inspect_policy_family(args.model.expanduser(), args.family_id)
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
    if args.command == "inspectProposalArtifact":
        return inspect_policy_proposal_artifact(args.artifact_dir.expanduser())
    if args.command == "evalProposalReplacementGate":
        return eval_policy_proposal_replacement_gate(args)
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


def replacement_rule_event(args: argparse.Namespace) -> dict[str, Any]:
    source_pattern = str(args.source_pattern)
    target_text = str(args.target_text)
    if not source_pattern.strip():
        raise SystemExit("addReplacementRule requires non-empty --source-pattern")
    if not target_text.strip():
        raise SystemExit("addReplacementRule requires non-empty --target-text")
    if strict_text_key(source_pattern) == strict_text_key(target_text):
        raise SystemExit("addReplacementRule source and target normalize to the same text")

    source_text = args.source_text or source_pattern
    positive_examples = parse_positive_examples(args.positive)
    negative_examples = parse_negative_examples(args.negative)
    if args.positive_text:
        positive_examples.append(
            {
                "text": args.positive_text,
                "context": args.positive_context or "",
                "expectedText": args.expected_text or replace_text(args.positive_text, source_pattern, target_text),
            }
        )
    if not positive_examples:
        positive_examples.append(
            {
                "text": source_text,
                "context": "",
                "expectedText": replace_text(source_text, source_pattern, target_text),
            }
        )
    if args.negative_text:
        negative_examples.append(
            {
                "text": args.negative_text,
                "context": args.negative_context or "",
                "expectedText": args.negative_text,
                "forbiddenText": target_text,
            }
        )

    family_id = str(getattr(args, "family_id", "") or "").strip()
    payload = {
        "ruleType": "unlockedReplacement",
        "rowPk": args.row_pk,
        "sourceText": source_text,
        "sourcePattern": source_pattern,
        "targetText": target_text,
        "ruleName": args.rule_name or f"manual-replacement:{short_digest(source_pattern + '->' + target_text)}",
        "examples": {
            "positive": positive_examples,
            "negative": negative_examples,
        },
        "provenance": {
            "manualLabel": "confirmed-unlocked-replacement-rule",
            "evidenceTier": "T4_GOLD",
            "note": args.note,
        },
    }
    if family_id:
        validate_family_id(family_id)
        payload["familyId"] = family_id
        payload["familyRole"] = str(getattr(args, "family_role", "") or "alias").strip() or "alias"
        payload["familyReason"] = str(getattr(args, "family_reason", "") or getattr(args, "note", "") or "").strip()
    return make_event(args.actor, "addReplacementRule", payload)


def replacement_family_event(args: argparse.Namespace) -> dict[str, Any]:
    family_id = str(args.family_id).strip()
    target_text = str(args.target_text)
    aliases = compact_alias_strings(args.alias)
    validate_family_id(family_id)
    if not target_text.strip():
        raise SystemExit("addReplacementFamily requires non-empty --target-text")
    if not aliases:
        raise SystemExit("addReplacementFamily requires at least one --alias")
    if target_text in aliases:
        raise SystemExit("addReplacementFamily alias must not exactly equal --target-text")

    strict_equivalent_aliases = [
        alias for alias in aliases if strict_text_key(alias) == strict_text_key(target_text)
    ]
    if strict_equivalent_aliases and not args.allow_strict_equivalent_alias:
        raise SystemExit(
            "addReplacementFamily strict-equivalent aliases require --allow-strict-equivalent-alias"
        )
    for alias in aliases:
        if len(strict_text_key(alias)) < 2 and not contains_ascii_token(alias):
            raise SystemExit("addReplacementFamily aliases must not be single non-ASCII characters")

    positive_examples = parse_positive_examples(args.positive)
    if not positive_examples:
        positive_examples = [
            {"text": alias, "context": "", "expectedText": target_text}
            for alias in aliases
        ]

    payload = {
        "ruleType": "replacementFamily",
        "familyId": family_id,
        "targetText": target_text,
        "aliases": aliases,
        "rowPk": args.row_pk,
        "ruleNamePrefix": args.rule_name_prefix or f"family:{family_id}",
        "allowStrictEquivalentAlias": bool(args.allow_strict_equivalent_alias),
        "examples": {
            "positive": positive_examples,
            "negative": parse_negative_examples(args.negative),
        },
        "provenance": {
            "manualLabel": "confirmed-replacement-family",
            "evidenceTier": "T4_GOLD",
            "note": args.note,
        },
    }
    return make_event(args.actor, "addReplacementFamily", payload)


def tag_policy_family_event(args: argparse.Namespace) -> dict[str, Any]:
    family_id = str(args.family_id).strip()
    validate_family_id(family_id)
    policy_ids = compact_strings(args.policy_id)
    source_pattern = str(args.source_pattern or "").strip()
    target_text = str(args.target_text or "").strip()
    if not policy_ids and not (source_pattern and target_text):
        raise SystemExit("tagPolicyFamily requires --policy-id or both --source-pattern and --target-text")
    payload = {
        "policyIds": policy_ids,
        "sourcePattern": source_pattern or None,
        "targetText": target_text or None,
        "familyId": family_id,
        "familyRole": str(args.family_role or "alias").strip() or "alias",
        "reason": str(args.reason or "").strip(),
        "note": args.note,
    }
    return make_event(args.actor, "tagPolicyFamily", payload)


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


def inspect_policy_proposal_artifact(artifact_dir: Path) -> dict[str, Any]:
    manifest_path = artifact_dir / POLICY_PROPOSAL_MANIFEST_FILE
    report_path = artifact_dir / POLICY_PROPOSAL_REPORT_FILE
    ranker_path = artifact_dir / POLICY_PROPOSAL_MODEL_FILE
    safety_gate_report_path = policy_proposal_safety_gate_report_path(artifact_dir)
    failures: list[dict[str, Any]] = []

    for required_path in [manifest_path, report_path, ranker_path]:
        if not required_path.exists():
            failures.append(
                {
                    "kind": "missingProposalArtifactFile",
                    "path": str(required_path),
                    "passed": False,
                }
            )

    manifest = load_json_object(manifest_path) if manifest_path.exists() else {}
    report = load_json_object(report_path) if report_path.exists() else {}
    manifest_boundary = string_list(manifest.get("safetyBoundary"))
    report_boundary = string_list(report.get("safetyBoundary"))
    combined_boundary = manifest_boundary + report_boundary

    manifest_intended_use = str(manifest.get("intendedUse") or "")
    report_intended_use = str(report.get("intendedUse") or "")
    if "not a Voco runtime model" not in manifest_intended_use:
        failures.append(
            {
                "kind": "proposalManifestRuntimeBoundaryMissing",
                "field": "intendedUse",
                "passed": False,
            }
        )
    if "not a Voco runtime" not in report_intended_use:
        failures.append(
            {
                "kind": "proposalReportRuntimeBoundaryMissing",
                "field": "intendedUse",
                "passed": False,
            }
        )
    if not any("full-db.auto-apply-model.json" in item and "runtime" in item.lower() for item in manifest_boundary):
        failures.append(
            {
                "kind": "proposalManifestCompiledRuntimeBoundaryMissing",
                "field": "safetyBoundary",
                "passed": False,
            }
        )
    if not any("proposal" in item.lower() and "apply" in item.lower() for item in combined_boundary):
        failures.append(
            {
                "kind": "proposalApplyBoundaryMissing",
                "field": "safetyBoundary",
                "passed": False,
            }
        )
    if not any("replay" in item.lower() and "compiled" in item.lower() for item in combined_boundary):
        failures.append(
            {
                "kind": "proposalReplayCompileGateBoundaryMissing",
                "field": "safetyBoundary",
                "passed": False,
            }
        )

    unsafe_apply_false_positives: dict[str, Any] = {}
    for split in ["valid", "test"]:
        split_report = report.get(split) if isinstance(report.get(split), dict) else {}
        count = split_report.get("unsafeApplyFalsePositiveCount")
        unsafe_apply_false_positives[split] = count
        if count != 0:
            failures.append(
                {
                    "kind": "unsafeApplyFalsePositive",
                    "split": split,
                    "count": count,
                    "passed": False,
                }
            )

    safety_gate = inspect_policy_proposal_safety_gate(safety_gate_report_path, failures)
    return {
        "artifactDir": str(artifact_dir),
        "role": "shadow/proposal contract fixture",
        "productionRuntimeAllowed": False,
        "runtimeModelFileName": DEFAULT_ACTIVE_MODEL.name,
        "rankerModel": str(ranker_path),
        "manifest": str(manifest_path),
        "report": str(report_path),
        "datasetType": manifest.get("datasetType"),
        "intendedUse": manifest_intended_use,
        "proposalCount": ((manifest.get("counts") or {}).get("proposals") if isinstance(manifest.get("counts"), dict) else None),
        "decisionCounts": ((manifest.get("counts") or {}).get("decisions") if isinstance(manifest.get("counts"), dict) else None),
        "applyThreshold": report.get("applyThreshold"),
        "unsafeApplyFalsePositiveCounts": unsafe_apply_false_positives,
        "safetyBoundary": combined_boundary,
        "proposalSafetyGate": safety_gate,
        "failed": bool(failures),
        "failures": failures,
    }


def inspect_policy_proposal_safety_gate(report_path: Path, failures: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not report_path.exists():
        return None

    report = load_json_object(report_path)
    ranker_gate = report.get("rankerGate") if isinstance(report.get("rankerGate"), dict) else {}
    candidate_replay = report.get("candidateReplay") if isinstance(report.get("candidateReplay"), dict) else {}
    raw_input_replay = report.get("rawInputReplay") if isinstance(report.get("rawInputReplay"), dict) else {}
    active_diff = report.get("activeModelDiff") if isinstance(report.get("activeModelDiff"), dict) else {}
    readiness = report.get("readiness") if isinstance(report.get("readiness"), dict) else {}
    runtime_boundary = report.get("runtimeBoundaryAudit") if isinstance(report.get("runtimeBoundaryAudit"), dict) else {}

    if readiness.get("productionRuntimeAllowed") is not False:
        failures.append(
            {
                "kind": "proposalSafetyGateRuntimeBoundaryMissing",
                "field": "readiness.productionRuntimeAllowed",
                "passed": False,
            }
        )
    if runtime_boundary.get("joblibActivationAllowed") is not False:
        failures.append(
            {
                "kind": "proposalSafetyGateJoblibActivationBoundaryMissing",
                "field": "runtimeBoundaryAudit.joblibActivationAllowed",
                "passed": False,
            }
        )
    if runtime_boundary.get("rankerModelIsRuntimeModel") is not False:
        failures.append(
            {
                "kind": "proposalSafetyGateRankerRuntimeBoundaryMissing",
                "field": "runtimeBoundaryAudit.rankerModelIsRuntimeModel",
                "passed": False,
            }
        )
    if runtime_boundary.get("installOrActivateCommandEmitted") is not False:
        failures.append(
            {
                "kind": "proposalSafetyGateInstallBoundaryMissing",
                "field": "runtimeBoundaryAudit.installOrActivateCommandEmitted",
                "passed": False,
            }
        )
    if runtime_boundary.get("candidateModelFilename") != DEFAULT_ACTIVE_MODEL.name:
        failures.append(
            {
                "kind": "proposalSafetyGateCandidateFilenameUnexpected",
                "field": "runtimeBoundaryAudit.candidateModelFilename",
                "passed": False,
            }
        )

    candidate_readiness = candidate_replay.get("readiness") if isinstance(candidate_replay.get("readiness"), dict) else {}
    raw_readiness = raw_input_replay.get("readiness") if isinstance(raw_input_replay.get("readiness"), dict) else {}
    return {
        "report": str(report_path),
        "schema": report.get("schema"),
        "proposalCount": ranker_gate.get("proposalCount"),
        "predictedApplyCount": ranker_gate.get("predictedApplyCount"),
        "acceptedForCompileCount": ranker_gate.get("acceptedForCompileCount"),
        "unsafeApplyFalsePositiveCount": ranker_gate.get("unsafeApplyFalsePositiveCount"),
        "applyMissCount": ranker_gate.get("applyMissCount"),
        "candidateReplayPass": candidate_readiness.get("autoApplyModelReady"),
        "candidateUnexpectedChanges": count_report_items(candidate_replay, "unexpectedChanges", "unexpectedChangeCount"),
        "candidateSentinelFailures": count_report_items(candidate_replay, "sentinelFailures", "sentinelFailureCount"),
        "candidateInheritedBaselineUnexpectedChanges": count_report_items(candidate_replay, "inheritedBaselineUnexpectedChanges"),
        "candidateAcceptedManualCorpusChanges": count_report_items(candidate_replay, "acceptedManualCorpusChanges"),
        "rawInputReplayPass": raw_readiness.get("rawInputReplayPass"),
        "rawUnexpectedChanges": count_report_items(raw_input_replay, "unexpectedChanges", "unexpectedChangeCount"),
        "rawSentinelFailures": count_report_items(raw_input_replay, "sentinelFailures", "sentinelFailureCount"),
        "rawInheritedBaselineUnexpectedChanges": count_report_items(raw_input_replay, "inheritedBaselineUnexpectedChanges"),
        "rawAcceptedManualCorpusChanges": count_report_items(raw_input_replay, "acceptedManualCorpusChanges"),
        "activePolicyCounts": active_diff.get("activePolicyCounts"),
        "candidatePolicyCounts": active_diff.get("candidatePolicyCounts"),
        "policyCountDelta": active_diff.get("policyCountDelta"),
        "addedPolicyCount": active_diff.get("addedPolicyCount"),
        "removedPolicyCount": active_diff.get("removedPolicyCount"),
        "changedPolicyCount": active_diff.get("changedPolicyCount"),
        "candidateIsSubsetOfActive": active_diff.get("candidateIsSubsetOfActive"),
        "candidateCoversActiveApplyPolicies": active_diff.get("candidateCoversActiveApplyPolicies"),
        "droppedActiveApplyPolicyCount": active_diff.get("droppedActiveApplyPolicyCount"),
        "droppedActiveApplyPolicyIds": active_diff.get("droppedActiveApplyPolicyIds") if isinstance(active_diff.get("droppedActiveApplyPolicyIds"), list) else [],
        "dryRunSafetyGatePass": readiness.get("dryRunSafetyGatePass"),
        "productionRuntimeAllowed": readiness.get("productionRuntimeAllowed"),
        "releaseReady": readiness.get("releaseReady"),
        "blockers": readiness.get("blockers") if isinstance(readiness.get("blockers"), list) else [],
        "warnings": readiness.get("warnings") if isinstance(readiness.get("warnings"), list) else [],
        "runtimeBoundaryAudit": {
            "candidateModelFilename": runtime_boundary.get("candidateModelFilename"),
            "candidateModelFilenameAllowed": runtime_boundary.get("candidateModelFilenameAllowed"),
            "installOrActivateCommandEmitted": runtime_boundary.get("installOrActivateCommandEmitted"),
            "joblibActivationAllowed": runtime_boundary.get("joblibActivationAllowed"),
            "rankerModelIsRuntimeModel": runtime_boundary.get("rankerModelIsRuntimeModel"),
            "productionRuntimeAllowed": runtime_boundary.get("productionRuntimeAllowed"),
        },
    }


def count_report_items(report: dict[str, Any], key: str, count_key: str | None = None) -> int:
    if count_key and isinstance(report.get(count_key), int):
        return int(report[count_key])
    value = report.get(key)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, list):
        return len(value)
    return 0


def policy_proposal_safety_gate_report_path(artifact_dir: Path) -> Path:
    release_report = artifact_dir / POLICY_PROPOSAL_RELEASE_GATE_DIR / POLICY_PROPOSAL_SAFETY_GATE_REPORT_FILE
    if release_report.exists():
        return release_report
    return artifact_dir / POLICY_PROPOSAL_SAFETY_GATE_DIR / POLICY_PROPOSAL_SAFETY_GATE_REPORT_FILE


def eval_policy_proposal_replacement_gate(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = args.artifact_dir.expanduser()
    release_dir = artifact_dir / POLICY_PROPOSAL_RELEASE_GATE_DIR
    safety_report_path = release_dir / POLICY_PROPOSAL_SAFETY_GATE_REPORT_FILE
    accepted_path = release_dir / "proposals.accepted.jsonl"
    if not safety_report_path.exists():
        raise FileNotFoundError(f"Missing proposal safety gate report: {safety_report_path}")
    if not accepted_path.exists():
        raise FileNotFoundError(f"Missing accepted proposal materialization: {accepted_path}")

    replaylab_root = args.replaylab_root.expanduser()
    safety_report = load_json_object(safety_report_path)
    report_input = safety_report.get("input") if isinstance(safety_report.get("input"), dict) else {}
    active_model_path = expanded_optional_path(getattr(args, "active_model", None)) or resolve_replaylab_path(
        report_input.get("activeCompiledModel"),
        replaylab_root,
    )
    corpus_dir = resolve_replaylab_path(report_input.get("corpusDir"), replaylab_root)
    if not active_model_path:
        raise ValueError("replacement gate requires input.activeCompiledModel or --active-model")
    if not corpus_dir:
        raise ValueError("replacement gate requires input.corpusDir")

    output_dir = expanded_optional_path(getattr(args, "output_dir", None)) or (
        REPO_ROOT / "artifacts" / artifact_dir.name / POLICY_PROPOSAL_REPLACEMENT_GATE_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    active_model = load_model(active_model_path)
    accepted_rows = load_jsonl(accepted_path)
    ranker_only_model = materialize_ranker_only_replacement_candidate(active_model, accepted_rows, artifact_dir)
    ranker_only_model_path = output_dir / DEFAULT_ACTIVE_MODEL.name
    write_model(ranker_only_model_path, ranker_only_model)

    backend = load_replaylab_backend(replaylab_root)
    if not backend:
        raise RuntimeError(f"ReplayLab replay backend unavailable under {replaylab_root}")

    cleaned_path = corpus_dir / "full-db.cleaned.jsonl"
    raw_path = corpus_dir / "full-db.raw.jsonl"
    trainable_path = corpus_dir / "full-db.trainable-pairs.jsonl"
    records = load_jsonl(cleaned_path)
    active_cleaned = backend["auto_apply"].replay_model(records, active_model)
    ranker_cleaned = backend["auto_apply"].replay_model(records, ranker_only_model)
    filter_accepted_manual_corpus_changes(active_cleaned, active_model)
    filter_accepted_manual_corpus_changes(ranker_cleaned, ranker_only_model)

    active_raw: dict[str, Any] | None = None
    ranker_raw: dict[str, Any] | None = None
    raw_report_path: Path | None = None
    if not getattr(args, "skip_raw_input_replay", False):
        active_raw = backend["raw_eval"].evaluate_raw_input(raw_path, cleaned_path, trainable_path, active_model_path)
        ranker_raw = backend["raw_eval"].evaluate_raw_input(raw_path, cleaned_path, trainable_path, ranker_only_model_path)
        filter_accepted_manual_corpus_changes(active_raw, active_model)
        filter_accepted_manual_corpus_changes(ranker_raw, ranker_only_model)
        raw_report_path = output_dir / "ranker-only.full-db.auto-apply-raw-input.report.json"
        raw_report_path.write_text(json.dumps(ranker_raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    active_diff = replacement_active_diff(active_model, ranker_only_model)
    cleaned_comparison = replay_comparison(active_cleaned, ranker_cleaned)
    raw_comparison = replay_comparison(active_raw, ranker_raw) if active_raw and ranker_raw else None
    readiness = replacement_readiness(active_diff, cleaned_comparison, raw_comparison)
    report = {
        "schema": "voco.policy-proposal-replacement-gate.v1",
        "generatedAt": now_iso(),
        "input": {
            "artifactDir": str(artifact_dir),
            "activeCompiledModel": str(active_model_path),
            "acceptedProposals": str(accepted_path),
            "corpusDir": str(corpus_dir),
            "preserveActiveSafetyGate": str(safety_report_path),
        },
        "outputs": {
            "report": str(output_dir / POLICY_PROPOSAL_REPLACEMENT_GATE_REPORT_FILE),
            "summary": str(output_dir / "proposal-replacement-gate.summary.md"),
            "rankerOnlyCandidateModel": str(ranker_only_model_path),
            "rankerOnlyRawInputReplayReport": str(raw_report_path) if raw_report_path else None,
        },
        "candidates": {
            "activeCompiledRulesBaseline": model_summary(active_model, active_model_path),
            "preserveActiveCandidate": {
                "sourceReport": str(safety_report_path),
                "releaseReady": ((safety_report.get("readiness") or {}).get("releaseReady") if isinstance(safety_report.get("readiness"), dict) else None),
                "productionRuntimeAllowed": ((safety_report.get("readiness") or {}).get("productionRuntimeAllowed") if isinstance(safety_report.get("readiness"), dict) else None),
                "activeModelDiff": safety_report.get("activeModelDiff"),
            },
            "rankerOnlyCandidate": model_summary(ranker_only_model, ranker_only_model_path),
        },
        "rankerOnlyVsActive": active_diff,
        "cleanedReplayComparison": cleaned_comparison,
        "rawInputReplayComparison": raw_comparison,
        "readiness": readiness,
        "runtimeBoundaryAudit": {
            "rankerModelIsRuntimeModel": False,
            "joblibActivationAllowed": False,
            "installOrActivateCommandEmitted": False,
            "productionRuntimeAllowed": False,
            "candidateModelFilename": DEFAULT_ACTIVE_MODEL.name,
            "candidateModelFilenameAllowed": True,
            "replacementCandidateIsDryRunOnly": True,
        },
        "failed": not bool(readiness["replacementReady"]),
    }

    report_path = output_dir / POLICY_PROPOSAL_REPLACEMENT_GATE_REPORT_FILE
    summary_path = output_dir / "proposal-replacement-gate.summary.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(replacement_gate_summary(report), encoding="utf-8")
    return report


def resolve_replaylab_path(value: Any, replaylab_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else replaylab_root / path


def model_summary(model: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "modelPath": str(path),
        "policyCounts": model.get("policyCounts"),
        "policyTypeCounts": model.get("policyTypeCounts"),
        "totalPolicyCount": len(model.get("policies") or []),
    }


def materialize_ranker_only_replacement_candidate(
    active_model: dict[str, Any],
    accepted_rows: list[dict[str, Any]],
    artifact_dir: Path,
) -> dict[str, Any]:
    policies: list[dict[str, Any]] = []
    seen_policy_ids: set[str] = set()
    for row in accepted_rows:
        policy = row.get("materializedPolicy") if isinstance(row.get("materializedPolicy"), dict) else None
        if not policy or policy.get("autoApplyMode") != "apply":
            continue
        policy_id = str(policy.get("policyId") or "")
        if not policy_id or policy_id in seen_policy_ids:
            continue
        seen_policy_ids.add(policy_id)
        policies.append(copy.deepcopy(policy))

    model = {
        "generatedAt": now_iso(),
        "intendedUse": "ranker-only replacement gate dry-run; not installed and not production runtime approval",
        "modelType": "voco-policy-proposal-ranker-only-materialized-candidate",
        "sourceActiveModelGeneratedAt": active_model.get("generatedAt"),
        "sourceArtifact": str(artifact_dir),
        "safetyContract": list(active_model.get("safetyContract") or []),
        "protectedTermAllowlistGuards": copy.deepcopy(active_model.get("protectedTermAllowlistGuards") or []),
        "proposalReplacementGate": {
            "candidateStrategy": "ranker-only-predicted-apply",
            "productionRuntimeAllowed": False,
            "joblibRuntimeAllowed": False,
        },
        "policies": policies,
    }
    model["policyCounts"] = dict(Counter(str(policy.get("autoApplyMode") or "unknown") for policy in policies))
    model["policyTypeCounts"] = dict(Counter(str(policy.get("policyType") or "unknown") for policy in policies))
    return model


def replacement_active_diff(active_model: dict[str, Any], candidate_model: dict[str, Any]) -> dict[str, Any]:
    active_apply = apply_policies_by_id(active_model)
    candidate_apply = apply_policies_by_id(candidate_model)
    dropped_ids = sorted(set(active_apply) - set(candidate_apply))
    added_ids = sorted(set(candidate_apply) - set(active_apply))
    changed_ids = sorted(
        policy_id
        for policy_id in set(active_apply).intersection(candidate_apply)
        if policy_runtime_fingerprint(active_apply[policy_id]) != policy_runtime_fingerprint(candidate_apply[policy_id])
    )
    return {
        "activePolicyCounts": active_model.get("policyCounts"),
        "candidatePolicyCounts": candidate_model.get("policyCounts"),
        "activePolicyTypeCounts": active_model.get("policyTypeCounts"),
        "candidatePolicyTypeCounts": candidate_model.get("policyTypeCounts"),
        "droppedActiveApplyPolicyCount": len(dropped_ids),
        "droppedActiveApplyPolicyIds": dropped_ids,
        "addedPolicyCount": len(added_ids),
        "addedPolicyIds": added_ids,
        "changedPolicyCount": len(changed_ids),
        "changedPolicyIds": changed_ids,
        "candidateCoversActiveApplyPolicies": not dropped_ids,
        "rankerOnlyCanReplaceActiveRules": not dropped_ids and not changed_ids,
    }


def apply_policies_by_id(model: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(policy.get("policyId")): policy
        for policy in model.get("policies") or []
        if policy.get("autoApplyMode") == "apply" and policy.get("policyId")
    }


def policy_runtime_fingerprint(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "autoApplyMode": policy.get("autoApplyMode"),
        "policyType": policy.get("policyType"),
        "sourcePattern": policy.get("sourcePattern"),
        "targetText": policy.get("targetText"),
        "inputStrictKey": policy.get("inputStrictKey"),
        "targetStrictKey": policy.get("targetStrictKey"),
        "exactInputRequired": policy.get("exactInputRequired"),
        "contextRequired": policy.get("contextRequired"),
        "contextTokensAny": policy.get("contextTokensAny") or [],
        "contextAliasesAny": policy.get("contextAliasesAny") or [],
        "requireAlias": policy.get("requireAlias"),
        "contextFromContextOnly": policy.get("contextFromContextOnly"),
    }


def replay_comparison(active_replay: dict[str, Any], candidate_replay: dict[str, Any]) -> dict[str, Any]:
    active_rows = replay_rows_by_pk(active_replay)
    candidate_rows = replay_rows_by_pk(candidate_replay)
    active_matching = {key for key, row in active_rows.items() if row.get("matchesCleaned")}
    candidate_matching = {key for key, row in candidate_rows.items() if row.get("matchesCleaned")}
    lost_matching = sorted(active_matching - candidate_matching)
    gained_matching = sorted(candidate_matching - active_matching)
    metrics = {
        "rowCount": metric_pair(active_replay, candidate_replay, "rowCount"),
        "applyPolicyCount": metric_pair(active_replay, candidate_replay, "applyPolicyCount"),
        "candidateFireCount": metric_pair(active_replay, candidate_replay, "candidateFireCount"),
        "rowFireCount": metric_pair(active_replay, candidate_replay, "rowFireCount"),
        "changedRows": metric_pair(active_replay, candidate_replay, "changedRows"),
        "rowsMatchingCleanedText": metric_pair(active_replay, candidate_replay, "rowsMatchingCleanedText"),
        "unexpectedChanges": metric_pair_len(active_replay, candidate_replay, "unexpectedChanges", "unexpectedChangeCount"),
        "sentinelFailures": metric_pair_len(active_replay, candidate_replay, "sentinelFailures", "sentinelFailureCount"),
    }
    regressions = []
    improvements = []
    if metrics["rowsMatchingCleanedText"]["delta"] < 0:
        regressions.append(
            {
                "kind": "rowsMatchingCleanedTextDropped",
                "delta": metrics["rowsMatchingCleanedText"]["delta"],
                "lostRowCount": len(lost_matching),
                "lostRowPks": lost_matching[:50],
            }
        )
    if metrics["candidateFireCount"]["delta"] < 0:
        regressions.append({"kind": "candidateFireCountDropped", "delta": metrics["candidateFireCount"]["delta"]})
    if metrics["unexpectedChanges"]["delta"] > 0:
        regressions.append({"kind": "unexpectedChangesIncreased", "delta": metrics["unexpectedChanges"]["delta"]})
    if metrics["sentinelFailures"]["delta"] > 0:
        regressions.append({"kind": "sentinelFailuresIncreased", "delta": metrics["sentinelFailures"]["delta"]})
    if metrics["rowsMatchingCleanedText"]["delta"] > 0:
        improvements.append(
            {
                "kind": "rowsMatchingCleanedTextImproved",
                "delta": metrics["rowsMatchingCleanedText"]["delta"],
                "gainedRowCount": len(gained_matching),
                "gainedRowPks": gained_matching[:50],
            }
        )
    if metrics["unexpectedChanges"]["delta"] < 0:
        improvements.append({"kind": "unexpectedChangesReduced", "delta": metrics["unexpectedChanges"]["delta"]})
    if metrics["sentinelFailures"]["delta"] < 0:
        improvements.append({"kind": "sentinelFailuresReduced", "delta": metrics["sentinelFailures"]["delta"]})
    return {
        "active": compact_replay_report(active_replay),
        "rankerOnly": compact_replay_report(candidate_replay),
        "metrics": metrics,
        "regressions": regressions,
        "improvements": improvements,
        "lostMatchingCleanedRowCount": len(lost_matching),
        "lostMatchingCleanedRowPks": lost_matching[:100],
        "gainedMatchingCleanedRowCount": len(gained_matching),
        "gainedMatchingCleanedRowPks": gained_matching[:100],
    }


def replay_rows_by_pk(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in report.get("rowResults") or []:
        row_pk = int_or_none(row.get("rowPk")) if isinstance(row, dict) else None
        if row_pk is not None:
            rows[row_pk] = row
    return rows


def metric_pair(active: dict[str, Any], candidate: dict[str, Any], key: str) -> dict[str, Any]:
    active_value = int(active.get(key) or 0)
    candidate_value = int(candidate.get(key) or 0)
    return {"active": active_value, "rankerOnly": candidate_value, "delta": candidate_value - active_value}


def metric_pair_len(active: dict[str, Any], candidate: dict[str, Any], key: str, count_key: str) -> dict[str, Any]:
    active_value = count_report_items(active, key, count_key)
    candidate_value = count_report_items(candidate, key, count_key)
    return {"active": active_value, "rankerOnly": candidate_value, "delta": candidate_value - active_value}


def replacement_readiness(
    active_diff: dict[str, Any],
    cleaned_comparison: dict[str, Any],
    raw_comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    if active_diff["droppedActiveApplyPolicyCount"]:
        blockers.append(
            {
                "kind": "droppedActiveApplyPolicies",
                "count": active_diff["droppedActiveApplyPolicyCount"],
                "samplePolicyIds": active_diff["droppedActiveApplyPolicyIds"][:25],
            }
        )
    if not active_diff["candidateCoversActiveApplyPolicies"]:
        blockers.append({"kind": "doesNotCoverActiveApplyBehavior"})
    blockers.extend(cleaned_comparison.get("regressions") or [])
    if raw_comparison:
        blockers.extend({"surface": "rawInputReplay", **item} for item in raw_comparison.get("regressions") or [])
    return {
        "replacementReady": not blockers,
        "productionRuntimeAllowed": False,
        "reason": (
            "ranker-only candidate matched or improved active compiled-rule behavior in dry-run replay"
            if not blockers
            else "ranker-only candidate drops or regresses active compiled-rule behavior; keep ranker as proposal/shadow gate"
        ),
        "blockers": blockers,
    }


def replacement_gate_summary(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    diff = report["rankerOnlyVsActive"]
    cleaned = report["cleanedReplayComparison"]
    raw = report.get("rawInputReplayComparison")
    lines = [
        "# Proposal Replacement Gate Dry Run",
        "",
        f"- schema: `{report['schema']}`",
        f"- replacementReady: `{str(readiness['replacementReady']).lower()}`",
        f"- productionRuntimeAllowed: `{str(readiness['productionRuntimeAllowed']).lower()}`",
        f"- reason: {readiness['reason']}",
        f"- droppedActiveApplyPolicyCount: {diff['droppedActiveApplyPolicyCount']}",
        f"- candidateCoversActiveApplyPolicies: `{str(diff['candidateCoversActiveApplyPolicies']).lower()}`",
        "",
        "## Cleaned Replay",
        replay_metric_line(cleaned, "candidateFireCount"),
        replay_metric_line(cleaned, "changedRows"),
        replay_metric_line(cleaned, "rowsMatchingCleanedText"),
        replay_metric_line(cleaned, "unexpectedChanges"),
        replay_metric_line(cleaned, "sentinelFailures"),
    ]
    if raw:
        lines.extend(
            [
                "",
                "## Raw Input Replay",
                replay_metric_line(raw, "candidateFireCount"),
                replay_metric_line(raw, "changedRows"),
                replay_metric_line(raw, "rowsMatchingCleanedText"),
                replay_metric_line(raw, "unexpectedChanges"),
                replay_metric_line(raw, "sentinelFailures"),
            ]
        )
    return "\n".join(lines) + "\n"


def replay_metric_line(comparison: dict[str, Any], key: str) -> str:
    metric = comparison["metrics"][key]
    return f"- {key}: active {metric['active']} / ranker-only {metric['rankerOnly']} / delta {metric['delta']}"


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
    strip_proposal_candidate_metadata(model)
    policies = [copy.deepcopy(policy) for policy in model.get("policies") or []]
    overlay_policy_count = 0
    tombstone_count = 0
    family_tag_count = 0
    family_tag_misses: list[dict[str, Any]] = []

    for event in events:
        action = str(event.get("action") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if action == "addCorrection":
            policy = exact_policy_from_event(event)
            overlay_policy_count += upsert_policy(policies, policy, event)
        elif action == "addContextLockedRule":
            policy = context_policy_from_event(event)
            overlay_policy_count += upsert_policy(policies, policy, event)
        elif action == "addReplacementRule":
            policy = replacement_policy_from_event(event)
            overlay_policy_count += upsert_policy(policies, policy, event)
        elif action == "addReplacementFamily":
            for policy in replacement_family_policies_from_event(event):
                overlay_policy_count += upsert_policy(policies, policy, event)
        elif action == "tagPolicyFamily":
            tag_result = tag_policy_family(policies, payload, event)
            family_tag_count += int(tag_result["matchedPolicyCount"])
            if not tag_result["matchedPolicyCount"]:
                family_tag_misses.append(tag_result)
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
    family_summary = policy_family_summary(policies)
    if family_summary:
        model["controlPlaneFamilies"] = family_summary
    append_safety_contract(model)
    model["schemaVersion"] = EVALUATION_CONTRACT_SCHEMA_VERSION
    model["actionCommandGuards"] = copy.deepcopy(DEFAULT_ACTION_COMMAND_GUARDS)
    model["autoApplyModelVersion"] = f"control-compiled-{now_iso()}"
    model["generatedAt"] = model.get("generatedAt") or now_iso()
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
        "familyTagCount": family_tag_count,
        "familyTagMissCount": len(family_tag_misses),
        "familyTagMisses": family_tag_misses,
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
        "familyTagCount": family_tag_count,
        "familyTagMissCount": len(family_tag_misses),
        "familyTagMisses": family_tag_misses,
    }
    return model, report


def strip_proposal_candidate_metadata(model: dict[str, Any]) -> None:
    for key in (
        "proposalSafetyGate",
        "proposalReplacementGate",
        "promotionPolicyGuard",
        "replayReadiness",
        "sourceActiveModelGeneratedAt",
    ):
        model.pop(key, None)
    intended_use = str(model.get("intendedUse") or "")
    if "dry-run" in intended_use.lower() or "do not install" in intended_use.lower():
        model["intendedUse"] = "local Voco control-plane patched auto-apply runtime model"


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


def replacement_policy_from_event(event: dict[str, Any]) -> dict[str, Any]:
    payload = event["payload"]
    source_pattern = str(payload["sourcePattern"])
    target_text = str(payload["targetText"])
    source_text = str(payload.get("sourceText") or source_pattern)
    row_pk = payload.get("rowPk")
    evidence_rows = [int(row_pk)] if row_pk else []
    policy_id_key = json.dumps(
        {
            "sourcePattern": source_pattern,
            "targetText": target_text,
            "ruleName": payload.get("ruleName"),
            "ruleType": "unlockedReplacement",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    policy = {
        "policyId": f"manual-replacement-{short_digest(policy_id_key, length=16)}",
        "policyType": "scopedReplacement",
        "autoApplyMode": "apply",
        "decisionReason": "manual unlocked scoped replacement from control-plane evidence",
        "source": source_text,
        "target": target_text,
        "sourcePattern": source_pattern,
        "targetText": target_text,
        "lockName": payload.get("ruleName") or "manual-unlocked-replacement",
        "contextRequired": False,
        "contextTokensAny": [],
        "contextAliasesAny": [],
        "contextFromContextOnly": False,
        "requireAlias": False,
        "scopedSourcePhrase": source_pattern,
        "scopeWindow": "manual unlocked replacement; ASCII sources use runtime token-boundary matching",
        "evidenceRows": evidence_rows,
        "trainableRows": evidence_rows,
        "reviewRows": [],
        "evidenceCount": len(evidence_rows) or 1,
        "trainableEvidenceCount": len(evidence_rows) or 1,
        "reviewEvidenceCount": 0,
        "riskFlagCounts": {"manualUnlockedReplacement": 1},
        "labelTierCounts": {"T4_GOLD": 1},
        "cleanedSourceCounts": {"manualControlPlane": 1},
        "pairContextRequiredRows": [],
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
    apply_policy_family_metadata(policy, payload, event)
    return policy


def replacement_family_policies_from_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    payload = event["payload"]
    aliases = compact_alias_strings(payload.get("aliases") or [])
    target_text = str(payload["targetText"])
    row_pk = payload.get("rowPk")
    evidence_rows = [int(row_pk)] if row_pk else []
    family_id = str(payload["familyId"])
    policies: list[dict[str, Any]] = []
    for alias in aliases:
        policy_id_key = json.dumps(
            {
                "familyId": family_id,
                "sourcePattern": alias,
                "targetText": target_text,
                "ruleNamePrefix": payload.get("ruleNamePrefix"),
                "ruleType": "replacementFamilyAlias",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        policies.append(
            {
                "policyId": f"manual-replacement-family-{short_digest(policy_id_key, length=16)}",
                "policyType": "scopedReplacement",
                "autoApplyMode": "apply",
                "decisionReason": "manual replacement family alias from control-plane evidence",
                "source": alias,
                "target": target_text,
                "sourcePattern": alias,
                "targetText": target_text,
                "lockName": f"{payload.get('ruleNamePrefix') or f'family:{family_id}'}:{short_digest(alias, length=8)}",
                "contextRequired": False,
                "contextTokensAny": [],
                "contextAliasesAny": [],
                "contextFromContextOnly": False,
                "requireAlias": False,
                "scopedSourcePhrase": alias,
                "scopeWindow": "manual replacement family alias; ASCII sources use runtime token-boundary matching",
                "evidenceRows": evidence_rows,
                "trainableRows": evidence_rows,
                "reviewRows": [],
                "evidenceCount": len(evidence_rows) or 1,
                "trainableEvidenceCount": len(evidence_rows) or 1,
                "reviewEvidenceCount": 0,
                "riskFlagCounts": {"manualReplacementFamily": 1},
                "labelTierCounts": {"T4_GOLD": 1},
                "cleanedSourceCounts": {"manualControlPlane": 1},
                "pairContextRequiredRows": [],
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
                "familyId": family_id,
                "familyRole": "alias",
                "familyReason": str((payload.get("provenance") or {}).get("note") or ""),
                "familyTagEventIds": [event["eventId"]],
                "familyTaggedAt": event.get("createdAt"),
                "familyAliasCount": len(aliases),
            }
        )
    return policies


def apply_policy_family_metadata(policy: dict[str, Any], payload: dict[str, Any], event: dict[str, Any]) -> None:
    family_id = str(payload.get("familyId") or "").strip()
    if not family_id:
        return
    policy["familyId"] = family_id
    policy["familyRole"] = str(payload.get("familyRole") or "alias").strip() or "alias"
    reason = str(payload.get("familyReason") or payload.get("reason") or "").strip()
    if reason:
        policy["familyReason"] = reason
    event_ids = list(policy.get("familyTagEventIds") or [])
    if event.get("eventId") not in event_ids:
        event_ids.append(event["eventId"])
    policy["familyTagEventIds"] = sorted(set(event_ids))
    policy["familyTaggedAt"] = event.get("createdAt")


def upsert_policy(policies: list[dict[str, Any]], new_policy: dict[str, Any], event: dict[str, Any]) -> int:
    for index, policy in enumerate(policies):
        if policy_identity(policy) == policy_identity(new_policy):
            if policy.get("autoApplyMode") != new_policy.get("autoApplyMode") or isinstance(policy.get("tombstone"), dict):
                superseded_ids = list(policy.get("controlEvidenceEventIds") or [])
                superseded_tombstone = policy.get("tombstone") if isinstance(policy.get("tombstone"), dict) else None
                replacement = copy.deepcopy(new_policy)
                if superseded_ids:
                    replacement["supersededControlEvidenceEventIds"] = superseded_ids
                if superseded_tombstone:
                    replacement["supersededTombstone"] = superseded_tombstone
                policies[index] = replacement
                return 0
            ids = list(policy.get("controlEvidenceEventIds") or [])
            if event["eventId"] not in ids:
                ids.append(event["eventId"])
            policy["controlEvidenceEventIds"] = ids
            merge_policy_family_metadata(policy, new_policy)
            policy["decisionReason"] = str(policy.get("decisionReason") or "") + "; reinforced by manual control-plane evidence"
            return 0
    policies.append(new_policy)
    return 1


def merge_policy_family_metadata(policy: dict[str, Any], source_policy: dict[str, Any]) -> None:
    family_id = str(source_policy.get("familyId") or "").strip()
    if not family_id:
        return
    policy["familyId"] = family_id
    policy["familyRole"] = str(source_policy.get("familyRole") or "alias").strip() or "alias"
    for key in ("familyReason", "familyAliasCount"):
        if source_policy.get(key) is not None:
            policy[key] = source_policy[key]
    if source_policy.get("familyTaggedAt"):
        policy["familyTaggedAt"] = source_policy["familyTaggedAt"]
    event_ids = list(policy.get("familyTagEventIds") or [])
    event_ids.extend(source_policy.get("familyTagEventIds") or [])
    policy["familyTagEventIds"] = sorted(set(str(event_id) for event_id in event_ids if str(event_id).strip()))


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


def tag_policy_family(
    policies: list[dict[str, Any]],
    payload: dict[str, Any],
    event: dict[str, Any],
) -> dict[str, Any]:
    matched_policy_ids: list[str] = []
    for policy in policies:
        if not family_selector_matches_policy(payload, policy):
            continue
        apply_policy_family_metadata(policy, payload, event)
        matched_policy_ids.append(str(policy.get("policyId") or ""))
    return {
        "eventId": event.get("eventId"),
        "familyId": payload.get("familyId"),
        "policyIds": compact_strings(payload.get("policyIds") or []),
        "sourcePattern": payload.get("sourcePattern"),
        "targetText": payload.get("targetText"),
        "matchedPolicyCount": len(matched_policy_ids),
        "matchedPolicyIds": sorted(matched_policy_ids),
    }


def family_selector_matches_policy(payload: dict[str, Any], policy: dict[str, Any]) -> bool:
    policy_ids = set(compact_strings(payload.get("policyIds") or []))
    policy_id = str(policy.get("policyId") or "")
    if policy_ids and policy_id in policy_ids:
        return True
    source_pattern = str(payload.get("sourcePattern") or "")
    target_text = str(payload.get("targetText") or "")
    if source_pattern and target_text:
        return (
            strict_text_key(str(policy.get("sourcePattern") or policy.get("source") or "")) == strict_text_key(source_pattern)
            and strict_text_key(str(policy.get("targetText") or policy.get("target") or "")) == strict_text_key(target_text)
        )
    return False


def policy_family_summary(policies: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for policy in policies:
        family_id = str(policy.get("familyId") or "").strip()
        if family_id:
            grouped[family_id].append(policy)
    return {
        family_id: summarize_policy_family(family_id, family_policies)
        for family_id, family_policies in sorted(grouped.items())
    }


def summarize_policy_family(family_id: str, policies: list[dict[str, Any]]) -> dict[str, Any]:
    source_patterns = sorted(
        set(str(policy.get("sourcePattern") or policy.get("source") or "") for policy in policies if str(policy.get("sourcePattern") or policy.get("source") or "").strip())
    )
    target_texts = sorted(
        set(str(policy.get("targetText") or policy.get("target") or "") for policy in policies if str(policy.get("targetText") or policy.get("target") or "").strip())
    )
    policy_ids = sorted(str(policy.get("policyId") or "") for policy in policies if str(policy.get("policyId") or "").strip())
    return {
        "familyId": family_id,
        "policyCount": len(policies),
        "autoApplyModeCounts": dict(Counter(str(policy.get("autoApplyMode") or "unknown") for policy in policies)),
        "policyTypeCounts": dict(Counter(str(policy.get("policyType") or "unknown") for policy in policies)),
        "familyRoleCounts": dict(Counter(str(policy.get("familyRole") or "unknown") for policy in policies)),
        "targetTextCount": len(target_texts),
        "targetTextSamples": target_texts[:20],
        "sourcePatternCount": len(source_patterns),
        "sourcePatternSamples": source_patterns[:20],
        "policyIdSamples": policy_ids[:20],
    }


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
        "manual unlocked replacements are reserved for closed-form strings with explicit examples and no context requirement",
        "manual tombstones preserve provenance by marking policies blocked or replaced instead of deleting evidence",
        "policy family tags are metadata-only and must not change runtime matching behavior",
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
    active_control_event_ids = control_event_ids_for_policies(apply_policies)
    protected_guards = protected_term_allowlist_guards(model)
    failures: list[dict[str, Any]] = []
    positive_results = validate_positive_examples(events, apply_policies, active_control_event_ids, protected_guards)
    negative_results = validate_negative_examples(events, apply_policies, active_control_event_ids, protected_guards)
    failures.extend(item for item in positive_results if not item["passed"])
    failures.extend(item for item in negative_results if not item["passed"])
    exact_conflicts = exact_apply_conflicts(apply_policies)
    failures.extend(exact_conflicts)
    manual_context_failures = manual_context_lock_failures(apply_policies)
    failures.extend(manual_context_failures)
    manual_replacement_failures = manual_replacement_rule_failures(apply_policies)
    failures.extend(manual_replacement_failures)
    family_metadata_failures = family_metadata_failures_for_model(model)
    failures.extend(family_metadata_failures)
    count_report = policy_count_report(model, base_model)
    failures.extend(count_report["failures"])
    corpus_reports = []
    if not skip_corpus_replay:
        for name, corpus_dir in [("currentRaw", current_corpus_dir), ("rerawPre12022", reraw_corpus_dir)]:
            report = corpus_replay_report(
                name,
                corpus_dir,
                model,
                model_path,
                replaylab_root,
                skip_raw_input_replay,
                base_model=base_model,
            )
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
        "manualReplacementFailures": manual_replacement_failures,
        "familyMetadataFailures": family_metadata_failures,
        "policyCounts": model.get("policyCounts") or {},
        "policyTypeCounts": model.get("policyTypeCounts") or {},
        "policyCountReport": count_report,
        "corpusReplay": corpus_reports,
        "failures": failures,
    }


def control_event_ids_for_policies(policies: list[dict[str, Any]]) -> set[str]:
    event_ids: set[str] = set()
    for policy in policies:
        for event_id in policy.get("controlEvidenceEventIds") or []:
            if isinstance(event_id, str) and event_id:
                event_ids.add(event_id)
    return event_ids


def should_validate_examples_for_event(event: dict[str, Any], active_control_event_ids: set[str]) -> bool:
    action = str(event.get("action") or "")
    if action in {"addCorrection", "addContextLockedRule", "addReplacementRule", "addReplacementFamily"}:
        event_id = str(event.get("eventId") or "")
        return bool(event_id and event_id in active_control_event_ids)
    return True


def validate_positive_examples(
    events: list[dict[str, Any]],
    apply_policies: list[dict[str, Any]],
    active_control_event_ids: set[str],
    protected_guards: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for event in events:
        if not should_validate_examples_for_event(event, active_control_event_ids):
            continue
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
    active_control_event_ids: set[str],
    protected_guards: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for event in events:
        if not should_validate_examples_for_event(event, active_control_event_ids):
            continue
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


def manual_replacement_rule_failures(apply_policies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for policy in apply_policies:
        policy_id = str(policy.get("policyId") or "")
        if not policy_id.startswith("manual-replacement-"):
            continue
        source = str(policy.get("sourcePattern") or "")
        target = str(policy.get("targetText") or "")
        if policy.get("policyType") != "scopedReplacement":
            failures.append({"kind": "manualReplacementUnexpectedPolicyType", "policyId": policy_id, "passed": False})
        if policy.get("contextRequired") is not False:
            failures.append({"kind": "manualReplacementMustNotRequireContext", "policyId": policy_id, "passed": False})
        if policy.get("contextTokensAny") or policy.get("contextAliasesAny") or policy.get("requireAlias"):
            failures.append({"kind": "manualReplacementHasContextLockFields", "policyId": policy_id, "passed": False})
        if not source.strip() or not target.strip():
            failures.append({"kind": "manualReplacementMissingSourceOrTarget", "policyId": policy_id, "passed": False})
        if manual_replacement_noop_key(source) == manual_replacement_noop_key(target):
            failures.append({"kind": "manualReplacementNoOp", "policyId": policy_id, "passed": False})
        if len(strict_text_key(source)) < 2 and not contains_ascii_token(source):
            failures.append({"kind": "manualReplacementSourceTooShort", "policyId": policy_id, "sourcePattern": source, "passed": False})
    return failures


def family_metadata_failures_for_model(model: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    control_plane = model.get("controlPlane") if isinstance(model.get("controlPlane"), dict) else {}
    for miss in control_plane.get("familyTagMisses") or []:
        failures.append(
            {
                "kind": "familyTagMatchedNoPolicies",
                "eventId": miss.get("eventId"),
                "familyId": miss.get("familyId"),
                "policyIds": miss.get("policyIds") or [],
                "sourcePattern": miss.get("sourcePattern"),
                "targetText": miss.get("targetText"),
                "passed": False,
            }
        )
    for policy in model.get("policies") or []:
        family_id = str(policy.get("familyId") or "").strip()
        if family_id and not FAMILY_ID_RE.match(family_id):
            failures.append(
                {
                    "kind": "invalidPolicyFamilyId",
                    "policyId": policy.get("policyId"),
                    "familyId": family_id,
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
    *,
    base_model: dict[str, Any] | None,
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
    baseline_cleaned_report: dict[str, Any] | None = None
    if base_model:
        baseline_cleaned_report = (
            backend["auto_apply"].replay_model(records, base_model)
            if backend and not protected_term_allowlist_guards(base_model)
            else local_corpus_replay(records, base_model)
        )
        filter_accepted_manual_corpus_changes(baseline_cleaned_report, base_model)
        suppress_inherited_baseline_corpus_changes(cleaned_report, baseline_cleaned_report)
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
        if base_model:
            suppress_inherited_baseline_policy_fires(raw_report, base_model)
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
        "baselineCleanedReplay": compact_replay_report(baseline_cleaned_report) if baseline_cleaned_report else None,
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


def suppress_inherited_baseline_corpus_changes(report: dict[str, Any], baseline_report: dict[str, Any]) -> None:
    unexpected = list(report.get("unexpectedChanges") or [])
    baseline_unexpected = list(baseline_report.get("unexpectedChanges") or [])
    if not unexpected or not baseline_unexpected:
        return

    baseline_keys = {corpus_change_key(item) for item in baseline_unexpected}
    inherited: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for item in unexpected:
        if corpus_change_key(item) in baseline_keys:
            inherited.append(item)
        else:
            remaining.append(item)

    report["baselineUnexpectedChanges"] = len(baseline_unexpected)
    report["inheritedBaselineUnexpectedChanges"] = inherited
    report["unexpectedChanges"] = remaining
    if report.get("sentinelFailures") or remaining:
        return

    readiness = report.get("readiness") if isinstance(report.get("readiness"), dict) else {}
    if "rawInputReplayPass" in readiness:
        readiness["rawInputReplayPass"] = True
        readiness["reason"] = "raw input replay passed after inherited baseline changes were ignored"
    elif "autoApplyModelReady" in readiness:
        readiness["autoApplyModelReady"] = True
        readiness["reason"] = "cleaned corpus replay passed; only inherited active-model changes were ignored"


def suppress_inherited_baseline_policy_fires(report: dict[str, Any], base_model: dict[str, Any]) -> None:
    unexpected = list(report.get("unexpectedChanges") or [])
    if not unexpected:
        return

    base_apply_policy_ids = {
        str(policy.get("policyId"))
        for policy in base_model.get("policies") or []
        if policy.get("autoApplyMode") == "apply" and policy.get("policyId")
    }
    inherited: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for item in unexpected:
        fire_ids = [
            str((fire if isinstance(fire, dict) else {}).get("policyId") or "")
            for fire in item.get("fires") or []
        ]
        if fire_ids and all(policy_id in base_apply_policy_ids for policy_id in fire_ids):
            inherited.append(item)
        else:
            remaining.append(item)

    report["baselineUnexpectedChanges"] = len(unexpected)
    report["inheritedBaselineUnexpectedChanges"] = inherited
    report["unexpectedChanges"] = remaining
    if report.get("sentinelFailures") or remaining:
        return

    readiness = report.get("readiness") if isinstance(report.get("readiness"), dict) else {}
    if "rawInputReplayPass" in readiness:
        readiness["rawInputReplayPass"] = True
        readiness["reason"] = "raw input replay passed; only inherited active-model policy fires were ignored"
    elif "autoApplyModelReady" in readiness:
        readiness["autoApplyModelReady"] = True
        readiness["reason"] = "cleaned corpus replay passed; only inherited active-model policy fires were ignored"


def corpus_change_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int_or_none(item.get("rowPk")),
        strict_text_key(str(item.get("before") or "")),
        strict_text_key(str(item.get("after") or "")),
        strict_text_key(str(item.get("cleanedText") or "")),
    )


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
            if manual_replacement_policy_accepts_change(policy_id, policy, item):
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


def manual_replacement_policy_accepts_change(
    policy_id: str,
    policy: dict[str, Any],
    item: dict[str, Any],
) -> bool:
    if not policy_id.startswith("manual-replacement-"):
        return False
    if policy.get("policyType") != "scopedReplacement" or policy.get("contextRequired") is not False:
        return False
    source = str(policy.get("sourcePattern") or "")
    target = str(policy.get("targetText") or "")
    if not source or not target:
        return False
    if not contains_ascii_token(source) or not contains_ascii_token(target):
        return False

    before = str(item.get("before") or "")
    after = str(item.get("after") or "")
    cleaned = str(item.get("cleanedText") or "")
    if not before or not after or not cleaned:
        return False
    if strict_text_key(after) == strict_text_key(cleaned):
        return False
    return strict_text_key(replace_policy_source(before, source, target)) == strict_text_key(after)


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
        "baselineUnexpectedChanges": report.get("baselineUnexpectedChanges"),
        "inheritedBaselineUnexpectedChanges": len(report.get("inheritedBaselineUnexpectedChanges") or []),
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


def proposal_activation_guard(
    model: dict[str, Any],
    model_path: Path,
    active_model: Path,
    activation_manifest: Path | None,
    replaylab_root: Path,
) -> dict[str, Any]:
    proposal_replacement_gate = model.get("proposalReplacementGate") if isinstance(model.get("proposalReplacementGate"), dict) else {}
    proposal_safety_gate = model.get("proposalSafetyGate") if isinstance(model.get("proposalSafetyGate"), dict) else {}
    model_type = str(model.get("modelType") or "")
    candidate_strategy = str(
        proposal_replacement_gate.get("candidateStrategy")
        or proposal_safety_gate.get("candidateStrategy")
        or ""
    )
    is_proposal_candidate = bool(proposal_replacement_gate or proposal_safety_gate or "proposal" in model_type)

    if candidate_strategy == "ranker-only-predicted-apply":
        return {
            "failed": True,
            "reason": "ranker-only proposal candidates cannot be activated as Voco runtime models",
            "candidateStrategy": candidate_strategy,
            "productionRuntimeAllowed": False,
        }
    if not is_proposal_candidate:
        return {
            "failed": False,
            "candidateStrategy": None,
            "requiresActivationManifest": False,
            "productionRuntimeAllowed": True,
            "reason": "standard compiled Voco model activation",
        }

    if candidate_strategy != "preserve-active":
        return {
            "failed": True,
            "reason": "proposal candidate activation requires preserve-active strategy",
            "candidateStrategy": candidate_strategy or None,
            "productionRuntimeAllowed": False,
        }
    if not activation_manifest:
        return {
            "failed": True,
            "reason": "preserve-active proposal candidates require an explicit Jason approval activation manifest",
            "candidateStrategy": candidate_strategy,
            "productionRuntimeAllowed": False,
        }

    manifest_path = activation_manifest.expanduser()
    if not manifest_path.exists():
        return {
            "failed": True,
            "reason": "activation manifest not found",
            "manifest": str(manifest_path),
            "candidateStrategy": candidate_strategy,
            "productionRuntimeAllowed": False,
        }
    manifest = load_json_object(manifest_path)
    failures = proposal_activation_manifest_failures(
        manifest,
        model_path,
        active_model,
        candidate_strategy,
        manifest_path,
        replaylab_root,
    )
    if failures:
        return {
            "failed": True,
            "reason": "activation manifest failed proposal runtime guard",
            "manifest": str(manifest_path),
            "candidateStrategy": candidate_strategy,
            "failures": failures,
            "productionRuntimeAllowed": False,
        }
    return {
        "failed": False,
        "reason": "preserve-active proposal candidate approved for this activation transaction",
        "manifest": str(manifest_path),
        "candidateStrategy": candidate_strategy,
        "approvedBy": manifest.get("approvedBy"),
        "approvedAt": manifest.get("approvedAt"),
        "productionRuntimeAllowed": True,
    }


def proposal_activation_manifest_failures(
    manifest: dict[str, Any],
    model_path: Path,
    active_model: Path,
    candidate_strategy: str,
    manifest_path: Path,
    replaylab_root: Path,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    required_equals = {
        "schema": "voco.policy-proposal-runtime-activation.v1",
        "candidateStrategy": candidate_strategy,
        "runtimeActivationEligible": True,
        "requiresJasonApproval": True,
    }
    for field, expected in required_equals.items():
        if manifest.get(field) != expected:
            failures.append({"field": field, "expected": expected, "actual": manifest.get(field)})
    if not str(manifest.get("approvalToken") or "").strip():
        failures.append({"field": "approvalToken", "reason": "missing approval token"})
    if str(manifest.get("approvedBy") or "").strip().lower() not in {"jason"}:
        failures.append({"field": "approvedBy", "reason": "Jason approval is required", "actual": manifest.get("approvedBy")})
    if not str(manifest.get("approvedAt") or "").strip():
        failures.append({"field": "approvedAt", "reason": "missing approval timestamp"})
    candidate_sha = manifest.get("candidateModelSha256")
    actual_candidate_sha = sha256_file(model_path)
    if candidate_sha != actual_candidate_sha:
        failures.append({"field": "candidateModelSha256", "expected": actual_candidate_sha, "actual": candidate_sha})
    manifest_candidate_path = manifest.get("candidateModelPath")
    if manifest_candidate_path and not manifest_path_matches_candidate(
        manifest_candidate_path,
        model_path,
        manifest_path,
        replaylab_root,
    ):
        failures.append({"field": "candidateModelPath", "expected": str(model_path), "actual": manifest_candidate_path})
    source_active_sha = manifest.get("sourceActiveModelSha256")
    if source_active_sha and active_model.exists():
        actual_active_sha = sha256_file(active_model)
        if source_active_sha != actual_active_sha:
            failures.append({"field": "sourceActiveModelSha256", "expected": actual_active_sha, "actual": source_active_sha})
    allowed_command = str(manifest.get("allowedActivationCommand") or "")
    if "activateModel" not in allowed_command:
        failures.append({"field": "allowedActivationCommand", "reason": "manifest must explicitly allow activateModel"})
    return failures


def manifest_path_matches_candidate(
    manifest_candidate_path: Any,
    model_path: Path,
    manifest_path: Path,
    replaylab_root: Path,
) -> bool:
    raw_path = Path(str(manifest_candidate_path)).expanduser()
    candidates = [raw_path] if raw_path.is_absolute() else [replaylab_root / raw_path, manifest_path.parent / raw_path]
    expected = model_path.expanduser().resolve()
    return any(candidate.expanduser().resolve() == expected for candidate in candidates)


def activate_model_command(args: argparse.Namespace) -> dict[str, Any]:
    model_path = args.model.expanduser()
    active_model = args.active_model.expanduser()
    evidence_store = args.evidence_store.expanduser()
    backup_dir = expanded_optional_path(getattr(args, "backup_dir", None))
    backup_retention = backup_retention_from_args(args)
    model = load_model(model_path)
    activation_guard = proposal_activation_guard(
        model,
        model_path,
        active_model,
        getattr(args, "activation_manifest", None),
        args.replaylab_root.expanduser(),
    )
    if activation_guard.get("failed"):
        return {"model": str(model_path), "activationGuard": activation_guard, "failed": True}
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
    model_sha = sha256_file(model_path)
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
            "modelSha256": model_sha,
            "activeModel": str(active_model),
            "previousActiveModelSha256": sha256_file(active_model) if active_model.exists() else None,
            "backup": str(backup_path) if backup_path else None,
            "backupMode": "directory" if backup_dir else "none",
            "backupDirectory": str(backup_dir) if backup_dir else None,
            "backupRetention": backup_retention if backup_dir else None,
            "validationReady": True,
            "activationGuard": activation_guard,
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
        "activationGuard": activation_guard,
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


def list_policy_families(model_path: Path, limit: int) -> dict[str, Any]:
    model = load_model(model_path)
    summary = policy_family_summary(model.get("policies") or [])
    families = sorted(
        summary.values(),
        key=lambda item: (-int(item.get("policyCount") or 0), str(item.get("familyId") or "")),
    )
    return {
        "model": str(model_path),
        "familyCount": len(families),
        "families": families[: max(0, limit)],
    }


def inspect_policy_family(model_path: Path, family_id: str) -> dict[str, Any]:
    validate_family_id(family_id)
    model = load_model(model_path)
    policies = [
        policy
        for policy in model.get("policies") or []
        if str(policy.get("familyId") or "") == family_id
    ]
    return {
        "model": str(model_path),
        "family": summarize_policy_family(family_id, policies) if policies else None,
        "policies": [
            {
                "policyId": policy.get("policyId"),
                "autoApplyMode": policy.get("autoApplyMode"),
                "policyType": policy.get("policyType"),
                "familyRole": policy.get("familyRole"),
                "sourcePattern": policy.get("sourcePattern") or policy.get("source"),
                "targetText": policy.get("targetText") or policy.get("target"),
                "lockName": policy.get("lockName"),
                "contextRequired": policy.get("contextRequired"),
                "contextTokensAny": policy.get("contextTokensAny") or [],
                "contextAliasesAny": policy.get("contextAliasesAny") or [],
                "familyTagEventIds": policy.get("familyTagEventIds") or [],
            }
            for policy in sorted(policies, key=lambda item: str(item.get("policyId") or ""))
        ],
    }


def strict_text_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").strip().casefold()
    return STRICT_SPACE_RE.sub(" ", normalized)


def manual_replacement_noop_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").strip()
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
    if path.name == POLICY_PROPOSAL_MODEL_FILE or path.suffix == ".joblib":
        raise ValueError(
            "proposal ranker artifacts are not compiled Voco runtime models; "
            "use full-db.auto-apply-model.json for production auto-apply"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


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


def compact_alias_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        key = unicodedata.normalize("NFKC", item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def validate_family_id(value: str) -> None:
    if not FAMILY_ID_RE.match(value):
        raise SystemExit(
            "family id must be an ASCII slug: letters/numbers plus dot, underscore, colon, or hyphen"
        )


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
        "familyMetadataFailures": len(report.get("familyMetadataFailures") or []),
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
