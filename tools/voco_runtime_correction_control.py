#!/usr/bin/env python3
"""
Control-plane guard for Voco runtime correction artifacts.

This intentionally mirrors the Swift runtime loader contract. It lets us
validate a ReplayLab runtime model artifact from the command line before any
live app setting is enabled or any file is installed into Application Support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


APP_SUPPORT = Path.home() / "Library/Application Support/com.jasonchien.Voco"
DEFAULT_RUNTIME_MODEL_DIR = APP_SUPPORT / "RuntimeCorrectionModels"
RUNTIME_ARTIFACT_FILE = "runtime-correction-artifact.json"
ALLOWED_JASON_APPROVERS = {"Jason", "Jason Chien", "Jianrui Cheng"}


class RuntimeCorrectionArtifactError(ValueError):
    pass


def main() -> int:
    args = parse_args()
    try:
        if args.command == "validateArtifact":
            result = validate_artifact(
                args.artifact.expanduser(),
                replay_cases_path=expanded_optional_path(args.replay_cases),
                replay_report_path=expanded_optional_path(args.replay_report),
            )
        elif args.command == "replayGate":
            result = replay_gate_command(args)
        elif args.command == "installArtifact":
            result = install_artifact_command(args)
        else:
            raise RuntimeCorrectionArtifactError(f"Unknown command: {args.command}")
    except RuntimeCorrectionArtifactError as error:
        result = {"failed": True, "ready": False, "error": str(error)}

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_human(result)
    return 1 if result.get("failed") else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate or install Voco runtime correction artifacts.")
    parser.add_argument("--json", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validateArtifact")
    validate.add_argument("--artifact", type=Path, required=True)
    validate.add_argument("--replay-cases", type=Path)
    validate.add_argument("--replay-report", type=Path)

    replay = subparsers.add_parser("replayGate")
    replay.add_argument("--artifact", type=Path, required=True)
    replay.add_argument("--replay-cases", type=Path, required=True)
    replay.add_argument("--report", type=Path)

    install = subparsers.add_parser("installArtifact")
    install.add_argument("--artifact", type=Path, required=True)
    install.add_argument("--target-dir", type=Path, default=DEFAULT_RUNTIME_MODEL_DIR)
    install.add_argument("--backup-dir", type=Path)
    install.add_argument("--replay-cases", type=Path)
    install.add_argument("--replay-report", type=Path)
    install.add_argument("--commit-install", action="store_true")

    return parser.parse_args()


def expanded_optional_path(path: Path | None) -> Path | None:
    return path.expanduser() if path else None


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as error:
        raise RuntimeCorrectionArtifactError(f"File not found: {path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeCorrectionArtifactError(f"Invalid JSON: {path}: {error}") from error


def sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeCorrectionArtifactError(message)


def safe_relative_path(value: Any, field_name: str) -> Path:
    require(isinstance(value, str), f"{field_name} must be a string")
    require(value != "", f"{field_name} is required")
    require(not value.lower().endswith(".joblib"), f"{field_name} must not point to a joblib ranker")
    path = Path(value)
    require(not path.is_absolute(), f"{field_name} must be relative")
    parts = value.split("/")
    require(
        all(part not in ("", ".", "..") for part in parts),
        f"{field_name} must be a safe relative path",
    )
    return path


def validate_artifact(
    artifact_path: Path,
    *,
    replay_cases_path: Path | None = None,
    replay_report_path: Path | None = None,
) -> dict[str, Any]:
    artifact_path = artifact_path.resolve()
    require(artifact_path.suffix != ".joblib", "Runtime correction artifact must be a manifest, not a joblib ranker")
    artifact = load_json(artifact_path)
    require(isinstance(artifact, dict), "Runtime correction artifact must be a JSON object")
    require(artifact.get("schema") == "voco.runtime-correction-model.v1", "Unsupported runtime correction artifact schema")

    mode = artifact.get("runtimeMode")
    if mode == "shadow":
        result = validate_shadow_artifact(artifact_path, artifact)
    elif mode == "gatedApply":
        result = validate_gated_apply_artifact(artifact_path, artifact)
    else:
        raise RuntimeCorrectionArtifactError(f"Unsupported runtime correction artifact mode: {mode}")

    if replay_cases_path:
        require(mode == "gatedApply", "Voco-side runtime replay gate only supports gatedApply artifacts")
        replay_report = run_runtime_replay_gate(artifact_path, artifact, replay_cases_path)
        write_json_report_if_requested(replay_report, replay_report_path)
        require(
            replay_report["readiness"]["deployReady"] is True,
            "Voco-side runtime replay gate failed: " + ", ".join(replay_report["readiness"]["blockers"]),
        )
        result["vocoReplayGate"] = replay_report
    return result


def validate_shadow_artifact(artifact_path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    model = object_field(artifact, "model")
    approval = object_field(artifact, "approval")
    source_ranker = object_field(artifact, "sourceRanker")
    safety = object_field(artifact, "safety")

    require(approval.get("allowedModes") == ["shadow"], "Shadow artifact must only allow shadow mode")
    require(approval.get("runtimeAllowed") is False, "Shadow artifact must not allow runtime apply")
    require(model.get("portableRuntime") is False, "Shadow artifact must not include a portable runtime model")
    require(model.get("format") == "none", "Shadow artifact model format must be none")
    require(source_ranker.get("runtimeUsableDirectly") is False, "Source ranker must not be runtime usable directly")
    validate_common_safety(safety)
    validate_decision_schema(artifact, require_apply=False)
    validate_candidate_generator(artifact)

    return base_result(artifact_path, artifact, ready=True)


def validate_gated_apply_artifact(artifact_path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    artifact_dir = artifact_path.parent
    model = object_field(artifact, "model")
    approval = object_field(artifact, "approval")
    source_ranker = object_field(artifact, "sourceRanker")
    safety = object_field(artifact, "safety")
    threshold = object_field(artifact, "thresholdConfig")
    readiness = object_field(artifact, "runtimeReadiness")

    require(approval.get("runtimeAllowed") is True, "Gated apply artifact must be runtimeAllowed")
    require("gatedApply" in list_field(approval, "allowedModes"), "Gated apply must be in allowedModes")
    require(approval.get("requiresJasonApprovalForApply") is True, "Gated apply requires Jason approval")
    require(approval.get("approvedBy") in ALLOWED_JASON_APPROVERS, "approvedBy is not an accepted Jason approver")
    require(bool(approval.get("approvedAt")), "approvedAt is required")
    require(bool(approval.get("approvalToken")), "approvalToken is required")

    require(model.get("portableRuntime") is True, "Gated apply requires a portable runtime model")
    require(model.get("format") == "candidate-spans-v1", "Portable model format must be candidate-spans-v1")
    model_relative_path = safe_relative_path(model.get("path", ""), "model.path")
    require(model_relative_path.suffix != ".joblib", "Portable runtime model must not be a joblib ranker")
    require(bool(model.get("sha256")), "model.sha256 is required")
    model_path = artifact_dir / model_relative_path
    require(model_path.exists(), f"Portable runtime model not found: {model_relative_path}")
    actual_model_sha = sha256_hex(model_path)
    require(actual_model_sha == model.get("sha256"), "Portable runtime model sha256 mismatch")
    candidate_model = validate_candidate_span_model(model_path)

    require(source_ranker.get("runtimeUsableDirectly") is False, "Source ranker must not be runtime usable directly")
    validate_common_safety(safety)
    require(safety.get("notWorseThanCompiledJson") is True, "Safety contract must assert notWorseThanCompiledJson")
    require(float(threshold.get("gatedApply", 0)) >= 0.97, "gatedApply threshold must be at least 0.97")
    require(readiness.get("baselineReplayPass") is True, "baselineReplayPass must be true")
    require(readiness.get("gatedApplyReplayPass") is True, "gatedApplyReplayPass must be true")
    require(readiness.get("notWorseThanCompiledJson") is True, "runtimeReadiness.notWorseThanCompiledJson must be true")
    require(readiness.get("unsafeApplyFalsePositiveCount") == 0, "unsafeApplyFalsePositiveCount must be 0")
    require(readiness.get("finalTextRegressionCount") == 0, "finalTextRegressionCount must be 0")
    require(readiness.get("actionCommandBypassVerified") is True, "actionCommandBypassVerified must be true")
    validate_decision_schema(artifact, require_apply=True)
    validate_candidate_generator(artifact)

    result = base_result(artifact_path, artifact, ready=True)
    result.update(
        {
            "candidateSpanCount": len(candidate_model.get("candidates", [])),
            "modelPath": str(model_path),
            "modelRelativePath": str(model_relative_path),
            "modelSha256": actual_model_sha,
            "thresholdGatedApply": threshold.get("gatedApply"),
            "runtimeReadiness": readiness,
        }
    )
    return result


def validate_common_safety(safety: dict[str, Any]) -> None:
    require(safety.get("actionCommandBypass") is True, "actionCommandBypass must be true")
    require(safety.get("compiledJsonLoaderMayLoadJoblib") is False, "compiledJsonLoaderMayLoadJoblib must be false")
    require(safety.get("artifactMissingFallback") == "return-post-rule-text", "artifactMissingFallback must return post-rule text")
    require(safety.get("timeoutFallback") == "return-post-rule-text", "timeoutFallback must return post-rule text")
    require(safety.get("jsonExactRulePriority") is True, "jsonExactRulePriority must be true")


def validate_decision_schema(artifact: dict[str, Any], *, require_apply: bool) -> None:
    decision = object_field(artifact, "decisionSchema")
    actions = set(list_field(decision, "actions"))
    require(decision.get("schema") == "voco.runtime-correction-decision.v1", "Unsupported decision schema")
    required_actions = {"noop", "block"}
    if require_apply:
        required_actions.add("apply")
    require(required_actions.issubset(actions), f"Decision schema actions must include {sorted(required_actions)}")
    require(decision.get("requiresEvidenceEvent") is True, "Decision schema must require evidence events")
    require(decision.get("requiresReasonCodes") is True, "Decision schema must require reason codes")
    require(decision.get("requiresScore") is True, "Decision schema must require score")


def validate_candidate_generator(artifact: dict[str, Any]) -> None:
    generator = object_field(artifact, "candidateGenerator")
    require(generator.get("required") is True, "candidateGenerator.required must be true")
    require(generator.get("schema") == "voco.runtime-candidate-generator.v1", "Unsupported candidate generator schema")


def validate_candidate_span_model(model_path: Path) -> dict[str, Any]:
    model = load_json(model_path)
    require(isinstance(model, dict), "Candidate span model must be a JSON object")
    require(model.get("schema") == "voco.runtime-candidate-spans.v1", "Unsupported candidate span model schema")
    candidates = model.get("candidates")
    require(isinstance(candidates, list), "Candidate span model candidates must be a list")
    for index, candidate in enumerate(candidates):
        require(isinstance(candidate, dict), f"Candidate #{index} must be an object")
        source = candidate.get("source")
        target = candidate.get("target")
        score = candidate.get("score")
        require(isinstance(source, str) and source, f"Candidate #{index} source is required")
        require(isinstance(target, str) and target, f"Candidate #{index} target is required")
        require(source != target, f"Candidate #{index} source and target must differ")
        require(isinstance(score, (int, float)) and 0 <= score <= 1, f"Candidate #{index} score must be between 0 and 1")
    return model


def replay_gate_command(args: argparse.Namespace) -> dict[str, Any]:
    artifact_path = args.artifact.expanduser().resolve()
    artifact = load_json(artifact_path)
    require(isinstance(artifact, dict), "Runtime correction artifact must be a JSON object")
    validate_artifact(artifact_path)
    require(artifact.get("runtimeMode") == "gatedApply", "Runtime replay gate only supports gatedApply artifacts")
    report = run_runtime_replay_gate(artifact_path, artifact, args.replay_cases.expanduser())
    write_json_report_if_requested(report, expanded_optional_path(args.report))
    require(
        report["readiness"]["deployReady"] is True,
        "Voco-side runtime replay gate failed: " + ", ".join(report["readiness"]["blockers"]),
    )
    return report


def write_json_report_if_requested(report: dict[str, Any], report_path: Path | None) -> None:
    if not report_path:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def run_runtime_replay_gate(
    artifact_path: Path,
    artifact: dict[str, Any],
    replay_cases_path: Path,
) -> dict[str, Any]:
    artifact_dir = artifact_path.parent
    model = object_field(artifact, "model")
    model_path = artifact_dir / safe_relative_path(model.get("path", ""), "model.path")
    candidate_model = validate_candidate_span_model(model_path)
    threshold = float(object_field(artifact, "thresholdConfig").get("gatedApply", 1.0))
    safety = object_field(artifact, "safety")
    cases = load_replay_cases(replay_cases_path)

    details: list[dict[str, Any]] = []
    candidate_event_count = 0
    candidate_fire_count = 0
    changed_rows = 0
    baseline_rows_matching_expected = 0
    rows_matching_expected = 0
    improvement_count = 0
    missed_improvement_count = 0
    final_text_regression_count = 0
    unsafe_apply_false_positive_count = 0
    action_command_bypass_failure_count = 0
    deterministic_rule_override_count = 0
    protected_term_override_count = 0

    for index, case in enumerate(cases):
        case_id = str(case.get("id") or f"case-{index + 1}")
        post_rule_text = required_string(case, "postRuleText", case_id)
        expected_final_text = required_string(case, "expectedFinalText", case_id)
        raw_transcript = string_field(case, "rawTranscript", default=post_rule_text)
        canonicalized_text = string_field(case, "canonicalizedText", default=post_rule_text)
        context_hints = list_of_strings(case.get("contextHints", []), f"{case_id}.contextHints")
        app_mode = case.get("appMode")
        require(app_mode is None or isinstance(app_mode, str), f"{case_id}.appMode must be a string")
        deterministic_rule_fires = list_field_if_present(case, "deterministicRuleFires")
        protected_term_hits = list_of_strings(case.get("protectedTermHits", []), f"{case_id}.protectedTermHits")
        action_command = bool(case.get("actionCommand", False))
        allow_runtime_change = bool(case.get("allowRuntimeChange", expected_final_text != post_rule_text))
        unsafe_to_apply = bool(case.get("unsafeToApply", False))

        candidates = runtime_candidates_for_case(
            candidate_model,
            raw_transcript=raw_transcript,
            post_rule_text=post_rule_text,
            context_hints=context_hints,
            app_mode=app_mode,
        )
        if candidates:
            candidate_event_count += 1
        decision = evaluate_runtime_case(
            artifact=artifact,
            threshold=threshold,
            candidates=candidates,
            post_rule_text=post_rule_text,
            action_command=action_command,
            deterministic_rule_fires=deterministic_rule_fires,
            protected_term_hits=protected_term_hits,
        )
        final_text = decision["finalText"]
        baseline_matches_expected = post_rule_text == expected_final_text
        runtime_matches_expected = final_text == expected_final_text
        changed = final_text != post_rule_text
        if decision["chosenAction"] == "apply":
            candidate_fire_count += 1
        if changed:
            changed_rows += 1
        if baseline_matches_expected:
            baseline_rows_matching_expected += 1
        if runtime_matches_expected:
            rows_matching_expected += 1
        if baseline_matches_expected and not runtime_matches_expected:
            final_text_regression_count += 1
        if not baseline_matches_expected and runtime_matches_expected:
            improvement_count += 1
        if not baseline_matches_expected and not runtime_matches_expected:
            missed_improvement_count += 1
        if (unsafe_to_apply or not allow_runtime_change) and changed:
            unsafe_apply_false_positive_count += 1
        if action_command and changed:
            action_command_bypass_failure_count += 1
        if deterministic_rule_fires and safety.get("jsonExactRulePriority") is True and changed:
            deterministic_rule_override_count += 1
        if protected_term_hits and changed:
            protected_term_override_count += 1

        details.append(
            {
                "id": case_id,
                "postRuleText": post_rule_text,
                "expectedFinalText": expected_final_text,
                "finalText": final_text,
                "chosenAction": decision["chosenAction"],
                "fallbackReason": decision["fallbackReason"],
                "candidateCount": len(candidates),
                "changed": changed,
                "baselineMatchesExpected": baseline_matches_expected,
                "runtimeMatchesExpected": runtime_matches_expected,
                "regression": baseline_matches_expected and not runtime_matches_expected,
                "improvement": not baseline_matches_expected and runtime_matches_expected,
            }
        )

    case_count = len(cases)
    runtime_replay_pass = case_count > 0 and rows_matching_expected == case_count
    not_worse_than_compiled_json = (
        final_text_regression_count == 0
        and unsafe_apply_false_positive_count == 0
        and action_command_bypass_failure_count == 0
        and deterministic_rule_override_count == 0
        and protected_term_override_count == 0
    )
    blockers: list[str] = []
    if case_count == 0:
        blockers.append("replay case set is empty")
    if not runtime_replay_pass:
        blockers.append("runtime replay does not match expected final text for every case")
    if not not_worse_than_compiled_json:
        blockers.append("runtime replay is worse than compiled JSON baseline")

    return {
        "schema": "voco.runtime-correction-replay-gate.v1",
        "artifactId": artifact.get("artifactId"),
        "artifactPath": str(artifact_path),
        "artifactSha256": sha256_hex(artifact_path),
        "replayCasesPath": str(replay_cases_path),
        "caseCount": case_count,
        "candidateEventCount": candidate_event_count,
        "candidateFireCount": candidate_fire_count,
        "changedRows": changed_rows,
        "baselineRowsMatchingExpectedFinalText": baseline_rows_matching_expected,
        "rowsMatchingExpectedFinalText": rows_matching_expected,
        "improvementCount": improvement_count,
        "missedImprovementCount": missed_improvement_count,
        "finalTextRegressionCount": final_text_regression_count,
        "unsafeApplyFalsePositiveCount": unsafe_apply_false_positive_count,
        "actionCommandBypassFailureCount": action_command_bypass_failure_count,
        "deterministicRuleOverrideCount": deterministic_rule_override_count,
        "protectedTermOverrideCount": protected_term_override_count,
        "readiness": {
            "runtimeReplayPass": runtime_replay_pass,
            "notWorseThanCompiledJson": not_worse_than_compiled_json,
            "deployReady": runtime_replay_pass and not_worse_than_compiled_json,
            "blockers": blockers,
        },
        "details": details,
    }


def load_replay_cases(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise RuntimeCorrectionArtifactError(f"Replay cases not found: {path}") from error

    if path.suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise RuntimeCorrectionArtifactError(f"Invalid replay cases JSON: {path}: {error}") from error
        cases = data.get("cases") if isinstance(data, dict) else data
        require(isinstance(cases, list), "Replay cases JSON must be a list or an object with cases")
        require(all(isinstance(case, dict) for case in cases), "Replay cases must be JSON objects")
        return cases

    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeCorrectionArtifactError(f"Invalid replay cases JSONL line {line_number}: {error}") from error
        require(isinstance(case, dict), f"Replay case line {line_number} must be a JSON object")
        cases.append(case)
    return cases


def runtime_candidates_for_case(
    candidate_model: dict[str, Any],
    *,
    raw_transcript: str,
    post_rule_text: str,
    context_hints: list[str],
    app_mode: str | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for candidate in candidate_model.get("candidates", []):
        source = candidate["source"]
        if source not in post_rule_text:
            continue
        raw_contains = candidate.get("rawContains")
        if raw_contains and raw_contains not in raw_transcript:
            continue
        post_rule_contains = candidate.get("postRuleContains")
        if post_rule_contains and post_rule_contains not in post_rule_text:
            continue
        context_hint_contains = candidate.get("contextHintContains")
        if context_hint_contains:
            has_context = any(context_hint_contains in hint for hint in context_hints)
            has_context = has_context or (isinstance(app_mode, str) and context_hint_contains in app_mode)
            if not has_context:
                continue
        candidates.append(
            {
                "source": source,
                "target": candidate["target"],
                "score": float(candidate.get("score", 0)),
            }
        )
    return candidates


def evaluate_runtime_case(
    *,
    artifact: dict[str, Any],
    threshold: float,
    candidates: list[dict[str, Any]],
    post_rule_text: str,
    action_command: bool,
    deterministic_rule_fires: list[Any],
    protected_term_hits: list[str],
) -> dict[str, Any]:
    safety = object_field(artifact, "safety")
    if action_command:
        return {"chosenAction": "block", "fallbackReason": "action-command-bypass", "finalText": post_rule_text}
    if deterministic_rule_fires and safety.get("jsonExactRulePriority") is True:
        return {"chosenAction": "block", "fallbackReason": "deterministic-rule-priority", "finalText": post_rule_text}
    if protected_term_hits:
        return {"chosenAction": "block", "fallbackReason": "protected-term-bypass", "finalText": post_rule_text}

    eligible = [candidate for candidate in candidates if candidate["score"] >= threshold]
    if not eligible:
        return {"chosenAction": "noop", "fallbackReason": "no-candidate-above-gated-threshold", "finalText": post_rule_text}
    candidate = sorted(eligible, key=lambda item: item["score"], reverse=True)[0]
    if candidate["source"] not in post_rule_text:
        return {"chosenAction": "noop", "fallbackReason": "candidate-source-not-found", "finalText": post_rule_text}
    final_text = post_rule_text.replace(candidate["source"], candidate["target"])
    if final_text == post_rule_text:
        return {"chosenAction": "noop", "fallbackReason": "candidate-does-not-change-output", "finalText": post_rule_text}
    return {"chosenAction": "apply", "fallbackReason": "", "finalText": final_text}


def required_string(case: dict[str, Any], field: str, case_id: str) -> str:
    value = case.get(field)
    require(isinstance(value, str) and value != "", f"{case_id}.{field} is required")
    return value


def string_field(case: dict[str, Any], field: str, *, default: str) -> str:
    value = case.get(field, default)
    require(isinstance(value, str), f"{field} must be a string")
    return value


def list_field_if_present(case: dict[str, Any], field: str) -> list[Any]:
    value = case.get(field, [])
    require(isinstance(value, list), f"{field} must be a list")
    return value


def list_of_strings(value: Any, field_name: str) -> list[str]:
    require(isinstance(value, list), f"{field_name} must be a list")
    require(all(isinstance(item, str) for item in value), f"{field_name} must contain only strings")
    return value


def install_artifact_command(args: argparse.Namespace) -> dict[str, Any]:
    artifact_path = args.artifact.expanduser().resolve()
    validation = validate_artifact(artifact_path)
    target_dir = args.target_dir.expanduser()
    model_path = Path(validation["modelPath"]).resolve() if validation.get("modelPath") else None
    model_relative_path = Path(validation["modelRelativePath"]) if validation.get("modelRelativePath") else None
    dry_run = not args.commit_install

    result = dict(validation)
    result.update(
        {
            "targetDir": str(target_dir),
            "dryRun": dry_run,
            "installed": False,
            "artifactInstallPath": str(target_dir / RUNTIME_ARTIFACT_FILE),
            "modelInstallPath": str(target_dir / model_relative_path) if model_relative_path else None,
            "backupDir": None,
        }
    )
    if dry_run:
        if getattr(args, "replay_cases", None):
            replay_report = run_runtime_replay_gate(artifact_path, load_json(artifact_path), args.replay_cases.expanduser())
            write_json_report_if_requested(replay_report, expanded_optional_path(getattr(args, "replay_report", None)))
            result["vocoReplayGate"] = replay_report
        return result
    require(
        validation.get("runtimeMode") == "gatedApply" and validation.get("productionRuntimeAllowed") is True,
        "Committed runtime correction install requires a production-allowed gatedApply artifact",
    )
    replay_cases_path = expanded_optional_path(getattr(args, "replay_cases", None))
    require(replay_cases_path is not None, "Committed runtime correction install requires Voco-side replay cases")
    replay_report = run_runtime_replay_gate(artifact_path, load_json(artifact_path), replay_cases_path)
    write_json_report_if_requested(replay_report, expanded_optional_path(getattr(args, "replay_report", None)))
    require(
        replay_report["readiness"]["deployReady"] is True,
        "Committed runtime correction install requires passing Voco-side replay gate",
    )

    backup_dir = args.backup_dir.expanduser() if args.backup_dir else default_backup_dir(target_dir)
    backup_existing_runtime_files(target_dir, backup_dir, next_model_relative_path=model_relative_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact_path, target_dir / RUNTIME_ARTIFACT_FILE)
    if model_path and model_relative_path:
        model_install_path = target_dir / model_relative_path
        model_install_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, model_install_path)
    result["installed"] = True
    result["backupDir"] = str(backup_dir)
    result["vocoReplayGate"] = replay_report
    return result


def backup_existing_runtime_files(
    target_dir: Path,
    backup_dir: Path,
    *,
    next_model_relative_path: Path | None = None,
) -> None:
    existing: set[Path] = set()
    artifact_path = target_dir / RUNTIME_ARTIFACT_FILE
    if artifact_path.is_file():
        existing.add(artifact_path)
        current_model_path = installed_model_path_from_artifact(artifact_path)
        if current_model_path and current_model_path.is_file():
            existing.add(current_model_path)
    if next_model_relative_path:
        next_model_path = target_dir / next_model_relative_path
        if next_model_path.is_file():
            existing.add(next_model_path)
    existing.update(path for path in target_dir.glob("runtime-candidate-*") if path.is_file())
    if not existing:
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        relative = path.relative_to(target_dir)
        destination = backup_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def installed_model_path_from_artifact(artifact_path: Path) -> Path | None:
    try:
        artifact = load_json(artifact_path)
        model = artifact.get("model", {})
        relative = safe_relative_path(model.get("path", ""), "model.path")
        return artifact_path.parent / relative
    except RuntimeCorrectionArtifactError:
        return None


def default_backup_dir(target_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return target_dir / "backups" / f"{stamp}-before-runtime-correction-install"


def object_field(data: dict[str, Any], field: str) -> dict[str, Any]:
    value = data.get(field)
    require(isinstance(value, dict), f"{field} must be an object")
    return value


def list_field(data: dict[str, Any], field: str) -> list[Any]:
    value = data.get(field)
    require(isinstance(value, list), f"{field} must be a list")
    return value


def base_result(artifact_path: Path, artifact: dict[str, Any], *, ready: bool) -> dict[str, Any]:
    return {
        "failed": False,
        "ready": ready,
        "schema": artifact.get("schema"),
        "artifactId": artifact.get("artifactId"),
        "runtimeMode": artifact.get("runtimeMode"),
        "artifactPath": str(artifact_path),
        "artifactSha256": sha256_hex(artifact_path),
        "productionRuntimeAllowed": artifact.get("approval", {}).get("runtimeAllowed") is True,
    }


def print_human(result: dict[str, Any]) -> None:
    if result.get("failed"):
        print(f"failed: {result.get('error')}")
        return
    print(f"ready: {result.get('ready')}")
    print(f"runtimeMode: {result.get('runtimeMode')}")
    print(f"artifact: {result.get('artifactPath')}")
    if result.get("modelPath"):
        print(f"model: {result.get('modelPath')}")
    if "dryRun" in result:
        print(f"dryRun: {result.get('dryRun')}")
        print(f"installed: {result.get('installed')}")


if __name__ == "__main__":
    raise SystemExit(main())
