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
            result = validate_artifact(args.artifact.expanduser())
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

    install = subparsers.add_parser("installArtifact")
    install.add_argument("--artifact", type=Path, required=True)
    install.add_argument("--target-dir", type=Path, default=DEFAULT_RUNTIME_MODEL_DIR)
    install.add_argument("--backup-dir", type=Path)
    install.add_argument("--commit-install", action="store_true")

    return parser.parse_args()


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
    path = Path(value)
    require(value != "", f"{field_name} is required")
    require(not path.is_absolute(), f"{field_name} must be relative")
    require(".." not in path.parts, f"{field_name} must not traverse outside the artifact directory")
    return path


def validate_artifact(artifact_path: Path) -> dict[str, Any]:
    artifact_path = artifact_path.resolve()
    require(artifact_path.suffix != ".joblib", "Runtime correction artifact must be a manifest, not a joblib ranker")
    artifact = load_json(artifact_path)
    require(isinstance(artifact, dict), "Runtime correction artifact must be a JSON object")
    require(artifact.get("schema") == "voco.runtime-correction-model.v1", "Unsupported runtime correction artifact schema")

    mode = artifact.get("runtimeMode")
    if mode == "shadow":
        return validate_shadow_artifact(artifact_path, artifact)
    if mode == "gatedApply":
        return validate_gated_apply_artifact(artifact_path, artifact)
    raise RuntimeCorrectionArtifactError(f"Unsupported runtime correction artifact mode: {mode}")


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
        return result
    require(
        validation.get("runtimeMode") == "gatedApply" and validation.get("productionRuntimeAllowed") is True,
        "Committed runtime correction install requires a production-allowed gatedApply artifact",
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
