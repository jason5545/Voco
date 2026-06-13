#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import voco_runtime_correction_control as control


class VocoRuntimeCorrectionControlTests(unittest.TestCase):
    def test_valid_gated_apply_artifact_passes_cli_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = write_gated_apply_fixture(root)

            result = control.validate_artifact(artifact)

            self.assertTrue(result["ready"])
            self.assertEqual(result["runtimeMode"], "gatedApply")
            self.assertTrue(result["productionRuntimeAllowed"])
            self.assertEqual(result["candidateSpanCount"], 1)

    def test_joblib_is_rejected_as_runtime_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            joblib = Path(tmp) / "proposal-ranker-model.joblib"
            joblib.write_bytes(b"\x80\x04")

            with self.assertRaisesRegex(control.RuntimeCorrectionArtifactError, "not a joblib"):
                control.validate_artifact(joblib)

    def test_not_worse_readiness_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = write_gated_apply_fixture(Path(tmp), not_worse=False)

            with self.assertRaisesRegex(control.RuntimeCorrectionArtifactError, "notWorseThanCompiledJson"):
                control.validate_artifact(artifact)

    def test_unsafe_model_paths_are_rejected_by_cli_guard(self):
        unsafe_paths = [
            "../runtime-candidate-spans.json",
            "/private/tmp/runtime-candidate-spans.json",
            "models//runtime-candidate-spans.json",
            "models/./runtime-candidate-spans.json",
            "runtime-candidate-spans.JOBLIB",
        ]
        for unsafe_path in unsafe_paths:
            with self.subTest(unsafe_path=unsafe_path):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    artifact = write_gated_apply_fixture(root)
                    patch_artifact_model_path(artifact, unsafe_path)

                    with self.assertRaisesRegex(control.RuntimeCorrectionArtifactError, "model.path"):
                        control.validate_artifact(artifact)

    def test_replay_gate_reports_runtime_improvement_without_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = write_gated_apply_fixture(
                root / "artifact",
                candidates=[
                    {
                        "id": "runtime-direct-output",
                        "source": "直接改輸出",
                        "target": "直接改 final output",
                        "score": 0.99,
                    },
                    {
                        "id": "action-command-danger",
                        "source": "全部刪除",
                        "target": "全部都刪掉",
                        "score": 0.999,
                    },
                    {
                        "id": "rule-baseline-danger",
                        "source": "程式",
                        "target": "城市",
                        "score": 0.999,
                    },
                ],
            )
            replay_cases = write_replay_cases_fixture(
                root,
                [
                    {
                        "id": "new-gap-can-improve",
                        "postRuleText": "runtime 小模型直接改輸出",
                        "expectedFinalText": "runtime 小模型直接改 final output",
                        "allowRuntimeChange": True,
                    },
                    {
                        "id": "action-command-bypass",
                        "postRuleText": "全部刪除",
                        "expectedFinalText": "全部刪除",
                        "actionCommand": True,
                        "allowRuntimeChange": False,
                    },
                    {
                        "id": "compiled-json-baseline-priority",
                        "postRuleText": "程式角色",
                        "expectedFinalText": "程式角色",
                        "allowRuntimeChange": False,
                        "deterministicRuleFires": [{"policyId": "city-to-program"}],
                    },
                ],
            )

            report = control.run_runtime_replay_gate(artifact, json.loads(artifact.read_text()), replay_cases)

            self.assertTrue(report["readiness"]["deployReady"])
            self.assertEqual(report["caseCount"], 3)
            self.assertEqual(report["candidateEventCount"], 3)
            self.assertEqual(report["candidateFireCount"], 1)
            self.assertEqual(report["improvementCount"], 1)
            self.assertEqual(report["finalTextRegressionCount"], 0)
            self.assertEqual(report["unsafeApplyFalsePositiveCount"], 0)
            self.assertEqual(report["actionCommandBypassFailureCount"], 0)
            self.assertEqual(report["deterministicRuleOverrideCount"], 0)

    def test_replay_gate_blocks_regression_against_compiled_json_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = write_gated_apply_fixture(
                root / "artifact",
                candidates=[
                    {
                        "id": "rule-baseline-danger",
                        "source": "程式",
                        "target": "城市",
                        "score": 0.999,
                    }
                ],
            )
            replay_cases = write_replay_cases_fixture(
                root,
                [
                    {
                        "id": "compiled-json-output-must-not-regress",
                        "postRuleText": "程式角色",
                        "expectedFinalText": "程式角色",
                        "allowRuntimeChange": False,
                    }
                ],
            )

            report = control.run_runtime_replay_gate(artifact, json.loads(artifact.read_text()), replay_cases)

            self.assertFalse(report["readiness"]["deployReady"])
            self.assertEqual(report["finalTextRegressionCount"], 1)
            self.assertEqual(report["unsafeApplyFalsePositiveCount"], 1)
            self.assertIn("runtime replay is worse than compiled JSON baseline", report["readiness"]["blockers"])
            with self.assertRaisesRegex(control.RuntimeCorrectionArtifactError, "Voco-side runtime replay gate failed"):
                control.validate_artifact(artifact, replay_cases_path=replay_cases)

    def test_committed_install_requires_voco_side_replay_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = write_gated_apply_fixture(root / "artifact")
            target_dir = root / "runtime"
            args = type(
                "Args",
                (),
                {
                    "artifact": artifact,
                    "target_dir": target_dir,
                    "backup_dir": None,
                    "commit_install": True,
                },
            )()

            with self.assertRaisesRegex(control.RuntimeCorrectionArtifactError, "requires Voco-side replay cases"):
                control.install_artifact_command(args)

    def test_dry_run_install_does_not_write_runtime_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = write_gated_apply_fixture(root / "artifact")
            target_dir = root / "runtime"
            args = type(
                "Args",
                (),
                {
                    "artifact": artifact,
                    "target_dir": target_dir,
                    "backup_dir": None,
                    "commit_install": False,
                },
            )()

            result = control.install_artifact_command(args)

            self.assertTrue(result["ready"])
            self.assertTrue(result["dryRun"])
            self.assertFalse(result["installed"])
            self.assertFalse(target_dir.exists())

    def test_committed_install_rejects_shadow_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = write_shadow_fixture(root / "artifact")
            target_dir = root / "runtime"
            args = type(
                "Args",
                (),
                {
                    "artifact": artifact,
                    "target_dir": target_dir,
                    "backup_dir": None,
                    "commit_install": True,
                },
            )()

            with self.assertRaisesRegex(control.RuntimeCorrectionArtifactError, "production-allowed gatedApply"):
                control.install_artifact_command(args)
            self.assertFalse(target_dir.exists())

    def test_committed_install_writes_only_valid_gated_apply_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = write_gated_apply_fixture(root / "artifact")
            replay_cases = write_replay_cases_fixture(root)
            target_dir = root / "runtime"
            args = type(
                "Args",
                (),
                {
                    "artifact": artifact,
                    "target_dir": target_dir,
                    "backup_dir": None,
                    "replay_cases": replay_cases,
                    "replay_report": root / "runtime-replay-gate.report.json",
                    "commit_install": True,
                },
            )()

            result = control.install_artifact_command(args)

            self.assertTrue(result["installed"])
            self.assertFalse(result["dryRun"])
            self.assertTrue((target_dir / control.RUNTIME_ARTIFACT_FILE).exists())
            self.assertTrue((target_dir / "runtime-candidate-spans.json").exists())
            self.assertTrue((root / "runtime-replay-gate.report.json").exists())
            self.assertTrue(result["vocoReplayGate"]["readiness"]["deployReady"])

    def test_committed_install_preserves_nested_model_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = write_gated_apply_fixture(
                root / "artifact",
                model_relative_path=Path("models/runtime-candidate-spans.json"),
            )
            replay_cases = write_replay_cases_fixture(root)
            target_dir = root / "runtime"
            args = type(
                "Args",
                (),
                {
                    "artifact": artifact,
                    "target_dir": target_dir,
                    "backup_dir": None,
                    "replay_cases": replay_cases,
                    "replay_report": None,
                    "commit_install": True,
                },
            )()

            result = control.install_artifact_command(args)

            self.assertEqual(result["modelRelativePath"], "models/runtime-candidate-spans.json")
            self.assertEqual(result["modelInstallPath"], str(target_dir / "models/runtime-candidate-spans.json"))
            self.assertTrue((target_dir / "models/runtime-candidate-spans.json").exists())
            installed_artifact = json.loads((target_dir / control.RUNTIME_ARTIFACT_FILE).read_text(encoding="utf-8"))
            self.assertEqual(installed_artifact["model"]["path"], "models/runtime-candidate-spans.json")

    def test_existing_installed_runtime_files_are_backed_up_with_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "runtime"
            target_dir.mkdir()
            write_gated_apply_fixture(
                target_dir,
                model_relative_path=Path("models/old-runtime-candidate-spans.json"),
            )
            backup_dir = root / "backup"
            artifact = write_gated_apply_fixture(
                root / "artifact",
                model_relative_path=Path("models/new-runtime-candidate-spans.json"),
            )
            replay_cases = write_replay_cases_fixture(root)
            args = type(
                "Args",
                (),
                {
                    "artifact": artifact,
                    "target_dir": target_dir,
                    "backup_dir": backup_dir,
                    "replay_cases": replay_cases,
                    "replay_report": None,
                    "commit_install": True,
                },
            )()

            result = control.install_artifact_command(args)

            self.assertTrue(result["installed"])
            self.assertTrue((backup_dir / control.RUNTIME_ARTIFACT_FILE).exists())
            self.assertTrue((backup_dir / "models/old-runtime-candidate-spans.json").exists())
            self.assertTrue((target_dir / "models/new-runtime-candidate-spans.json").exists())


def write_shadow_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema": "voco.runtime-correction-model.v1",
        "artifactId": "runtime-correction-shadow-test",
        "runtimeMode": "shadow",
        "intendedUse": "runtime shadow contract",
        "model": {
            "format": "none",
            "modelType": "shadow-contract-only",
            "path": "",
            "portableRuntime": False,
            "sha256": "",
        },
        "approval": {
            "allowedModes": ["shadow"],
            "runtimeAllowed": False,
        },
        "sourceRanker": {
            "runtimeUsableDirectly": False,
        },
        "safety": {
            "actionCommandBypass": True,
            "artifactMissingFallback": "return-post-rule-text",
            "compiledJsonLoaderMayLoadJoblib": False,
            "jsonExactRulePriority": True,
            "timeoutFallback": "return-post-rule-text",
        },
        "decisionSchema": {
            "schema": "voco.runtime-correction-decision.v1",
            "actions": ["noop", "block"],
            "requiresEvidenceEvent": True,
            "requiresReasonCodes": True,
            "requiresScore": True,
        },
        "candidateGenerator": {
            "required": True,
            "schema": "voco.runtime-candidate-generator.v1",
            "sha256": "candidate-generator-test-sha",
        },
    }
    artifact_path = root / "runtime-correction-artifact.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact_path


def write_gated_apply_fixture(
    root: Path,
    *,
    not_worse: bool = True,
    model_relative_path: Path = Path("runtime-candidate-spans.json"),
    candidates: list[dict] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    model = {
        "schema": "voco.runtime-candidate-spans.v1",
        "candidates": candidates or [
            {
                "id": "runtime-direct-output",
                "source": "直接改輸出",
                "target": "直接改 final output",
                "score": 0.99,
            }
        ],
    }
    model_path = root / model_relative_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    model_sha = control.sha256_hex(model_path)
    artifact = {
        "schema": "voco.runtime-correction-model.v1",
        "artifactId": "runtime-correction-gated-apply-test",
        "runtimeMode": "gatedApply",
        "intendedUse": "runtime gated apply correction contract",
        "model": {
            "format": "candidate-spans-v1",
            "modelType": "candidate-ranker",
            "path": str(model_relative_path),
            "portableRuntime": True,
            "sha256": model_sha,
        },
        "approval": {
            "allowedModes": ["gatedApply"],
            "approvedAt": "2026-06-13T12:00:00Z",
            "approvedBy": "Jason",
            "approvalToken": "jason-approved-runtime-gated-apply-test",
            "requiresJasonApprovalForApply": True,
            "runtimeAllowed": True,
        },
        "sourceRanker": {
            "runtimeUsableDirectly": False,
        },
        "safety": {
            "actionCommandBypass": True,
            "artifactMissingFallback": "return-post-rule-text",
            "compiledJsonLoaderMayLoadJoblib": False,
            "jsonExactRulePriority": True,
            "notWorseThanCompiledJson": not_worse,
            "timeoutFallback": "return-post-rule-text",
        },
        "decisionSchema": {
            "schema": "voco.runtime-correction-decision.v1",
            "actions": ["noop", "block", "apply"],
            "requiresEvidenceEvent": True,
            "requiresReasonCodes": True,
            "requiresScore": True,
        },
        "candidateGenerator": {
            "required": True,
            "schema": "voco.runtime-candidate-generator.v1",
            "sha256": "candidate-generator-test-sha",
        },
        "thresholdConfig": {
            "shadow": 0.0,
            "suggest": 0.85,
            "gatedApply": 0.97,
        },
        "runtimeReadiness": {
            "actionCommandBypassVerified": True,
            "baselineReplayPass": True,
            "finalTextRegressionCount": 0 if not_worse else 1,
            "gatedApplyReplayPass": not_worse,
            "notWorseThanCompiledJson": not_worse,
            "unsafeApplyFalsePositiveCount": 0,
        },
    }
    artifact_path = root / "runtime-correction-artifact.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact_path


def write_replay_cases_fixture(root: Path, cases: list[dict] | None = None) -> Path:
    replay_cases = cases or [
        {
            "id": "new-gap-can-improve",
            "postRuleText": "runtime 小模型直接改輸出",
            "expectedFinalText": "runtime 小模型直接改 final output",
            "allowRuntimeChange": True,
        }
    ]
    path = root / "runtime-replay-cases.jsonl"
    path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False, sort_keys=True) for case in replay_cases) + "\n",
        encoding="utf-8",
    )
    return path


def patch_artifact_model_path(artifact_path: Path, model_path: str) -> None:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["model"]["path"] = model_path
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
