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


def write_gated_apply_fixture(root: Path, *, not_worse: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    model = {
        "schema": "voco.runtime-candidate-spans.v1",
        "candidates": [
            {
                "id": "runtime-direct-output",
                "source": "直接改輸出",
                "target": "直接改 final output",
                "score": 0.99,
            }
        ],
    }
    model_path = root / "runtime-candidate-spans.json"
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
            "path": model_path.name,
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


if __name__ == "__main__":
    unittest.main()
