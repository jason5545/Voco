#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import voco_auto_apply_control as control


def tiny_base_model() -> dict:
    return {
        "generatedAt": "2026-06-11T00:00:00Z",
        "modelType": "test",
        "policyCounts": {"apply": 0},
        "policyTypeCounts": {},
        "safetyContract": [],
        "policies": [],
        "mergedReplayReadiness": {"mergedAutoApplyModelReady": True, "failures": []},
    }


class VocoAutoApplyControlTests(unittest.TestCase):
    def test_append_compile_and_validate_exact_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.jsonl"
            base = root / "base.json"
            model_path = root / "compiled/full-db.auto-apply-model.json"
            base.write_text(json.dumps(tiny_base_model()), encoding="utf-8")
            args = Namespace(
                actor="test",
                source_text="Love Report",
                target_text="lab repo",
                row_pk=12813,
                context="",
                note=None,
            )
            event = control.correction_event(args)
            control.append_event(evidence, event)
            model, _report = control.compile_model(
                control.load_model(base),
                control.load_events(evidence),
                base_model_path=base,
                evidence_store=evidence,
            )
            control.write_model(model_path, model)
            validation = control.validate_model(
                model,
                control.load_events(evidence),
                model_path=model_path,
                base_model=control.load_model(base),
                replaylab_root=root / "missing-replaylab",
                current_corpus_dir=root / "missing-current",
                reraw_corpus_dir=root / "missing-reraw",
                skip_corpus_replay=True,
                skip_raw_input_replay=True,
            )
            self.assertTrue(validation["ready"])
            self.assertEqual(validation["positiveExamples"][0]["actualText"], "lab repo")
            self.assertEqual(model["policyCounts"]["apply"], 1)

    def test_context_locked_rule_passes_positive_and_negative_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.jsonl"
            base = root / "base.json"
            model_path = root / "compiled/full-db.auto-apply-model.json"
            base.write_text(json.dumps(tiny_base_model()), encoding="utf-8")
            args = Namespace(
                actor="test",
                source_pattern="充電",
                target_text="重新建立",
                source_text="應該是要充電的才對吧？",
                row_pk=12799,
                lock_name="rebuild-small-model",
                context_token=["小模型", "ReplayLab", "auto-apply"],
                context_alias=[],
                context_from_context_only=False,
                require_alias=False,
                positive=[],
                negative=[],
                positive_text="應該是要充電的才對吧？",
                positive_context="小模型 ReplayLab",
                expected_text="應該是要重新建立的才對吧？",
                negative_text="手機要充電。",
                negative_context="手機",
                note=None,
            )
            control.append_event(evidence, control.context_locked_rule_event(args))
            model, _report = control.compile_model(
                control.load_model(base),
                control.load_events(evidence),
                base_model_path=base,
                evidence_store=evidence,
            )
            control.write_model(model_path, model)
            validation = control.validate_model(
                model,
                control.load_events(evidence),
                model_path=model_path,
                base_model=control.load_model(base),
                replaylab_root=root / "missing-replaylab",
                current_corpus_dir=root / "missing-current",
                reraw_corpus_dir=root / "missing-reraw",
                skip_corpus_replay=True,
                skip_raw_input_replay=True,
            )
            self.assertTrue(validation["ready"])
            self.assertEqual(validation["positiveExamples"][0]["actualText"], "應該是要重新建立的才對吧？")
            self.assertEqual(validation["negativeExamples"][0]["actualText"], "手機要充電。")

    def test_promoted_suggest_tombstone_counts_as_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.jsonl"
            base = root / "base.json"
            base_model = tiny_base_model()
            base_model["policyCounts"] = {"suggest": 2}
            base_model["policies"] = [
                {
                    "policyId": "policy-githubcode吉他>github",
                    "policyType": "scopedReplacement",
                    "autoApplyMode": "suggest",
                    "sourcePattern": "吉他",
                    "targetText": "Github",
                    "contextTokensAny": ["github", "repo"],
                    "contextAliasesAny": [],
                },
                {
                    "policyId": "policy-counseling智商>諮商",
                    "policyType": "scopedReplacement",
                    "autoApplyMode": "suggest",
                    "sourcePattern": "智商",
                    "targetText": "諮商",
                    "contextTokensAny": ["心理師"],
                    "contextAliasesAny": [],
                },
            ]
            base.write_text(json.dumps(base_model, ensure_ascii=False), encoding="utf-8")

            control.append_event(
                evidence,
                control.disable_rule_event(
                    Namespace(
                        actor="test",
                        policy_id="policy-githubcode吉他>github",
                        source_pattern=None,
                        target_text=None,
                        reason="Promote suggest-only GitHub domain correction into manual context-locked apply rule.",
                        disposition=None,
                    )
                ),
            )
            control.append_event(
                evidence,
                control.disable_rule_event(
                    Namespace(
                        actor="test",
                        policy_id="policy-counseling智商>諮商",
                        source_pattern=None,
                        target_text=None,
                        reason="Replace broad suggest-only counseling correction with narrower phrase-level context locks.",
                        disposition=None,
                    )
                ),
            )

            model, report = control.compile_model(
                control.load_model(base),
                control.load_events(evidence),
                base_model_path=base,
                evidence_store=evidence,
            )

            self.assertEqual(model["policyCounts"]["replaced"], 1)
            self.assertEqual(model["policyCounts"]["blocked"], 1)
            self.assertEqual(report["tombstoneDispositionCounts"], {"replaced": 1, "blocked": 1})

    def test_manual_context_corpus_drift_is_reported_but_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.jsonl"
            base = root / "base.json"
            model_path = root / "compiled/full-db.auto-apply-model.json"
            corpus = root / "corpus"
            corpus.mkdir()
            base.write_text(json.dumps(tiny_base_model()), encoding="utf-8")
            args = Namespace(
                actor="test",
                source_pattern="轉路",
                target_text="轉錄",
                source_text="轉路",
                row_pk=12291,
                lock_name="migrated-swift-transcription",
                context_token=["轉錄", "技能", "ASR"],
                context_alias=[],
                context_from_context_only=False,
                require_alias=False,
                positive=["重新轉路的技能||Voco 轉錄 技能||重新轉錄的技能"],
                negative=["這條轉路很危險||道路"],
                positive_text=None,
                positive_context=None,
                expected_text=None,
                negative_text=None,
                negative_context=None,
                note=None,
            )
            control.append_event(evidence, control.context_locked_rule_event(args))
            model, _report = control.compile_model(
                control.load_model(base),
                control.load_events(evidence),
                base_model_path=base,
                evidence_store=evidence,
            )
            control.write_model(model_path, model)
            (corpus / "full-db.cleaned.jsonl").write_text(
                json.dumps(
                    {
                        "rowPk": 12291,
                        "rawOpenCC": "重新轉路的技能",
                        "cleanedText": "重新轉路的技能",
                        "requiresReview": True,
                        "riskFlags": ["storedOutputDisagreesWithRawDerivedCleaned"],
                        "context": {"before": [{"rawOpenCC": "Voco 轉錄 技能 ASR"}]},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            validation = control.validate_model(
                model,
                control.load_events(evidence),
                model_path=model_path,
                base_model=control.load_model(base),
                replaylab_root=root / "missing-replaylab",
                current_corpus_dir=corpus,
                reraw_corpus_dir=root / "missing-reraw",
                skip_corpus_replay=False,
                skip_raw_input_replay=True,
            )

            self.assertTrue(validation["ready"])
            replay = validation["corpusReplay"][0]["cleanedReplay"]
            self.assertEqual(replay["unexpectedChanges"], 0)
            self.assertEqual(replay["acceptedManualCorpusChanges"], 1)
            self.assertEqual(replay["originalUnexpectedChanges"], 1)

    def test_activate_and_rollback_use_backups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.jsonl"
            active = root / "full-db.auto-apply-model.json"
            candidate = root / "candidate.json"
            active.write_text(json.dumps(tiny_base_model()), encoding="utf-8")
            new_model = tiny_base_model()
            new_model["policyCounts"] = {"apply": 1}
            new_model["policies"] = [
                {
                    "policyId": "test-policy",
                    "policyType": "exactTrainablePair",
                    "autoApplyMode": "apply",
                    "sourcePattern": "get report",
                    "targetText": "git repo",
                    "inputStrictKey": "get report",
                    "targetStrictKey": "git repo",
                    "exactInputRequired": True,
                    "contextTokensAny": [],
                    "contextAliasesAny": [],
                    "reviewGateConflictRows": [],
                }
            ]
            candidate.write_text(json.dumps(new_model), encoding="utf-8")
            activate_args = Namespace(
                actor="test",
                model=candidate,
                active_model=active,
                base_model=active,
                evidence_store=evidence,
                replaylab_root=root / "missing-replaylab",
                backup_suffix="test",
                current_corpus_dir=root / "missing-current",
                reraw_corpus_dir=root / "missing-reraw",
                skip_corpus_replay=True,
                skip_raw_input_replay=True,
            )
            activated = control.activate_model_command(activate_args)
            self.assertFalse(activated.get("failed"))
            self.assertEqual(json.loads(active.read_text(encoding="utf-8"))["policyCounts"]["apply"], 1)
            rollback_args = Namespace(
                actor="test",
                active_model=active,
                backup=Path(activated["backup"]),
                list=False,
                reason="test rollback",
                evidence_store=evidence,
            )
            rolled_back = control.rollback_model_command(rollback_args)
            self.assertFalse(rolled_back.get("failed"))
            self.assertEqual(json.loads(active.read_text(encoding="utf-8"))["policyCounts"]["apply"], 0)
            actions = [event["action"] for event in control.load_events(evidence)]
            self.assertEqual(actions, ["activateModel", "rollbackModel"])


if __name__ == "__main__":
    unittest.main()
