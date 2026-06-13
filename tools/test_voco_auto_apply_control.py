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

    def test_protected_mingde_guard_matches_runtime_policy_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "full-db.auto-apply-model.json"
            model = tiny_base_model()
            model["policyCounts"] = {"apply": 1}
            model["policyTypeCounts"] = {"scopedReplacement": 1}
            model["protectedTermAllowlistGuards"] = [
                {
                    "guardId": "protected-term-allowlist.mingde",
                    "reason": control.PROTECTED_TERM_GUARD_REASON,
                    "term": "明德",
                    "allowedPhrases": ["明德捷運站", "明德水庫", "明德路", "施明德"],
                }
            ]
            model["policies"] = [
                {
                    "policyId": "scoped-fixture-mingde-recent-change",
                    "policyType": "scopedReplacement",
                    "autoApplyMode": "apply",
                    "sourcePattern": "最明德變更",
                    "targetText": "最近的變更",
                    "contextTokensAny": ["變更", "自動學習", "昨天晚上", "最近"],
                    "contextAliasesAny": [],
                    "contextRequired": True,
                    "sourceSlices": ["controlEvidence"],
                }
            ]
            control.write_model(model_path, model)

            allowed = control.explain_rule_match(model_path, "明德捷運站。", "")
            self.assertFalse(allowed["blocked"])
            self.assertEqual(allowed["outputText"], "明德捷運站。")
            self.assertEqual(allowed["guardBlocks"], [])

            scoped = control.explain_rule_match(model_path, "我們最明德變更應該有加了自動學習。", "")
            self.assertFalse(scoped["blocked"])
            self.assertEqual(scoped["outputText"], "我們最近的變更應該有加了自動學習。")
            self.assertEqual([fire["policyId"] for fire in scoped["applied"]], ["scoped-fixture-mingde-recent-change"])
            self.assertEqual(scoped["guardBlocks"], [])

            blocked = control.explain_rule_match(model_path, "這個明德變更怪怪的。", "")
            self.assertTrue(blocked["blocked"])
            self.assertEqual(blocked["outputText"], "這個明德變更怪怪的。")
            self.assertEqual(blocked["applied"], [])
            self.assertEqual(blocked["guardBlocks"][0]["guardId"], "protected-term-allowlist.mingde")
            self.assertEqual(blocked["guardBlocks"][0]["reason"], control.PROTECTED_TERM_GUARD_REASON)

            replay = control.local_corpus_replay(
                [
                    {
                        "rowPk": 12762,
                        "rawOpenCC": "這個明德變更怪怪的。",
                        "cleanedText": "這個明德變更怪怪的。",
                    }
                ],
                model,
            )
            self.assertEqual(replay["guardBlockedRows"], 1)
            self.assertEqual(replay["rowResults"][0]["guardBlocks"][0]["guardId"], "protected-term-allowlist.mingde")

    def test_protected_mingde_guard_is_model_declared_not_cli_global(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "full-db.auto-apply-model.json"
            model = tiny_base_model()
            control.write_model(model_path, model)

            result = control.explain_rule_match(model_path, "這個明德變更怪怪的。", "")
            self.assertFalse(result["blocked"])
            self.assertEqual(result["guardBlocks"], [])
            self.assertEqual(result["outputText"], "這個明德變更怪怪的。")

    def test_upsert_protected_term_allowlist_guard_writes_model_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "full-db.auto-apply-model.json"
            evidence = root / "evidence.jsonl"
            control.write_model(model_path, tiny_base_model())

            result = control.upsert_protected_term_allowlist_guard_command(
                Namespace(
                    actor="test",
                    evidence_store=evidence,
                    model=model_path,
                    guard_id="protected-term-allowlist.mingde",
                    term="明德",
                    allowed_phrase=["明德路", "明德捷運站", "施明德", "明德水庫"],
                    reason=control.PROTECTED_TERM_GUARD_REASON,
                    backup_suffix="test",
                )
            )

            self.assertEqual(result["guardCount"], 1)
            updated = control.load_model(model_path)
            self.assertEqual(updated["protectedTermAllowlistGuards"][0]["term"], "明德")
            self.assertEqual(updated["protectedTermAllowlistGuards"][0]["allowedPhrases"], ["明德路", "明德捷運站", "施明德", "明德水庫"])
            self.assertIn("protected term allowlist guards must be declared in the model artifact", "\n".join(updated["safetyContract"]))
            self.assertIsNone(result["backup"])
            self.assertEqual(result["backupMode"], "none")
            self.assertEqual(list(root.glob("full-db.auto-apply-model.json.bak-*")), [])
            self.assertEqual(len(control.load_events(evidence)), 1)

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

    def test_manual_exact_same_row_silver_target_change_is_accepted(self):
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
                source_text="平日要完善了。",
                target_text="平日要晚上了。",
                row_pk=10574,
                context="",
                note=None,
            )
            control.append_event(evidence, control.correction_event(args))
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
                        "rowPk": 10574,
                        "rawOpenCC": "平日要完善了。",
                        "cleanedText": "平日要完善了。",
                        "requiresReview": False,
                        "riskFlags": [],
                        "context": {"before": []},
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

    def test_unrelated_manual_exact_same_text_on_another_row_still_fails(self):
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
                source_text="平日要完善了。",
                target_text="平日要晚上了。",
                row_pk=10574,
                context="",
                note=None,
            )
            control.append_event(evidence, control.correction_event(args))
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
                        "rowPk": 10575,
                        "rawOpenCC": "平日要完善了。",
                        "cleanedText": "平日要完善了。",
                        "requiresReview": False,
                        "riskFlags": [],
                        "context": {"before": []},
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

            self.assertFalse(validation["ready"])
            self.assertEqual(validation["corpusReplay"][0]["cleanedReplay"]["unexpectedChanges"], 1)
            self.assertEqual(validation["corpusReplay"][0]["cleanedReplay"]["acceptedManualCorpusChanges"], 0)
            self.assertEqual(validation["failures"][0]["kind"], "unexpectedCorpusChanges")

    def test_scoped_replacement_overreach_on_non_review_row_still_fails(self):
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
                        "requiresReview": False,
                        "riskFlags": [],
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

            self.assertFalse(validation["ready"])
            self.assertEqual(validation["corpusReplay"][0]["cleanedReplay"]["unexpectedChanges"], 1)
            self.assertEqual(validation["corpusReplay"][0]["cleanedReplay"]["acceptedManualCorpusChanges"], 0)
            self.assertEqual(validation["failures"][0]["kind"], "unexpectedCorpusChanges")

    def test_manual_context_review_risk_corpus_drift_remains_allowed(self):
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

    def test_exact_conflict_requires_replacing_old_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.json"
            old_source = "平日要完善了。"
            old_policy_id = "exact-pair-512819eb3c3a5e4c"
            base_model = tiny_base_model()
            base_model["policyCounts"] = {"apply": 1}
            base_model["policyTypeCounts"] = {"exactTrainablePair": 1}
            base_model["policies"] = [
                {
                    "policyId": old_policy_id,
                    "policyType": "exactTrainablePair",
                    "autoApplyMode": "apply",
                    "sourcePattern": old_source,
                    "targetText": old_source,
                    "inputStrictKey": control.strict_text_key(old_source),
                    "targetStrictKey": control.strict_text_key(old_source),
                    "exactInputRequired": True,
                    "contextTokensAny": [],
                    "contextAliasesAny": [],
                }
            ]
            base.write_text(json.dumps(base_model, ensure_ascii=False), encoding="utf-8")

            conflict_evidence = root / "conflict/evidence.jsonl"
            control.append_event(
                conflict_evidence,
                control.correction_event(
                    Namespace(
                        actor="test",
                        source_text=old_source,
                        target_text="平日要晚上了。",
                        row_pk=10574,
                        context="",
                        note=None,
                    )
                ),
            )
            conflict_model, _report = control.compile_model(
                control.load_model(base),
                control.load_events(conflict_evidence),
                base_model_path=base,
                evidence_store=conflict_evidence,
            )
            conflict_validation = control.validate_model(
                conflict_model,
                control.load_events(conflict_evidence),
                model_path=root / "conflict/model.json",
                base_model=control.load_model(base),
                replaylab_root=root / "missing-replaylab",
                current_corpus_dir=root / "missing-current",
                reraw_corpus_dir=root / "missing-reraw",
                skip_corpus_replay=True,
                skip_raw_input_replay=True,
            )
            self.assertFalse(conflict_validation["ready"])
            self.assertEqual(len(conflict_validation["exactApplyConflicts"]), 1)

            replaced_evidence = root / "replaced/evidence.jsonl"
            control.append_event(
                replaced_evidence,
                control.disable_rule_event(
                    Namespace(
                        actor="test",
                        policy_id=old_policy_id,
                        source_pattern=None,
                        target_text=None,
                        reason="Replace stale silver target with Jason confirmed manual exact correction.",
                        disposition="replaced",
                    )
                ),
            )
            control.append_event(
                replaced_evidence,
                control.correction_event(
                    Namespace(
                        actor="test",
                        source_text=old_source,
                        target_text="平日要晚上了。",
                        row_pk=10574,
                        context="",
                        note=None,
                    )
                ),
            )
            replaced_model, _report = control.compile_model(
                control.load_model(base),
                control.load_events(replaced_evidence),
                base_model_path=base,
                evidence_store=replaced_evidence,
            )
            replaced_validation = control.validate_model(
                replaced_model,
                control.load_events(replaced_evidence),
                model_path=root / "replaced/model.json",
                base_model=control.load_model(base),
                replaylab_root=root / "missing-replaylab",
                current_corpus_dir=root / "missing-current",
                reraw_corpus_dir=root / "missing-reraw",
                skip_corpus_replay=True,
                skip_raw_input_replay=True,
            )
            self.assertTrue(replaced_validation["ready"])
            self.assertEqual(len(replaced_validation["exactApplyConflicts"]), 0)
            self.assertEqual(replaced_model["policyCounts"]["replaced"], 1)

    def test_activate_defaults_to_no_backup_and_rollback_requires_explicit_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.jsonl"
            active = root / "full-db.auto-apply-model.json"
            candidate = root / "candidate.json"
            active.write_text(json.dumps(tiny_base_model()), encoding="utf-8")
            explicit_backup = root / "explicit-active-backup.json"
            explicit_backup.write_text(active.read_text(encoding="utf-8"), encoding="utf-8")
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
                backup_dir=None,
                backup_retention=3,
                current_corpus_dir=root / "missing-current",
                reraw_corpus_dir=root / "missing-reraw",
                skip_corpus_replay=True,
                skip_raw_input_replay=True,
            )
            activated = control.activate_model_command(activate_args)
            self.assertFalse(activated.get("failed"))
            self.assertIsNone(activated["backup"])
            self.assertEqual(activated["backupMode"], "none")
            self.assertEqual(list(root.glob("full-db.auto-apply-model.json.bak-*")), [])
            self.assertEqual(json.loads(active.read_text(encoding="utf-8"))["policyCounts"]["apply"], 1)

            implicit_rollback_args = Namespace(
                actor="test",
                active_model=active,
                backup=None,
                backup_dir=None,
                list=False,
                reason="implicit rollback should fail",
                evidence_store=evidence,
                pre_rollback_backup_dir=None,
                pre_rollback_backup_retention=3,
            )
            implicit_rollback = control.rollback_model_command(implicit_rollback_args)
            self.assertTrue(implicit_rollback.get("failed"))
            self.assertIn("explicit --backup", implicit_rollback["reason"])

            rollback_args = Namespace(
                actor="test",
                active_model=active,
                backup=explicit_backup,
                backup_dir=None,
                list=False,
                reason="test rollback",
                evidence_store=evidence,
                pre_rollback_backup_dir=None,
                pre_rollback_backup_retention=3,
            )
            rolled_back = control.rollback_model_command(rollback_args)
            self.assertFalse(rolled_back.get("failed"))
            self.assertIsNone(rolled_back["preRollbackBackup"])
            self.assertEqual(json.loads(active.read_text(encoding="utf-8"))["policyCounts"]["apply"], 0)
            actions = [event["action"] for event in control.load_events(evidence)]
            self.assertEqual(actions, ["activateModel", "rollbackModel"])

    def test_activate_backup_dir_uses_retention_without_app_support_bak_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.jsonl"
            active = root / "full-db.auto-apply-model.json"
            backup_dir = root / "voco-active-model-backups"
            active.write_text(json.dumps(tiny_base_model()), encoding="utf-8")

            for index in range(5):
                candidate = root / f"candidate-{index}.json"
                model = tiny_base_model()
                model["policies"] = [
                    {
                        "policyId": f"test-policy-{index}",
                        "policyType": "exactTrainablePair",
                        "autoApplyMode": "apply",
                        "sourcePattern": f"get report {index}",
                        "targetText": f"git repo {index}",
                        "inputStrictKey": f"get report {index}",
                        "targetStrictKey": f"git repo {index}",
                        "exactInputRequired": True,
                        "contextTokensAny": [],
                        "contextAliasesAny": [],
                        "reviewGateConflictRows": [],
                    }
                ]
                control.write_model(candidate, model)
                activated = control.activate_model_command(
                    Namespace(
                        actor="test",
                        model=candidate,
                        active_model=active,
                        base_model=active,
                        evidence_store=evidence,
                        replaylab_root=root / "missing-replaylab",
                        backup_suffix="test",
                        backup_dir=backup_dir,
                        backup_retention=3,
                        current_corpus_dir=root / "missing-current",
                        reraw_corpus_dir=root / "missing-reraw",
                        skip_corpus_replay=True,
                        skip_raw_input_replay=True,
                    )
                )
                self.assertFalse(activated.get("failed"))
                self.assertEqual(activated["backupMode"], "directory")
                self.assertEqual(activated["backupDirectory"], str(backup_dir))

            self.assertEqual(list(root.glob("full-db.auto-apply-model.json.bak-*")), [])
            backups = sorted(backup_dir.glob("full-db.auto-apply-model.json.bak-*"))
            self.assertEqual(len(backups), 3)

            listed = control.rollback_model_command(
                Namespace(
                    actor="test",
                    active_model=active,
                    backup=None,
                    backup_dir=backup_dir,
                    list=True,
                    reason="list backups",
                    evidence_store=evidence,
                    pre_rollback_backup_dir=None,
                    pre_rollback_backup_retention=3,
                )
            )
            self.assertEqual(len(listed["backups"]), 3)

    def test_policy_proposal_ranker_artifact_is_shadow_contract_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = write_policy_proposal_ranker_fixture(root)

            result = control.run_command(
                Namespace(command="inspectProposalArtifact", artifact_dir=artifact_dir)
            )

            self.assertFalse(result["failed"])
            self.assertEqual(result["role"], "shadow/proposal contract fixture")
            self.assertFalse(result["productionRuntimeAllowed"])
            self.assertEqual(result["runtimeModelFileName"], "full-db.auto-apply-model.json")
            self.assertEqual(result["proposalCount"], 4898)
            self.assertEqual(result["decisionCounts"], {"apply": 4550, "block": 93, "abstain": 255})
            self.assertEqual(result["unsafeApplyFalsePositiveCounts"], {"valid": 0, "test": 0})
            self.assertTrue(any("full-db.auto-apply-model.json" in item for item in result["safetyBoundary"]))
            self.assertTrue(any("proposal" in item.lower() and "apply" in item.lower() for item in result["safetyBoundary"]))
            self.assertTrue(any("replay" in item.lower() and "compiled" in item.lower() for item in result["safetyBoundary"]))
            safety_gate = result["proposalSafetyGate"]
            self.assertIsNotNone(safety_gate)
            self.assertEqual(safety_gate["schema"], "voco.policy-proposal-safety-gate.v2")
            self.assertEqual(safety_gate["proposalCount"], 4898)
            self.assertEqual(safety_gate["predictedApplyCount"], 4524)
            self.assertEqual(safety_gate["acceptedForCompileCount"], 4524)
            self.assertEqual(safety_gate["unsafeApplyFalsePositiveCount"], 0)
            self.assertEqual(safety_gate["applyMissCount"], 26)
            self.assertTrue(safety_gate["candidateReplayPass"])
            self.assertTrue(safety_gate["rawInputReplayPass"])
            self.assertEqual(safety_gate["candidateUnexpectedChanges"], 0)
            self.assertEqual(safety_gate["rawUnexpectedChanges"], 0)
            self.assertEqual(safety_gate["candidateInheritedBaselineUnexpectedChanges"], 1)
            self.assertEqual(safety_gate["rawInheritedBaselineUnexpectedChanges"], 1)
            self.assertEqual(safety_gate["candidateAcceptedManualCorpusChanges"], 0)
            self.assertEqual(safety_gate["rawAcceptedManualCorpusChanges"], 0)
            self.assertTrue(safety_gate["dryRunSafetyGatePass"])
            self.assertFalse(safety_gate["productionRuntimeAllowed"])
            self.assertTrue(safety_gate["releaseReady"])
            self.assertEqual(safety_gate["policyCountDelta"], {"apply": 0, "blocked": 0, "replaced": 0})
            self.assertEqual(safety_gate["addedPolicyCount"], 0)
            self.assertEqual(safety_gate["changedPolicyCount"], 0)
            self.assertEqual(safety_gate["droppedActiveApplyPolicyCount"], 0)
            self.assertEqual(safety_gate["droppedActiveApplyPolicyIds"], [])
            self.assertTrue(safety_gate["candidateCoversActiveApplyPolicies"])
            self.assertTrue(safety_gate["candidateIsSubsetOfActive"])
            self.assertEqual(safety_gate["blockers"], [])
            self.assertIn("dry-run candidate is not an install approval", safety_gate["warnings"])
            self.assertFalse(safety_gate["runtimeBoundaryAudit"]["joblibActivationAllowed"])
            self.assertFalse(safety_gate["runtimeBoundaryAudit"]["installOrActivateCommandEmitted"])
            self.assertFalse(safety_gate["runtimeBoundaryAudit"]["rankerModelIsRuntimeModel"])

    def test_ranker_joblib_cannot_be_loaded_or_activated_as_runtime_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = write_policy_proposal_ranker_fixture(root)
            ranker = artifact_dir / "proposal-ranker-model.joblib"
            evidence = root / "evidence.jsonl"
            active = root / "full-db.auto-apply-model.json"
            active.write_text(json.dumps(tiny_base_model()), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not compiled Voco runtime models"):
                control.load_model(ranker)

            with self.assertRaisesRegex(ValueError, "not compiled Voco runtime models"):
                control.activate_model_command(
                    Namespace(
                        actor="test",
                        model=ranker,
                        active_model=active,
                        base_model=active,
                        evidence_store=evidence,
                        replaylab_root=root / "missing-replaylab",
                        backup_suffix="test",
                        backup_dir=None,
                        backup_retention=3,
                        current_corpus_dir=root / "missing-current",
                        reraw_corpus_dir=root / "missing-reraw",
                        skip_corpus_replay=True,
                        skip_raw_input_replay=True,
                    )
                )

            self.assertEqual(json.loads(active.read_text(encoding="utf-8"))["policyCounts"], {"apply": 0})
            self.assertEqual(control.load_events(evidence), [])


def write_policy_proposal_ranker_fixture(root: Path) -> Path:
    artifact_dir = root / "artifacts/policy-proposal-model-20260613-active-122458"
    artifact_dir.mkdir(parents=True)
    manifest = {
        "datasetType": "post-asr-policy-proposal-decision",
        "intendedUse": "train/evaluate a proposal classifier or ranker; not a Voco runtime model",
        "counts": {
            "proposals": 4898,
            "decisions": {"apply": 4550, "block": 93, "abstain": 255},
            "splits": {"train": 3938, "valid": 483, "test": 477},
        },
        "files": {
            "all": "artifacts/policy-proposal-model-20260613-active-122458/proposal-all.jsonl",
            "train": "artifacts/policy-proposal-model-20260613-active-122458/proposal-train.jsonl",
            "valid": "artifacts/policy-proposal-model-20260613-active-122458/proposal-valid.jsonl",
            "test": "artifacts/policy-proposal-model-20260613-active-122458/proposal-test.jsonl",
        },
        "mergedModel": "artifacts/active-auto-apply-model-snapshots/20260613-122458-current-active-after-13168-cloi-cli/full-db.auto-apply-model.json",
        "safetyBoundary": [
            "Rows are training/evaluation examples for proposal decisions only.",
            "Voco runtime must continue to load compiled full-db.auto-apply-model.json, not model outputs.",
            "A generated proposal must pass replay gates before it can be compiled into runtime JSON.",
        ],
    }
    report = {
        "applyThreshold": 0.6,
        "datasetDir": "artifacts/policy-proposal-model-20260613-active-122458",
        "intendedUse": "rank/classify post-ASR policy proposals before replay; not a Voco runtime auto-apply model",
        "labels": ["apply", "suggest", "block", "abstain"],
        "modelType": "tfidf-charword-logistic-regression-policy-proposal-ranker",
        "safetyBoundary": [
            "Predicted apply is only a proposal decision.",
            "A generated proposal must pass ReplayLab gates before it is compiled into runtime JSON.",
            "The current dataset has only three suggest examples, all in train; suggest metrics are not meaningful yet.",
        ],
        "valid": {"rows": 483, "unsafeApplyFalsePositiveCount": 0, "applyMissCount": 6},
        "test": {"rows": 477, "unsafeApplyFalsePositiveCount": 0, "applyMissCount": 2},
    }
    (artifact_dir / "dataset-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (artifact_dir / "proposal-ranker-report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    (artifact_dir / "proposal-ranker-model.joblib").write_bytes(b"\x80\x04proposal-ranker-fixture")
    safety_gate_dir = artifact_dir / "proposal-release-gate-dry-run"
    safety_gate_dir.mkdir()
    safety_gate = {
        "schema": "voco.policy-proposal-safety-gate.v2",
        "rankerGate": {
            "proposalCount": 4898,
            "predictedApplyCount": 4524,
            "acceptedForCompileCount": 4524,
            "unsafeApplyFalsePositiveCount": 0,
            "applyMissCount": 26,
        },
        "candidateReplay": {
            "applyPolicyCount": 4524,
            "candidateFireCount": 560,
            "changedRows": 140,
            "sentinelFailures": [],
            "unexpectedChanges": [],
            "inheritedBaselineUnexpectedChanges": [{"rowPk": 12291}],
            "acceptedManualCorpusChanges": [],
            "readiness": {"autoApplyModelReady": True},
        },
        "rawInputReplay": {
            "applyPolicyCount": 4524,
            "candidateFireCount": 560,
            "changedRows": 140,
            "sentinelFailures": [],
            "unexpectedChanges": [],
            "inheritedBaselineUnexpectedChanges": [{"rowPk": 12291}],
            "acceptedManualCorpusChanges": [],
            "readiness": {"rawInputReplayPass": True},
        },
        "activeModelDiff": {
            "activePolicyCounts": {"apply": 4550, "blocked": 1, "replaced": 17},
            "candidatePolicyCounts": {"apply": 4550, "blocked": 1, "replaced": 17},
            "policyCountDelta": {"apply": 0, "blocked": 0, "replaced": 0},
            "addedPolicyCount": 0,
            "removedPolicyCount": 0,
            "changedPolicyCount": 0,
            "candidateCoversActiveApplyPolicies": True,
            "droppedActiveApplyPolicyCount": 0,
            "droppedActiveApplyPolicyIds": [],
            "candidateIsSubsetOfActive": True,
        },
        "readiness": {
            "dryRunSafetyGatePass": True,
            "productionRuntimeAllowed": False,
            "releaseReady": True,
            "blockers": [],
            "warnings": [
                "dry-run candidate is not an install approval",
                "ranker artifact is evaluated only as proposal/shadow fixture",
                "suggest has no valid/test support; do not treat suggest as a release signal",
            ],
        },
        "runtimeBoundaryAudit": {
            "candidateModelFilename": "full-db.auto-apply-model.json",
            "candidateModelFilenameAllowed": True,
            "installOrActivateCommandEmitted": False,
            "joblibActivationAllowed": False,
            "rankerModelIsRuntimeModel": False,
            "productionRuntimeAllowed": False,
        },
    }
    (safety_gate_dir / "proposal-safety-gate.report.json").write_text(json.dumps(safety_gate, ensure_ascii=False), encoding="utf-8")
    return artifact_dir


if __name__ == "__main__":
    unittest.main()
