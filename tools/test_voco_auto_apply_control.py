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

    def test_unlocked_replacement_rule_compiles_without_context_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.jsonl"
            base = root / "base.json"
            model_path = root / "compiled/full-db.auto-apply-model.json"
            base.write_text(json.dumps(tiny_base_model()), encoding="utf-8")
            args = Namespace(
                actor="test",
                source_pattern="A 三八零",
                target_text="A380",
                source_text="這一家的 A 三八零 很好。",
                row_pk=13560,
                rule_name="aircraft-model-a380",
                positive=[],
                negative=[],
                positive_text="這一家的 A 三八零 很好。",
                positive_context="",
                expected_text="這一家的 A380 很好。",
                negative_text="XA 三八零B 不應該改。",
                negative_context="",
                note=None,
            )
            control.append_event(evidence, control.replacement_rule_event(args))
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
            self.assertEqual(validation["positiveExamples"][0]["actualText"], "這一家的 A380 很好。")
            self.assertEqual(validation["negativeExamples"][0]["actualText"], "XA 三八零B 不應該改。")
            policy = model["policies"][0]
            self.assertTrue(policy["policyId"].startswith("manual-replacement-"))
            self.assertEqual(policy["policyType"], "scopedReplacement")
            self.assertFalse(policy["contextRequired"])
            self.assertEqual(policy["contextTokensAny"], [])
            self.assertEqual(policy["contextAliasesAny"], [])

    def test_readded_replacement_after_tombstone_validates_only_new_event_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.jsonl"
            base = root / "base.json"
            model_path = root / "compiled/full-db.auto-apply-model.json"
            base.write_text(json.dumps(tiny_base_model()), encoding="utf-8")
            old_args = Namespace(
                actor="test",
                source_pattern="A 三八零",
                target_text="A380",
                source_text="就有人把 A 三八零批評的一文不值。",
                row_pk=13557,
                rule_name="aircraft-model-a380",
                positive=[],
                negative=[],
                positive_text="就有人把 A 三八零批評的一文不值。",
                positive_context="",
                expected_text="bad expected text",
                negative_text=None,
                negative_context="",
                note=None,
            )
            new_args = Namespace(
                actor="test",
                source_pattern="A 三八零",
                target_text="A380",
                source_text="這架 A 三八零 很穩。",
                row_pk=13557,
                rule_name="aircraft-model-a380-v2",
                positive=[],
                negative=[],
                positive_text="這架 A 三八零 很穩。",
                positive_context="",
                expected_text="這架 A380 很穩。",
                negative_text=None,
                negative_context="",
                note=None,
            )
            control.append_event(evidence, control.replacement_rule_event(old_args))
            control.append_event(
                evidence,
                control.disable_rule_event(
                    Namespace(
                        actor="test",
                        policy_id=None,
                        source_pattern="A 三八零",
                        target_text="A380",
                        reason="Replace overlapping replacement example.",
                        disposition="replaced",
                    )
                ),
            )
            control.append_event(evidence, control.replacement_rule_event(new_args))
            events = control.load_events(evidence)
            model, _report = control.compile_model(
                control.load_model(base),
                events,
                base_model_path=base,
                evidence_store=evidence,
            )
            control.write_model(model_path, model)
            validation = control.validate_model(
                model,
                events,
                model_path=model_path,
                base_model=control.load_model(base),
                replaylab_root=root / "missing-replaylab",
                current_corpus_dir=root / "missing-current",
                reraw_corpus_dir=root / "missing-reraw",
                skip_corpus_replay=True,
                skip_raw_input_replay=True,
            )

            self.assertTrue(validation["ready"])
            self.assertEqual(len(validation["positiveExamples"]), 1)
            self.assertEqual(validation["positiveExamples"][0]["actualText"], "這架 A380 很穩。")
            policy = model["policies"][0]
            self.assertEqual(policy["controlEvidenceEventIds"], [events[-1]["eventId"]])
            self.assertEqual(policy["supersededControlEvidenceEventIds"], [events[0]["eventId"], events[1]["eventId"]])

    def test_unlocked_ascii_replacement_corpus_drift_is_accepted_with_replay_cap(self):
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
                source_pattern="A三五零",
                target_text="A350",
                source_text="如果A三五零的話，有貨機就好了。",
                row_pk=13553,
                rule_name="aircraft-model-a350",
                positive=[],
                negative=[],
                positive_text="如果A三五零的話，有貨機就好了。",
                positive_context="",
                expected_text="如果A350的話，有貨機就好了。",
                negative_text=None,
                negative_context="",
                note=None,
            )
            control.append_event(evidence, control.replacement_rule_event(args))
            events = control.load_events(evidence)
            model, _report = control.compile_model(
                control.load_model(base),
                events,
                base_model_path=base,
                evidence_store=evidence,
            )
            control.write_model(model_path, model)
            (corpus / "full-db.cleaned.jsonl").write_text(
                json.dumps(
                    {
                        "rowPk": 11646,
                        "rawOpenCC": "你剛才提到A三二零，那是不是說A三五零就沒有這個東西了？",
                        "cleanedText": "你剛才提到A三二零，那是不是說A三五零就沒有這個東西了？",
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
                events,
                model_path=model_path,
                base_model=control.load_model(base),
                replaylab_root=root / "missing-replaylab",
                current_corpus_dir=corpus,
                reraw_corpus_dir=root / "missing-reraw",
                skip_corpus_replay=False,
                skip_raw_input_replay=True,
            )

            self.assertTrue(validation["ready"])
            cleaned = validation["corpusReplay"][0]["cleanedReplay"]
            self.assertEqual(cleaned["acceptedManualCorpusChanges"], 1)
            self.assertEqual(cleaned["unexpectedChanges"], 0)

    def test_inherited_baseline_policy_fire_is_suppressed_from_raw_replay_failures(self):
        report = {
            "sentinelFailures": [],
            "unexpectedChanges": [
                {
                    "rowPk": 10802,
                    "before": "成都。",
                    "after": "程度。",
                    "cleanedText": "成都。",
                    "fires": [{"policyId": "temporary-replacement-guard-9779e48749170520"}],
                },
                {
                    "rowPk": 13553,
                    "before": "如果A三五零的話，有貨機就好了。",
                    "after": "如果A350的話，有貨機就好了。",
                    "cleanedText": "如果A三五零的話，有貨機就好了。",
                    "fires": [{"policyId": "manual-replacement-new"}],
                },
            ],
            "readiness": {"rawInputReplayPass": False, "reason": "raw input replay produced unexpected changes"},
        }
        base_model = tiny_base_model()
        base_model["policies"] = [
            {
                "policyId": "temporary-replacement-guard-9779e48749170520",
                "autoApplyMode": "apply",
            }
        ]

        control.suppress_inherited_baseline_policy_fires(report, base_model)

        self.assertEqual(len(report["inheritedBaselineUnexpectedChanges"]), 1)
        self.assertEqual(report["unexpectedChanges"][0]["rowPk"], 13553)
        self.assertFalse(report["readiness"]["rawInputReplayPass"])

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

    def test_policy_proposal_replacement_gate_fails_when_ranker_only_drops_active_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replaylab_root = root / "replaylab"
            artifact_dir = replaylab_root / "artifacts/probe"
            release_dir = artifact_dir / "proposal-release-gate-dry-run"
            corpus_dir = replaylab_root / "artifacts/corpus"
            output_dir = root / "replacement"
            release_dir.mkdir(parents=True)
            corpus_dir.mkdir(parents=True)
            for name in ["full-db.cleaned.jsonl", "full-db.raw.jsonl", "full-db.trainable-pairs.jsonl"]:
                (corpus_dir / name).write_text('{"rowPk": 1, "rawASR": "a", "rawOpenCC": "a", "cleanedText": "A"}\n', encoding="utf-8")

            active_model_path = root / "active/full-db.auto-apply-model.json"
            active_model_path.parent.mkdir()
            policy_a = proposal_policy("policy-a", "a", "A", 1)
            policy_b = proposal_policy("policy-b", "b", "B", 2)
            active_model = tiny_base_model()
            active_model["policies"] = [policy_a, policy_b]
            active_model["policyCounts"] = {"apply": 2}
            active_model["policyTypeCounts"] = {"exactTrainablePair": 2}
            control.write_model(active_model_path, active_model)

            safety_gate = {
                "schema": "voco.policy-proposal-safety-gate.v2",
                "input": {"activeCompiledModel": str(active_model_path), "corpusDir": str(corpus_dir)},
                "readiness": {"releaseReady": True, "productionRuntimeAllowed": False},
                "activeModelDiff": {"candidateCoversActiveApplyPolicies": True, "droppedActiveApplyPolicyCount": 0},
            }
            (release_dir / "proposal-safety-gate.report.json").write_text(json.dumps(safety_gate), encoding="utf-8")
            accepted = {
                "proposalId": "policy-a",
                "sourcePolicyId": "policy-a",
                "compileGateDecision": "accepted",
                "materializedPolicy": policy_a,
                "ranker": {"prediction": "apply"},
            }
            (release_dir / "proposals.accepted.jsonl").write_text(json.dumps(accepted, ensure_ascii=False) + "\n", encoding="utf-8")

            original_backend = control.load_replaylab_backend
            control.load_replaylab_backend = lambda _root: fake_replacement_backend()
            try:
                result = control.run_command(
                    Namespace(
                        command="evalProposalReplacementGate",
                        artifact_dir=artifact_dir,
                        output_dir=output_dir,
                        active_model=None,
                        skip_raw_input_replay=False,
                        replaylab_root=replaylab_root,
                    )
                )
            finally:
                control.load_replaylab_backend = original_backend

            self.assertTrue(result["failed"])
            self.assertEqual(result["schema"], "voco.policy-proposal-replacement-gate.v1")
            self.assertFalse(result["readiness"]["replacementReady"])
            self.assertFalse(result["readiness"]["productionRuntimeAllowed"])
            self.assertEqual(result["rankerOnlyVsActive"]["droppedActiveApplyPolicyCount"], 1)
            self.assertEqual(result["rankerOnlyVsActive"]["droppedActiveApplyPolicyIds"], ["policy-b"])
            self.assertFalse(result["rankerOnlyVsActive"]["candidateCoversActiveApplyPolicies"])
            self.assertEqual(result["cleanedReplayComparison"]["metrics"]["rowsMatchingCleanedText"]["delta"], -1)
            self.assertEqual(result["rawInputReplayComparison"]["metrics"]["candidateFireCount"]["delta"], -1)
            self.assertTrue(any(item["kind"] == "droppedActiveApplyPolicies" for item in result["readiness"]["blockers"]))
            self.assertFalse(result["runtimeBoundaryAudit"]["productionRuntimeAllowed"])
            self.assertTrue((output_dir / "proposal-replacement-gate.report.json").exists())
            self.assertTrue((output_dir / "full-db.auto-apply-model.json").exists())

    def test_ranker_only_candidate_cannot_activate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active.json"
            evidence = root / "evidence.jsonl"
            candidate = root / "ranker-only/full-db.auto-apply-model.json"
            active.write_text(json.dumps(tiny_base_model()), encoding="utf-8")
            model = proposal_candidate_model("ranker-only-predicted-apply")
            control.write_model(candidate, model)

            result = control.activate_model_command(activation_args(root, candidate, active, evidence))

            self.assertTrue(result["failed"])
            self.assertIn("ranker-only", result["activationGuard"]["reason"])
            self.assertFalse(result["activationGuard"]["productionRuntimeAllowed"])
            self.assertEqual(json.loads(active.read_text(encoding="utf-8"))["policyCounts"], {"apply": 0})

    def test_preserve_active_candidate_requires_approval_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active.json"
            evidence = root / "evidence.jsonl"
            candidate = root / "preserve/full-db.auto-apply-model.json"
            active.write_text(json.dumps(tiny_base_model()), encoding="utf-8")
            control.write_model(candidate, proposal_candidate_model("preserve-active"))

            result = control.activate_model_command(activation_args(root, candidate, active, evidence))

            self.assertTrue(result["failed"])
            self.assertIn("approval activation manifest", result["activationGuard"]["reason"])
            self.assertFalse(result["activationGuard"]["productionRuntimeAllowed"])

    def test_preserve_active_candidate_rejects_manifest_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active.json"
            evidence = root / "evidence.jsonl"
            candidate = root / "preserve/full-db.auto-apply-model.json"
            manifest = root / "activation.json"
            active.write_text(json.dumps(tiny_base_model()), encoding="utf-8")
            control.write_model(candidate, proposal_candidate_model("preserve-active"))
            write_activation_manifest(manifest, candidate, active, candidate_sha="not-the-real-sha")

            result = control.activate_model_command(
                activation_args(root, candidate, active, evidence, activation_manifest=manifest)
            )

            self.assertTrue(result["failed"])
            self.assertEqual(result["activationGuard"]["failures"][0]["field"], "candidateModelSha256")
            self.assertEqual(json.loads(active.read_text(encoding="utf-8"))["policyCounts"], {"apply": 0})

    def test_preserve_active_candidate_with_matching_manifest_can_activate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / "active.json"
            evidence = root / "evidence.jsonl"
            candidate = root / "preserve/full-db.auto-apply-model.json"
            manifest = root / "activation.json"
            active.write_text(json.dumps(tiny_base_model()), encoding="utf-8")
            control.write_model(candidate, proposal_candidate_model("preserve-active"))
            write_activation_manifest(manifest, candidate, active)
            candidate_sha = control.sha256_file(candidate)

            result = control.activate_model_command(
                activation_args(root, candidate, active, evidence, activation_manifest=manifest)
            )

            self.assertFalse(result.get("failed"))
            self.assertTrue(result["activationGuard"]["productionRuntimeAllowed"])
            self.assertEqual(result["activationGuard"]["approvedBy"], "Jason")
            self.assertEqual(control.sha256_file(candidate), candidate_sha)
            self.assertEqual(control.sha256_file(active), candidate_sha)
            self.assertEqual(json.loads(active.read_text(encoding="utf-8"))["policyCounts"], {"apply": 1})

    def test_preserve_active_manifest_accepts_replaylab_relative_candidate_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replaylab_root = root / "ReplayLab"
            active = root / "active.json"
            evidence = root / "evidence.jsonl"
            candidate = replaylab_root / "artifacts/probe/proposal-release-gate-dry-run/full-db.auto-apply-model.json"
            manifest = replaylab_root / "artifacts/probe/proposal-release-gate-dry-run/activation.json"
            active.write_text(json.dumps(tiny_base_model()), encoding="utf-8")
            control.write_model(candidate, proposal_candidate_model("preserve-active"))
            write_activation_manifest(
                manifest,
                candidate,
                active,
                candidate_path_value="artifacts/probe/proposal-release-gate-dry-run/full-db.auto-apply-model.json",
            )

            result = control.activate_model_command(
                activation_args(root, candidate, active, evidence, activation_manifest=manifest, replaylab_root=replaylab_root)
            )

            self.assertFalse(result.get("failed"))
            self.assertTrue(result["activationGuard"]["productionRuntimeAllowed"])

    def test_control_plane_compile_strips_proposal_activation_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_path = root / "proposal-base.json"
            active = root / "active.json"
            evidence = root / "evidence.jsonl"
            candidate = root / "compiled/full-db.auto-apply-model.json"
            base_model = proposal_candidate_model("preserve-active")
            base_model["intendedUse"] = "dry-run candidate only; do not install without separate ReplayLab and Jason approval"
            base_model["promotionPolicyGuard"] = {"blockedPolicies": [{"policyId": "stale-audit"}]}
            control.write_model(base_path, base_model)
            active.write_text(json.dumps(tiny_base_model()), encoding="utf-8")
            evidence.write_text("", encoding="utf-8")

            model, _report = control.compile_model(
                control.load_model(base_path),
                control.load_events(evidence),
                base_model_path=base_path,
                evidence_store=evidence,
            )
            control.write_model(candidate, model)

            self.assertEqual(model["modelType"], "control_plane_patched_auto_apply_model")
            self.assertNotIn("proposalSafetyGate", model)
            self.assertNotIn("promotionPolicyGuard", model)
            self.assertNotIn("replayReadiness", model)
            self.assertNotIn("sourceActiveModelGeneratedAt", model)
            self.assertNotIn("dry-run", model.get("intendedUse", ""))

            result = control.activate_model_command(activation_args(root, candidate, active, evidence))

            self.assertFalse(result.get("failed"))
            self.assertEqual(result["activationGuard"]["reason"], "standard compiled Voco model activation")


def proposal_policy(policy_id: str, source: str, target: str, row_pk: int) -> dict:
    return {
        "policyId": policy_id,
        "autoApplyMode": "apply",
        "policyType": "exactTrainablePair",
        "exactInputRequired": True,
        "inputStrictKey": control.strict_text_key(source),
        "targetStrictKey": control.strict_text_key(target),
        "sourcePattern": source,
        "targetText": target,
        "contextTokensAny": [],
        "contextAliasesAny": [],
        "contextRequired": False,
        "contextFromContextOnly": False,
        "requireAlias": False,
        "evidenceRows": [row_pk],
        "trainableRows": [row_pk],
    }


def proposal_candidate_model(strategy: str) -> dict:
    model = tiny_base_model()
    model["modelType"] = "voco-policy-proposal-candidate"
    model["proposalSafetyGate"] = {
        "schema": "voco.policy-proposal-safety-gate.v2",
        "candidateStrategy": strategy,
        "releaseReady": strategy == "preserve-active",
        "productionRuntimeAllowed": False,
    }
    model["policies"] = [proposal_policy("policy-a", "a", "A", 1)]
    model["policyCounts"] = {"apply": 1}
    model["policyTypeCounts"] = {"exactTrainablePair": 1}
    if strategy == "ranker-only-predicted-apply":
        model["proposalReplacementGate"] = {
            "candidateStrategy": "ranker-only-predicted-apply",
            "productionRuntimeAllowed": False,
        }
    return model


def activation_args(
    root: Path,
    candidate: Path,
    active: Path,
    evidence: Path,
    *,
    activation_manifest: Path | None = None,
    replaylab_root: Path | None = None,
) -> Namespace:
    return Namespace(
        actor="test",
        model=candidate,
        active_model=active,
        base_model=active,
        evidence_store=evidence,
        replaylab_root=replaylab_root or root / "missing-replaylab",
        backup_suffix="test",
        backup_dir=None,
        backup_retention=3,
        activation_manifest=activation_manifest,
        current_corpus_dir=root / "missing-current",
        reraw_corpus_dir=root / "missing-reraw",
        skip_corpus_replay=True,
        skip_raw_input_replay=True,
    )


def write_activation_manifest(
    path: Path,
    candidate: Path,
    active: Path,
    *,
    candidate_sha: str | None = None,
    candidate_path_value: str | None = None,
) -> None:
    manifest = {
        "schema": "voco.policy-proposal-runtime-activation.v1",
        "artifactId": "test-proposal-artifact",
        "replaylabCommit": "0828c7b",
        "candidateStrategy": "preserve-active",
        "candidateModelPath": candidate_path_value or str(candidate),
        "candidateModelSha256": candidate_sha or control.sha256_file(candidate),
        "sourceActiveModelPath": str(active),
        "sourceActiveModelSha256": control.sha256_file(active),
        "safetyGateReportPath": "proposal-release-gate-dry-run/proposal-safety-gate.report.json",
        "safetyGateReportSha256": "fixture",
        "runtimeActivationEligible": True,
        "requiresJasonApproval": True,
        "approvedBy": "Jason",
        "approvedAt": "2026-06-13T16:38:04+08:00",
        "approvalToken": "jason-approved-fixture",
        "allowedActivationCommand": "voco_auto_apply_control.py activateModel --model full-db.auto-apply-model.json",
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fake_replacement_backend() -> dict:
    class AutoApply:
        @staticmethod
        def replay_model(_records, model):
            apply_count = int((model.get("policyCounts") or {}).get("apply") or 0)
            row_results = [
                {"rowPk": row_pk, "matchesCleaned": True, "fires": [{"policyId": f"policy-{name}"}]}
                for row_pk, name in [(1, "a"), (2, "b")][:apply_count]
            ]
            return {
                "rowCount": 2,
                "applyPolicyCount": apply_count,
                "candidateFireCount": apply_count,
                "rowFireCount": apply_count,
                "changedRows": apply_count,
                "rowsMatchingCleanedText": apply_count,
                "unexpectedChanges": [],
                "sentinelFailures": [],
                "rowResults": row_results,
                "readiness": {"autoApplyModelReady": True},
            }

    class RawEval:
        @staticmethod
        def evaluate_raw_input(_raw_path, _cleaned_path, _trainable_path, model_path):
            model = control.load_model(Path(model_path))
            apply_count = int((model.get("policyCounts") or {}).get("apply") or 0)
            row_results = [
                {"rowPk": row_pk, "matchesCleaned": True, "fires": [{"policyId": f"policy-{name}"}]}
                for row_pk, name in [(1, "a"), (2, "b")][:apply_count]
            ]
            return {
                "rowCount": 2,
                "rawAsrRowCount": 2,
                "applyPolicyCount": apply_count,
                "candidateFireCount": apply_count,
                "rowFireCount": apply_count,
                "changedRows": apply_count,
                "rowsMatchingCleanedText": apply_count,
                "unexpectedChanges": [],
                "sentinelFailures": [],
                "rowResults": row_results,
                "readiness": {"rawInputReplayPass": True},
            }

    return {"auto_apply": AutoApply, "raw_eval": RawEval}


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
