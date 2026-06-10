#!/usr/bin/env python3
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


candidates = load_module("voco_shadow_candidates", ROOT / "voco_shadow_candidates.py")
replay = load_module("voco_shadow_replay", ROOT / "voco_shadow_replay.py")


class Phase2AShadowCandidateTests(unittest.TestCase):
    def test_69_lun_is_preserved_and_negative_blocked(self):
        event = pipeline_event(
            raw="69 輪",
            final="69 輪",
            llm="69轮",
            length_bucket="1_4",
        )

        enriched = candidates.enrich_event(event)
        shadow_candidates = enriched["shadowCandidates"]

        self.assertEqual(shadow_candidates[0]["source"], "raw")
        self.assertEqual(shadow_candidates[0]["text"], "69 輪")
        self.assertEqual({candidate["source"] for candidate in shadow_candidates}, {"raw"})
        self.assertTrue(all(candidate["text"] == "69 輪" for candidate in shadow_candidates))
        self.assertTrue(all(candidate["blockedBecauseNegativeEvidence"] for candidate in shadow_candidates))
        self.assertFalse(enriched["safety"]["autoApplied"])
        self.assertFalse(enriched["safety"]["wouldHaveChangedFinalOutput"])

    def test_llm_only_candidate_can_appear_but_is_not_trusted(self):
        event = pipeline_event(
            raw="Ripple内的Markdown。",
            final="repo 內的 Markdown。",
            llm="repo 內的 Markdown。",
            route="reviewSuggested",
            user_target="repo 內的 Markdown。",
            user_source="reviewCandidate",
        )

        enriched = candidates.enrich_event(event)
        llm_candidates = [
            candidate for candidate in enriched["shadowCandidates"]
            if candidate["source"] == "llm"
        ]

        self.assertTrue(llm_candidates)
        self.assertTrue(all(candidate["blockedBecauseLlmOnly"] for candidate in llm_candidates))
        confirmed = next(candidate for candidate in enriched["shadowCandidates"] if candidate["source"] == "confirmedExact")
        self.assertTrue(all(candidate["rank"] > confirmed["rank"] for candidate in llm_candidates))
        self.assertTrue(all(candidate["score"] < confirmed["score"] for candidate in llm_candidates))
        self.assertNotEqual(replay.top_trusted_candidate(enriched)["source"], "llm")
        self.assertFalse(enriched["safety"]["autoApplied"])
        self.assertFalse(enriched["safety"]["wouldHaveChangedFinalOutput"])

    def test_direct_insertion_final_is_not_confirmed_exact(self):
        event = pipeline_event(
            raw="然后，你可以先做手机。",
            final="然後，你可以先做手機。",
            llm=None,
        )

        enriched = candidates.enrich_event(event)
        sources = {candidate["source"] for candidate in enriched["shadowCandidates"]}

        self.assertNotIn("confirmedExact", sources)
        self.assertIn("zhPhonetic", sources)
        self.assertTrue(
            all(candidate["requiresReview"] for candidate in enriched["shadowCandidates"] if candidate["source"] != "raw")
        )
        zh_candidate = next(candidate for candidate in enriched["shadowCandidates"] if candidate["source"] == "zhPhonetic")
        self.assertTrue(zh_candidate["blockedBecauseShortPhraseRisk"])
        self.assertFalse(enriched["safety"]["autoApplied"])
        self.assertFalse(enriched["safety"]["wouldHaveChangedFinalOutput"])

    def test_short_phrase_wider_context_candidate_stays_review_only(self):
        event = pipeline_event(
            raw="做手机。",
            final="做手機。",
            llm=None,
            length_bucket="1_4",
            ui_context={
                "recentContextCandidate": "做收集。",
                "contextWindowRawBefore": [
                    "请先把你看不出来的原来的句子列出来，我跟你说原来的句子是什么意思。",
                    "不用重跑，重新辨识。",
                ],
                "contextWindowRawAfter": ["增加 Markdown。这样，我们过几天回来看结果的时候，你才知道。"],
            },
        )

        enriched = candidates.enrich_event(event)
        context_candidate = next(
            candidate for candidate in enriched["shadowCandidates"]
            if candidate["source"] == "recentContext"
        )

        self.assertEqual(context_candidate["text"], "做收集。")
        self.assertEqual(context_candidate["evidenceTier"], "T1_WEAK_INTERACTION")
        self.assertTrue(context_candidate["blockedBecauseShortPhraseRisk"])
        self.assertTrue(context_candidate["requiresReview"])
        self.assertNotIn("confirmedExact", {candidate["source"] for candidate in enriched["shadowCandidates"]})
        self.assertFalse(enriched["safety"]["autoApplied"])
        self.assertFalse(enriched["safety"]["wouldHaveChangedFinalOutput"])

    def test_cross_script_label_requires_combined_script_and_phonetic_change(self):
        mixed = candidates.enrich_event(
            pipeline_event(
                raw="Ripple内的Markdown。",
                final="repo 內的 Markdown。",
                llm=None,
            )
        )
        mixed_sources = {candidate["source"] for candidate in mixed["shadowCandidates"]}
        self.assertIn("zhPhonetic", mixed_sources)
        self.assertIn("enPhonetic", mixed_sources)
        self.assertIn("crossScript", mixed_sources)
        self.assertNotIn("domainLexicon", mixed_sources)

        zh_only = candidates.enrich_event(
            pipeline_event(
                raw="然后，你可以先做手机。",
                final="然後，你可以先做手機。",
                llm=None,
            )
        )
        zh_only_sources = {candidate["source"] for candidate in zh_only["shadowCandidates"]}
        self.assertIn("zhPhonetic", zh_only_sources)
        self.assertNotIn("crossScript", zh_only_sources)
        self.assertNotIn("domainLexicon", zh_only_sources)

    def test_short_phrase_candidates_are_review_only_by_default(self):
        event = pipeline_event(
            raw="Ripple内",
            final="repo 內",
            llm="repo 內",
            length_bucket="1_4",
        )

        enriched = candidates.enrich_event(event)
        changed_candidates = [
            candidate for candidate in enriched["shadowCandidates"]
            if candidate["source"] != "raw"
        ]

        self.assertTrue(changed_candidates)
        self.assertTrue(all(candidate["blockedBecauseShortPhraseRisk"] for candidate in changed_candidates))
        self.assertTrue(all(candidate["requiresReview"] for candidate in changed_candidates))
        self.assertFalse(enriched["safety"]["autoApplied"])
        self.assertFalse(enriched["safety"]["wouldHaveChangedFinalOutput"])

    def test_replay_reports_phase2a_candidate_metrics(self):
        event = candidates.enrich_event(
            pipeline_event(
                raw="增加 Markdown。这样，我们过几天回来看结果的时候，你才知道。",
                final="增加 Markdown。這樣，我們過幾天回來看結果的時候，你才知道。",
                llm="增加 Markdown。這樣，我們過幾天回來看結果的時候，你才知道。",
                route="reviewSuggested",
                user_target="增加 Markdown。這樣，我們過幾天回來看結果的時候，你才知道。",
                user_source="reviewCandidate",
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase2a.jsonl"
            path.write_text(json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")
            loaded, warnings = replay.load_events([path])
            report = replay.build_report(loaded, [path], warnings)

        metrics = report["metrics"]
        for key in [
            "shadowTop1WouldHaveMatchedUserCorrection",
            "shadowTop3WouldHaveMatchedUserCorrection",
            "blockedBecauseLlmOnlyCount",
            "blockedBecauseShortPhraseRiskCount",
            "blockedBecauseNoiseSuspectedCount",
            "blockedBecauseNegativeEvidenceCount",
            "potentialReviewSavingsPer100",
            "potentialWrongCandidatePer100",
        ]:
            self.assertIn(key, metrics)

        self.assertEqual(report["safetyAudit"]["autoAppliedCount"], 0)
        self.assertEqual(report["safetyAudit"]["wouldHaveChangedFinalOutputCount"], 0)
        self.assertTrue(report["safetyAudit"]["phase1SafetyPass"])
        self.assertGreaterEqual(metrics["blockedBecauseLlmOnlyCount"], 1)
        self.assertEqual(metrics["potentialReviewSavingsPer100"], 100.0)
        self.assertEqual(metrics["potentialWrongCandidatePer100"], 0.0)


def pipeline_event(
    raw: str,
    final: str,
    llm: str | None = None,
    route: str = "directInsertion",
    length_bucket: str = "5_15",
    user_target: str | None = None,
    user_selected: str | None = None,
    user_source: str = "none",
    ui_context: dict | None = None,
):
    default_ui_context = {
        "activeAppBundleId": None,
        "windowTitleHash": None,
        "focusedElementRole": None,
        "selectionTextBefore": None,
        "selectionTextAfter": None,
        "anchorBeforeHash": None,
        "anchorAfterHash": None,
    }
    if ui_context:
        default_ui_context.update(ui_context)

    return {
        "schemaVersion": 1,
        "eventType": "pipelineSnapshot",
        "eventId": "event-1",
        "utteranceId": "utt-1",
        "transcriptionDbId": "db-1",
        "featureFlags": {
            "VocoPhoneticShadowLoggingEnabled": True,
            "VocoPhoneticCandidateApplicationEnabled": False,
        },
        "audio": {
            "audioAssetId": "sample.wav",
            "durationMs": 1000,
            "sampleRate": 16000,
            "audioHashPrefix": "abc",
        },
        "pipeline": {
            "asrEngine": "Qwen3-ASR",
            "rawASR": raw,
            "afterOpenCC": raw,
            "afterPinyinCorrector": None,
            "afterHomophoneCorrection": None,
            "afterNasalCorrection": None,
            "afterPersonalCorrection": raw,
            "llmEnhanced": llm,
            "finalInserted": final,
            "route": route,
            "confidenceScore": 0.9,
            "latencyMs": 120,
        },
        "classification": {
            "lengthBucket": length_bucket,
            "scriptMode": "mixedZhEn",
            "languageMode": "codeSwitch",
            "isCommandLike": False,
            "isTechnicalTermCandidate": True,
            "evidenceTier": "NONE",
            "noiseFlags": [],
            "isPurePhoneticCandidate": False,
        },
        "phonetics": {},
        "userAction": {
            "source": user_source,
            "targetText": user_target,
            "selectedCandidateText": user_selected,
        },
        "uiContext": default_ui_context,
        "shadowCandidates": [],
        "safety": {
            "autoApplied": False,
            "wouldHaveChangedFinalOutput": False,
            "blockedBecauseLlmOnly": False,
            "blockedBecauseShortPhraseRisk": False,
            "blockedBecauseNoiseSuspected": False,
            "blockedBecauseNegativeEvidence": False,
        },
    }


if __name__ == "__main__":
    unittest.main()
