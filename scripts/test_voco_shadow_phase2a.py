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
                raw="Ripple内的Markdown。",
                final="repo 內的 Markdown。",
                llm="repo 內的 Markdown。",
                route="reviewSuggested",
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
):
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
        "userAction": {"source": "none"},
        "uiContext": {},
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
