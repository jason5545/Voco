//
//  VoiceInkTests.swift
//  VoiceInkTests
//
//  Created by Prakash Joshi on 15/10/2024.
//

import Foundation
import SwiftData
import Testing
@testable import Voco

struct VoiceInkTests {

    @Test func validatorRejectsPromptLeakage() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "這是正體中文語音輸入，請修正。",
            original: "請修正這段文字"
        )

        #expect(result.isValid == false)
        #expect(result.reasons.contains(where: { $0.contains("blacklist") }))
    }

    @Test func validatorPreservesProtectedTechnicalTerms() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "我剛剛用差值被統一整理這段程式。",
            original: "我剛剛用 Chat GPT 整理這段程式。",
            protectedTerms: ["ChatGPT"]
        )

        #expect(result.isValid == false)
        #expect(result.reasons.contains(where: { $0.contains("dropped-term") }))
    }

    @Test func validatorAllowsScriptNormalizationAndPunctuation() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "網頁版。",
            original: "网页版"
        )

        #expect(result.isValid == true)
    }

    @Test func validatorAllowsChineseNumeralsConvertedToDigitsInsideTechTerms() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "我的 M Max 128GB 的筆電是在吃電池的。",
            original: "我的 M Max 一二八GB 的筆電是在吃電池的。"
        )

        #expect(result.isValid == true)
    }

    @Test func validatorAllowsVocabularyTermWithInsertedDigit() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "我的 M5 Max 128GB 的筆電是在吃電池的。",
            original: "我的 M Max 一二八GB 的筆電是在吃電池的。",
            wordReplacements: [],
            customVocabulary: ["M5 Max", "M5 Max 128GB"]
        )

        #expect(result.isValid == true)
    }

    @Test func validatorAllowsSingleChineseNumeralConvertedToDigit() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "它的 M 跟 Max 之間少了一個 5。",
            original: "它的M跟Max之間少了一個五。"
        )

        #expect(result.isValid == true)
    }

    @Test func validatorAllowsChineseDecimalConvertedToDigits() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "版本是 3.5。",
            original: "版本是三點五。"
        )

        #expect(result.isValid == true)
    }

    @Test func protectionListCoversWordsInsideProtectedPhrases() async throws {
        let chars = Array("到家了接上 AC 電源")
        let offset = try #require(chars.firstIndex(of: "家"))

        #expect(
            CorrectionProtectionList.shared.containsProtectedPhrase(
                in: chars,
                covering: offset,
                length: 1
            ) == true
        )
    }

    @Test func validatorRejectsAggressiveShortRewrite() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "網葉斑",
            original: "网页版"
        )

        #expect(result.isValid == false)
        #expect(result.reasons.contains(where: { $0 == "short-edit-budget" }))
    }

    @Test func canonicalizationNormalizesVocoDevelopmentTerms() async throws {
        let service = VocoCanonicalizationService()

        #expect(service.normalize("我現在用 voice ink 的 fork 做 voco").normalizedText == "我現在用 VoiceInk 的 fork 做 VOCO")
        #expect(service.normalize("我現在用 qwen three asr 的 mlx 版本").normalizedText == "我現在用 Qwen3-ASR 的 MLX 版本")
        #expect(service.normalize("我還是會留 whisper.cpp 支援").normalizedText == "我還是會留 whisper.cpp 支援")
    }

    @Test func canonicalizationSuggestsInactiveContextTermsWithoutAutoReplacing() async throws {
        let service = VocoCanonicalizationService()

        let inactive = service.normalize("我剛剛用 voice anc 測了一下", activeContextIDs: [])
        #expect(inactive.normalizedText == "我剛剛用 voice anc 測了一下")
        #expect(inactive.replacements.isEmpty)

        let suggestion = try #require(inactive.suggestions.first)
        #expect(suggestion.originalText == "voice anc")
        #expect(suggestion.replacementText == "VoiceInk")
        #expect(suggestion.termID == "product.voiceink")
        #expect(suggestion.reason == "inactive-context-suggestion")

        let active = service.normalize("我剛剛用 voice anc 測了一下")
        #expect(active.normalizedText == "我剛剛用 VoiceInk 測了一下")
        #expect(active.replacements.contains { $0.replacementText == "VoiceInk" })
    }

    @Test func canonicalizationNormalizesMixedLanguageMusicTermsWithContext() async throws {
        let service = VocoCanonicalizationService()

        #expect(service.normalize("我很喜歡 lisa 的紅蓮花").normalizedText == "我很喜歡 LiSA 的紅蓮華")
        #expect(service.normalize("我覺得 lisa 的 homura 很難唱").normalizedText == "我覺得 LiSA 的炎很難唱")
    }

    @Test func canonicalizationDoesNotOverreachAmbiguousTerms() async throws {
        let service = VocoCanonicalizationService()

        let neutral = service.normalize("今天看到火很大")
        #expect(neutral.normalizedText == "今天看到火很大")
        #expect(neutral.replacements.isEmpty)

        let ambiguous = service.normalize("今天看到焰很大")
        #expect(ambiguous.normalizedText == "今天看到焰很大")
        #expect(ambiguous.replacements.isEmpty)
        #expect(ambiguous.suggestions.contains(where: { $0.replacementText == "炎" }))
    }

    @Test func canonicalizationUsesPowerModeContextHintsForAmbiguousTerms() async throws {
        let service = VocoCanonicalizationService()
        let text = "今天看到 homura 很亮"

        let neutral = service.normalize(text)
        #expect(neutral.normalizedText == text)
        #expect(neutral.replacements.isEmpty)
        #expect(neutral.suggestions.contains(where: { $0.replacementText == "炎" }))

        let powerMode = PowerModeConfig(
            name: "LiSA music notes",
            emoji: "M",
            appConfigs: [AppConfig(bundleIdentifier: "com.apple.Music", appName: "Music")],
            urlConfigs: [URLConfig(url: "youtube.com")],
            isAIEnhancementEnabled: false,
            selectedLanguage: "auto"
        )
        let contextual = service.normalize(
            text,
            contextHints: VocoCanonicalizationService.powerModeContextHints(from: powerMode)
        )

        #expect(contextual.normalizedText == "今天看到炎很亮")
        #expect(contextual.replacements.first?.termID == "song.homura")
        #expect(contextual.replacements.first?.replacementText == "炎")
    }

    @Test func canonicalizationDoesNotExpandCanonicalCJKPhrases() async throws {
        let service = VocoCanonicalizationService()

        #expect(service.normalize("我昨天又看了鬼滅之刃").normalizedText == "我昨天又看了鬼滅之刃")
    }

    @Test func confidenceGateKeepsCleanCanonicalizationOnDirectRoute() async throws {
        let result = VocoCanonicalizationService().normalize("我現在用 voice ink 的 fork 做 voco")
        let assessment = VocoConfidenceGateService().assess(normalizationResult: result, rawTranscript: result.originalText)

        #expect(assessment.route == .directInsertion)
        #expect(assessment.score > 0.85)
        #expect(assessment.selectedCandidate == "我現在用 VoiceInk 的 fork 做 VOCO")
        #expect(assessment.candidates.first == "我現在用 VoiceInk 的 fork 做 VOCO")
        #expect(assessment.labelForCandidate(at: 0) == "Recommended")
    }

    @Test func confidenceGateSuggestsReviewForUnresolvedAmbiguousTerms() async throws {
        let result = VocoCanonicalizationService().normalize("今天看到焰很大")
        let assessment = VocoConfidenceGateService().assess(normalizationResult: result, rawTranscript: result.originalText)

        #expect(result.normalizedText == "今天看到焰很大")
        #expect(result.suggestions.contains(where: { $0.replacementText == "炎" }))
        #expect(assessment.route == .reviewSuggested)
        #expect(assessment.reasons.contains("unresolved-suggestions"))
        #expect(assessment.candidates.contains("今天看到炎很大"))
        #expect(assessment.candidateLabels == ["Recommended", "With suggestions"])
        #expect(assessment.hypothesisDetails.map(\.source) == [.autoContext, .suggestedRepair])
        #expect(assessment.hypothesisDetails[1].appliedTermIDs.contains("song.homura"))
        #expect((assessment.hypothesisDetails[1].divergenceFromRecommended ?? 0) > 0)
    }

    @Test func confidenceGateRoutesInactiveContextSuggestionsToReview() async throws {
        let result = VocoCanonicalizationService().normalize("我剛剛用 voice anc 測了一下", activeContextIDs: [])
        let assessment = VocoConfidenceGateService().assess(normalizationResult: result, rawTranscript: result.originalText)

        #expect(result.replacements.isEmpty)
        #expect(result.suggestions.contains { $0.replacementText == "VoiceInk" })
        #expect(assessment.route == .reviewSuggested)
        #expect(assessment.candidates == ["我剛剛用 voice anc 測了一下", "我剛剛用 VoiceInk 測了一下"])
        #expect(assessment.candidateLabels == ["Recommended", "With suggestions"])
        #expect(assessment.hypothesisDetails[1].appliedTermIDs.contains("product.voiceink"))
        #expect((assessment.hypothesisDetails[1].divergenceFromRecommended ?? 0) > 0)
    }

    @Test func confidenceGateSuggestsReviewForHeavyNormalization() async throws {
        let result = VocoCanonicalizationService().normalize("voice ink voco qwen three asr mlx")
        let assessment = VocoConfidenceGateService().assess(normalizationResult: result, rawTranscript: result.originalText)

        #expect(result.replacements.count >= 4)
        #expect(assessment.route == .reviewSuggested)
        #expect(assessment.reasons.contains("heavy-normalization"))
    }

    @Test func confidenceGateKeepsMinorRawCleanupDriftOnDirectRoute() async throws {
        let result = VocoCanonicalizationService().normalize("我現在用 VoiceInk")
        let assessment = VocoConfidenceGateService().assess(
            normalizationResult: result,
            rawTranscript: "我現在用 VoiceInk。"
        )

        #expect(assessment.route == .directInsertion)
        #expect(assessment.reasons.contains("raw-cleanup-drift"))
        #expect(!assessment.reasons.contains("raw-cleanup-significant"))
    }

    @Test func confidenceGateRoutesSignificantRawCleanupDriftToReview() async throws {
        let result = VocoCanonicalizationService().normalize("我今天要測 VoiceInk")
        let assessment = VocoConfidenceGateService().assess(
            normalizationResult: result,
            rawTranscript: "我今天要測 voice anc 然後後面還有一大段錯字"
        )

        #expect(assessment.route == .reviewSuggested)
        #expect(assessment.reasons.contains("raw-cleanup-significant"))
        #expect(assessment.candidates == [
            "我今天要測 VoiceInk",
            "我今天要測 voice anc 然後後面還有一大段錯字",
        ])
        #expect(assessment.candidateLabels == ["Recommended", "Raw ASR"])
        #expect(assessment.hypothesisDetails.map(\.source) == [.autoContext, .rawASR])
    }

    @Test func hypothesisManagerAddsSegmentRescueForRawDriftWithCanonicalTerms() async throws {
        let result = VocoCanonicalizationService().normalize("我今天要測 voice anc")
        let assessment = VocoConfidenceGateService().assess(
            normalizationResult: result,
            rawTranscript: "我今天要測 voice anc 然後後面還有一大段錯字"
        )

        #expect(assessment.route == .reviewSuggested)
        #expect(assessment.reasons.contains("raw-cleanup-significant"))
        #expect(assessment.candidates == [
            "我今天要測 VoiceInk",
            "我今天要測 VoiceInk 然後後面還有一大段錯字",
            "我今天要測 voice anc",
            "我今天要測 voice anc 然後後面還有一大段錯字",
        ])
        #expect(assessment.candidateLabels == ["Recommended", "Segment rescue", "Original", "Raw ASR"])

        let rescue = try #require(assessment.hypothesisDetails.first { $0.source == .segmentRescue })
        #expect(rescue.text == "我今天要測 VoiceInk 然後後面還有一大段錯字")
        #expect(rescue.label == "Segment rescue")
        #expect(rescue.appliedTermIDs == ["product.voiceink"])
        #expect(rescue.requiresReview)
        #expect(rescue.reasons.contains("segment-rescue"))
        #expect((rescue.divergenceFromRecommended ?? 0) > 0)
        #expect(VocoHypothesisDisplayFormatter.summary(for: rescue)?.contains("Segment rescue") == true)
        #expect(VocoHypothesisDisplayFormatter.summary(for: rescue)?.contains("Delta") == true)
    }

    @Test func hypothesisManagerKeepsRawASRAsTraceableCandidate() async throws {
        let result = VocoCanonicalizationService().normalize("我現在用 voice ink 的 fork 做 voco")
        let assessment = VocoConfidenceGateService().assess(
            normalizationResult: result,
            rawTranscript: "raw qwen transcript"
        )

        #expect(assessment.hypothesisDetails.map(\.label).contains("Raw ASR"))
        #expect(assessment.hypothesisDetails.last?.source == .rawASR)
        #expect(assessment.hypothesisDetails.last?.text == "raw qwen transcript")
        #expect(assessment.hypothesisDetails.first?.sourceDisplayName == "AUTO + context")
    }

    @Test func transcriptionStoresConfidenceGateMetadata() async throws {
        let result = VocoCanonicalizationService().normalize("我現在用 qwen three asr 的 mlx 版本")
        let assessment = VocoConfidenceGateService().assess(normalizationResult: result, rawTranscript: result.originalText)
        let transcription = Transcription(text: "", duration: 0)

        transcription.recordASRMetadata(
            rawTranscript: result.originalText,
            normalizationResult: result,
            confidenceAssessment: assessment,
            asrEngineID: "qwen3:Qwen3-ASR",
            languageMode: "auto"
        )

        #expect(transcription.normalizedTranscript == "我現在用 Qwen3-ASR 的 MLX 版本")
        #expect(transcription.confidenceRoute == VocoConfidenceRoute.directInsertion.rawValue)
        #expect(transcription.reviewTriggers.isEmpty)
        #expect(transcription.hypotheses.first == "我現在用 Qwen3-ASR 的 MLX 版本")
        #expect(transcription.hypothesisLabels.first == "Recommended")
        #expect(transcription.hypothesisDetails.first?.source == .autoContext)
        #expect(transcription.hypothesisDetails.first?.appliedTermIDs.contains("model.qwen3-asr") == true)
        #expect(transcription.selectedCandidate == "我現在用 Qwen3-ASR 的 MLX 版本")
        #expect(transcription.candidateSelectionSource == nil)
    }

    @Test func csvExportPreservesContextAwareSessionMetadata() async throws {
        let sourceTranscriptionID = try #require(UUID(uuidString: "11111111-2222-3333-4444-555555555555"))
        let transcription = Transcription(
            text: "我現在用 VoiceInk",
            duration: 1.25,
            enhancedText: "他說 \"VOCO, good\"",
            audioFileURL: nil,
            transcriptionModelName: "Qwen3-ASR",
            aiEnhancementModelName: "Local model",
            promptName: "dictation",
            transcriptionDuration: 0.5,
            enhancementDuration: 0.25,
            rawTranscript: "我現在用 voice ink",
            normalizedTranscript: "我現在用 VoiceInk",
            finalPastedText: "他說 \"VOCO, good\" ",
            pasteCommandPosted: true,
            activeContextIDs: [
                VocoCanonicalizationService.defaultContextPackID,
                "power-mode:123",
            ],
            canonicalizationReplacements: [
                VocoReplacement(
                    originalText: "voice ink",
                    replacementText: "VoiceInk",
                    termID: "product.voiceink",
                    confidence: 0.97,
                    reason: "alias-match",
                    rangeStart: 4,
                    rangeLength: 9
                ),
            ],
            canonicalizationSuggestions: [
                VocoReplacement(
                    originalText: "voco",
                    replacementText: "VOCO",
                    termID: "product.voco",
                    confidence: 0.55,
                    reason: "inactive-context-suggestion",
                    rangeStart: nil,
                    rangeLength: nil
                ),
            ],
            asrEngineID: "qwen3:Qwen3-ASR",
            languageMode: "auto",
            confidenceScore: 0.86,
            selectedCandidate: "我現在用 VoiceInk",
            candidateSelectionSource: .timeoutFallback,
            userCorrectionDistance: 0.12,
            sourceTranscriptionID: sourceTranscriptionID,
            retranscriptionSourceText: "我現在用 voice anc",
            retranscriptionAnalysis: RetranscriptionAnalysis(
                editDistance: 4,
                changeRatio: 0.3333,
                confidenceDelta: 0.2,
                changeCategory: .meaningfulChange
            )
        )
        transcription.confidenceRoute = VocoConfidenceRoute.reviewSuggested.rawValue
        transcription.confidenceReasons = ["alias-match", "raw-cleanup-drift"]
        transcription.reviewTriggers = [
            VocoReviewTrigger(
                id: "unresolved-suggestions",
                reason: "unresolved-suggestions",
                detail: "1 suggestion"
            ),
            VocoReviewTrigger(
                id: "raw-cleanup-significant",
                reason: "raw-cleanup-significant",
                detail: "Raw cleanup changed text"
            ),
        ]
        transcription.hypothesisLabels = ["Recommended", "Segment rescue", "Raw ASR"]
        transcription.hypotheses = ["我現在用 VoiceInk", "我現在用 VoiceInk rescue", "我現在用 voice ink"]
        transcription.hypothesisDetails = [
            VocoHypothesis(
                id: "autoContext",
                text: "我現在用 VoiceInk",
                label: "Recommended",
                source: .autoContext,
                confidenceScore: 0.86,
                reasons: ["alias-match", "raw-cleanup-drift"],
                activeContextIDs: [
                    VocoCanonicalizationService.defaultContextPackID,
                    "power-mode:123",
                ],
                appliedTermIDs: ["product.voiceink"],
                requiresReview: true
            ),
            VocoHypothesis(
                id: "segmentRescue",
                text: "我現在用 VoiceInk rescue",
                label: "Segment rescue",
                source: .segmentRescue,
                confidenceScore: 0.86,
                divergenceFromRecommended: 0.25,
                reasons: ["segment-rescue", "raw-cleanup-drift"],
                activeContextIDs: [
                    VocoCanonicalizationService.defaultContextPackID,
                ],
                appliedTermIDs: ["product.voiceink"],
                requiresReview: true
            ),
            VocoHypothesis(
                id: "rawASR",
                text: "我現在用 voice ink",
                label: "Raw ASR",
                source: .rawASR,
                confidenceScore: 0.86,
                divergenceFromRecommended: 0.18,
                reasons: ["raw-cleanup-drift"],
                activeContextIDs: [],
                appliedTermIDs: [],
                requiresReview: false
            ),
        ]

        let csv = VoiceInkCSVExportService().generateCSV(for: [transcription])
        let candidateDivergence = try #require(SessionMetric.candidateDivergenceRatio(in: transcription.hypothesisDetails))
        let candidateDivergenceText = String(format: "%.3f", candidateDivergence)

        #expect(csv.contains("Original Transcript,Raw Transcript,Normalized Transcript"))
        #expect(csv.contains("Final Pasted Text"))
        #expect(csv.contains("Paste Command Posted"))
        #expect(csv.contains("ASR Engine ID"))
        #expect(csv.contains("我現在用 voice ink"))
        #expect(csv.contains("我現在用 VoiceInk"))
        #expect(csv.contains("\"他說 \"\"VOCO, good\"\"\""))
        #expect(csv.contains("\"他說 \"\"VOCO, good\"\" \""))
        #expect(csv.contains("true"))
        #expect(csv.contains("qwen3:Qwen3-ASR"))
        #expect(csv.contains("auto"))
        #expect(csv.contains("builtin.voco-development | power-mode:123"))
        #expect(csv.contains("VOCO Development | Power Mode"))
        #expect(csv.contains("voice ink -> VoiceInk [product.voiceink, 97%, alias-match]"))
        #expect(csv.contains("voco -> VOCO [product.voco, 55%, inactive-context-suggestion]"))
        #expect(csv.contains("86%"))
        #expect(csv.contains("reviewSuggested"))
        #expect(csv.contains("Alias match | Cleanup drift"))
        #expect(csv.contains("Review Triggers"))
        #expect(csv.contains("Needs choice (1 suggestion) | Cleanup changed text (Raw cleanup changed text)"))
        #expect(csv.contains("Recommended: 我現在用 VoiceInk | Segment rescue: 我現在用 VoiceInk rescue | Raw ASR: 我現在用 voice ink"))
        #expect(csv.contains("Candidate Details"))
        #expect(csv.contains("Recommended / AUTO + context: Confidence 86%"))
        #expect(csv.contains("Segment rescue / Segment rescue: Confidence 86% · Delta 25%"))
        #expect(csv.contains("Terms product.voiceink"))
        #expect(csv.contains("Contexts VOCO Development, Power Mode"))
        #expect(csv.contains("Review required"))
        #expect(csv.contains("Raw ASR / Raw ASR: Confidence 86% · Delta 18%"))
        #expect(csv.contains("Candidate Source Counts"))
        #expect(csv.contains("Review Required Candidates"))
        #expect(csv.contains("Candidate Divergence Ratio"))
        #expect(csv.contains("Selected Candidate Source"))
        #expect(csv.contains("AUTO + context: 1 | Segment rescue: 1 | Raw ASR: 1"))
        #expect(csv.contains("2,\(candidateDivergenceText),我現在用 VoiceInk,AUTO + context,Timeout fallback"))
        #expect(csv.contains("Candidate Selection Source"))
        #expect(csv.contains("Timeout fallback"))
        #expect(csv.contains("0.120"))
        #expect(csv.contains("Retranscription Source ID"))
        #expect(csv.contains("11111111-2222-3333-4444-555555555555"))
        #expect(csv.contains("我現在用 voice anc"))
        #expect(csv.contains("meaningfulChange"))
        #expect(csv.contains("0.333"))
        #expect(csv.contains("meaningfulChange,0.333,4,0.200"))
        #expect(csv.contains("0.200"))
    }

    @Test func historyAssistiveBadgesSummarizeContextAwareSignals() async throws {
        let replacement = VocoReplacement(
            originalText: "voice ink",
            replacementText: "VoiceInk",
            termID: "product.voiceink",
            confidence: 0.97,
            reason: "alias-match",
            rangeStart: 4,
            rangeLength: 9
        )
        let suggestion = VocoReplacement(
            originalText: "焰",
            replacementText: "炎",
            termID: "song.homura",
            confidence: 0.68,
            reason: "context-required",
            rangeStart: 4,
            rangeLength: 1
        )
        let transcription = Transcription(
            text: "我現在用 VoiceInk",
            duration: 1.25,
            activeContextIDs: [
                VocoCanonicalizationService.defaultContextPackID,
                "power-mode:123",
            ],
            canonicalizationReplacements: [replacement, replacement],
            canonicalizationSuggestions: [suggestion],
            selectedCandidate: "我現在用 VoiceInk",
            candidateSelectionSource: .timeoutFallback,
            retranscriptionAnalysis: RetranscriptionAnalysis(
                editDistance: 6,
                changeRatio: 0.24,
                confidenceDelta: 0.18,
                changeCategory: .meaningfulChange
            )
        )
        transcription.confidenceRoute = VocoConfidenceRoute.reviewSuggested.rawValue
        transcription.reviewTriggers = [
            VocoReviewTrigger(
                id: "unresolved-suggestions",
                reason: "unresolved-suggestions",
                detail: "1 suggestion"
            ),
        ]

        let allBadges = TranscriptionAssistiveBadge.badges(for: transcription, limit: 10)
        #expect(allBadges.map(\.title) == [
            "Needs choice",
            "Timeout",
            "Re-run 24%",
            "2 fixes",
            "1 choice",
            "2 contexts",
        ])
        #expect(allBadges.map(\.tone) == [
            .orange,
            .orange,
            .purple,
            .accent,
            .orange,
            .secondary,
        ])
        #expect(TranscriptionAssistiveBadge.badges(for: transcription).map(\.title) == [
            "Needs choice",
            "Timeout",
            "Re-run 24%",
        ])

        let directCanonicalized = Transcription(
            text: "我現在用 VoiceInk",
            duration: 0.5,
            canonicalizationReplacements: [replacement]
        )

        #expect(TranscriptionAssistiveBadge.badges(for: directCanonicalized).map(\.title) == ["1 fix"])
    }

    @Test func historyReviewBadgeSummarizesTriggerSpecificity() async throws {
        let review = Transcription(text: "今天看到焰很大", duration: 0.4)
        review.confidenceRoute = VocoConfidenceRoute.reviewSuggested.rawValue
        review.reviewTriggers = [
            VocoReviewTrigger(
                id: "low-confidence-score",
                reason: "low-confidence-score",
                detail: "Score 60% below 78%"
            ),
            VocoReviewTrigger(
                id: "unresolved-suggestions",
                reason: "unresolved-suggestions",
                detail: "1 suggestion"
            ),
            VocoReviewTrigger(
                id: "unresolved-suggestions",
                reason: "unresolved-suggestions",
                detail: "duplicate ignored"
            ),
        ]

        let legacyReview = Transcription(text: "今天看到焰很大", duration: 0.4)
        legacyReview.confidenceRoute = VocoConfidenceRoute.reviewSuggested.rawValue

        #expect(TranscriptionAssistiveBadge.badges(for: review, limit: 1).first?.title == "2 signals")
        #expect(TranscriptionAssistiveBadge.badges(for: legacyReview, limit: 1).first?.title == "Review")
    }

    @Test func transcriptionDictationMetadataIncludesReviewSignals() async throws {
        let empty = Transcription(text: "一般歷史", duration: 0.2)
        let routeOnly = Transcription(text: "需要確認", duration: 0.2)
        routeOnly.confidenceRoute = VocoConfidenceRoute.reviewSuggested.rawValue

        let triggerOnly = Transcription(text: "需要確認", duration: 0.2)
        triggerOnly.reviewTriggers = [
            VocoReviewTrigger(
                id: "low-confidence-score",
                reason: "low-confidence-score",
                detail: "Score 60% below 78%"
            ),
        ]

        #expect(!empty.hasDictationMetadata)
        #expect(routeOnly.hasDictationMetadata)
        #expect(triggerOnly.hasDictationMetadata)
    }

    @Test func historyDisplayTextPrefersFinalUserOutput() async throws {
        let rawOnly = Transcription(text: "raw qwen transcript", duration: 0.2)
        let normalizedOnly = Transcription(
            text: "voice ink",
            duration: 0.2,
            normalizedTranscript: "VoiceInk"
        )
        let selected = Transcription(
            text: "今天看到焰很大",
            duration: 0.2,
            normalizedTranscript: "今天看到焰很大",
            selectedCandidate: "今天看到炎很大"
        )
        let enhanced = Transcription(
            text: "voice ink",
            duration: 0.2,
            enhancedText: "VoiceInk enhanced",
            normalizedTranscript: "VoiceInk"
        )
        let pasted = Transcription(
            text: "voice ink",
            duration: 0.2,
            enhancedText: "VoiceInk enhanced",
            finalPastedText: " VoiceInk pasted "
        )

        #expect(rawOnly.historyDisplayText == "raw qwen transcript")
        #expect(normalizedOnly.historyDisplayText == "VoiceInk")
        #expect(selected.historyDisplayText == "今天看到炎很大")
        #expect(enhanced.historyDisplayText == "VoiceInk enhanced")
        #expect(pasted.historyDisplayText == "VoiceInk pasted")
    }

    @Test func transcriptionDetailTextsTraceMeaningfulOutputVariants() async throws {
        let selected = Transcription(
            text: "今天看到焰很大",
            duration: 0.2,
            rawTranscript: "今天看到焰很大",
            normalizedTranscript: "今天看到焰很大",
            finalPastedText: " 今天看到炎很大 ",
            selectedCandidate: "今天看到炎很大"
        )
        let enhanced = Transcription(
            text: "voice ink",
            duration: 0.2,
            enhancedText: "VoiceInk enhanced",
            rawTranscript: "voice ink",
            normalizedTranscript: "VoiceInk",
            finalPastedText: "VoiceInk enhanced."
        )
        let original = Transcription(text: "一般文字", duration: 0.2)
        let normalizedWithoutRaw = Transcription(
            text: "voice ink",
            duration: 0.2,
            normalizedTranscript: "VoiceInk"
        )

        #expect(selected.detailDisplayTexts.map(\.label) == ["Normalized", "Selected"])
        #expect(selected.detailDisplayTexts.map(\.text) == ["今天看到焰很大", "今天看到炎很大"])
        #expect(enhanced.detailDisplayTexts.map(\.label) == ["Raw ASR", "Normalized", "Enhanced", "Pasted"])
        #expect(enhanced.detailDisplayTexts.map(\.isEnhanced) == [false, false, true, true])
        #expect(original.detailDisplayTexts.map(\.label) == ["Original"])
        #expect(normalizedWithoutRaw.detailDisplayTexts.map(\.label) == ["Original", "Normalized"])
    }

    @Test @MainActor func sessionMetricRecorderCapturesDictationMetadata() async throws {
        let context = try makeSessionMetricContext()
        let output = makeSessionMetricDictationOutput()
        let sourceTranscriptionID = UUID()
        let retranscriptionAnalysis = RetranscriptionAnalysis(
            editDistance: 4,
            changeRatio: 0.25,
            confidenceDelta: 0.18,
            changeCategory: .meaningfulChange
        )
        let transcription = Transcription(
            text: output.result.normalizedText,
            duration: 2.0,
            transcriptionModelName: "Qwen3-ASR",
            transcriptionDuration: 0.5,
            rawTranscript: output.result.originalText,
            normalizedTranscript: output.result.normalizedText,
            activeContextIDs: output.result.activeContextIDs,
            canonicalizationReplacements: output.result.replacements,
            canonicalizationSuggestions: output.result.suggestions,
            asrEngineID: "qwen3:Qwen3-ASR",
            languageMode: "auto",
            confidenceAssessment: output.assessment,
            candidateSelectionSource: .userSelection,
            userCorrectionDistance: 0.12,
            sourceTranscriptionID: sourceTranscriptionID,
            retranscriptionAnalysis: retranscriptionAnalysis,
            transcriptionStatus: .completed
        )
        context.insert(transcription)

        let inserted = try SessionMetricRecorder.recordRecorderSession(
            transcription: transcription,
            model: nil,
            in: context,
            timestamp: transcription.timestamp
        )

        let metric = try #require(try context.fetch(FetchDescriptor<SessionMetric>()).first)
        #expect(inserted)
        #expect(metric.transcriptionId == transcription.id)
        #expect(metric.asrEngineID == "qwen3:Qwen3-ASR")
        #expect(metric.languageMode == "auto")
        #expect(metric.activeContextIDs == output.result.activeContextIDs)
        #expect(metric.canonicalizationReplacementCount == 1)
        #expect(metric.canonicalizationSuggestionCount == 1)
        #expect(metric.confidenceScore == output.assessment.score)
        #expect(metric.confidenceRoute == VocoConfidenceRoute.reviewSuggested.rawValue)
        #expect(metric.confidenceReasons == output.assessment.reasons)
        #expect(metric.reviewTriggerCount == output.assessment.reviewTriggers.count)
        #expect(metric.reviewTriggerIDs == output.assessment.reviewTriggers.map(\.id))
        #expect(metric.reviewTriggerSummaries == SessionMetric.reviewTriggerSummaries(from: output.assessment.reviewTriggers))
        #expect(metric.candidateCount == output.assessment.candidates.count)
        #expect(metric.candidateSourceCounts[VocoHypothesisSource.autoContext.rawValue] == 1)
        #expect(metric.candidateSourceCounts[VocoHypothesisSource.suggestedRepair.rawValue] == 1)
        #expect(metric.candidateSourceCounts[VocoHypothesisSource.originalCleaned.rawValue] == 1)
        #expect(metric.reviewRequiredCandidateCount == 2)
        #expect(metric.candidateDivergenceRatio != nil)
        #expect((metric.candidateDivergenceRatio ?? 0) > 0)
        #expect(metric.selectedCandidateHypothesisSource == VocoHypothesisSource.autoContext.rawValue)
        #expect(metric.selectedCandidate == output.assessment.selectedCandidate)
        #expect(metric.candidateSelectionSource == VocoCandidateSelectionSource.userSelection.rawValue)
        #expect(metric.userCorrectionDistance == 0.12)
        #expect(metric.sourceTranscriptionID == sourceTranscriptionID)
        #expect(metric.retranscriptionChangeCategory == RetranscriptionChangeCategory.meaningfulChange.rawValue)
        #expect(metric.retranscriptionChangeRatio == 0.25)
        #expect(metric.retranscriptionEditDistance == 4)
        #expect(metric.retranscriptionConfidenceDelta == 0.18)
    }

    @Test @MainActor func sessionMetricRecorderCountsFinalPastedTextWhenAvailable() async throws {
        let context = try makeSessionMetricContext()
        let finalPastedText = "hello world "
        let transcription = Transcription(
            text: "hello",
            duration: 1.0,
            finalPastedText: finalPastedText,
            pasteCommandPosted: true,
            transcriptionStatus: .completed
        )
        context.insert(transcription)

        let inserted = try SessionMetricRecorder.recordRecorderSession(
            transcription: transcription,
            model: nil,
            in: context,
            timestamp: transcription.timestamp
        )

        let metric = try #require(try context.fetch(FetchDescriptor<SessionMetric>()).first)
        #expect(inserted)
        #expect(metric.wordCount == 2)
        #expect(metric.finalPastedWordCount == 2)
        #expect(metric.finalPastedCharacterCount == finalPastedText.count)
        #expect(metric.pasteCommandPosted == true)
    }

    @Test func sessionMetricBackfillCapturesDictationMetadataFromTranscription() async throws {
        let output = makeSessionMetricDictationOutput()
        let finalPastedText = "hello world "
        let sourceTranscriptionID = UUID()
        let retranscriptionAnalysis = RetranscriptionAnalysis(
            editDistance: 1,
            changeRatio: 0.08,
            confidenceDelta: -0.05,
            changeCategory: .minorChange
        )
        let transcription = Transcription(
            text: output.result.normalizedText,
            duration: 2.0,
            rawTranscript: output.result.originalText,
            normalizedTranscript: output.result.normalizedText,
            finalPastedText: finalPastedText,
            pasteCommandPosted: false,
            activeContextIDs: output.result.activeContextIDs,
            canonicalizationReplacements: output.result.replacements,
            canonicalizationSuggestions: output.result.suggestions,
            asrEngineID: "qwen3:Qwen3-ASR",
            languageMode: "auto",
            confidenceAssessment: output.assessment,
            candidateSelectionSource: .timeoutFallback,
            sourceTranscriptionID: sourceTranscriptionID,
            retranscriptionAnalysis: retranscriptionAnalysis,
            transcriptionStatus: .completed
        )
        let metric = SessionMetric(
            transcriptionId: transcription.id,
            wordCount: 3,
            audioDuration: 2.0,
            transcriptionModelName: "Qwen3-ASR",
            transcriptionDuration: 0.5,
            speedFactor: 4.0,
            powerModeName: nil,
            aiEnhancementModelName: nil,
            enhancementDuration: nil
        )

        metric.recordDictationMetadata(from: transcription)

        #expect(metric.activeContextIDs == output.result.activeContextIDs)
        #expect(metric.canonicalizationReplacementCount == 1)
        #expect(metric.canonicalizationSuggestionCount == 1)
        #expect(metric.confidenceRoute == VocoConfidenceRoute.reviewSuggested.rawValue)
        #expect(metric.confidenceReasons == output.assessment.reasons)
        #expect(metric.reviewTriggerCount == output.assessment.reviewTriggers.count)
        #expect(metric.reviewTriggerIDs == output.assessment.reviewTriggers.map(\.id))
        #expect(metric.reviewTriggerSummaries == SessionMetric.reviewTriggerSummaries(from: output.assessment.reviewTriggers))
        #expect(metric.candidateCount == output.assessment.candidates.count)
        #expect(metric.candidateSourceCounts[VocoHypothesisSource.autoContext.rawValue] == 1)
        #expect(metric.candidateSourceCounts[VocoHypothesisSource.suggestedRepair.rawValue] == 1)
        #expect(metric.candidateSourceCounts[VocoHypothesisSource.originalCleaned.rawValue] == 1)
        #expect(metric.reviewRequiredCandidateCount == 2)
        #expect(metric.selectedCandidateHypothesisSource == VocoHypothesisSource.autoContext.rawValue)
        #expect(metric.selectedCandidate == output.assessment.selectedCandidate)
        #expect(metric.candidateSelectionSource == VocoCandidateSelectionSource.timeoutFallback.rawValue)
        #expect(metric.wordCount == 2)
        #expect(metric.finalPastedWordCount == 2)
        #expect(metric.finalPastedCharacterCount == finalPastedText.count)
        #expect(metric.pasteCommandPosted == false)
        #expect(metric.sourceTranscriptionID == sourceTranscriptionID)
        #expect(metric.retranscriptionChangeCategory == RetranscriptionChangeCategory.minorChange.rawValue)
        #expect(metric.retranscriptionChangeRatio == 0.08)
        #expect(metric.retranscriptionEditDistance == 1)
        #expect(metric.retranscriptionConfidenceDelta == -0.05)
    }

    @Test func assistiveSignalSummaryCountsContextAwareMetrics() async throws {
        let direct = SessionMetric(
            transcriptionId: UUID(),
            wordCount: 4,
            audioDuration: 2.0,
            transcriptionModelName: "Qwen3-ASR",
            transcriptionDuration: 0.5,
            speedFactor: 4.0,
            powerModeName: nil,
            aiEnhancementModelName: nil,
            enhancementDuration: nil,
            canonicalizationReplacementCount: 2,
            canonicalizationSuggestionCount: 0,
            confidenceScore: 0.9,
            confidenceRoute: VocoConfidenceRoute.directInsertion.rawValue,
            candidateSourceCounts: [
                VocoHypothesisSource.autoContext.rawValue: 1,
            ],
            reviewRequiredCandidateCount: 0,
            selectedCandidateHypothesisSource: VocoHypothesisSource.autoContext.rawValue,
            candidateSelectionSource: VocoCandidateSelectionSource.userSelection.rawValue,
            retranscriptionChangeCategory: RetranscriptionChangeCategory.unchanged.rawValue,
            retranscriptionChangeRatio: 0,
            retranscriptionEditDistance: 0,
            retranscriptionConfidenceDelta: 0.02,
            pasteCommandPosted: true
        )
        let review = SessionMetric(
            transcriptionId: UUID(),
            wordCount: 5,
            audioDuration: 3.0,
            transcriptionModelName: "Qwen3-ASR",
            transcriptionDuration: 0.8,
            speedFactor: 3.75,
            powerModeName: nil,
            aiEnhancementModelName: nil,
            enhancementDuration: nil,
            canonicalizationReplacementCount: 0,
            canonicalizationSuggestionCount: 3,
            confidenceScore: 0.6,
            confidenceRoute: VocoConfidenceRoute.reviewSuggested.rawValue,
            reviewTriggerCount: 2,
            reviewTriggerIDs: [
                "unresolved-suggestions",
                "low-confidence-score",
            ],
            reviewTriggerSummaries: [
                "Needs choice (3 suggestions)",
                "Low score (Score 60% below 78%)",
            ],
            candidateSourceCounts: [
                VocoHypothesisSource.suggestedRepair.rawValue: 1,
                VocoHypothesisSource.segmentRescue.rawValue: 1,
                VocoHypothesisSource.rawASR.rawValue: 1,
            ],
            reviewRequiredCandidateCount: 2,
            candidateDivergenceRatio: 0.25,
            selectedCandidateHypothesisSource: VocoHypothesisSource.suggestedRepair.rawValue,
            candidateSelectionSource: VocoCandidateSelectionSource.timeoutFallback.rawValue,
            retranscriptionChangeCategory: RetranscriptionChangeCategory.meaningfulChange.rawValue,
            retranscriptionChangeRatio: 0.24,
            retranscriptionEditDistance: 6,
            retranscriptionConfidenceDelta: 0.18,
            pasteCommandPosted: false
        )
        let legacy = SessionMetric(
            transcriptionId: UUID(),
            wordCount: 2,
            audioDuration: 1.0,
            transcriptionModelName: nil,
            transcriptionDuration: nil,
            speedFactor: nil,
            powerModeName: nil,
            aiEnhancementModelName: nil,
            enhancementDuration: nil
        )

        let summary = AssistiveSignalSummary(metrics: [direct, review, legacy])

        #expect(summary.hasData)
        #expect(summary.sessionCount == 3)
        #expect(summary.confidenceRouteSampleCount == 2)
        #expect(summary.directInsertionCount == 1)
        #expect(summary.reviewSuggestedCount == 1)
        #expect(summary.directInsertionRate == 0.5)
        #expect(summary.reviewSuggestedRate == 0.5)
        #expect(summary.confidenceScoreSampleCount == 2)
        #expect(abs((summary.averageConfidenceScore ?? 0) - 0.75) < 0.001)
        #expect(summary.reviewTriggerSessionCount == 1)
        #expect(summary.reviewTriggerCount == 2)
        #expect(summary.reviewTriggerCounts["unresolved-suggestions"] == 1)
        #expect(summary.reviewTriggerCounts["low-confidence-score"] == 1)
        #expect(summary.reviewTriggerSummaryCounts["Needs choice (3 suggestions)"] == 1)
        #expect(summary.reviewTriggerSummaryCounts["Low score (Score 60% below 78%)"] == 1)
        #expect(summary.reviewTriggerDetail == "1 session / Low score (Score 60% below 78%) 1, Needs choice (3 suggestions) 1")
        #expect(summary.candidateSelectionCount == 2)
        #expect(summary.userSelectionCount == 1)
        #expect(summary.timeoutFallbackCount == 1)
        #expect(summary.fallbackSelectionCount == 1)
        #expect(summary.candidateSourceSampleCount == 2)
        #expect(summary.candidateSourceCandidateCount == 4)
        #expect(summary.candidateSourceCounts[VocoHypothesisSource.autoContext.rawValue] == 1)
        #expect(summary.candidateSourceCounts[VocoHypothesisSource.suggestedRepair.rawValue] == 1)
        #expect(summary.candidateSourceCounts[VocoHypothesisSource.segmentRescue.rawValue] == 1)
        #expect(summary.candidateSourceCounts[VocoHypothesisSource.rawASR.rawValue] == 1)
        #expect(summary.reviewRequiredCandidateCount == 2)
        #expect(summary.candidateDivergenceRatioSampleCount == 1)
        #expect(abs((summary.averageCandidateDivergenceRatio ?? 0) - 0.25) < 0.001)
        #expect(summary.selectedCandidateSourceCounts[VocoHypothesisSource.autoContext.rawValue] == 1)
        #expect(summary.selectedCandidateSourceCounts[VocoHypothesisSource.suggestedRepair.rawValue] == 1)
        #expect(summary.candidateSourceDetail == "2 review / AUTO + context 1, Suggestion pass 1, Segment rescue 1 / avg delta 25%")
        #expect(summary.canonicalizedSessionCount == 1)
        #expect(summary.suggestedSessionCount == 1)
        #expect(summary.totalCanonicalizationReplacementCount == 2)
        #expect(summary.totalCanonicalizationSuggestionCount == 3)
        #expect(summary.retranscriptionSampleCount == 2)
        #expect(summary.unchangedRetranscriptionCount == 1)
        #expect(summary.minorRetranscriptionCount == 0)
        #expect(summary.meaningfulRetranscriptionCount == 1)
        #expect(summary.meaningfulRetranscriptionRate == 0.5)
        #expect(summary.retranscriptionChangeRatioSampleCount == 2)
        #expect(abs((summary.averageRetranscriptionChangeRatio ?? 0) - 0.12) < 0.001)
        #expect(summary.retranscriptionConfidenceDeltaSampleCount == 2)
        #expect(abs((summary.averageRetranscriptionConfidenceDelta ?? 0) - 0.10) < 0.001)
        #expect(summary.retranscriptionDetail == "1 meaningful / 2 analyzed, avg change 12%, avg confidence +10%")
        #expect(summary.pasteCommandSampleCount == 2)
        #expect(summary.pasteCommandPostedCount == 1)
        #expect(summary.pasteCommandPostedRate == 0.5)
    }

    @Test @MainActor func canonicalizationPipelineUsesSingleAssessmentForTranscriptionMetadata() async throws {
        let context = try makeCanonicalizationPipelineContext()
        let transcription = Transcription(text: "", duration: 0)
        context.insert(transcription)

        let model = try #require(TranscriptionModelRegistry.models.first)
        let output = VocoCanonicalizationPipeline.normalizeWithAssessment(
            "我剛剛用 voice anc 測了一下",
            rawTranscript: "我剛剛用 voice anc 測了一下",
            model: model,
            modelContext: context,
            transcription: transcription
        )

        #expect(transcription.normalizedTranscript == output.normalizationResult.normalizedText)
        #expect(transcription.activeContextIDs == output.normalizationResult.activeContextIDs)
        #expect(transcription.canonicalizationReplacements == output.normalizationResult.replacements)
        #expect(transcription.canonicalizationSuggestions == output.normalizationResult.suggestions)
        #expect(transcription.confidenceScore == output.confidenceAssessment.score)
        #expect(transcription.confidenceRoute == output.confidenceAssessment.route.rawValue)
        #expect(transcription.confidenceReasons == output.confidenceAssessment.reasons)
        #expect(transcription.reviewTriggers == output.confidenceAssessment.reviewTriggers)
        #expect(transcription.hypotheses == output.confidenceAssessment.candidates)
        #expect(transcription.hypothesisLabels == output.confidenceAssessment.candidateLabels)
        #expect(transcription.hypothesisDetails == output.confidenceAssessment.hypothesisDetails)
        #expect(transcription.selectedCandidate == output.confidenceAssessment.selectedCandidate)
        #expect(transcription.asrEngineID == VocoCanonicalizationPipeline.asrEngineID(for: model))
        #expect(transcription.languageMode == VocoCanonicalizationPipeline.selectedLanguageMode())
    }

    @Test func candidateReviewDisplaysReadableReasonsAndLabels() async throws {
        let hypothesis = VocoHypothesis(
            id: "suggestedRepair",
            text: "今天看到炎很大",
            label: "With suggestions",
            source: .suggestedRepair,
            confidenceScore: 0.62,
            reasons: ["unresolved-suggestions"],
            activeContextIDs: [],
            appliedTermIDs: ["song.homura"],
            requiresReview: true
        )
        let review = VocoCandidateReview(
            candidates: ["今天看到焰很大", "今天看到炎很大"],
            candidateLabels: ["Recommended", "With suggestions"],
            hypotheses: [
                VocoHypothesis(
                    id: "autoContext",
                    text: "今天看到焰很大",
                    label: "Recommended",
                    source: .autoContext,
                    confidenceScore: 0.62,
                    reasons: ["unresolved-suggestions"],
                    activeContextIDs: [],
                    appliedTermIDs: [],
                    requiresReview: true
                ),
                hypothesis,
            ],
            confidenceScore: 0.62,
            reasons: ["unresolved-suggestions", "high-risk-term", "unresolved-suggestions"],
            reviewTriggers: [
                VocoReviewTrigger(
                    id: "low-confidence-score",
                    reason: "low-confidence-score",
                    detail: "Score 62% below 78%"
                ),
                VocoReviewTrigger(
                    id: "unresolved-suggestions",
                    reason: "unresolved-suggestions",
                    detail: "1 suggestion"
                ),
                VocoReviewTrigger(
                    id: "unresolved-suggestions",
                    reason: "unresolved-suggestions",
                    detail: "duplicate ignored"
                ),
            ]
        )

        #expect(review.defaultCandidate == "今天看到焰很大")
        #expect(review.timeoutFallbackCandidate == "今天看到焰很大")
        #expect(VocoCandidateReview.timeoutSeconds == 20)
        #expect(VocoCandidateReview.shouldRefreshTimeout(forTypedCandidate: " 今天看到火焰很大 "))
        #expect(!VocoCandidateReview.shouldRefreshTimeout(forTypedCandidate: "   "))
        #expect(review.keyboardShortcutForCandidate(at: 0) == "1")
        #expect(review.keyboardShortcutForCandidate(at: 1) == "2")
        #expect(review.keyboardShortcutForCandidate(at: 5) == nil)
        #expect(review.labelForCandidate(at: 1) == "With suggestions")
        #expect(review.labelForCandidate(at: 4) == "Candidate")
        #expect(review.sourceDisplayNameForCandidate(at: 1) == "Suggestion pass")
        #expect(review.displayReasons == ["Needs choice", "High-risk term"])
        #expect(review.displayReviewSignals == [
            "Low score (Score 62% below 78%)",
            "Needs choice (1 suggestion)",
        ])
    }

    @Test func signalDisplayFormatterCoversDictationAndFeedbackReasons() async throws {
        let displayReasons = VocoSignalDisplayFormatter.displayReasons(for: [
            "unresolved-suggestions",
            "inactive-context-suggestion",
            "context-required",
            "alias-match",
            "candidate-override",
            "candidate-confirmed",
            "candidate-timeout-fallback",
            "candidate-dismissed-fallback",
            "candidate-auto-fallback",
            "raw-cleanup-significant",
            "retranscription-meaningfulChange",
            "user-substitution",
            "unknown-signal",
            "unresolved-suggestions",
        ])

        #expect(displayReasons == [
            "Needs choice",
            "Inactive context",
            "Needs context",
            "Alias match",
            "Candidate changed",
            "Candidate confirmed",
            "Timeout fallback",
            "Dismissed fallback",
            "Automatic fallback",
            "Cleanup changed text",
            "Retranscription meaningful",
            "User substitution",
            "Unknown signal",
        ])
    }

    @Test func hypothesisDisplayFormatterSummarizesPersistedCandidateDetails() async throws {
        let hypothesis = VocoHypothesis(
            id: "suggestedRepair",
            text: "今天看到炎很大",
            label: "With suggestions",
            source: .suggestedRepair,
            confidenceScore: 0.624,
            divergenceFromRecommended: 0.18,
            reasons: ["unresolved-suggestions", "high-risk-term", "unresolved-suggestions"],
            activeContextIDs: [
                VocoCanonicalizationService.defaultContextPackID,
                "power-mode:1234",
                VocoCanonicalizationService.defaultContextPackID,
            ],
            appliedTermIDs: ["song.homura", "song.homura", "artist.lisa"],
            requiresReview: true
        )

        #expect(
            VocoHypothesisDisplayFormatter.summary(for: hypothesis) ==
                "Confidence 62% · Delta 18% · Needs choice, High-risk term · Terms song.homura, artist.lisa · Contexts VOCO Development, Power Mode · Review required"
        )

        let empty = VocoHypothesis(
            id: "clean",
            text: "乾淨候選",
            label: "Recommended",
            source: .autoContext,
            confidenceScore: nil,
            reasons: [],
            activeContextIDs: [],
            appliedTermIDs: [],
            requiresReview: false
        )
        #expect(VocoHypothesisDisplayFormatter.summary(for: empty) == nil)
    }

    @Test func hypothesisDecodesLegacyPayloadWithoutDivergence() async throws {
        let json = """
        {
          "id": "autoContext",
          "text": "我現在用 VoiceInk",
          "label": "Recommended",
          "source": "autoContext",
          "confidenceScore": 0.9,
          "reasons": ["alias-match"],
          "activeContextIDs": ["builtin.voco-development"],
          "appliedTermIDs": ["product.voiceink"],
          "requiresReview": false
        }
        """

        let hypothesis = try JSONDecoder().decode(VocoHypothesis.self, from: Data(json.utf8))

        #expect(hypothesis.source == .autoContext)
        #expect(hypothesis.divergenceFromRecommended == nil)
        #expect(hypothesis.text == "我現在用 VoiceInk")
    }

    @Test func confidenceAssessmentDecodesLegacyPayloadWithoutReviewTriggers() async throws {
        let json = """
        {
          "score": 0.86,
          "route": "reviewSuggested",
          "reasons": ["unresolved-suggestions"],
          "candidates": ["今天看到焰很大", "今天看到炎很大"],
          "candidateLabels": ["Recommended", "With suggestions"],
          "hypothesisDetails": [],
          "selectedCandidate": "今天看到焰很大"
        }
        """

        let assessment = try JSONDecoder().decode(VocoConfidenceAssessment.self, from: Data(json.utf8))

        #expect(assessment.route == .reviewSuggested)
        #expect(assessment.reviewTriggers.isEmpty)
        #expect(assessment.candidates.count == 2)
    }

    @Test func candidateReviewPayloadKeepsOnlyActionableCandidates() async throws {
        let hypothesis = VocoHypothesis(
            id: "suggestedRepair",
            text: "今天看到炎很大",
            label: "With suggestions",
            source: .suggestedRepair,
            confidenceScore: 0.7,
            reasons: ["unresolved-suggestions"],
            activeContextIDs: [],
            appliedTermIDs: ["song.homura"],
            requiresReview: true
        )
        let reviewTriggers = [
            VocoReviewTrigger(
                id: "unresolved-suggestions",
                reason: "unresolved-suggestions",
                detail: "1 suggestion"
            ),
        ]
        let assessment = VocoConfidenceAssessment(
            score: 0.7,
            route: .reviewSuggested,
            reasons: ["unresolved-suggestions"],
            reviewTriggers: reviewTriggers,
            candidates: [" 今天看到炎很大 ", "今天看到炎很大", ""],
            candidateLabels: ["With suggestions", "Duplicate", "Empty"],
            hypothesisDetails: [hypothesis],
            selectedCandidate: "今天看到焰很大"
        )

        let review = try #require(VocoCandidateReviewService.review(for: assessment))

        #expect(review.candidates == ["今天看到焰很大", "今天看到炎很大"])
        #expect(review.candidateLabels == ["Recommended", "With suggestions"])
        #expect(review.defaultCandidate == "今天看到焰很大")
        #expect(review.sourceDisplayNameForCandidate(at: 1) == "Suggestion pass")
        #expect(review.reviewTriggers == reviewTriggers)
        #expect(review.displayReviewSignals == ["Needs choice (1 suggestion)"])
    }

    @Test func candidateReviewPayloadRequiresReviewRouteAndAlternative() async throws {
        let duplicateOnly = VocoConfidenceAssessment(
            score: 0.7,
            route: .reviewSuggested,
            reasons: ["unresolved-suggestions"],
            candidates: ["今天看到焰很大", " 今天看到焰很大 "],
            selectedCandidate: "今天看到焰很大"
        )
        let directRoute = VocoConfidenceAssessment(
            score: 0.7,
            route: .directInsertion,
            reasons: ["unresolved-suggestions"],
            candidates: ["今天看到焰很大", "今天看到炎很大"],
            selectedCandidate: "今天看到焰很大"
        )

        #expect(VocoCandidateReviewService.review(for: duplicateOnly) == nil)
        #expect(VocoCandidateReviewService.review(for: directRoute) == nil)
    }

    @Test func correctionFeedbackCapturesCandidateOverride() async throws {
        let result = VocoCanonicalizationService().normalize("今天看到焰很大")
        let assessment = VocoConfidenceGateService().assess(normalizationResult: result, rawTranscript: result.originalText)
        let signal = try #require(
            CorrectionFeedbackService.candidateSelectionSignal(
                normalizationResult: result,
                assessment: assessment,
                selectedCandidate: "今天看到炎很大",
                rawTranscript: result.originalText
            )
        )

        #expect(signal.kind == .candidateSelection)
        #expect(signal.reason == "candidate-override")
        #expect(signal.sourceText == "今天看到焰很大")
        #expect(signal.proposedText == "今天看到焰很大")
        #expect(signal.acceptedText == "今天看到炎很大")
        #expect(signal.termIDs.contains("song.homura"))
        #expect((signal.changeRatio ?? 0) > 0)
        #expect(signal.isCorrectiveSignal)
    }

    @Test func correctionFeedbackCapturesTypedCandidateRescue() async throws {
        let result = VocoCanonicalizationService().normalize("今天看到焰很大")
        let assessment = VocoConfidenceGateService().assess(normalizationResult: result, rawTranscript: result.originalText)
        let signal = try #require(
            CorrectionFeedbackService.candidateSelectionSignal(
                normalizationResult: result,
                assessment: assessment,
                selectedCandidate: "今天看到火焰很大",
                rawTranscript: result.originalText
            )
        )

        #expect(signal.kind == .candidateSelection)
        #expect(signal.reason == "candidate-custom")
        #expect(signal.proposedText == "今天看到焰很大")
        #expect(signal.acceptedText == "今天看到火焰很大")
        #expect(signal.termIDs.contains("song.homura"))
        #expect(signal.isCorrectiveSignal)
    }

    @Test func correctionFeedbackClassifiesCandidateConfirmationAsNonCorrective() async throws {
        let result = VocoCanonicalizationService().normalize("我現在用 voice anc")
        let assessment = VocoConfidenceGateService().assess(normalizationResult: result, rawTranscript: result.originalText)
        let signal = try #require(
            CorrectionFeedbackService.candidateSelectionSignal(
                normalizationResult: result,
                assessment: assessment,
                selectedCandidate: "我現在用 VoiceInk",
                rawTranscript: result.originalText
            )
        )

        #expect(signal.kind == .candidateSelection)
        #expect(signal.reason == "candidate-confirmed")
        #expect(signal.acceptedText == "我現在用 VoiceInk")
        #expect(signal.isCorrectiveSignal == false)
    }

    @Test func correctionFeedbackClassifiesCandidateTimeoutFallbackAsNonCorrective() async throws {
        let result = VocoCanonicalizationService().normalize("我現在用 voice anc")
        let assessment = VocoConfidenceGateService().assess(normalizationResult: result, rawTranscript: result.originalText)
        let signal = try #require(
            CorrectionFeedbackService.candidateSelectionSignal(
                normalizationResult: result,
                assessment: assessment,
                selectedCandidate: "我現在用 VoiceInk",
                rawTranscript: result.originalText,
                selectionSource: .timeoutFallback
            )
        )

        #expect(signal.kind == .candidateSelection)
        #expect(signal.reason == "candidate-timeout-fallback")
        #expect(signal.acceptedText == "我現在用 VoiceInk")
        #expect((signal.changeRatio ?? 0) > 0)
        #expect(signal.isCorrectiveSignal == false)
    }

    @Test func transcriptionStoresCandidateFeedback() async throws {
        let result = VocoCanonicalizationService().normalize("今天看到焰很大")
        let assessment = VocoConfidenceGateService().assess(normalizationResult: result, rawTranscript: result.originalText)
        let transcription = Transcription(text: "今天看到焰很大", duration: 0)

        transcription.recordCandidateReviewFeedback(
            normalizationResult: result,
            confidenceAssessment: assessment,
            selectedCandidate: "今天看到炎很大",
            rawTranscript: result.originalText
        )

        let signal = try #require(transcription.correctionFeedback.first)
        #expect(signal.kind == .candidateSelection)
        #expect(signal.acceptedText == "今天看到炎很大")
        #expect(signal.reason == "candidate-override")
    }

    @Test func candidateReviewAcceptanceStoresTypedRescue() async throws {
        let result = VocoCanonicalizationService().normalize("今天看到焰很大")
        let assessment = VocoConfidenceGateService().assess(normalizationResult: result, rawTranscript: result.originalText)
        let transcription = Transcription(text: result.normalizedText, duration: 0)

        let signal = try #require(
            VocoCandidateReviewService.acceptCandidate(
                "今天看到火焰很大",
                for: transcription,
                normalizationResult: result,
                confidenceAssessment: assessment,
                rawTranscript: result.originalText
            )
        )

        #expect(transcription.text == "今天看到火焰很大")
        #expect(transcription.normalizedTranscript == result.normalizedText)
        #expect(transcription.selectedCandidate == "今天看到火焰很大")
        #expect(signal.reason == "candidate-custom")
        #expect(transcription.correctionFeedback.first?.reason == "candidate-custom")
        #expect(transcription.userCorrectionDistance != nil)
        let customIndex = try #require(transcription.hypotheses.firstIndex(of: "今天看到火焰很大"))
        #expect(transcription.hypothesisLabels[customIndex] == "Typed correction")
        #expect(transcription.hypothesisDetails[customIndex].source == .customRescue)
        #expect(transcription.hypothesisDetails[customIndex].requiresReview == false)
        #expect(transcription.hypotheses.count <= 5)
    }

    @Test func candidateReviewAcceptanceUpdatesTranscriptAndFeedback() async throws {
        let result = VocoCanonicalizationService().normalize("今天看到焰很大")
        let assessment = VocoConfidenceGateService().assess(normalizationResult: result, rawTranscript: result.originalText)
        let transcription = Transcription(text: result.normalizedText, duration: 0)

        transcription.recordASRMetadata(
            rawTranscript: result.originalText,
            normalizationResult: result,
            confidenceAssessment: assessment,
            asrEngineID: "qwen3:Qwen3-ASR",
            languageMode: "auto"
        )

        let signal = try #require(
            VocoCandidateReviewService.acceptCandidate(
                "今天看到炎很大",
                for: transcription,
                normalizationResult: result,
                confidenceAssessment: assessment,
                rawTranscript: result.originalText
            )
        )

        #expect(transcription.text == "今天看到炎很大")
        #expect(transcription.normalizedTranscript == result.normalizedText)
        #expect(transcription.selectedCandidate == "今天看到炎很大")
        #expect(transcription.candidateSelectionSource == VocoCandidateSelectionSource.userSelection.rawValue)
        #expect(transcription.userCorrectionDistance == signal.changeRatio)
        #expect(transcription.correctionFeedback.count == 1)
        #expect(signal.reason == "candidate-override")
        #expect(signal.isCorrectiveSignal)
        #expect(signal.termIDs.contains("song.homura"))
    }

    @Test func candidateReviewAcceptancePreservesAutoNormalizedTranscript() async throws {
        let result = VocoCanonicalizationService().normalize("今天看到焰很大")
        let assessment = VocoConfidenceGateService().assess(normalizationResult: result, rawTranscript: result.originalText)
        let transcription = Transcription(text: result.normalizedText, duration: 0)

        transcription.recordASRMetadata(
            rawTranscript: result.originalText,
            normalizationResult: result,
            confidenceAssessment: assessment,
            asrEngineID: "qwen3:Qwen3-ASR",
            languageMode: "auto"
        )

        VocoCandidateReviewService.acceptCandidate(
            "今天看到火焰很大",
            for: transcription,
            normalizationResult: result,
            confidenceAssessment: assessment,
            rawTranscript: result.originalText
        )

        #expect(transcription.rawTranscript == result.originalText)
        #expect(transcription.normalizedTranscript == result.normalizedText)
        #expect(transcription.text == "今天看到火焰很大")
        #expect(transcription.selectedCandidate == "今天看到火焰很大")
    }

    @Test func candidateReviewConfirmationDoesNotSetCorrectionDistance() async throws {
        let result = VocoCanonicalizationService().normalize("我現在用 voice ink")
        let assessment = VocoConfidenceGateService().assess(normalizationResult: result, rawTranscript: result.originalText)
        let transcription = Transcription(text: result.normalizedText, duration: 0)

        transcription.recordASRMetadata(
            rawTranscript: result.originalText,
            normalizationResult: result,
            confidenceAssessment: assessment,
            asrEngineID: "qwen3:Qwen3-ASR",
            languageMode: "auto"
        )

        let signal = try #require(
            VocoCandidateReviewService.acceptCandidate(
                "我現在用 VoiceInk",
                for: transcription,
                normalizationResult: result,
                confidenceAssessment: assessment,
                rawTranscript: result.originalText
            )
        )

        #expect(signal.reason == "candidate-confirmed")
        #expect(signal.isCorrectiveSignal == false)
        #expect(transcription.text == "我現在用 VoiceInk")
        #expect(transcription.normalizedTranscript == "我現在用 VoiceInk")
        #expect(transcription.selectedCandidate == "我現在用 VoiceInk")
        #expect(transcription.candidateSelectionSource == VocoCandidateSelectionSource.userSelection.rawValue)
        #expect(transcription.userCorrectionDistance == nil)
        #expect(transcription.correctionFeedback.count == 1)
    }

    @Test func candidateReviewTimeoutFallbackUpdatesTranscriptWithoutCorrectionDistance() async throws {
        let result = VocoCanonicalizationService().normalize("我現在用 voice anc")
        let assessment = VocoConfidenceGateService().assess(normalizationResult: result, rawTranscript: result.originalText)
        let transcription = Transcription(text: result.originalText, duration: 0)

        transcription.recordASRMetadata(
            rawTranscript: result.originalText,
            normalizationResult: result,
            confidenceAssessment: assessment,
            asrEngineID: "qwen3:Qwen3-ASR",
            languageMode: "auto"
        )

        let signal = try #require(
            VocoCandidateReviewService.acceptCandidate(
                "我現在用 VoiceInk",
                for: transcription,
                normalizationResult: result,
                confidenceAssessment: assessment,
                rawTranscript: result.originalText,
                selectionSource: .timeoutFallback
            )
        )

        #expect(signal.reason == "candidate-timeout-fallback")
        #expect(signal.isCorrectiveSignal == false)
        #expect(transcription.text == "我現在用 VoiceInk")
        #expect(transcription.normalizedTranscript == "我現在用 VoiceInk")
        #expect(transcription.selectedCandidate == "我現在用 VoiceInk")
        #expect(transcription.candidateSelectionSource == VocoCandidateSelectionSource.timeoutFallback.rawValue)
        #expect(transcription.userCorrectionDistance == nil)
        #expect(transcription.correctionFeedback.count == 1)
    }

    @Test func persistedCandidateReviewDoesNotDuplicateSameAcceptedCandidate() async throws {
        let result = VocoCanonicalizationService().normalize("今天看到焰很大")
        let assessment = VocoConfidenceGateService().assess(normalizationResult: result, rawTranscript: result.originalText)
        let transcription = Transcription(text: result.normalizedText, duration: 0)

        transcription.recordASRMetadata(
            rawTranscript: result.originalText,
            normalizationResult: result,
            confidenceAssessment: assessment,
            asrEngineID: "qwen3:Qwen3-ASR",
            languageMode: "auto"
        )

        _ = VocoCandidateReviewService.acceptPersistedCandidate("今天看到炎很大", for: transcription)
        _ = VocoCandidateReviewService.acceptPersistedCandidate("今天看到炎很大", for: transcription)

        #expect(transcription.text == "今天看到炎很大")
        #expect(transcription.normalizedTranscript == result.normalizedText)
        #expect(transcription.selectedCandidate == "今天看到炎很大")
        #expect(transcription.candidateSelectionSource == VocoCandidateSelectionSource.userSelection.rawValue)
        #expect(transcription.correctionFeedback.count == 1)
        #expect(transcription.correctionFeedback.first?.acceptedText == "今天看到炎很大")
    }

    @Test func candidateReviewAcceptanceKeepsCustomRescueWithinPersistedLimit() async throws {
        let candidates = ["first", "second", "third", "fourth", "fifth"]
        let details = candidates.enumerated().map { index, candidate in
            VocoHypothesis(
                id: "candidate.\(index)",
                text: candidate,
                label: "Candidate \(index + 1)",
                source: .autoContext,
                confidenceScore: 0.62,
                reasons: ["unresolved-suggestions"],
                activeContextIDs: [VocoCanonicalizationService.defaultContextPackID],
                appliedTermIDs: [],
                requiresReview: true
            )
        }
        let assessment = VocoConfidenceAssessment(
            score: 0.62,
            route: .reviewSuggested,
            reasons: ["unresolved-suggestions"],
            candidates: candidates,
            candidateLabels: details.map(\.label),
            hypothesisDetails: details,
            selectedCandidate: "first"
        )
        let result = VocoNormalizationResult(
            originalText: "first",
            normalizedText: "first",
            activeContextIDs: [VocoCanonicalizationService.defaultContextPackID],
            replacements: [],
            suggestions: []
        )
        let transcription = Transcription(text: "first", duration: 0)

        transcription.recordASRMetadata(
            rawTranscript: result.originalText,
            normalizationResult: result,
            confidenceAssessment: assessment,
            asrEngineID: "qwen3:Qwen3-ASR",
            languageMode: "auto"
        )

        _ = VocoCandidateReviewService.acceptCandidate(
            "typed rescue",
            for: transcription,
            normalizationResult: result,
            confidenceAssessment: assessment,
            rawTranscript: result.originalText
        )

        #expect(transcription.hypotheses.count == 5)
        #expect(transcription.hypotheses.contains("typed rescue"))
        #expect(!transcription.hypotheses.contains("fifth"))
        let customIndex = try #require(transcription.hypotheses.firstIndex(of: "typed rescue"))
        #expect(transcription.hypothesisDetails[customIndex].source == .customRescue)
        #expect(transcription.hypothesisDetails[customIndex].sourceDisplayName == "Custom rescue")
    }

    @Test @MainActor func correctionRiskProfileCountsOnlyCorrectiveFeedback() async throws {
        let context = try makeTranscriptionContext()
        let now = Date()

        let first = Transcription(text: "voice ink", duration: 0, transcriptionStatus: .completed)
        first.timestamp = now
        first.recordCorrectionFeedback(
            CorrectionFeedbackSignal(
                kind: .candidateSelection,
                sourceText: "voice ink",
                proposedText: "VoiceInk",
                acceptedText: "VoiceInk",
                reason: "candidate-confirmed",
                termIDs: ["product.voiceink"]
            )
        )
        context.insert(first)

        let second = Transcription(text: "voice anc", duration: 0, transcriptionStatus: .completed)
        second.timestamp = now.addingTimeInterval(-60)
        second.recordCorrectionFeedback(
            CorrectionFeedbackSignal(
                kind: .candidateSelection,
                sourceText: "voice anc",
                proposedText: "VoiceInk",
                acceptedText: "VoiceInk",
                reason: "candidate-override",
                termIDs: ["product.voiceink"]
            )
        )
        context.insert(second)

        let third = Transcription(text: "voice inc", duration: 0, transcriptionStatus: .completed)
        third.timestamp = now.addingTimeInterval(-90)
        third.recordCorrectionFeedback(
            CorrectionFeedbackSignal(
                kind: .candidateSelection,
                sourceText: "voice inc",
                proposedText: "VoiceInk",
                acceptedText: "VoiceInk",
                reason: "candidate-override",
                termIDs: ["product.voiceink"]
            )
        )
        context.insert(third)

        let clean = Transcription(text: "沒有修正", duration: 0, transcriptionStatus: .completed)
        clean.timestamp = now.addingTimeInterval(-120)
        context.insert(clean)

        let old = Transcription(text: "很久以前", duration: 0, transcriptionStatus: .completed)
        old.timestamp = now.addingTimeInterval(-60 * 60 * 24 * 30)
        old.recordCorrectionFeedback(
            CorrectionFeedbackSignal(
                kind: .candidateSelection,
                sourceText: "old",
                acceptedText: "old",
                reason: "candidate-confirmed",
                termIDs: ["product.voiceink"]
            )
        )
        context.insert(old)

        try context.save()

        let profile = VocoCorrectionRiskService.profile(in: context, now: now, lookbackDays: 14)
        #expect(profile.recentSessionCount == 4)
        #expect(profile.correctedSessionCount == 2)
        #expect(abs(profile.recentCorrectionRate - 0.5) < 0.0001)
        #expect(profile.highRiskTermIDs == ["product.voiceink"])
        #expect(profile.hasElevatedCorrectionRate)
    }

    @Test func confidenceGateUsesRecentCorrectionRiskForReview() async throws {
        let result = VocoCanonicalizationService().normalize("我現在用 voice ink 做測試")
        let riskProfile = VocoCorrectionRiskProfile(
            recentSessionCount: 4,
            correctedSessionCount: 2,
            recentCorrectionRate: 0.5,
            highRiskTermIDs: ["product.voiceink"],
            lookbackDays: 14,
            minimumSampleCount: 3
        )
        let assessment = VocoConfidenceGateService().assess(
            normalizationResult: result,
            rawTranscript: result.originalText,
            correctionRiskProfile: riskProfile
        )

        #expect(assessment.route == .reviewSuggested)
        #expect(assessment.reasons.contains("recent-correction-rate"))
        #expect(assessment.reasons.contains("recent-term-corrections"))
        #expect(assessment.reviewTriggers.map(\.id).contains("recent-correction-rate"))
        #expect(assessment.reviewTriggers.map(\.id).contains("recent-term-corrections"))
        #expect(
            VocoReviewTriggerDisplayFormatter
                .summaries(for: assessment.reviewTriggers)
                .contains("Recent corrections (50% recent correction rate)")
        )
        #expect(assessment.correctionRiskProfile == riskProfile)
        #expect(assessment.hypothesisDetails.first?.reasons.contains("recent-term-corrections") == true)
    }

    @Test func transcriptionStoresCorrectionRiskMetadata() async throws {
        let result = VocoCanonicalizationService().normalize("我現在用 voice ink 做測試")
        let riskProfile = VocoCorrectionRiskProfile(
            recentSessionCount: 5,
            correctedSessionCount: 2,
            recentCorrectionRate: 0.4,
            highRiskTermIDs: ["product.voiceink"],
            lookbackDays: 14,
            minimumSampleCount: 3
        )
        let assessment = VocoConfidenceGateService().assess(
            normalizationResult: result,
            rawTranscript: result.originalText,
            correctionRiskProfile: riskProfile
        )
        let transcription = Transcription(text: "", duration: 0)

        transcription.recordASRMetadata(
            rawTranscript: result.originalText,
            normalizationResult: result,
            confidenceAssessment: assessment,
            asrEngineID: "qwen3:Qwen3-ASR",
            languageMode: "auto"
        )

        #expect(transcription.correctionRiskRate == 0.4)
        #expect(transcription.correctionRiskSampleCount == 5)
        #expect(transcription.correctionRiskCorrectedCount == 2)
        #expect(transcription.correctionRiskTermIDs == ["product.voiceink"])
        #expect(transcription.reviewTriggers.map(\.id).contains("recent-correction-rate"))
        #expect(transcription.reviewTriggers.map(\.id).contains("recent-term-corrections"))
    }

    @Test func contextPackEnabledIDsDefaultAndPersist() async throws {
        let suiteName = "VocoCanonicalizationTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }

        #expect(VocoCanonicalizationService.enabledContextPackIDs(defaults: defaults) == VocoCanonicalizationService.defaultActiveContextIDs)

        VocoCanonicalizationService.setEnabledContextPackIDs([], defaults: defaults)
        #expect(VocoCanonicalizationService.enabledContextPackIDs(defaults: defaults).isEmpty)

        VocoCanonicalizationService.setEnabledContextPackIDs(["custom"], defaults: defaults)
        #expect(VocoCanonicalizationService.enabledContextPackIDs(defaults: defaults) == ["custom"])
    }

    @Test func backupFileDecodesLegacyGeneralSettingsWithoutContextPackSelection() throws {
        let data = Data(
            """
            {
              "version": "1.79",
              "generalSettings": {
                "llmUserContext": "mixed language dictation"
              }
            }
            """.utf8
        )

        let backup = try JSONDecoder().decode(BackupFile.self, from: data)

        #expect(backup.generalSettings?.llmUserContext == "mixed language dictation")
        #expect(backup.generalSettings?.enabledContextPackIDs == nil)
    }

    @Test func backupFilePreservesContextPackSelection() throws {
        let expectedIDs = [
            VocoCanonicalizationService.defaultContextPackID,
            "custom.personal",
        ]
        let data = Data(
            """
            {
              "version": "1.79",
              "generalSettings": {
                "enabledContextPackIDs": [
                  "\(expectedIDs[0])",
                  "\(expectedIDs[1])"
                ]
              }
            }
            """.utf8
        )

        let backup = try JSONDecoder().decode(BackupFile.self, from: data)
        let encoded = try JSONEncoder().encode(backup)
        let roundTripped = try JSONDecoder().decode(BackupFile.self, from: encoded)

        #expect(backup.generalSettings?.enabledContextPackIDs == expectedIDs)
        #expect(roundTripped.generalSettings?.enabledContextPackIDs == expectedIDs)
    }

    @Test func contextPackDisplayMetadataIsReadable() async throws {
        let pack = try #require(VocoCanonicalizationService.builtInContextPacks.first)

        #expect(pack.displayName == "VOCO Development")
        #expect(pack.aliasCount > pack.terms.count)
        #expect(pack.contextRequiredTermCount > 0)
        #expect(pack.canonicalPreview.contains("VoiceInk"))

        let names = VocoCanonicalizationService.contextDisplayNames(for: [
            VocoCanonicalizationService.defaultContextPackID,
            "power-mode:123",
            "custom.context",
        ])

        #expect(names == ["VOCO Development", "Power Mode", "custom.context"])
    }

    @Test func powerModeDefaultLanguageStaysAutoFirst() throws {
        #expect(PowerModeConfig.defaultSelectedLanguage(storedLanguage: nil) == "auto")
        #expect(PowerModeConfig.defaultSelectedLanguage(storedLanguage: "ja") == "ja")
    }

    @Test func canonicalizationUsesEnabledWordReplacementsAsPersonalDictionaryTerms() async throws {
        let terms = VocoCanonicalizationService.wordReplacementTerms(
            from: [
                WordReplacement(originalText: "snow mode, snowmode", replacementText: "SnowMode"),
                WordReplacement(originalText: "ghost term", replacementText: "GhostTerm", isEnabled: false),
            ]
        )

        let term = try #require(terms.first)
        #expect(terms.count == 1)
        #expect(term.canonical == "SnowMode")
        #expect(term.aliases == ["snow mode", "snowmode"])
        #expect(term.type == "word-replacement")
        #expect(term.contexts == ["personal-dictionary"])

        let result = VocoCanonicalizationService(contextPacks: []).normalize(
            "我今天開 snow mode 不是 ghost term",
            activeContextIDs: [],
            additionalTerms: terms
        )
        #expect(result.normalizedText == "我今天開 SnowMode 不是 ghost term")
        #expect(result.replacements.count == 1)
        #expect(result.replacements.first?.originalText == "snow mode")
        #expect(result.replacements.first?.replacementText == "SnowMode")
        #expect(result.replacements.first?.termID.hasPrefix("word-replacement.") == true)
    }

    @Test func personalStyleGuardAllowsPlainMixedLanguageEditing() async throws {
        let result = PersonalStyleGuardService().validate(
            response: "我覺得這個 approach unstable，要 go around。",
            original: "我覺得這個 approach unstable 要 go around"
        )

        #expect(result.isValid)
    }

    @Test func personalStyleGuardRejectsAssistantOpeners() async throws {
        let result = PersonalStyleGuardService().validate(
            response: "以下是我整理後的版本：我覺得這個改法可以跑。",
            original: "我覺得這個改法可以跑"
        )

        #expect(result.isValid == false)
        #expect(result.reasons.contains(where: { $0.hasPrefix("assistant-opener") }))
    }

    @Test func personalStyleGuardRejectsUnrequestedListFormatting() async throws {
        let result = PersonalStyleGuardService().validate(
            response: "我覺得這個改法可以跑，但我不太喜歡。\n- 它把問題藏到呼叫端\n- 之後每個地方都要補判斷",
            original: "我覺得這個改法可以跑但我不太喜歡它把問題藏到呼叫端之後每個地方都要補判斷"
        )

        #expect(result.isValid == false)
        #expect(result.reasons.contains("introduced-structured-format"))
    }

    @Test func personalStyleGuardAllowsRequestedListFormatting() async throws {
        let result = PersonalStyleGuardService().validate(
            response: "重點如下：\n- 保留 raw transcript\n- 記錄 normalized transcript",
            original: "幫我條列重點保留 raw transcript 還有記錄 normalized transcript"
        )

        #expect(result.isValid)
    }

    @Test func personalStyleGuardRejectsDroppedMixedLanguageTerms() async throws {
        let result = PersonalStyleGuardService().validate(
            response: "我覺得這次進場不穩，應該重來。",
            original: "我覺得這次 approach unstable 應該 go around"
        )

        #expect(result.isValid == false)
        #expect(result.reasons.contains(where: { $0.hasPrefix("dropped-mixed-language-term") }))
    }

    @Test func transcriptionStoresStyleGuardRejection() async throws {
        let transcription = Transcription(text: "我覺得這個改法可以跑", duration: 0)

        transcription.recordStyleGuardRejection(
            response: "以下是我整理後的版本：我覺得這個改法可以跑。",
            reasons: ["assistant-opener:以下是"]
        )

        #expect(transcription.styleGuardRejectedText == "以下是我整理後的版本：我覺得這個改法可以跑。")
        #expect(transcription.styleGuardReasons == ["assistant-opener:以下是"])
    }

    @Test func retranscriptionAnalyticsDetectsChangeCategoriesAndConfidenceDelta() async throws {
        let unchanged = RetranscriptionAnalyticsService.analyze(
            sourceText: "我現在用 VoiceInk",
            retranscribedText: "我現在用 VoiceInk",
            sourceConfidenceScore: 0.8,
            retranscribedConfidenceScore: 0.9
        )

        #expect(unchanged.changeCategory == .unchanged)
        #expect(unchanged.editDistance == 0)
        #expect(abs((unchanged.confidenceDelta ?? 0) - 0.1) < 0.0001)

        let meaningful = RetranscriptionAnalyticsService.analyze(
            sourceText: "我今天要測試 voice anc",
            retranscribedText: "我今天要測試 VoiceInk 的 Qwen3-ASR",
            sourceConfidenceScore: nil,
            retranscribedConfidenceScore: nil
        )

        #expect(meaningful.changeCategory == .meaningfulChange)
        #expect(meaningful.changeRatio > 0.12)
        #expect(meaningful.confidenceDelta == nil)
    }

    @Test func transcriptionStoresRetranscriptionAnalysis() async throws {
        let source = Transcription(
            text: "我今天要測試 voice anc",
            duration: 1,
            confidenceScore: 0.6
        )
        let retranscribed = Transcription(
            text: "我今天要測試 VoiceInk",
            duration: 1,
            confidenceScore: 0.9
        )

        retranscribed.recordRetranscriptionAnalysis(source: source)

        let analysis = try #require(retranscribed.retranscriptionAnalysis)
        #expect(retranscribed.sourceTranscriptionID == source.id)
        #expect(retranscribed.retranscriptionSourceText == source.text)
        #expect(abs((analysis.confidenceDelta ?? 0) - 0.3) < 0.0001)
        #expect(retranscribed.userCorrectionDistance == analysis.changeRatio)

        let signal = try #require(retranscribed.correctionFeedback.first)
        #expect(signal.kind == .retranscriptionChange)
        #expect(signal.sourceText == "我今天要測試 voice anc")
        #expect(signal.acceptedText == "我今天要測試 VoiceInk")
        #expect(signal.reason == "retranscription-\(analysis.changeCategory.rawValue)")
    }

    @Test func correctionFeedbackCapturesUserSubstitution() async throws {
        let signal = try #require(
            CorrectionFeedbackService.userSubstitutionSignal(
                WordSubstitution(original: "voice anc", replacement: "VoiceInk")
            )
        )

        #expect(signal.kind == .userSubstitution)
        #expect(signal.sourceText == "voice anc")
        #expect(signal.acceptedText == "VoiceInk")
        #expect(signal.reason == "user-substitution")
    }

    @Test @MainActor func correctionFeedbackLearningStagesCandidateOverride() async throws {
        let context = try makeDictionaryContext()
        let signal = CorrectionFeedbackSignal(
            kind: .candidateSelection,
            sourceText: "我剛剛用 voice anc 測一下",
            proposedText: "我剛剛用 voice anc 測一下",
            acceptedText: "我剛剛用 VoiceInk 測一下",
            confidenceScore: 0.64,
            changeRatio: 0.12,
            reason: "candidate-override"
        )

        let staged = CorrectionFeedbackLearningService.stageLearningCandidates(from: signal, in: context)

        #expect(staged.count == 1)
        #expect(staged.first?.original == "voice anc")
        #expect(staged.first?.replacement == "VoiceInk")

        let entries = try context.fetch(FetchDescriptor<WordReplacement>())
        let entry = try #require(entries.first)
        #expect(entry.originalText == "voice anc")
        #expect(entry.replacementText == "VoiceInk")
        #expect(entry.source == WordReplacement.sourceCorrectionFeedback)
        #expect(entry.isEnabled == false)
        #expect(entry.hitCount == 1)
    }

    @Test @MainActor func correctionFeedbackLearningStagesTypedCJKSpanRescue() async throws {
        let context = try makeDictionaryContext()
        let signal = CorrectionFeedbackSignal(
            kind: .candidateSelection,
            sourceText: "今天看到焰很大",
            proposedText: "今天看到焰很大",
            acceptedText: "今天看到火焰很大",
            confidenceScore: 0.61,
            changeRatio: 0.14,
            reason: "candidate-custom"
        )

        let staged = CorrectionFeedbackLearningService.stageLearningCandidates(from: signal, in: context)

        #expect(staged.count == 1)
        #expect(staged.first?.original == "焰")
        #expect(staged.first?.replacement == "火焰")

        let entry = try #require(try context.fetch(FetchDescriptor<WordReplacement>()).first)
        #expect(entry.originalText == "焰")
        #expect(entry.replacementText == "火焰")
        #expect(entry.source == WordReplacement.sourceCorrectionFeedback)
        #expect(entry.isEnabled == false)
    }

    @Test @MainActor func correctionFeedbackLearningStagesTypedKanaCanonicalRescue() async throws {
        let context = try makeDictionaryContext()
        let signal = CorrectionFeedbackSignal(
            kind: .candidateSelection,
            sourceText: "我喜歡あけぼし",
            proposedText: "我喜歡あけぼし",
            acceptedText: "我喜歡明け星",
            confidenceScore: 0.58,
            changeRatio: 0.2,
            reason: "candidate-custom"
        )

        let staged = CorrectionFeedbackLearningService.stageLearningCandidates(from: signal, in: context)

        #expect(staged.count == 1)
        #expect(staged.first?.original == "あけぼし")
        #expect(staged.first?.replacement == "明け星")

        let entry = try #require(try context.fetch(FetchDescriptor<WordReplacement>()).first)
        #expect(entry.originalText == "あけぼし")
        #expect(entry.replacementText == "明け星")
        #expect(entry.source == WordReplacement.sourceCorrectionFeedback)
        #expect(entry.isEnabled == false)
    }

    @Test @MainActor func correctionFeedbackLearningSkipsBroadCharacterRewrite() async throws {
        let context = try makeDictionaryContext()
        let signal = CorrectionFeedbackSignal(
            kind: .candidateSelection,
            sourceText: "今天看到焰很大",
            proposedText: "今天看到焰很大",
            acceptedText: "我想改成另一句完全不同",
            confidenceScore: 0.4,
            changeRatio: 0.8,
            reason: "candidate-custom"
        )

        let staged = CorrectionFeedbackLearningService.stageLearningCandidates(from: signal, in: context)
        let entries = try context.fetch(FetchDescriptor<WordReplacement>())

        #expect(staged.isEmpty)
        #expect(entries.isEmpty)
    }

    @Test @MainActor func correctionFeedbackLearningSkipsCandidateConfirmation() async throws {
        let context = try makeDictionaryContext()
        let signal = CorrectionFeedbackSignal(
            kind: .candidateSelection,
            sourceText: "voice ink",
            proposedText: "VoiceInk",
            acceptedText: "VoiceInk",
            confidenceScore: 0.82,
            changeRatio: 0.1,
            reason: "candidate-confirmed"
        )

        let staged = CorrectionFeedbackLearningService.stageLearningCandidates(from: signal, in: context)
        let entries = try context.fetch(FetchDescriptor<WordReplacement>())

        #expect(staged.isEmpty)
        #expect(entries.isEmpty)
    }

    @Test @MainActor func correctionFeedbackLearningSkipsCandidateTimeoutFallback() async throws {
        let context = try makeDictionaryContext()
        let signal = CorrectionFeedbackSignal(
            kind: .candidateSelection,
            sourceText: "我剛剛用 voice anc 測一下",
            proposedText: "我剛剛用 VoiceInk 測一下",
            acceptedText: "我剛剛用 VoiceInk 測一下",
            confidenceScore: 0.64,
            changeRatio: 0.12,
            reason: "candidate-timeout-fallback"
        )

        let staged = CorrectionFeedbackLearningService.stageLearningCandidates(from: signal, in: context)
        let entries = try context.fetch(FetchDescriptor<WordReplacement>())

        #expect(staged.isEmpty)
        #expect(entries.isEmpty)
    }

    @Test @MainActor func correctionFeedbackLearningPromotesRepeatedRetranscriptionCorrection() async throws {
        let context = try makeDictionaryContext()
        let signal = CorrectionFeedbackSignal(
            kind: .retranscriptionChange,
            sourceText: "我今天要測 voice anc",
            acceptedText: "我今天要測 VoiceInk",
            confidenceScore: 0.9,
            changeRatio: 0.18,
            reason: "retranscription-meaningfulChange"
        )

        for _ in 0..<3 {
            _ = CorrectionFeedbackLearningService.stageLearningCandidates(from: signal, in: context)
        }

        let entries = try context.fetch(FetchDescriptor<WordReplacement>())
        let entry = try #require(entries.first)
        #expect(entry.originalText == "voice anc")
        #expect(entry.replacementText == "VoiceInk")
        #expect(entry.source == WordReplacement.sourceCorrectionFeedback)
        #expect(entry.hitCount == 3)
        #expect(entry.isEnabled)
    }

    @Test func wordReplacementLearningStateDisplaysProgressAndApproves() async throws {
        let replacement = WordReplacement(
            originalText: "voice anc",
            replacementText: "VoiceInk",
            isEnabled: false,
            source: WordReplacement.sourceCorrectionFeedback
        )
        replacement.hitCount = 2

        #expect(replacement.isLearningCandidate)
        #expect(replacement.sourceDisplayName == "Feedback")
        #expect(replacement.learningProgressLabel == "2/3")

        replacement.approveLearningCandidate()

        #expect(replacement.isEnabled)
        #expect(replacement.source == WordReplacement.sourceUser)
        #expect(replacement.sourceDisplayName == "User")
        #expect(replacement.learningProgressLabel == nil)
    }

    @Test func editModePollingStateCoalescesAcrossRestartWhilePollIsInFlight() async throws {
        var state = EditModePollingState()

        let firstGenerationResult = state.startPolling()
        let firstGeneration = try #require(firstGenerationResult)
        #expect(state.beginNextRefresh() == firstGeneration)
        #expect(state.enqueueRefresh() == .queuedBehindInFlight)

        state.stopPolling()
        #expect(state.shouldApplyResult(for: firstGeneration) == false)

        let secondGenerationResult = state.startPolling()
        let secondGeneration = try #require(secondGenerationResult)
        #expect(secondGeneration > firstGeneration)
        #expect(state.enqueueRefresh() == .coalesced)

        state.finishRefresh()
        #expect(state.beginNextRefresh() == secondGeneration)
    }

    @Test func editModePollingStateIgnoresRefreshRequestsWhenStopped() async throws {
        var state = EditModePollingState()

        #expect(state.enqueueRefresh() == .ignoredStopped)

        let generationResult = state.startPolling()
        let generation = try #require(generationResult)
        #expect(state.shouldContinuePolling(expectedGeneration: generation) == true)

        state.stopPolling()
        #expect(state.shouldContinuePolling(expectedGeneration: generation) == false)
        #expect(state.shouldApplyResult(for: generation) == false)
        #expect(state.enqueueRefresh() == .ignoredStopped)
    }

}

@MainActor
private func makeDictionaryContext() throws -> ModelContext {
    let schema = Schema([VocabularyWord.self, WordReplacement.self])
    let config = ModelConfiguration(
        "dictionary-test-\(UUID().uuidString)",
        schema: schema,
        isStoredInMemoryOnly: true
    )
    let container = try ModelContainer(for: schema, configurations: [config])
    return ModelContext(container)
}

@MainActor
private func makeTranscriptionContext() throws -> ModelContext {
    let schema = Schema([Transcription.self])
    let storeURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("transcription-test-\(UUID().uuidString).store")
    let config = ModelConfiguration(
        "transcription-test-\(UUID().uuidString)",
        schema: schema,
        url: storeURL,
        cloudKitDatabase: .none
    )
    let container = try ModelContainer(for: schema, configurations: [config])
    return ModelContext(container)
}

@MainActor
private func makeCanonicalizationPipelineContext() throws -> ModelContext {
    let schema = Schema([Transcription.self, VocabularyWord.self, WordReplacement.self])
    let storeURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("canonicalization-pipeline-test-\(UUID().uuidString).store")
    let config = ModelConfiguration(
        "canonicalization-pipeline-test-\(UUID().uuidString)",
        schema: schema,
        url: storeURL,
        cloudKitDatabase: .none
    )
    let container = try ModelContainer(for: schema, configurations: [config])
    return ModelContext(container)
}

@MainActor
private func makeSessionMetricContext() throws -> ModelContext {
    let schema = Schema([Transcription.self, SessionMetric.self])
    let storeURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("session-metric-test-\(UUID().uuidString).store")
    let config = ModelConfiguration(
        "session-metric-test-\(UUID().uuidString)",
        schema: schema,
        url: storeURL,
        cloudKitDatabase: .none
    )
    let container = try ModelContainer(for: schema, configurations: [config])
    return ModelContext(container)
}

private func makeSessionMetricDictationOutput() -> (result: VocoNormalizationResult, assessment: VocoConfidenceAssessment) {
    let result = VocoNormalizationResult(
        originalText: "我現在用 voice ink 然後可能是 voco",
        normalizedText: "我現在用 VoiceInk 然後可能是 voco",
        activeContextIDs: [
            VocoCanonicalizationService.defaultContextPackID,
            "power-mode:123",
        ],
        replacements: [
            VocoReplacement(
                originalText: "voice ink",
                replacementText: "VoiceInk",
                termID: "product.voiceink",
                confidence: 0.97,
                reason: "alias-match",
                rangeStart: 4,
                rangeLength: 9
            ),
        ],
        suggestions: [
            VocoReplacement(
                originalText: "voco",
                replacementText: "VOCO",
                termID: "product.voco",
                confidence: 0.55,
                reason: "inactive-context-suggestion",
                rangeStart: 19,
                rangeLength: 4
            ),
        ]
    )
    let assessment = VocoConfidenceGateService().assess(
        normalizationResult: result,
        rawTranscript: result.originalText
    )
    return (result, assessment)
}
