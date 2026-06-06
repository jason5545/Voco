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
        #expect(transcription.hypotheses.first == "我現在用 Qwen3-ASR 的 MLX 版本")
        #expect(transcription.hypothesisLabels.first == "Recommended")
        #expect(transcription.hypothesisDetails.first?.source == .autoContext)
        #expect(transcription.hypothesisDetails.first?.appliedTermIDs.contains("model.qwen3-asr") == true)
        #expect(transcription.selectedCandidate == "我現在用 Qwen3-ASR 的 MLX 版本")
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
            reasons: ["unresolved-suggestions", "high-risk-term", "unresolved-suggestions"]
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
        let assessment = VocoConfidenceAssessment(
            score: 0.7,
            route: .reviewSuggested,
            reasons: ["unresolved-suggestions"],
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
        #expect(transcription.normalizedTranscript == "今天看到火焰很大")
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
        #expect(transcription.normalizedTranscript == "今天看到炎很大")
        #expect(transcription.selectedCandidate == "今天看到炎很大")
        #expect(transcription.userCorrectionDistance == signal.changeRatio)
        #expect(transcription.correctionFeedback.count == 1)
        #expect(signal.reason == "candidate-override")
        #expect(signal.isCorrectiveSignal)
        #expect(signal.termIDs.contains("song.homura"))
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
        #expect(transcription.normalizedTranscript == "今天看到炎很大")
        #expect(transcription.selectedCandidate == "今天看到炎很大")
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
    let config = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)
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
