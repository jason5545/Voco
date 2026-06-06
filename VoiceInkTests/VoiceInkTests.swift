//
//  VoiceInkTests.swift
//  VoiceInkTests
//
//  Created by Prakash Joshi on 15/10/2024.
//

import Foundation
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
    }

    @Test func confidenceGateSuggestsReviewForUnresolvedAmbiguousTerms() async throws {
        let result = VocoCanonicalizationService().normalize("今天看到焰很大")
        let assessment = VocoConfidenceGateService().assess(normalizationResult: result, rawTranscript: result.originalText)

        #expect(result.normalizedText == "今天看到焰很大")
        #expect(result.suggestions.contains(where: { $0.replacementText == "炎" }))
        #expect(assessment.route == .reviewSuggested)
        #expect(assessment.reasons.contains("unresolved-suggestions"))
        #expect(assessment.candidates.contains("今天看到炎很大"))
    }

    @Test func confidenceGateSuggestsReviewForHeavyNormalization() async throws {
        let result = VocoCanonicalizationService().normalize("voice ink voco qwen three asr mlx")
        let assessment = VocoConfidenceGateService().assess(normalizationResult: result, rawTranscript: result.originalText)

        #expect(result.replacements.count >= 4)
        #expect(assessment.route == .reviewSuggested)
        #expect(assessment.reasons.contains("heavy-normalization"))
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
        #expect(transcription.selectedCandidate == "我現在用 Qwen3-ASR 的 MLX 版本")
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
