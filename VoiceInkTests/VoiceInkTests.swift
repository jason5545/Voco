//
//  VoiceInkTests.swift
//  VoiceInkTests
//
//  Created by Prakash Joshi on 15/10/2024.
//

import Foundation
import ApplicationServices
import os
import SwiftData
import Testing
@testable import Voco

@Suite(.serialized)
struct VoiceInkTests {

    private func requireLoadedPinyinDatabase() async throws {
        for _ in 0..<100 {
            if PinyinDatabase.shared.isLoaded { return }
            try await Task.sleep(nanoseconds: 50_000_000)
        }

        try #require(PinyinDatabase.shared.isLoaded)
    }

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

    @Test func validatorRejectsInsertedDisallowedMingdeTerm() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "所以你整體看我的構音障礙到底到了什麼程度？我越來越懷疑自己比我自己明德嚴重了。",
            original: "所以你整體看我的過癮障礙到底到了什麼程度？我越來越懷疑自己比我自己想的嚴重了。",
            insertedProtectedTerms: ["明德"]
        )

        #expect(result.isValid == false)
        #expect(result.reasons.contains("inserted-protected-term:明德"))
    }

    @Test func validatorAllowsMingdeWhenOriginalAlreadyContainsTerm() async throws {
        let complaint = LLMResponseValidator.shared.validate(
            response: "你看那個明德又出來了。",
            original: "你看那個明德又出來了。",
            insertedProtectedTerms: ["明德"]
        )
        #expect(complaint.isValid == true)

        let station = LLMResponseValidator.shared.validate(
            response: "我們在明德捷運站碰面。",
            original: "我們在明德捷運站碰面。",
            insertedProtectedTerms: ["明德"]
        )
        #expect(station.isValid == true)

        let reservoir = LLMResponseValidator.shared.validate(
            response: "我們去明德水庫旁邊。",
            original: "我們去明德水庫旁邊。",
            insertedProtectedTerms: ["明德"]
        )
        #expect(reservoir.isValid == true)

        let juniorHighSchool = LLMResponseValidator.shared.validate(
            response: "我們在明德國中旁邊碰面。",
            original: "我們在明德國中旁邊碰面。",
            insertedProtectedTerms: ["明德"]
        )
        #expect(juniorHighSchool.isValid == true)
    }

    @Test func validatorRejectsInsertedMingdeEvenInsidePreviouslyAllowlistedPhrase() async throws {
        let station = LLMResponseValidator.shared.validate(
            response: "我們在明德捷運站碰面。",
            original: "我們在捷運站碰面。",
            insertedProtectedTerms: ["明德"]
        )
        #expect(station.isValid == false)
        #expect(station.reasons.contains("inserted-protected-term:明德"))

        let reservoir = LLMResponseValidator.shared.validate(
            response: "我們去明德水庫旁邊。",
            original: "我們去水庫旁邊。",
            insertedProtectedTerms: ["明德"]
        )
        #expect(reservoir.isValid == false)
        #expect(reservoir.reasons.contains("inserted-protected-term:明德"))

        let juniorHighSchool = LLMResponseValidator.shared.validate(
            response: "我們在明德國中旁邊碰面。",
            original: "我們在國中旁邊碰面。",
            insertedProtectedTerms: ["明德"]
        )
        #expect(juniorHighSchool.isValid == false)
        #expect(juniorHighSchool.reasons.contains("inserted-protected-term:明德"))
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

    @Test func validatorAllowsCorrectionToExplicitlySpelledLatinTerm() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "那個飛馳其實是英文的 phase，就是 phase 那個 phase。",
            original: "那個飛馳其實是英文的face，就是p h a s e那個face。"
        )

        #expect(result.isValid == true)
    }

    @Test func validatorStillRejectsLatinTermSwapWithoutSpellingEvidence() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "那個詞其實是 phase。",
            original: "那個詞其實是 face。"
        )

        #expect(result.isValid == false)
        #expect(result.reasons.contains(where: { $0.contains("dropped-term:face") }))
    }

    @Test func validatorAllowsSingleChineseNumeralConvertedToDigit() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "它的 M 跟 Max 之間少了一個 5。",
            original: "它的M跟Max之間少了一個五。"
        )

        #expect(result.isValid == true)
    }

    @Test func validatorAllowsChineseTensConvertedToDigits() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "因為你之前已經跑過 69 個飛馳了。",
            original: "因為你之前已經跑過六十九個飛馳了。"
        )

        #expect(result.isValid == true)
    }

    @Test func validatorAllowsLatinReplacementForSuspiciousCountedTerm() async throws {
        try await requireLoadedPinyinDatabase()

        let result = LLMResponseValidator.shared.validate(
            response: "我就問你一句話嘛，69 個 phase 到底有沒有這個詞嘛？",
            original: "我就問你一句話嘛，六十九個飛馳到底有沒有這個詞嘛？"
        )

        #expect(result.isValid == true)
    }

    @Test func validatorStillRejectsCommonCountedNounToLatin() async throws {
        try await requireLoadedPinyinDatabase()

        let result = LLMResponseValidator.shared.validate(
            response: "我有 69 個 phase 要處理。",
            original: "我有六十九個問題要處理。"
        )

        #expect(result.isValid == false)
        #expect(result.reasons.contains(where: { $0.contains("cross-script-substitution") }))
    }

    @Test func validatorAllowsChineseDecimalConvertedToDigits() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "版本是 3.5。",
            original: "版本是三點五。"
        )

        #expect(result.isValid == true)
    }

    @Test func validatorRejectsDroppedRetranscribeSkillPhrase() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "你再跑一次，重新轉錄的聲量。因為我發現最近這三筆又有東西可以改了。",
            original: "你再跑一次，重新轉錄的技能。因為我發現最近這三筆又有東西可以改了。",
            protectedTerms: CorrectionProtectionList.shared.allWords()
        )

        #expect(result.isValid == false)
        #expect(result.reasons.contains(where: { $0.contains("dropped-term:轉錄的技能") }))
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

        let assessmentChars = Array("身心障礙鑑定")
        let assessmentOffset = try #require(assessmentChars.firstIndex(of: "鑑"))

        #expect(CorrectionProtectionList.shared.contains("鑑定") == true)
        #expect(CorrectionProtectionList.shared.contains("轉錄") == true)
        #expect(CorrectionProtectionList.shared.contains("轉路") == true)
        #expect(CorrectionProtectionList.shared.containsProtectedTerm(in: "语音转录") == true)
        #expect(CorrectionProtectionList.shared.containsProtectedTerm(in: "你再跑一次转路的技能") == true)
        #expect(CorrectionProtectionList.shared.containsProtectedTerm(in: "你再跑一次轉錄的技能") == true)
        #expect(CorrectionProtectionList.shared.containsProtectedTerm(in: "retranscribe skill") == true)
        #expect(
            CorrectionProtectionList.shared.containsProtectedPhrase(
                in: assessmentChars,
                covering: assessmentOffset,
                length: 1
            ) == true
        )
    }

    @Test @MainActor func chinesePostProcessingPreservesAssessmentTerms() async throws {
        try await requireLoadedPinyinDatabase()

        let service = ChinesePostProcessingService.shared
        let oldOpenCC = service.isOpenCCEnabled
        let oldPinyin = service.isPinyinCorrectionEnabled
        let oldDataDriven = service.isDataDrivenCorrectionEnabled
        let oldNasal = service.isNasalCorrectionEnabled
        defer {
            service.isOpenCCEnabled = oldOpenCC
            service.isPinyinCorrectionEnabled = oldPinyin
            service.isDataDrivenCorrectionEnabled = oldDataDriven
            service.isNasalCorrectionEnabled = oldNasal
        }

        service.isOpenCCEnabled = true
        service.isPinyinCorrectionEnabled = true
        service.isDataDrivenCorrectionEnabled = true
        service.isNasalCorrectionEnabled = true

        let result = service.process("我不是要你鉴定什么的。如果真的要鉴定，我还有一个更严重的CP呢。")

        #expect(result.processedText.contains("鑑定"))
        #expect(!result.processedText.contains("簡訊"))
    }

    @Test @MainActor func chinesePostProcessingDoesNotPromoteModelPhrasesToMingde() async throws {
        try await requireLoadedPinyinDatabase()

        let service = ChinesePostProcessingService.shared
        let oldOpenCC = service.isOpenCCEnabled
        let oldPinyin = service.isPinyinCorrectionEnabled
        let oldDataDriven = service.isDataDrivenCorrectionEnabled
        let oldNasal = service.isNasalCorrectionEnabled
        defer {
            service.isOpenCCEnabled = oldOpenCC
            service.isPinyinCorrectionEnabled = oldPinyin
            service.isDataDrivenCorrectionEnabled = oldDataDriven
            service.isNasalCorrectionEnabled = oldNasal
        }

        service.isOpenCCEnabled = true
        service.isPinyinCorrectionEnabled = true
        service.isDataDrivenCorrectionEnabled = true
        service.isNasalCorrectionEnabled = true

        let localModel = service.process("我剛剛在訓練本地模型。")
        #expect(localModel.processedText.contains("本地模型"))

        let newModel = service.process("但是我們有試過從訓練好的 GPT 接著訓練新的模型，但是效果都不太好。")
        #expect(newModel.processedText.contains("新的模型"))

        let regularModel = service.process("因為我們現在已經有個規則性的模型來做這個事情了。")
        #expect(regularModel.processedText.contains("規則性的模型"))

        let newPrompt = service.process("我必須要開一個新的對話來避免對話污染。")
        #expect(newPrompt.processedText.contains("新的對話"))

        for result in [localModel, newModel, regularModel, newPrompt] {
            #expect(!result.processedText.contains("明德模型"))
        }
    }

    @Test @MainActor func chinesePostProcessingDoesNotPromoteCorrectRawThoughtPhraseToMingde() async throws {
        try await requireLoadedPinyinDatabase()

        let service = ChinesePostProcessingService.shared
        let oldOpenCC = service.isOpenCCEnabled
        let oldPinyin = service.isPinyinCorrectionEnabled
        let oldDataDriven = service.isDataDrivenCorrectionEnabled
        let oldNasal = service.isNasalCorrectionEnabled
        defer {
            service.isOpenCCEnabled = oldOpenCC
            service.isPinyinCorrectionEnabled = oldPinyin
            service.isDataDrivenCorrectionEnabled = oldDataDriven
            service.isNasalCorrectionEnabled = oldNasal
        }

        service.isOpenCCEnabled = true
        service.isPinyinCorrectionEnabled = true
        service.isDataDrivenCorrectionEnabled = true
        service.isNasalCorrectionEnabled = true

        let result = service.process("所以你整体看我的過癮障礙到底到了什麼程度？我越來越懷疑自己比我自己想的嚴重了。")

        #expect(result.processedText.contains("想的嚴重"))
        #expect(!result.processedText.contains("明德嚴重"))
    }

    @Test func validatorRejectsAggressiveShortRewrite() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "網葉斑",
            original: "网页版"
        )

        #expect(result.isValid == false)
        #expect(result.reasons.contains(where: { $0 == "short-edit-budget" }))
    }

    @Test func validatorRejectsCJKInsertionIntoLatinOnlyInput() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "Loading.\n漏頂",
            original: "Loading."
        )

        #expect(result.isValid == false)
        #expect(result.isRetryable == true)
        #expect(result.reasons.contains(where: { $0 == "latin-cjk-insertion" }))
    }

    @Test func canonicalizationNormalizesVocoDevelopmentTerms() async throws {
        let service = VocoCanonicalizationService()

        #expect(service.normalize("我現在用 voice ink 的 fork 做 voco").normalizedText == "我現在用 VoiceInk 的 fork 做 VOCO")
        #expect(service.normalize("我現在用 qwen three asr 的 mlx 版本").normalizedText == "我現在用 Qwen3-ASR 的 MLX 版本")
        #expect(service.normalize("我還是會留 whisper.cpp 支援").normalizedText == "我還是會留 whisper.cpp 支援")
        #expect(service.normalize("我現在跑 Mac OS 26 Tahoe").normalizedText == "我現在跑 macOS 26 Tahoe")
    }

    @Test func canonicalizationNormalizesCOIToCLIOnlyInCommandRepairContext() async throws {
        let service = VocoCanonicalizationService(autoApplyModelService: disabledAutoApplyModelService())

        let spaced = service.normalize("所以你可以用 C O I 再修一下。")
        #expect(spaced.normalizedText == "所以你可以用 CLI 再修一下。")
        #expect(spaced.replacements.contains {
            $0.termID == "tool.cli" &&
                $0.originalText == "C O I" &&
                $0.replacementText == "CLI"
        })

        let compact = service.normalize("所以你可以用 COI 再修一下。")
        #expect(compact.normalizedText == "所以你可以用 CLI 再修一下。")
        #expect(compact.replacements.contains {
            $0.termID == "tool.cli" &&
                $0.originalText == "COI" &&
                $0.replacementText == "CLI"
        })
    }

    @Test func canonicalizationDoesNotRewriteTrueCOIAcronymContext() async throws {
        let service = VocoCanonicalizationService(autoApplyModelService: disabledAutoApplyModelService())

        let result = service.normalize("這份 COI disclosure 要保留原本縮寫。")

        #expect(result.normalizedText == "這份 COI disclosure 要保留原本縮寫。")
        #expect(result.replacements.isEmpty)
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

    @Test func canonicalizationDoesNotUseVocabularyPhoneticsForPersonalVocabulary() async throws {
        try await requireLoadedPinyinDatabase()

        let service = VocoCanonicalizationService(contextPacks: [])
        let vocabulary = VocoCanonicalizationService.vocabularyTerms(from: ["明德", "王小明"])

        let regression = service.normalize(
            "我們最近的變更應該有加了自動學習的那個",
            additionalTerms: vocabulary
        )
        #expect(regression.normalizedText == "我們最近的變更應該有加了自動學習的那個")
        #expect(regression.replacements.isEmpty)

        let fullName = service.normalize("汪曉鳴", additionalTerms: vocabulary)
        #expect(fullName.normalizedText == "汪曉鳴")
        #expect(fullName.replacements.isEmpty)
        #expect(fullName.suggestions.isEmpty)
    }

    @Test func canonicalizationSuppressesKnownAmbiguousWordReplacementPair() async throws {
        let service = VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: disabledAutoApplyModelService()
        )
        let projectTerm = VocoCanonicalTerm(
            id: "word-replacement.test-project",
            canonical: "專案",
            aliases: ["圖案"],
            type: "word-replacement",
            contexts: ["personal-dictionary"],
            caseSensitive: true,
            autoReplaceThreshold: 0.97
        )
        let fileTerm = VocoCanonicalTerm(
            id: "word-replacement.test-file",
            canonical: "檔案",
            aliases: ["答案"],
            type: "word-replacement",
            contexts: ["personal-dictionary"],
            caseSensitive: true,
            autoReplaceThreshold: 0.97
        )

        let poster = service.normalize(
            "你沒有什麼特別的圖案或者是拍板要放上去嗎？",
            additionalTerms: [projectTerm]
        )
        #expect(poster.normalizedText == "你沒有什麼特別的圖案或者是拍板要放上去嗎？")
        #expect(poster.replacements.isEmpty)

        let importText = service.normalize(
            "把答案給我匯入會比較安全。",
            additionalTerms: [fileTerm]
        )
        #expect(importText.normalizedText == "把檔案給我匯入會比較安全。")
        #expect(importText.replacements.first?.replacementText == "檔案")
    }

    @Test func confidenceGateFlagsLocalCleanupRegressionWithoutSelectingRescue() async throws {
        try await requireLoadedPinyinDatabase()

        let result = VocoNormalizationResult(
            originalText: "這邊又有語音，語音辨識錯誤，所以你自己小振",
            normalizedText: "這邊又有語音，語音辨識錯誤，所以你自己小振",
            activeContextIDs: ["builtin.voco-development"],
            replacements: [],
            suggestions: []
        )

        let assessment = VocoConfidenceGateService.shared.assess(
            normalizationResult: result,
            rawTranscript: "这边又有语音，语音辨识错误，所以你自己修正。"
        )

        #expect(assessment.route == .reviewSuggested)
        #expect(assessment.reasons.contains("raw-cleanup-local-regression"))
        #expect(assessment.selectedCandidate == result.normalizedText)
        #expect(!assessment.hypothesisDetails.contains { $0.source == .customRescue })
    }

    @Test func confidenceGateFlagsSystemResourceCleanupRegressionWithoutSelectingRescue() async throws {
        try await requireLoadedPinyinDatabase()

        let result = VocoNormalizationResult(
            originalText: "而且，今天你知道 Mac OS 應該會有個支援耗盡的提示窗吧？但是現在的情況是完全沒有誒，是直接卡死呢。",
            normalizedText: "而且，今天你知道 Mac OS 應該會有個支援耗盡的提示窗吧？但是現在的情況是完全沒有誒，是直接卡死呢。",
            activeContextIDs: ["builtin.voco-development"],
            replacements: [],
            suggestions: []
        )

        let assessment = VocoConfidenceGateService.shared.assess(
            normalizationResult: result,
            rawTranscript: "而且，今天你知道 Mac OS 应该会有个资源耗尽的提示窗吧？但是现在的情况是完全没有诶，是直接卡死呢。"
        )

        #expect(assessment.route == .reviewSuggested)
        #expect(assessment.reasons.contains("raw-cleanup-local-regression"))
        #expect(assessment.selectedCandidate == result.normalizedText)
        #expect(!assessment.hypothesisDetails.contains { $0.source == .customRescue })
    }

    @Test func canonicalizationTreatsVocalAsContextRequiredVocoAlias() async throws {
        let service = VocoCanonicalizationService()

        let neutral = service.normalize("this vocal range is wide")
        #expect(neutral.normalizedText == "this vocal range is wide")
        #expect(neutral.replacements.isEmpty)
        let neutralSuggestion = try #require(neutral.suggestions.first)
        #expect(neutralSuggestion.originalText == "vocal")
        #expect(neutralSuggestion.replacementText == "VOCO")
        #expect(neutralSuggestion.termID == "product.voco.ambiguous-vocal")

        let contextual = service.normalize("我在 vocal dictation 模式測試")
        #expect(contextual.normalizedText == "我在 VOCO dictation 模式測試")
        let contextualReplacement = try #require(contextual.replacements.first)
        #expect(contextualReplacement.originalText == "vocal")
        #expect(contextualReplacement.replacementText == "VOCO")
        #expect(contextualReplacement.termID == "product.voco.ambiguous-vocal")
    }

    @Test func canonicalizationNormalizesEdgeCaseMisrecognition() async throws {
        let result = VocoCanonicalizationService().normalize("但是剛才就出現了一個 H case")

        #expect(result.normalizedText == "但是剛才就出現了一個 edge case")
        #expect(result.replacements.first?.termID == "term.edge-case")
    }

    @Test func canonicalizationNormalizesComfyUIOnlyWithImageGenerationContext() async throws {
        let service = VocoCanonicalizationService()

        let contextual = service.normalize("我記得那個 Config UI 它的用法就是流程圖這樣連來連去的，然後要一堆的 workflow。")
        #expect(contextual.normalizedText == "我記得那個 ComfyUI 它的用法就是流程圖這樣連來連去的，然後要一堆的 workflow。")
        #expect(contextual.replacements.first?.termID == "app.comfyui")

        let neutral = service.normalize("請先打開 config.yml 那個設定檔。")
        #expect(neutral.normalizedText == "請先打開 config.yml 那個設定檔。")
        #expect(neutral.replacements.isEmpty)
    }

    @Test func canonicalizationUsesModeContextHintsForAmbiguousTerms() async throws {
        let service = VocoCanonicalizationService()
        let text = "今天看到 homura 很亮"

        let neutral = service.normalize(text)
        #expect(neutral.normalizedText == text)
        #expect(neutral.replacements.isEmpty)
        #expect(neutral.suggestions.contains(where: { $0.replacementText == "炎" }))

        let mode = ModeConfig(
            name: "LiSA music notes",
            icon: .emoji("M"),
            appConfigs: [AppConfig(bundleIdentifier: "com.apple.Music", appName: "Music")],
            urlConfigs: [URLConfig(url: "youtube.com")],
            isAIEnhancementEnabled: false,
            selectedLanguage: "auto"
        )
        let contextual = service.normalize(
            text,
            contextHints: VocoCanonicalizationService.modeContextHints(from: mode)
        )

        #expect(contextual.normalizedText == "今天看到炎很亮")
        #expect(contextual.replacements.first?.termID == "song.homura")
        #expect(contextual.replacements.first?.replacementText == "炎")
    }

    @Test func canonicalizationUsesAppWindowContextHintsForAmbiguousTerms() async throws {
        let service = VocoCanonicalizationService()
        let text = "今天看到 homura 很亮"
        let hints = VocoCanonicalizationService.contextHints(
            mode: nil,
            appName: "Music",
            windowTitle: "LiSA playlist"
        )
        let contextual = service.normalize(text, contextHints: hints)

        #expect(contextual.normalizedText == "今天看到炎很亮")
        #expect(contextual.replacements.first?.termID == "song.homura")
    }

    @Test func canonicalizationDoesNotExpandCanonicalCJKPhrases() async throws {
        let service = VocoCanonicalizationService()

        #expect(service.normalize("我昨天又看了鬼滅之刃").normalizedText == "我昨天又看了鬼滅之刃")
    }

    @Test @MainActor func chinesePostProcessingRoutesSuspiciousCountedTermToLLM() async throws {
        try await requireLoadedPinyinDatabase()

        let service = ChinesePostProcessingService.shared
        let oldRouting = service.isConfidenceRoutingEnabled
        let oldProvider = service.lastModelProvider
        let oldAvgLogProb = service.lastAvgLogProb
        let oldUncertainWords = service.lastUncertainWords
        let oldWordConfidences = service.lastWordConfidences
        defer {
            service.isConfidenceRoutingEnabled = oldRouting
            service.lastModelProvider = oldProvider
            service.lastAvgLogProb = oldAvgLogProb
            service.lastUncertainWords = oldUncertainWords
            service.lastWordConfidences = oldWordConfidences
        }

        service.isConfidenceRoutingEnabled = true
        service.lastModelProvider = .qwen3
        service.lastAvgLogProb = -0.1
        service.lastUncertainWords = []
        service.lastWordConfidences = []

        let shouldSkip = service.shouldSkipLLMEnhancement(text: "我有六十九個飛馳要跑。")

        #expect(shouldSkip == false)
        #expect(service.lastUncertainWords.contains(where: { $0.text == "飛馳" }))
    }

    @Test func pinyinCorrectorFixesObviousTherapyMisrecognitions() async throws {
        try await requireLoadedPinyinDatabase()

        let therapyContext = CorrectionContext(
            recentTranscriptions: ["心理智商", "下個禮拜帶給他", "焦慮"],
            appName: nil,
            windowTitle: nil
        )

        #expect(PinyinCorrector.shared.correct("心理智商。").text == "心理諮商。")
        #expect(
            PinyinCorrector.shared.correct(
                "目前討論的結果是再努力一個月，同時繼續去做持倉。",
                context: therapyContext
            ).text == "目前討論的結果是再努力一個月，同時繼續去做諮商。"
        )
        #expect(
            PinyinCorrector.shared.correct(
                "但是我對智障是就不會這樣啊！",
                context: therapyContext
            ).text == "但是我對諮商師就不會這樣啊！"
        )
    }

    @Test func pinyinCorrectorFixesContextualSessionAndUITerms() async throws {
        try await requireLoadedPinyinDatabase()

        let sessionContext = CorrectionContext(
            recentTranscriptions: ["十個小時的 session 已經爆掉了"],
            appName: "Codex",
            windowTitle: "Voco retranscribe"
        )
        let uiContext = CorrectionContext(
            recentTranscriptions: ["app server 的 UI", "第三方 UI 接 Codex Server"],
            appName: "Codex",
            windowTitle: nil
        )

        #expect(PinyinCorrector.shared.correct("結果昨天八點多才回到夾，蘋果店早就關了。").text == "結果昨天八點多才回到家，蘋果店早就關了。")
        #expect(PinyinCorrector.shared.correct("這是從之前的氣聲接過來的。", context: sessionContext).text == "這是從之前的 session 接過來的。")
        #expect(PinyinCorrector.shared.correct("或許這並非有微弱，原因到底是什麼？", context: uiContext).text == "或許這並非 UI 問題，原因到底是什麼？")
    }

    @Test func pinyinCorrectorFixesRecentRetranscribeContextMisses() async throws {
        try await requireLoadedPinyinDatabase()

        let correctionContext = CorrectionContext(
            recentTranscriptions: ["這邊有語音辨識的錯誤"],
            appName: "Codex",
            windowTitle: "Voco retranscribe"
        )
        let freezeContext = CorrectionContext(
            recentTranscriptions: ["畫面全部都不動，連鍵盤、連滑鼠，什麼都不動"],
            appName: nil,
            windowTitle: nil
        )
        let systemContext = CorrectionContext(
            recentTranscriptions: ["Activity Monitor 正常，但是系統直接卡死"],
            appName: nil,
            windowTitle: "macOS resource issue"
        )
        let dataImportContext = CorrectionContext(
            recentTranscriptions: ["demo 的資料只有一筆", "欄位格式", "去年今年檔案", "準備匯入作業"],
            appName: nil,
            windowTitle: nil
        )
        let virtualizationContext = CorrectionContext(
            recentTranscriptions: ["Windows VM", "Virtual Machine", "虛擬機器", "串流日誌"],
            appName: "Codex",
            windowTitle: nil
        )
        let sessionContext = CorrectionContext(
            recentTranscriptions: ["SESSION 跑了七十幾輪", "Codex session 已經很長"],
            appName: "Codex",
            windowTitle: nil
        )
        let imageGenerationContext = CorrectionContext(
            recentTranscriptions: ["產圖介面是節點 workflow", "流程圖這樣連來連去"],
            appName: "Draw Things",
            windowTitle: nil
        )
        let templeDataContext = CorrectionContext(
            recentTranscriptions: ["廟裡的中元節資料", "功德金跟香油錢", "去年 Excel 檔案"],
            appName: nil,
            windowTitle: nil
        )
        let aiWritingContext = CorrectionContext(
            recentTranscriptions: ["Google Gemini", "這篇文章要去 AI 化", "AI 檢測器"],
            appName: "Codex",
            windowTitle: nil
        )
        let jobApplicationContext = CorrectionContext(
            recentTranscriptions: ["招聘 PDF", "個人自傳", "履歷要怎麼繳交"],
            appName: "Codex",
            windowTitle: nil
        )
        let appleFoundationModelContext = CorrectionContext(
            recentTranscriptions: ["Apple Intelligence", "LLM 增強", "插電與電池省電"],
            appName: "Codex",
            windowTitle: nil
        )
        let contextCollectionContext = CorrectionContext(
            recentTranscriptions: [
                "請先把你看不出來的原來的句子列出來",
                "不用重跑，重新辨識",
                "我們正在收集上下文跟原始句子"
            ],
            appName: "Codex",
            windowTitle: "Voco retranscribe"
        )
        let blockingContext = CorrectionContext(
            recentTranscriptions: [
                "喚醒阻塞跟 engine instance 有關",
                "我們在追具體的阻塞點原因",
                "2.0 merge 之前有調查模組"
            ],
            appName: "Codex",
            windowTitle: "Voco blocking investigation"
        )
        let schedulingContext = CorrectionContext(
            recentTranscriptions: [
                "Startup Trace 的排程已修復",
                "schedule 背景任務要加回去",
                "prewarm timer 跟啟動追蹤"
            ],
            appName: "Codex",
            windowTitle: "Voco scheduler"
        )
        let inputMethodContext = CorrectionContext(
            recentTranscriptions: [
                "我在用 RIME 鼠鬚管輸入法",
                "personal_dict 裡有很多人名",
                "想把詞庫帶進 Voco"
            ],
            appName: "Codex",
            windowTitle: "RIME vocabulary import"
        )
        #expect(PinyinCorrector.shared.correct("這邊又有語音，語音辨識錯誤，所以你自己小振", context: correctionContext).text == "這邊又有語音，語音辨識錯誤，所以你自己修正")
        #expect(PinyinCorrector.shared.correct("連大小雪都不會動了。", context: freezeContext).text == "連大寫鍵都不會動了。")
        #expect(PinyinCorrector.shared.correct("Mac OS 應該會有個支援耗盡的提示窗。", context: systemContext).text == "Mac OS 應該會有個資源耗盡的提示窗。")
        #expect(PinyinCorrector.shared.correct("漏頂對基本版的 M 五來說太大了。", context: systemContext).text == "loading 對基本版的 M5 來說太大了。")
        #expect(PinyinCorrector.shared.correct("你開始說吧，然後照流程後面再部署。", context: correctionContext).text == "你開始修正吧，然後照流程後面再部署。")
        #expect(PinyinCorrector.shared.correct("西成的總長是十三個小時四十九分鐘。", context: sessionContext).text == "session 的總長是十三個小時四十九分鐘。")
        #expect(PinyinCorrector.shared.correct("資料的，新就不重要了。", context: dataImportContext).text == "資料的新舊不重要了。")
        #expect(PinyinCorrector.shared.correct("闌尾的名稱。", context: dataImportContext).text == "欄位的名稱。")
        #expect(PinyinCorrector.shared.correct("雖然一比而已，看不出來什麼東西。", context: dataImportContext).text == "雖然一筆而已，看不出來什麼東西。")
        #expect(PinyinCorrector.shared.correct("他藍位的藍位辨識不出來。", context: dataImportContext).text == "他欄位的欄位辨識不出來。")
        #expect(PinyinCorrector.shared.correct("會變質成浪費，還有狼狽。", context: correctionContext).text == "會辨識成浪費，還有狼狽。")
        #expect(PinyinCorrector.shared.correct("浪費的名稱。", context: dataImportContext).text == "欄位的名稱。")
        #expect(PinyinCorrector.shared.correct("狼狽格式不對。", context: dataImportContext).text == "欄位格式不對。")
        #expect(PinyinCorrector.shared.correct("不少的浪費。", context: dataImportContext).text == "不少的浪費。")
        #expect(PinyinCorrector.shared.correct("查 Windows B M 有沒有問題。", context: virtualizationContext).text == "查 Windows VM 有沒有問題。")
        #expect(PinyinCorrector.shared.correct("Windows BM 跟本地的串流日誌都特別卡。", context: virtualizationContext).text == "Windows VM 跟本地的串流日誌都特別卡。")
        #expect(PinyinCorrector.shared.correct("Windows Virtual Machine 的 BM 很卡。", context: virtualizationContext).text == "Windows Virtual Machine 的 VM 很卡。")
        #expect(PinyinCorrector.shared.correct("語音系統很容易把 Windows Virtual Machine 的 VM 變吃成 BM。", context: virtualizationContext).text == "語音系統很容易把 Windows Virtual Machine 的 VM 辨識成 BM。")
        #expect(PinyinCorrector.shared.correct("這個 BM 是別的縮寫。", context: virtualizationContext).text == "這個 BM 是別的縮寫。")
        #expect(PinyinCorrector.shared.correct("我記得那個 Config UI 是流程圖這樣連來連去的。", context: imageGenerationContext).text == "我記得那個 ComfyUI 是流程圖這樣連來連去的。")
        #expect(PinyinCorrector.shared.correct("第一個要討論的是config.yml那邊。", context: imageGenerationContext).text == "第一個要討論的是ComfyUI那邊。")
        #expect(PinyinCorrector.shared.correct("請先打開 config.yml 那個設定檔。").text == "請先打開 config.yml 那個設定檔。")
        #expect(PinyinCorrector.shared.correct("就等您莊園前一星期的資料。", context: templeDataContext).text == "就等您中元節前一星期的資料。")
        #expect(PinyinCorrector.shared.correct("那您可以跟妙芳討論一下。", context: templeDataContext).text == "那您可以跟廟方討論一下。")
        #expect(PinyinCorrector.shared.correct("妙方的人去用用看。", context: templeDataContext).text == "廟方的人去用用看。")
        #expect(PinyinCorrector.shared.correct("這是一個妙方。").text == "這是一個妙方。")
        #expect(PinyinCorrector.shared.correct("目標是整車漆，跟人工看起來都不像是AI。", context: aiWritingContext).text == "目標是偵測器，跟人工看起來都不像是AI。")
        #expect(PinyinCorrector.shared.correct("以你的角度的話，這片AI的味道有多重？如果有很重的話，去一下AI好了。", context: aiWritingContext).text == "以你的角度的話，這篇AI的味道有多重？如果有很重的話，去 AI 化好了。")
        #expect(PinyinCorrector.shared.correct("你寫一個完整的，然後我交給居民，你去 DAI 為。", context: aiWritingContext).text == "你寫一個完整的，然後我交給 Gemini，你去 de-AI 化。")
        #expect(PinyinCorrector.shared.correct("個人自傳跟找教的方式。", context: jobApplicationContext).text == "個人自傳跟繳交的方式。")
        #expect(PinyinCorrector.shared.correct("個人自傳跟找工作的方法。", context: jobApplicationContext).text == "個人自傳跟繳交的方法。")
        #expect(PinyinCorrector.shared.correct("教教的方式。", context: jobApplicationContext).text == "繳交的方式。")
        #expect(PinyinCorrector.shared.correct("要自轉，或者是要怎麼角標自轉？是用新相機出去嗎？", context: jobApplicationContext).text == "要自傳，或者是要怎麼繳交自傳？是用信箱寄出去嗎？")
        #expect(PinyinCorrector.shared.correct("附近居民最近在做汽車整車漆，也會用新相機出去拍照。", context: jobApplicationContext).text == "附近居民最近在做汽車整車漆，也會用新相機出去拍照。")
        #expect(PinyinCorrector.shared.correct("Apple 的防雷圈 Moto 在這種情況下幫不幫得上忙？", context: appleFoundationModelContext).text == "Apple 的 Foundation Model 在這種情況下幫不幫得上忙？")
        #expect(PinyinCorrector.shared.correct("Foundation motto。", context: appleFoundationModelContext).text == "Foundation model。")
        #expect(PinyinCorrector.shared.correct("這款機車的 motto 是省電。", context: appleFoundationModelContext).text == "這款機車的 motto 是省電。")
        #expect(PinyinCorrector.shared.correct("不過，先跟我承認一下。").text == "不過，先跟我確認一下。")
        #expect(PinyinCorrector.shared.correct("不用重跑，重重新辨析。").text == "不用重跑，重新辨識。")
        #expect(PinyinCorrector.shared.correct("你再跑一次，重重新轉錄的技能。").text == "你再跑一次，重新轉錄的技能。")
        #expect(PinyinCorrector.shared.correct("如果他就確定是often，那我們就可以刪了。").text == "如果他就確定是 orphan，那我們就可以刪了。")
        #expect(PinyinCorrector.shared.correct("現在的這個應該也沒有到很凹頭。").text == "現在的這個應該也沒有到很 auto。")
        #expect(PinyinCorrector.shared.correct("這三天又補了屬於自己的失重。").text == "這三天又補了屬於自己的實作。")
        #expect(PinyinCorrector.shared.correct("而且還做了一些work on the resources.").text == "而且還做了一些Workaround 與實作.")
        #expect(PinyinCorrector.shared.correct("我承認這是失重造成的。").text == "我承認這是失重造成的。")
        #expect(PinyinCorrector.shared.correct("呃，你在現在這筆也是可以改的。你應該知道我在我在說什麼吧？").text == "呃，你在現在這筆也是可以改的。你應該知道我在說什麼吧？")
        #expect(PinyinCorrector.shared.correct("之前那個 69 輪的東西。").text == "之前那個 69 輪的東西。")
        #expect(PinyinCorrector.shared.correct("然後，你可以先做手機。", context: contextCollectionContext).text == "然後，你可以先做收集。")
        #expect(PinyinCorrector.shared.correct("做手機。", context: contextCollectionContext).text == "做收集。")
        #expect(PinyinCorrector.shared.correct("我想做手機 app。", context: contextCollectionContext).text == "我想做手機 app。")
        #expect(PinyinCorrector.shared.correct("我們上次測完之後有記到具體的堵塞點的原因嗎？因為陳英感覺跟上次很像。", context: blockingContext).text == "我們上次測完之後有記到具體的阻塞點的原因嗎？因為成因感覺跟上次很像。")
        #expect(PinyinCorrector.shared.correct("有新增一個組賽的調查模組，組賽的又更嚴重了。", context: blockingContext).text == "有新增一個阻塞的調查模組，阻塞的又更嚴重了。")
        #expect(PinyinCorrector.shared.correct("這場小組賽很精彩。", context: blockingContext).text == "這場小組賽很精彩。")
        #expect(PinyinCorrector.shared.correct("官方回覆已修復，所以你把那個陪存加回去，我試試看。").text == "官方回覆已修復，所以你把那個排程加回去，我試試看。")
        #expect(PinyinCorrector.shared.correct("官方回覆已修復，所以你把那個陪臣加回去，我試試看。", context: schedulingContext).text == "官方回覆已修復，所以你把那個排程加回去，我試試看。")
        #expect(PinyinCorrector.shared.correct("古書裡的陪臣。").text == "古書裡的陪臣。")
        #expect(PinyinCorrector.shared.correct("我目前在用 macOS 系統的 i iM 輸入法。", context: inputMethodContext).text == "我目前在用 macOS 系統的 RIME 輸入法。")
        #expect(PinyinCorrector.shared.correct("我那個輸入法指的是 RIME，也就是鼠須管輸入法。", context: inputMethodContext).text == "我那個輸入法指的是 RIME，也就是鼠鬚管輸入法。")
        #expect(PinyinCorrector.shared.correct("你覺得可以把它帶進這個城市裡面嗎？", context: inputMethodContext).text == "你覺得可以把它帶進這個程式裡面嗎？")
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

    @Test func confidenceGateRoutesProtectedTermReplacementToReview() async throws {
        let result = VocoNormalizationResult(
            originalText: "我是要說鑑定的鑑定哦。",
            normalizedText: "我是要說簡訊的簡訊哦。",
            activeContextIDs: ["builtin.voco-development"],
            replacements: [
                VocoReplacement(
                    originalText: "鑑定",
                    replacementText: "簡訊",
                    termID: "word-replacement.51aff0ef-3162-4f62-8f67-ed132b2b9053",
                    confidence: 0.97,
                    reason: "alias-match",
                    rangeStart: 4,
                    rangeLength: 2
                ),
            ],
            suggestions: []
        )

        let assessment = VocoConfidenceGateService().assess(
            normalizationResult: result,
            rawTranscript: "我是要说鉴定的鉴定哦。"
        )

        #expect(assessment.route == .reviewSuggested)
        #expect(assessment.reasons.contains("protected-term-replacement"))
        #expect(assessment.reviewTriggers.contains { $0.reason == "protected-term-replacement" })
        #expect(assessment.selectedCandidate == "我是要說簡訊的簡訊哦。")
        #expect(assessment.candidates.contains("我是要說鑑定的鑑定哦。"))
    }

    @Test func hypothesisManagerDoesNotCreateSegmentRescueForRawDriftWithCanonicalTerms() async throws {
        let result = VocoCanonicalizationService().normalize("我今天要測 voice anc")
        let assessment = VocoConfidenceGateService().assess(
            normalizationResult: result,
            rawTranscript: "我今天要測 voice anc 然後後面還有一大段錯字"
        )

        #expect(assessment.route == .reviewSuggested)
        #expect(assessment.reasons.contains("raw-cleanup-significant"))
        #expect(assessment.candidates == [
            "我今天要測 VoiceInk",
            "我今天要測 voice anc",
            "我今天要測 voice anc 然後後面還有一大段錯字",
        ])
        #expect(assessment.candidateLabels == ["Recommended", "Original", "Raw ASR"])
        #expect(!assessment.hypothesisDetails.contains { $0.source == .segmentRescue })
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

    @Test func transcriptionSyncsSelectedCandidateWithFinalPaste() async throws {
        let transcription = Transcription(
            text: "再跑一次轉怒的技能吧。",
            duration: 0,
            selectedCandidate: "再跑一次轉怒的技能吧。"
        )

        transcription.recordPasteAttempt(
            text: "再跑一次轉錄的技能吧。",
            didPostCommand: true
        )

        #expect(transcription.finalPastedText == "再跑一次轉錄的技能吧。")
        #expect(transcription.selectedCandidate == "再跑一次轉錄的技能吧。")
        #expect(transcription.candidateSelectionSource == VocoCandidateSelectionSource.finalPaste.rawValue)

        let reviewed = Transcription(
            text: "今天看到焰很大",
            duration: 0,
            selectedCandidate: "今天看到炎很大",
            candidateSelectionSource: .userSelection
        )
        reviewed.recordPasteAttempt(
            text: "今天看到炎很大。",
            didPostCommand: true
        )

        #expect(reviewed.selectedCandidate == "今天看到炎很大。")
        #expect(reviewed.candidateSelectionSource == VocoCandidateSelectionSource.userSelection.rawValue)
    }

    @Test func csvExportPreservesContextAwareSessionMetadata() async throws {
        let sourceTranscriptionID = try #require(UUID(uuidString: "11111111-2222-3333-4444-555555555555"))
        let feedbackCreatedAt = Date(timeIntervalSince1970: 1_700_000_000)
        let correctionFeedback = CorrectionFeedbackSignal(
            kind: .candidateSelection,
            sourceText: "我現在用 voice anc",
            proposedText: "我現在用 VoiceInk",
            acceptedText: "我現在用 VoiceInk",
            confidenceScore: 0.86,
            changeRatio: 0.12,
            reason: "candidate-override",
            termIDs: ["product.voiceink"],
            createdAt: feedbackCreatedAt
        )
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
            ),
            correctionFeedback: [correctionFeedback]
        )
        transcription.recordStyleGuardRejection(
            response: "以下是我整理後的版本：我現在用 VoiceInk。",
            reasons: [
                "assistant-opener:以下是",
                "dropped-mixed-language-term:Qwen3-ASR",
            ]
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
        #expect(csv.contains("Correction Feedback"))
        #expect(csv.contains("Style Guard Reasons"))
        #expect(csv.contains("Style Guard Rejected Text"))
        #expect(csv.contains("Candidate selection · Candidate changed · 86% · change 12% · Terms product.voiceink"))
        #expect(csv.contains("Source: 我現在用 voice anc; Proposed: 我現在用 VoiceInk; Accepted: 我現在用 VoiceInk"))
        #expect(csv.contains("assistant-opener:以下是 | dropped-mixed-language-term:Qwen3-ASR"))
        #expect(csv.contains("以下是我整理後的版本：我現在用 VoiceInk。"))
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

    @Test func historyAssistiveBadgesSurfaceFeedbackAndStyleGuardSignals() async throws {
        let correctiveFeedback = CorrectionFeedbackSignal(
            kind: .candidateSelection,
            sourceText: "我覺得 lisa 的 homura 很難唱",
            proposedText: "我覺得 LiSA 的炎很難唱",
            acceptedText: "我覺得 LiSA 的明け星很難唱",
            confidenceScore: 0.62,
            changeRatio: 0.18,
            reason: "candidate-override",
            termIDs: ["artist.lisa", "song.akeboshi"]
        )
        let passiveFeedback = CorrectionFeedbackSignal(
            kind: .candidateSelection,
            sourceText: "我現在用 voice ink",
            proposedText: "我現在用 VoiceInk",
            acceptedText: "我現在用 VoiceInk",
            confidenceScore: 0.82,
            changeRatio: 0.1,
            reason: "candidate-timeout-fallback",
            termIDs: ["product.voiceink"]
        )
        let transcription = Transcription(
            text: "我覺得 LiSA 的明け星很難唱",
            duration: 0.8,
            styleGuardReasons: [
                "assistant-opener:以下是",
                "dropped-mixed-language-term:Qwen3-ASR",
            ],
            styleGuardRejectedText: "以下是我整理後的版本：我覺得這首歌很難唱。",
            correctionFeedback: [correctiveFeedback, passiveFeedback]
        )

        let badges = TranscriptionAssistiveBadge.badges(for: transcription, limit: 10)

        #expect(badges.map(\.id) == [
            "correction-feedback",
            "style-guard",
        ])
        #expect(badges.map(\.title) == [
            "1 correction",
            "2 style flags",
        ])
        #expect(badges.map(\.tone) == [
            .green,
            .purple,
        ])
        #expect(TranscriptionAssistiveBadge.badges(for: transcription, limit: 1).map(\.title) == ["1 correction"])

        let passiveOnly = Transcription(
            text: "我現在用 VoiceInk",
            duration: 0.3,
            correctionFeedback: [passiveFeedback]
        )

        #expect(TranscriptionAssistiveBadge.badges(for: passiveOnly).map(\.title) == ["1 feedback signal"])
        #expect(TranscriptionAssistiveBadge.badges(for: passiveOnly).map(\.tone) == [.secondary])
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

    @Test @MainActor func lastTranscriptionSkipsNonReusableLatestRows() throws {
        let context = try makeTranscriptionContext()
        let reusable = Transcription(
            text: "usable transcript",
            duration: 0.2,
            transcriptionStatus: .completed
        )
        reusable.timestamp = Date(timeIntervalSince1970: 100)

        let failed = Transcription(
            text: "Transcription Failed: network",
            duration: 0.2,
            transcriptionStatus: .failed
        )
        failed.timestamp = Date(timeIntervalSince1970: 200)

        let pending = Transcription(
            text: "",
            duration: 0.2,
            transcriptionStatus: .pending
        )
        pending.timestamp = Date(timeIntervalSince1970: 300)

        let canceled = Transcription(
            text: Transcription.canceledTranscriptionText,
            duration: 0.2,
            transcriptionStatus: .canceled
        )
        canceled.timestamp = Date(timeIntervalSince1970: 400)

        context.insert(reusable)
        context.insert(failed)
        context.insert(pending)
        context.insert(canceled)
        try context.save()

        #expect(LastTranscriptionService.getLastTranscription(from: context)?.id == reusable.id)
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
        let correctiveFeedback = CorrectionFeedbackSignal(
            kind: .candidateSelection,
            sourceText: "voice anc",
            proposedText: "VoiceInk",
            acceptedText: "VoiceInk",
            confidenceScore: output.assessment.score,
            changeRatio: 0.2,
            reason: "candidate-override",
            termIDs: ["product.voiceink"]
        )
        let fallbackFeedback = CorrectionFeedbackSignal(
            kind: .candidateSelection,
            sourceText: "voice inc",
            proposedText: "VoiceInk",
            acceptedText: "VoiceInk",
            confidenceScore: output.assessment.score,
            changeRatio: 0.18,
            reason: "candidate-timeout-fallback",
            termIDs: ["product.voiceink"]
        )
        let styleGuardRejectedText = "以下是我整理後的版本：我現在用 VoiceInk。"
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
            styleGuardReasons: [
                "assistant-opener:以下是",
                "dropped-mixed-language-term:Qwen3-ASR",
            ],
            styleGuardRejectedText: styleGuardRejectedText,
            sourceTranscriptionID: sourceTranscriptionID,
            retranscriptionAnalysis: retranscriptionAnalysis,
            correctionFeedback: [correctiveFeedback, fallbackFeedback],
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
        #expect(metric.correctionFeedbackCount == 2)
        #expect(metric.correctiveFeedbackCount == 1)
        #expect(metric.correctionFeedbackReasons == ["candidate-override", "candidate-timeout-fallback"])
        #expect(metric.styleGuardReasonCount == 2)
        #expect(metric.styleGuardReasons == ["assistant-opener:以下是", "dropped-mixed-language-term:Qwen3-ASR"])
        #expect(metric.styleGuardRejectedCharacterCount == styleGuardRejectedText.count)
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

    @Test @MainActor func sessionMetricRefreshTracksPersistedCandidateSelection() async throws {
        let context = try makeSessionMetricContext()
        let result = VocoCanonicalizationService().normalize("今天看到焰很大")
        let assessment = VocoConfidenceGateService().assess(
            normalizationResult: result,
            rawTranscript: result.originalText
        )
        let transcription = Transcription(
            text: result.normalizedText,
            duration: 2.0,
            transcriptionModelName: "Qwen3-ASR",
            transcriptionDuration: 0.5,
            transcriptionStatus: .completed
        )

        transcription.recordASRMetadata(
            rawTranscript: result.originalText,
            normalizationResult: result,
            confidenceAssessment: assessment,
            asrEngineID: "qwen3:Qwen3-ASR",
            languageMode: "auto"
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
        #expect(metric.selectedCandidate == result.normalizedText)
        #expect(metric.candidateSelectionSource == nil)

        let signal = try #require(
            CorrectionFeedbackService.candidateSelectionSignal(
                normalizationResult: result,
                assessment: assessment,
                selectedCandidate: "今天看到炎很大",
                rawTranscript: result.originalText
            )
        )
        transcription.recordCorrectionFeedback(signal)
        transcription.text = "今天看到炎很大"
        transcription.selectedCandidate = "今天看到炎很大"
        transcription.recordCandidateSelectionSource(.userSelection)
        transcription.userCorrectionDistance = signal.changeRatio
        let refreshed = try SessionMetricRecorder.refreshExistingRecorderSessionMetric(
            transcription: transcription,
            in: context
        )

        #expect(refreshed)
        #expect(signal.reason == "candidate-override")
        #expect(transcription.selectedCandidate == "今天看到炎很大")
        #expect(metric.selectedCandidate == "今天看到炎很大")
        #expect(metric.candidateSelectionSource == VocoCandidateSelectionSource.userSelection.rawValue)
        #expect(metric.selectedCandidateHypothesisSource == VocoHypothesisSource.suggestedRepair.rawValue)
        #expect(metric.userCorrectionDistance == transcription.userCorrectionDistance)
        #expect(metric.wordCount == WordCounter.count(in: "今天看到炎很大"))
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
        let feedbackSignal = CorrectionFeedbackSignal(
            kind: .candidateSelection,
            sourceText: "voice anc",
            proposedText: "VoiceInk",
            acceptedText: "VoiceInk",
            confidenceScore: output.assessment.score,
            changeRatio: 0.2,
            reason: "candidate-override",
            termIDs: ["product.voiceink"]
        )
        let styleGuardRejectedText = "以下是我整理後的版本：hello world"
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
            styleGuardReasons: ["assistant-opener:以下是"],
            styleGuardRejectedText: styleGuardRejectedText,
            sourceTranscriptionID: sourceTranscriptionID,
            retranscriptionAnalysis: retranscriptionAnalysis,
            correctionFeedback: [feedbackSignal],
            transcriptionStatus: .completed
        )
        let metric = SessionMetric(
            transcriptionId: transcription.id,
            wordCount: 3,
            audioDuration: 2.0,
            transcriptionModelName: "Qwen3-ASR",
            transcriptionDuration: 0.5,
            speedFactor: 4.0,
            modeName: nil,
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
        #expect(metric.correctionFeedbackCount == 1)
        #expect(metric.correctiveFeedbackCount == 1)
        #expect(metric.correctionFeedbackReasons == ["candidate-override"])
        #expect(metric.styleGuardReasonCount == 1)
        #expect(metric.styleGuardReasons == ["assistant-opener:以下是"])
        #expect(metric.styleGuardRejectedCharacterCount == styleGuardRejectedText.count)
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

    @Test func sessionMetricMetadataDeduplicatesReviewTriggers() async throws {
        let transcription = Transcription(
            text: "需要確認",
            duration: 0.5,
            transcriptionStatus: .completed
        )
        transcription.reviewTriggers = [
            VocoReviewTrigger(
                id: "low-confidence-score",
                reason: "low-confidence-score",
                detail: "Score 60% below 78%"
            ),
            VocoReviewTrigger(
                id: "low-confidence-score",
                reason: "low-confidence-score",
                detail: "duplicate ignored"
            ),
            VocoReviewTrigger(
                id: "unresolved-suggestions",
                reason: "unresolved-suggestions",
                detail: "1 suggestion"
            ),
        ]

        let metric = SessionMetric(
            transcriptionId: transcription.id,
            wordCount: 1,
            audioDuration: 0.5,
            transcriptionModelName: "Qwen3-ASR",
            transcriptionDuration: nil,
            speedFactor: nil,
            modeName: nil,
            aiEnhancementModelName: nil,
            enhancementDuration: nil
        )

        metric.recordDictationMetadata(from: transcription)

        #expect(metric.reviewTriggerCount == 2)
        #expect(metric.reviewTriggerIDs == ["low-confidence-score", "unresolved-suggestions"])
        #expect(metric.reviewTriggerSummaries == [
            "Low score (Score 60% below 78%)",
            "Needs choice (1 suggestion)",
        ])

    }

    @Test func sessionMetricSelectedCandidateSourceUsesFoldedMatch() async throws {
        let hypotheses = [
            VocoHypothesis(
                id: "suggestedRepair",
                text: "LiSA",
                label: "With suggestions",
                source: .suggestedRepair,
                confidenceScore: 0.72,
                reasons: ["unresolved-suggestions"],
                activeContextIDs: [VocoCanonicalizationService.defaultContextPackID],
                appliedTermIDs: ["artist.lisa"],
                requiresReview: true
            ),
        ]

        #expect(
            SessionMetric.selectedCandidateHypothesisSource(
                in: hypotheses,
                selectedCandidate: " lisa "
            ) == VocoHypothesisSource.suggestedRepair.rawValue
        )
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

    @Test @MainActor func canonicalizationPipelinePassesAppWindowContextHints() async throws {
        let context = try makeCanonicalizationPipelineContext()
        let model = try #require(TranscriptionModelRegistry.models.first)

        let output = VocoCanonicalizationPipeline.normalizeWithAssessment(
            "今天看到 homura 很亮",
            rawTranscript: "今天看到 homura 很亮",
            model: model,
            modelContext: context,
            appName: "Music",
            windowTitle: "LiSA playlist"
        )

        #expect(output.normalizationResult.normalizedText == "今天看到炎很亮")
        #expect(output.normalizationResult.replacements.first?.termID == "song.homura")
    }

    @Test @MainActor func canonicalizationPipelineIncludesCorrectionRiskWithoutExistingTranscription() async throws {
        let context = try makeCanonicalizationPipelineContext()
        let replacement = WordReplacement(
            originalText: "jay son",
            replacementText: "Jason",
            isEnabled: true
        )
        context.insert(replacement)
        let termID = "word-replacement.\(replacement.id.uuidString.lowercased())"

        let now = Date()
        for (index, text) in ["jay son", "j son"].enumerated() {
            let transcription = Transcription(text: text, duration: 0, transcriptionStatus: .completed)
            transcription.timestamp = now.addingTimeInterval(TimeInterval(-index * 60))
            transcription.recordCorrectionFeedback(
                CorrectionFeedbackSignal(
                    kind: .candidateSelection,
                    sourceText: text,
                    proposedText: "Jason",
                    acceptedText: "Jason",
                    reason: "candidate-override",
                    termIDs: [termID]
                )
            )
            context.insert(transcription)
        }
        let confirmed = Transcription(text: "jay son", duration: 0, transcriptionStatus: .completed)
        confirmed.timestamp = now.addingTimeInterval(-180)
        confirmed.recordCorrectionFeedback(
            CorrectionFeedbackSignal(
                kind: .candidateSelection,
                sourceText: "jay son",
                proposedText: "Jason",
                acceptedText: "Jason",
                reason: "candidate-confirmed",
                termIDs: [termID]
            )
        )
        context.insert(confirmed)

        let clean = Transcription(text: "沒有修正", duration: 0, transcriptionStatus: .completed)
        clean.timestamp = now.addingTimeInterval(-240)
        context.insert(clean)
        try context.save()

        let model = try #require(TranscriptionModelRegistry.models.first)
        let output = VocoCanonicalizationPipeline.normalizeWithAssessment(
            "我現在用 jay son 做測試",
            rawTranscript: "我現在用 jay son 做測試",
            model: model,
            modelContext: context
        )

        #expect(output.normalizationResult.normalizedText == "我現在用 Jason 做測試")
        #expect(output.confidenceAssessment.route == .reviewSuggested)
        #expect(output.confidenceAssessment.correctionRiskProfile?.recentSessionCount == 4)
        #expect(output.confidenceAssessment.correctionRiskProfile?.correctedSessionCount == 2)
        #expect(output.confidenceAssessment.correctionRiskProfile?.highRiskTermIDs == [termID])
        #expect(output.confidenceAssessment.reasons.contains("recent-correction-rate"))
        #expect(output.confidenceAssessment.reasons.contains("recent-term-corrections"))
        #expect(output.confidenceAssessment.reviewTriggers.map(\.id).contains("recent-correction-rate"))
        #expect(output.confidenceAssessment.reviewTriggers.map(\.id).contains("recent-term-corrections"))
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
            "protected-term-replacement",
            "phonetic-correction-term",
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
            "Protected term changed",
            "Phonetic correction",
            "Cleanup changed text",
            "Retranscription meaningful",
            "User substitution",
            "Unknown signal",
        ])

        let styleGuardReasons = VocoSignalDisplayFormatter.displayStyleGuardReasons(for: [
            "assistant-opener:以下是",
            "dropped-mixed-language-term:Qwen3-ASR",
            "introduced-structured-format",
            "style-expansion",
            "assistant-opener:以下是",
        ])

        #expect(styleGuardReasons == [
            "Assistant opener (以下是)",
            "Dropped mixed language term (Qwen3-ASR)",
            "Structured formatting",
            "Style expansion",
        ])
        #expect(VocoSignalDisplayFormatter.displayStyleGuardReasonCategory(for: "assistant-opener:總而言之") == "Assistant opener")
        #expect(VocoSignalDisplayFormatter.displayStyleGuardReasonCategory(for: "dropped-mixed-language-term:Qwen3-ASR") == "Dropped mixed language term")
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

    @Test func correctionFeedbackClassifiesFoldedCandidateConfirmationAsNonCorrective() async throws {
        let result = VocoNormalizationResult(
            originalText: "voiceink",
            normalizedText: "VoiceInk",
            activeContextIDs: [VocoCanonicalizationService.defaultContextPackID],
            replacements: [],
            suggestions: []
        )
        let assessment = VocoConfidenceAssessment(
            score: 0.7,
            route: .reviewSuggested,
            reasons: ["unresolved-suggestions"],
            candidates: ["VoiceInk"],
            selectedCandidate: "VoiceInk"
        )
        let signal = try #require(
            CorrectionFeedbackService.candidateSelectionSignal(
                normalizationResult: result,
                assessment: assessment,
                selectedCandidate: " voiceink ",
                rawTranscript: result.originalText
            )
        )

        #expect(signal.kind == .candidateSelection)
        #expect(signal.reason == "candidate-confirmed")
        #expect(signal.acceptedText == "voiceink")
        #expect(signal.isCorrectiveSignal == false)
    }

    @Test func correctionFeedbackClassifiesFoldedCandidateOverride() async throws {
        let result = VocoNormalizationResult(
            originalText: "voiceink",
            normalizedText: "VoiceInk",
            activeContextIDs: [VocoCanonicalizationService.defaultContextPackID],
            replacements: [],
            suggestions: []
        )
        let assessment = VocoConfidenceAssessment(
            score: 0.7,
            route: .reviewSuggested,
            reasons: ["unresolved-suggestions"],
            candidates: ["VoiceInk", "VOCO"],
            selectedCandidate: "VoiceInk"
        )
        let signal = try #require(
            CorrectionFeedbackService.candidateSelectionSignal(
                normalizationResult: result,
                assessment: assessment,
                selectedCandidate: " voco ",
                rawTranscript: result.originalText
            )
        )

        #expect(signal.kind == .candidateSelection)
        #expect(signal.reason == "candidate-override")
        #expect(signal.acceptedText == "voco")
        #expect(signal.isCorrectiveSignal)
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

        let signal = CorrectionFeedbackService.candidateSelectionSignal(
            normalizationResult: result,
            assessment: assessment,
            selectedCandidate: "今天看到炎很大",
            rawTranscript: result.originalText
        )
        transcription.recordCorrectionFeedback(signal)

        let storedSignal = try #require(transcription.correctionFeedback.first)
        #expect(storedSignal.kind == .candidateSelection)
        #expect(storedSignal.acceptedText == "今天看到炎很大")
        #expect(storedSignal.reason == "candidate-override")
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
        #expect(backup.generalSettings?.personalStyleGuardEnabled == nil)
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

    @Test func backupFilePreservesPersonalStyleGuardSetting() throws {
        let data = Data(
            """
            {
              "version": "1.79",
              "generalSettings": {
                "personalStyleGuardEnabled": false
              }
            }
            """.utf8
        )

        let backup = try JSONDecoder().decode(BackupFile.self, from: data)
        let encoded = try JSONEncoder().encode(backup)
        let roundTripped = try JSONDecoder().decode(BackupFile.self, from: encoded)

        #expect(backup.generalSettings?.personalStyleGuardEnabled == false)
        #expect(roundTripped.generalSettings?.personalStyleGuardEnabled == false)
    }

    @Test func backupFilePreservesWordReplacementLearningMetadata() throws {
        let dateAdded = Date(timeIntervalSince1970: 1_700_000_000)
        let lastSeenDate = Date(timeIntervalSince1970: 1_700_003_600)
        let replacement = WordReplacement(
            originalText: "voice anc, voice inc",
            replacementText: "VoiceInk",
            dateAdded: dateAdded,
            isEnabled: false,
            source: WordReplacement.sourceCorrectionFeedback
        )
        replacement.hitCount = 2
        replacement.lastSeenDate = lastSeenDate

        let backup = BackupFile(
            version: "1.79",
            customPrompts: [],
            modeConfigs: [],
            modeShortcuts: nil,
            vocabularyWords: nil,
            wordReplacements: [
                replacement.originalText: replacement.replacementText,
            ],
            wordReplacementDetails: [WordReplacementBackup(replacement: replacement)],
            generalSettings: nil,
            customEmojis: nil,
            customCloudModels: nil
        )

        let encoded = try JSONEncoder().encode(backup)
        let roundTripped = try JSONDecoder().decode(BackupFile.self, from: encoded)
        let detail = try #require(roundTripped.wordReplacementDetails?.first)
        let restored = detail.makeReplacement()

        #expect(roundTripped.wordReplacements == ["voice anc, voice inc": "VoiceInk"])
        #expect(detail.originalText == "voice anc, voice inc")
        #expect(detail.replacementText == "VoiceInk")
        #expect(detail.isEnabled == false)
        #expect(detail.source == WordReplacement.sourceCorrectionFeedback)
        #expect(detail.hitCount == 2)
        #expect(detail.dateAdded == dateAdded)
        #expect(detail.lastSeenDate == lastSeenDate)
        #expect(restored.originalText == replacement.originalText)
        #expect(restored.replacementText == replacement.replacementText)
        #expect(restored.isEnabled == false)
        #expect(restored.source == WordReplacement.sourceCorrectionFeedback)
        #expect(restored.hitCount == 2)
        #expect(restored.lastSeenDate == lastSeenDate)
        #expect(restored.isLearningCandidate)
    }

    @Test @MainActor func backupImporterPreservesWordReplacementLearningMetadata() throws {
        let context = try makeDictionaryContext()
        let dateAdded = Date(timeIntervalSince1970: 1_700_000_000)
        let lastSeenDate = Date(timeIntervalSince1970: 1_700_003_600)
        let staged = WordReplacement(
            originalText: "voice anc, voice inc",
            replacementText: "VoiceInk",
            dateAdded: dateAdded,
            isEnabled: false,
            source: WordReplacement.sourceCorrectionFeedback
        )
        staged.hitCount = 2
        staged.lastSeenDate = lastSeenDate

        let backup = BackupFile(
            version: "1.79",
            customPrompts: [],
            modeConfigs: [],
            modeShortcuts: nil,
            vocabularyWords: nil,
            wordReplacements: [
                "voice anc, voice inc": "VoiceInk",
            ],
            wordReplacementDetails: [WordReplacementBackup(replacement: staged)],
            generalSettings: nil,
            customEmojis: nil,
            customCloudModels: nil
        )

        try BackupImporter.importDictionary(from: backup, modelContext: context)
        let imported = try #require(try context.fetch(FetchDescriptor<WordReplacement>()).first)

        #expect(imported.originalText == "voice anc, voice inc")
        #expect(imported.replacementText == "VoiceInk")
        #expect(imported.isEnabled == false)
        #expect(imported.source == WordReplacement.sourceCorrectionFeedback)
        #expect(imported.hitCount == 2)
        #expect(imported.lastSeenDate == lastSeenDate)
        #expect(imported.isLearningCandidate)
        #expect(VocoCanonicalizationService.wordReplacementTerms(from: [imported]).isEmpty)
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

    @Test func modeDefaultLanguageStaysAutoFirst() throws {
        let defaultMode = ModeConfig(name: "Default", isAIEnhancementEnabled: false)
        let japaneseMode = ModeConfig(
            name: "Japanese",
            isAIEnhancementEnabled: false,
            selectedLanguage: "ja"
        )

        #expect(defaultMode.selectedLanguage == TranscriptionLanguageSupport.defaultLanguageCode)
        #expect(japaneseMode.selectedLanguage == "ja")
    }

    @Test @MainActor func appDefaultsRegisterAutoFirstLanguage() throws {
        let suiteName = "AppDefaultsTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }

        AppDefaults.registerDefaults(defaults: defaults)

        #expect(
            defaults.string(forKey: TranscriptionLanguageSupport.selectedLanguageKey) ==
                TranscriptionLanguageSupport.defaultLanguageCode
        )
        #expect(TranscriptionLanguageSupport.selectedLanguage(in: defaults) == TranscriptionLanguageSupport.defaultLanguageCode)
        #expect(VocoCanonicalizationPipeline.selectedLanguageMode(defaults: defaults) == TranscriptionLanguageSupport.defaultLanguageCode)
    }

    @Test func languageFallbackPrefersAutoWhenModelSupportsIt() throws {
        let qwenModel = try #require(TranscriptionModelRegistry.models.first { $0.provider == .qwen3 })
        let nativeAppleModel = try #require(TranscriptionModelRegistry.models.first { $0.provider == .nativeApple })

        #expect(
            TranscriptionLanguageSupport.validLanguageOrFallback(nil, for: qwenModel) ==
                TranscriptionLanguageSupport.defaultLanguageCode
        )
        #expect(
            TranscriptionLanguageSupport.validLanguageOrFallback(
                TranscriptionLanguageSupport.defaultLanguageCode,
                for: qwenModel
            ) == TranscriptionLanguageSupport.defaultLanguageCode
        )
        #expect(
            TranscriptionLanguageSupport.validLanguageOrFallback(
                TranscriptionLanguageSupport.defaultLanguageCode,
                for: nativeAppleModel
            ) == "en-US"
        )
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

    @Test func canonicalizationSkipsWordReplacementsThatTouchProtectedTerms() async throws {
        let terms = VocoCanonicalizationService.wordReplacementTerms(
            from: [
                WordReplacement(originalText: "鑑定", replacementText: "簡訊"),
                WordReplacement(originalText: "鉴定", replacementText: "简讯"),
                WordReplacement(originalText: "轉錄", replacementText: "專案"),
                WordReplacement(originalText: "转路", replacementText: "專案"),
                WordReplacement(originalText: "语音转录", replacementText: "語音專案"),
                WordReplacement(originalText: "retranscribe", replacementText: "專案"),
            ]
        )

        #expect(terms.isEmpty)

        let result = VocoCanonicalizationService(contextPacks: []).normalize(
            "我是要說鑑定的鑑定哦。",
            activeContextIDs: [],
            additionalTerms: terms
        )
        #expect(result.normalizedText == "我是要說鑑定的鑑定哦。")
        #expect(result.replacements.isEmpty)
    }

    @Test func personalStyleGuardEnabledDefaultsToTrueAndPersists() throws {
        let suiteName = "PersonalStyleGuardTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }

        #expect(PersonalStyleGuardService.isEnabled(defaults: defaults))

        PersonalStyleGuardService.setEnabled(false, defaults: defaults)
        #expect(!PersonalStyleGuardService.isEnabled(defaults: defaults))

        PersonalStyleGuardService.setEnabled(true, defaults: defaults)
        #expect(PersonalStyleGuardService.isEnabled(defaults: defaults))
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

    @Test @MainActor func correctionFeedbackLearningSkipsProtectedSourceTerms() async throws {
        let context = try makeDictionaryContext()
        let signal = CorrectionFeedbackSignal(
            kind: .candidateSelection,
            sourceText: "你再跑一次轉錄的技能",
            proposedText: "你再跑一次專案的技能",
            acceptedText: "你再跑一次專案的技能",
            confidenceScore: 0.79,
            changeRatio: 0.2,
            reason: "candidate-override"
        )

        let staged = CorrectionFeedbackLearningService.stageLearningCandidates(from: signal, in: context)
        let entries = try context.fetch(FetchDescriptor<WordReplacement>())

        #expect(staged.count == 1)
        #expect(staged.first?.original == "轉錄")
        #expect(staged.first?.replacement == "專案")
        #expect(entries.isEmpty)

        let nearMissSignal = CorrectionFeedbackSignal(
            kind: .candidateSelection,
            sourceText: "你再跑一次转路的技能",
            proposedText: "你再跑一次專案的技能",
            acceptedText: "你再跑一次專案的技能",
            confidenceScore: 0.79,
            changeRatio: 0.2,
            reason: "candidate-override"
        )

        let nearMissStaged = CorrectionFeedbackLearningService.stageLearningCandidates(from: nearMissSignal, in: context)
        let entriesAfterNearMiss = try context.fetch(FetchDescriptor<WordReplacement>())

        #expect(nearMissStaged.count == 1)
        #expect(nearMissStaged.first?.original == "转路")
        #expect(nearMissStaged.first?.replacement == "專案")
        #expect(entriesAfterNearMiss.isEmpty)
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

    @Test @MainActor func correctionFeedbackLearningSkipsLLMOnlyEvidence() async throws {
        let context = try makeDictionaryContext()
        let signal = CorrectionFeedbackSignal(
            kind: .retranscriptionChange,
            sourceText: "修正",
            acceptedText: "小振",
            confidenceScore: 0.82,
            changeRatio: 0.4,
            reason: "llm-enhancement-difference"
        )

        let staged = CorrectionFeedbackLearningService.stageLearningCandidates(from: signal, in: context)
        let entries = try context.fetch(FetchDescriptor<WordReplacement>())

        #expect(staged.isEmpty)
        #expect(entries.isEmpty)
    }

    @Test @MainActor func correctionFeedbackLearningSkipsNegativeEvidenceOriginals() async throws {
        let context = try makeDictionaryContext()
        let signal = CorrectionFeedbackSignal(
            kind: .userSubstitution,
            sourceText: "69 輪",
            acceptedText: "六十九輪",
            confidenceScore: 0.9,
            changeRatio: 0.5,
            reason: "user-substitution"
        )

        let staged = CorrectionFeedbackLearningService.stageLearningCandidates(from: signal, in: context)
        let entries = try context.fetch(FetchDescriptor<WordReplacement>())

        #expect(staged.isEmpty)
        #expect(entries.isEmpty)
    }

    @Test @MainActor func correctionFeedbackLearningSkipsNoisyUserSubstitution() async throws {
        let context = try makeDictionaryContext()
        let signal = CorrectionFeedbackSignal(
            kind: .userSubstitution,
            sourceText: "陪存",
            acceptedText: "陪臣",
            confidenceScore: 0.9,
            changeRatio: 0.5,
            reason: "user-substitution"
        )

        let staged = CorrectionFeedbackLearningService.stageLearningCandidates(from: signal, in: context)
        let entries = try context.fetch(FetchDescriptor<WordReplacement>())

        #expect(staged.isEmpty)
        #expect(entries.isEmpty)
    }

    @Test @MainActor func correctionFeedbackLearningKeepsShortRetranscriptionReviewOnlyUnlessTechnical() async throws {
        let context = try makeDictionaryContext()
        let shortNonTechnical = CorrectionFeedbackSignal(
            kind: .retranscriptionChange,
            sourceText: "做手機。",
            acceptedText: "做收集。",
            confidenceScore: 0.9,
            changeRatio: 0.25,
            reason: "retranscription-meaningfulChange"
        )

        let staged = CorrectionFeedbackLearningService.stageLearningCandidates(from: shortNonTechnical, in: context)
        let entries = try context.fetch(FetchDescriptor<WordReplacement>())

        #expect(staged.isEmpty)
        #expect(entries.isEmpty)
    }

    @Test @MainActor func correctionFeedbackLearningSkipsNonTechnicalCrossLanguageReconstruction() async throws {
        let context = try makeDictionaryContext()
        let signal = CorrectionFeedbackSignal(
            kind: .userSubstitution,
            sourceText: "你好",
            acceptedText: "hello there",
            confidenceScore: 0.9,
            changeRatio: 0.8,
            reason: "user-substitution"
        )

        let staged = CorrectionFeedbackLearningService.stageLearningCandidates(from: signal, in: context)
        let entries = try context.fetch(FetchDescriptor<WordReplacement>())

        #expect(staged.isEmpty)
        #expect(entries.isEmpty)
    }

    @Test @MainActor func correctionFeedbackLearningKeepsRepeatedRetranscriptionCorrectionReviewOnly() async throws {
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
        #expect(!entry.isEnabled)
        #expect(entry.isLearningCandidate)
        #expect(entry.learningProgressLabel == "3/3")
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

    @Test func editModeCacheMustBelongToFrontmostApp() {
        #expect(EditModeDetectionPolicy.cacheMatchesFrontmostApp(cachedPID: 42, currentPID: 42))
        #expect(!EditModeDetectionPolicy.cacheMatchesFrontmostApp(cachedPID: 42, currentPID: 43))
        #expect(!EditModeDetectionPolicy.cacheMatchesFrontmostApp(cachedPID: nil, currentPID: 42))
        #expect(!EditModeDetectionPolicy.cacheMatchesFrontmostApp(cachedPID: 42, currentPID: nil))
    }

    @Test func editModeRequiresTrustedEditableSignalAndSelection() {
        #expect(EditModeDetectionPolicy.shouldEnterEditMode(hasTrustedEditableSignal: true, selectedText: "selected text"))
        #expect(!EditModeDetectionPolicy.shouldEnterEditMode(hasTrustedEditableSignal: false, selectedText: "selected text"))
        #expect(!EditModeDetectionPolicy.shouldEnterEditMode(hasTrustedEditableSignal: true, selectedText: nil))
        #expect(!EditModeDetectionPolicy.shouldEnterEditMode(hasTrustedEditableSignal: true, selectedText: " \n\t "))
    }

    @Test func editModeAXSelectionRequiresPositiveEditableRange() {
        let textAreaRole = kAXTextAreaRole as String
        let textFieldRole = kAXTextFieldRole as String

        #expect(EditableTextSelectionPolicy.resolve(observations: [
            EditableTextSelectionObservation(
                role: textAreaRole,
                selectedText: "old clipboard payload",
                selectedRangeLength: 0
            )
        ]) == .noSelection)

        #expect(EditableTextSelectionPolicy.resolve(observations: [
            EditableTextSelectionObservation(
                role: textAreaRole,
                selectedText: "stale AX payload",
                selectedRangeLength: nil
            )
        ]) == .unavailable)

        #expect(EditableTextSelectionPolicy.resolve(observations: [
            EditableTextSelectionObservation(
                role: textAreaRole,
                selectedText: "actual selection",
                selectedRangeLength: "actual selection".utf16.count
            )
        ]) == .selected("actual selection"))

        // Selecting the complete field is legitimate Edit Mode input. The old
        // fieldValue == selectedText heuristic incorrectly rejected this case.
        #expect(EditableTextSelectionPolicy.resolve(observations: [
            EditableTextSelectionObservation(
                role: textFieldRole,
                selectedText: "https://example.com",
                selectedRangeLength: "https://example.com".utf16.count
            )
        ]) == .selected("https://example.com"))

        #expect(EditableTextSelectionPolicy.resolve(observations: [
            EditableTextSelectionObservation(
                role: textAreaRole,
                selectedText: "\u{FFFC}\u{200B}",
                selectedRangeLength: 2
            )
        ]) == .unavailable)

        #expect(EditableTextSelectionPolicy.resolve(observations: [
            EditableTextSelectionObservation(
                role: kAXGroupRole as String,
                selectedText: "static transcript selection",
                selectedRangeLength: 27
            )
        ]) == .unavailable)
    }

    @Test func editModeDeepScanRejectsRetainedSelectionFromUnfocusedEditor() {
        let textAreaRole = kAXTextAreaRole as String

        #expect(EditableTextSelectionPolicy.resolve(
            observations: [
                EditableTextSelectionObservation(
                    role: textAreaRole,
                    selectedText: "retained old selection",
                    selectedRangeLength: 22,
                    isFocused: false
                ),
                EditableTextSelectionObservation(
                    role: textAreaRole,
                    selectedText: nil,
                    selectedRangeLength: 0,
                    isFocused: true
                ),
            ],
            requireFocusedElement: true
        ) == .noSelection)

        #expect(EditableTextSelectionPolicy.resolve(
            observations: [
                EditableTextSelectionObservation(
                    role: textAreaRole,
                    selectedText: "current selection",
                    selectedRangeLength: 17,
                    isFocused: true
                )
            ],
            requireFocusedElement: true
        ) == .selected("current selection"))
    }

    @Test func editModeMarkerSelectionRequiresOneEditableAncestor() {
        #expect(EditableMarkerSelectionPolicy.resolve(
            selectedText: "static transcript selection",
            rangeLength: 27,
            endpointsShareEditableAncestor: false,
            editableAncestorIsFocused: false
        ) == .noSelection)

        #expect(EditableMarkerSelectionPolicy.resolve(
            selectedText: "composer selection",
            rangeLength: 18,
            endpointsShareEditableAncestor: true,
            editableAncestorIsFocused: true
        ) == .selected("composer selection"))

        #expect(EditableMarkerSelectionPolicy.resolve(
            selectedText: "selection retained in hidden editor",
            rangeLength: 35,
            endpointsShareEditableAncestor: true,
            editableAncestorIsFocused: false
        ) == .noSelection)

        #expect(EditableMarkerSelectionPolicy.resolve(
            selectedText: nil,
            rangeLength: 0,
            endpointsShareEditableAncestor: true,
            editableAncestorIsFocused: true
        ) == .noSelection)
    }

    @Test func editModeSelectionSnapshotMustMatchCapturedProcess() {
        let snapshot = EditModeSelectionSnapshot(text: "selected", pid: 42)

        #expect(EditModeSelectionSnapshotPolicy.validated(snapshot, capturedAppPID: 42) == snapshot)
        #expect(EditModeSelectionSnapshotPolicy.validated(snapshot, capturedAppPID: 43) == nil)
        #expect(EditModeSelectionSnapshotPolicy.validated(snapshot, capturedAppPID: nil) == nil)
        #expect(EditModeSelectionSnapshotPolicy.validated(nil, capturedAppPID: 42) == nil)
    }

    @Test func editModeKnownElectronAppsUseFocusedWindowAXSearch() {
        #expect(EditModeDetectionPolicy.shouldSearchFocusedWindow(bundleID: "com.openai.codex"))
        #expect(EditModeDetectionPolicy.shouldSearchFocusedWindow(bundleID: "com.anthropic.claudefordesktop"))
        #expect(EditModeDetectionPolicy.shouldSearchFocusedWindow(bundleID: "com.microsoft.VSCode"))
        #expect(!EditModeDetectionPolicy.shouldSearchFocusedWindow(bundleID: "com.apple.TextEdit"))
    }

    @Test @MainActor func clipboardTransactionsKeepDelayedRestoreOutsideSelectionCapture() async {
        let coordinator = ClipboardTransactionCoordinator()
        let gate = TestAsyncGate()
        var events: [String] = []

        let selectionCapture = Task { @MainActor in
            await coordinator.withExclusiveAccess {
                events.append("selection-start")
                await gate.wait()
                events.append("selection-end")
            }
        }

        while events.isEmpty {
            await Task.yield()
        }

        let delayedRestore = Task { @MainActor in
            await coordinator.withExclusiveAccess {
                events.append("restore")
            }
        }

        for _ in 0..<10 {
            await Task.yield()
        }
        #expect(events == ["selection-start"])

        gate.open()
        await selectionCapture.value
        await delayedRestore.value
        #expect(events == ["selection-start", "selection-end", "restore"])
    }

    @Test func rapidVocoPastesKeepTheOriginalClipboardRestoreTarget() {
        var chain = ClipboardRestoreChain<String>()
        let firstSession = ClipboardPasteSessionIdentity(id: "paste-1", text: "first Voco output")
        let secondSession = ClipboardPasteSessionIdentity(id: "paste-2", text: "second Voco output")

        let firstBaseline = chain.originalSnapshotForNextPaste(
            currentSession: nil,
            makeSnapshot: { "user clipboard" }
        )
        chain.begin(session: firstSession, originalSnapshot: firstBaseline)

        let secondBaseline = chain.originalSnapshotForNextPaste(
            currentSession: firstSession,
            makeSnapshot: { "first Voco output" }
        )
        #expect(secondBaseline == "user clipboard")

        chain.begin(session: secondSession, originalSnapshot: secondBaseline)
        chain.clear(ifSessionMatches: firstSession)
        #expect(chain.activeSession == secondSession)
        chain.clear(ifSessionMatches: secondSession)
        #expect(chain.activeSession == nil)
    }

    @Test func recordingContextRejectsTransientVocoPasteSession() {
        #expect(ClipboardContextPolicy.userClipboardText(
            "previous transcription",
            pasteSessionID: "active-voco-session"
        ) == nil)
        #expect(ClipboardContextPolicy.userClipboardText(
            "user clipboard",
            pasteSessionID: nil
        ) == "user clipboard")
    }

    @Test @MainActor func canceledClipboardWaiterDoesNotStartAnotherCopyOperation() async {
        let coordinator = ClipboardTransactionCoordinator()
        let gate = TestAsyncGate()
        var holderStarted = false
        var canceledOperationRan = false

        let holder = Task { @MainActor in
            await coordinator.withExclusiveAccess {
                holderStarted = true
                await gate.wait()
            }
        }
        while !holderStarted {
            await Task.yield()
        }

        let canceledWaiter = Task { @MainActor in
            await coordinator.withExclusiveAccessUnlessCancelled {
                canceledOperationRan = true
                return true
            }
        }
        await Task.yield()
        canceledWaiter.cancel()
        gate.open()

        await holder.value
        let result = await canceledWaiter.value
        #expect(result == nil)
        #expect(!canceledOperationRan)
    }

    @Test @MainActor func editModeDetectionDeadlineDoesNotWaitForHungAXTask() async {
        let gate = TestAsyncGate()
        let detectionTask = Task { @MainActor in
            await gate.wait()
        }
        let startedAt = Date()

        let completed = await EditModeDetectionWaiter.wait(
            for: detectionTask,
            timeoutNanoseconds: 30_000_000
        )
        let elapsed = Date().timeIntervalSince(startedAt)

        #expect(!completed)
        #expect(elapsed < 0.2)
        #expect(detectionTask.isCancelled)

        gate.open()
        await detectionTask.value
    }

    @Test func editModeInitialDecisionAvoidsLiveAXWhenCacheAlreadyRulesOutEditMode() {
        let terminalBundleIDs: Set<String> = ["com.apple.Terminal"]

        #expect(EditModeDetectionPolicy.initialDecision(
            bundleID: "com.apple.Terminal",
            currentPID: 42,
            cachedPID: 42,
            cachedIsEditable: true,
            terminalBundleIDs: terminalBundleIDs
        ) == .clear)

        #expect(EditModeDetectionPolicy.initialDecision(
            bundleID: "com.apple.TextEdit",
            currentPID: nil,
            cachedPID: 42,
            cachedIsEditable: true,
            terminalBundleIDs: terminalBundleIDs
        ) == .clear)

        #expect(EditModeDetectionPolicy.initialDecision(
            bundleID: "com.apple.TextEdit",
            currentPID: 42,
            cachedPID: 42,
            cachedIsEditable: false,
            terminalBundleIDs: terminalBundleIDs
        ) == .clear)

        #expect(EditModeDetectionPolicy.initialDecision(
            bundleID: "com.openai.codex",
            currentPID: 42,
            cachedPID: 42,
            cachedIsEditable: false,
            terminalBundleIDs: terminalBundleIDs
        ) == .applyLive(searchFocusedWindow: true))

        #expect(EditModeDetectionPolicy.initialDecision(
            bundleID: "com.apple.TextEdit",
            currentPID: 42,
            cachedPID: 42,
            cachedIsEditable: true,
            terminalBundleIDs: terminalBundleIDs
        ) == .applyLive(searchFocusedWindow: false))
    }

    @Test func editModeDictionaryConfirmationTakesPriorityInRecorder() {
        #expect(
            RecorderSupplementaryPresentation.resolve(
                hasDictionaryConfirmation: true,
                hasAssistantResponse: true,
                hasLiveTranscript: true
            ) == .dictionaryConfirmation
        )
        #expect(
            RecorderSupplementaryPresentation.resolve(
                hasDictionaryConfirmation: false,
                hasAssistantResponse: true,
                hasLiveTranscript: true
            ) == .assistant
        )
    }

    @Test @MainActor func selectedTextNormalizationPreservesSelectionBoundaries() {
        #expect(SelectedTextService.normalized("  hello\n") == "  hello\n")
        #expect(SelectedTextService.normalized("\n\t ") == nil)
        #expect(SelectedTextService.normalized(nil) == nil)
    }

    @Test func recordingStartupDoesNotDeferStartBranchAfterPermission() throws {
        let testsURL = URL(fileURLWithPath: #filePath)
        let sourceURL = testsURL
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("VoiceInk/Transcription/Engine/VoiceInkEngine.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)

        let permissionGuard = try #require(source.range(of: "guard requestRecordPermission() else"))
        let startID = try #require(
            source.range(of: "let startID = UUID()", range: permissionGuard.upperBound..<source.endIndex)
        )
        let stateSet = try #require(
            source.range(of: "self.recordingState = .starting", range: startID.upperBound..<source.endIndex)
        )
        let preStartBootstrap = permissionGuard.upperBound..<stateSet.lowerBound

        #expect(!source.contains("requestRecordPermission {"))
        #expect(source.range(of: "Task { @MainActor [self] in", range: preStartBootstrap) == nil)
        #expect(source.contains("private func requestRecordPermission() -> Bool"))
    }

    @Test @MainActor func toggleShortcutAllowsSecondStopTapInsideCancelWindow() async throws {
        var isRecorderVisible = false
        var recordingState: RecordingState = .idle
        var toggleCount = 0

        let handler = RecordingShortcutModeHandler(
            canHandleShortcutAction: { true },
            isRecorderVisible: { isRecorderVisible },
            recordingState: { recordingState },
            toggleRecorderPanel: { _ in
                toggleCount += 1
                if toggleCount == 1 {
                    isRecorderVisible = true
                    recordingState = .recording
                }
            },
            cancelRecording: {}
        )

        await handler.handleKeyDown(
            action: .primaryRecording,
            eventTime: 1.0,
            mode: .toggle
        )
        await handler.handleKeyUp(
            action: .primaryRecording,
            eventTime: 1.05,
            mode: .toggle
        )
        #expect(toggleCount == 1)

        try await Task.sleep(nanoseconds: 600_000_000)

        await handler.handleKeyDown(
            action: .primaryRecording,
            eventTime: 2.0,
            mode: .toggle
        )
        await handler.handleKeyUp(
            action: .primaryRecording,
            eventTime: 2.05,
            mode: .toggle
        )
        #expect(toggleCount == 2)

        try await Task.sleep(nanoseconds: 200_000_000)

        await handler.handleKeyDown(
            action: .primaryRecording,
            eventTime: 2.2,
            mode: .toggle
        )

        #expect(toggleCount == 3)
    }

    @Test @MainActor func recordingShortcutSyntheticTriggerFlowHandlesHundredStartsQuickly() async {
        let triggerCount = 100
        var toggleCount = 0
        var cancelCount = 0
        var elapsedMilliseconds: [Double] = []
        elapsedMilliseconds.reserveCapacity(triggerCount)

        for iteration in 0..<triggerCount {
            var isRecorderVisible = false
            var recordingState: RecordingState = .idle

            let handler = RecordingShortcutModeHandler(
                canHandleShortcutAction: { true },
                isRecorderVisible: { isRecorderVisible },
                recordingState: { recordingState },
                toggleRecorderPanel: { _ in
                    toggleCount += 1
                    isRecorderVisible = true
                    recordingState = .recording
                },
                cancelRecording: {
                    cancelCount += 1
                }
            )

            let scheduledAt = DispatchTime.now().uptimeNanoseconds
            let elapsedMs: Double = await withCheckedContinuation { continuation in
                Task { @MainActor in
                    await handler.handleKeyDown(
                        action: .primaryRecording,
                        eventTime: TimeInterval(iteration),
                        mode: .toggle
                    )
                    await handler.handleKeyUp(
                        action: .primaryRecording,
                        eventTime: TimeInterval(iteration) + 0.02,
                        mode: .toggle
                    )

                    let finishedAt = DispatchTime.now().uptimeNanoseconds
                    continuation.resume(returning: Double(finishedAt - scheduledAt) / 1_000_000)
                }
            }

            elapsedMilliseconds.append(elapsedMs)
        }

        let totalMs = elapsedMilliseconds.reduce(0, +)
        let averageMs = totalMs / Double(triggerCount)
        let maxMs = elapsedMilliseconds.max() ?? 0

        print(
            String(
                format: "recordingShortcutSyntheticTrigger100 count=%d avgMs=%.3f maxMs=%.3f",
                triggerCount,
                averageMs,
                maxMs
            )
        )

        #expect(toggleCount == triggerCount)
        #expect(cancelCount == 0)
        #expect(averageMs < 20)
        #expect(maxMs < 200)
    }

    @Test @MainActor func recordingShortcutSyntheticTriggerStartsAfterLongIdleQuickly() async {
        var isRecorderVisible = false
        var recordingState: RecordingState = .idle
        var currentDate = Date(timeIntervalSinceReferenceDate: 1_000)
        var toggleCount = 0
        var cancelCount = 0

        let handler = RecordingShortcutModeHandler(
            canHandleShortcutAction: { true },
            isRecorderVisible: { isRecorderVisible },
            recordingState: { recordingState },
            toggleRecorderPanel: { _ in
                toggleCount += 1
                isRecorderVisible = true
                recordingState = .recording
            },
            cancelRecording: {
                cancelCount += 1
            },
            currentDate: {
                currentDate
            }
        )

        let firstScheduledAt = DispatchTime.now().uptimeNanoseconds
        let firstElapsedMs: Double = await withCheckedContinuation { continuation in
            Task { @MainActor in
                await handler.handleKeyDown(
                    action: .primaryRecording,
                    eventTime: 0,
                    mode: .toggle
                )
                await handler.handleKeyUp(
                    action: .primaryRecording,
                    eventTime: 0.02,
                    mode: .toggle
                )

                let finishedAt = DispatchTime.now().uptimeNanoseconds
                continuation.resume(returning: Double(finishedAt - firstScheduledAt) / 1_000_000)
            }
        }

        #expect(toggleCount == 1)

        isRecorderVisible = false
        recordingState = .idle
        currentDate = currentDate.addingTimeInterval(3 * 60 * 60)

        let idleScheduledAt = DispatchTime.now().uptimeNanoseconds
        let idleElapsedMs: Double = await withCheckedContinuation { continuation in
            Task { @MainActor in
                await handler.handleKeyDown(
                    action: .primaryRecording,
                    eventTime: 3 * 60 * 60,
                    mode: .toggle
                )
                await handler.handleKeyUp(
                    action: .primaryRecording,
                    eventTime: 3 * 60 * 60 + 0.02,
                    mode: .toggle
                )

                let finishedAt = DispatchTime.now().uptimeNanoseconds
                continuation.resume(returning: Double(finishedAt - idleScheduledAt) / 1_000_000)
            }
        }

        print(
            String(
                format: "recordingShortcutLongIdleTrigger idleHours=3 firstMs=%.3f idleMs=%.3f",
                firstElapsedMs,
                idleElapsedMs
            )
        )

        #expect(toggleCount == 2)
        #expect(cancelCount == 0)
        #expect(firstElapsedMs < 200)
        #expect(idleElapsedMs < 200)
    }

    @Test @MainActor func recorderPanelStartFlowOrdersPanelVisibilityBeforeToggleRecord() async {
        var events: [String] = []
        var isPanelVisible = false

        await RecorderPanelStartFlow.run(
            resetStopStateAndCancelModelCleanup: {
                events.append("resetStopStateAndCancelModelCleanup")
            },
            playStartSound: {
                events.append("playStartSound")
            },
            detectEditMode: {
                events.append("detectEditMode")
            },
            setRecorderPanelVisible: {
                events.append("setRecorderPanelVisible")
                isPanelVisible = true
            },
            toggleRecord: {
                events.append("toggleRecord")
                #expect(isPanelVisible)
            },
            beginTrace: {
                events.append("begin:\($0)")
            },
            checkpoint: {
                events.append("checkpoint:\($0)")
            }
        )

        #expect(events == [
            "begin:hotkey_press",
            "resetStopStateAndCancelModelCleanup",
            "checkpoint:cancelModelCleanup_done",
            "playStartSound",
            "checkpoint:playStartSound_done",
            "detectEditMode",
            "checkpoint:detectEditMode_done",
            "setRecorderPanelVisible",
            "checkpoint:isRecorderPanelVisible_set",
            "toggleRecord",
        ])
    }

    @Test @MainActor func recorderPanelStartFlowSyntheticTriggerHandlesHundredStartsQuickly() async {
        let triggerCount = 100
        var cleanupCount = 0
        var soundCount = 0
        var detectCount = 0
        var visibleCount = 0
        var toggleCount = 0
        var elapsedMilliseconds: [Double] = []
        elapsedMilliseconds.reserveCapacity(triggerCount)

        for _ in 0..<triggerCount {
            var isPanelVisible = false
            let scheduledAt = DispatchTime.now().uptimeNanoseconds
            let elapsedMs: Double = await withCheckedContinuation { continuation in
                Task { @MainActor in
                    await RecorderPanelStartFlow.run(
                        resetStopStateAndCancelModelCleanup: {
                            cleanupCount += 1
                        },
                        playStartSound: {
                            soundCount += 1
                        },
                        detectEditMode: {
                            detectCount += 1
                        },
                        setRecorderPanelVisible: {
                            visibleCount += 1
                            isPanelVisible = true
                        },
                        toggleRecord: {
                            #expect(isPanelVisible)
                            toggleCount += 1
                        },
                        beginTrace: { _ in },
                        checkpoint: { _ in }
                    )

                    let finishedAt = DispatchTime.now().uptimeNanoseconds
                    continuation.resume(returning: Double(finishedAt - scheduledAt) / 1_000_000)
                }
            }

            elapsedMilliseconds.append(elapsedMs)
        }

        let totalMs = elapsedMilliseconds.reduce(0, +)
        let averageMs = totalMs / Double(triggerCount)
        let maxMs = elapsedMilliseconds.max() ?? 0

        print(
            String(
                format: "recorderPanelStartFlowSynthetic100 count=%d avgMs=%.3f maxMs=%.3f",
                triggerCount,
                averageMs,
                maxMs
            )
        )

        #expect(cleanupCount == triggerCount)
        #expect(soundCount == triggerCount)
        #expect(detectCount == triggerCount)
        #expect(visibleCount == triggerCount)
        #expect(toggleCount == triggerCount)
        #expect(averageMs < 20)
        #expect(maxMs < 200)
    }

    @Test @MainActor func recorderPanelStartFlowSyntheticTriggerStartsAfterLongIdleQuickly() async {
        var cleanupCount = 0
        var soundCount = 0
        var detectCount = 0
        var visibleCount = 0
        var toggleCount = 0

        func runSyntheticStart() async -> Double {
            var isPanelVisible = false
            let scheduledAt = DispatchTime.now().uptimeNanoseconds

            return await withCheckedContinuation { continuation in
                Task { @MainActor in
                    await RecorderPanelStartFlow.run(
                        resetStopStateAndCancelModelCleanup: {
                            cleanupCount += 1
                        },
                        playStartSound: {
                            soundCount += 1
                        },
                        detectEditMode: {
                            detectCount += 1
                        },
                        setRecorderPanelVisible: {
                            visibleCount += 1
                            isPanelVisible = true
                        },
                        toggleRecord: {
                            #expect(isPanelVisible)
                            toggleCount += 1
                        },
                        beginTrace: { _ in },
                        checkpoint: { _ in }
                    )

                    let finishedAt = DispatchTime.now().uptimeNanoseconds
                    continuation.resume(returning: Double(finishedAt - scheduledAt) / 1_000_000)
                }
            }
        }

        let firstElapsedMs = await runSyntheticStart()
        let simulatedIdleHours = 3
        let idleElapsedMs = await runSyntheticStart()

        print(
            String(
                format: "recorderPanelStartFlowLongIdleTrigger idleHours=%d firstMs=%.3f idleMs=%.3f",
                simulatedIdleHours,
                firstElapsedMs,
                idleElapsedMs
            )
        )

        #expect(cleanupCount == 2)
        #expect(soundCount == 2)
        #expect(detectCount == 2)
        #expect(visibleCount == 2)
        #expect(toggleCount == 2)
        #expect(firstElapsedMs < 200)
        #expect(idleElapsedMs < 200)
    }

    @Test @MainActor func engineRecordingStartFlowOrdersStartingStateBeforeRecorderStart() async throws {
        var events: [String] = []
        var recordedFile: URL?
        var installedAudioCallback: ((Data) -> Void)?
        var recordingState: RecordingState = .idle
        var scheduledMute = false
        let expectedURL = URL(fileURLWithPath: "/tmp/voco-engine-start-flow-order.wav")

        let output = try await EngineRecordingStartFlow.run(
            makeRecordingURL: {
                events.append("makeRecordingURL")
                return expectedURL
            },
            setRecordedFile: {
                events.append("setRecordedFile")
                recordedFile = $0
            },
            setAudioChunkCallback: {
                events.append("setAudioChunkCallback")
                installedAudioCallback = $0
            },
            setRecordingStateStarting: {
                events.append("setRecordingStateStarting")
                recordingState = .starting
            },
            scheduleSystemMute: {
                events.append("scheduleSystemMute")
                scheduledMute = true
            },
            startRecording: {
                events.append("startRecording")
                #expect($0 == expectedURL)
                #expect(recordingState == .starting)
                #expect(scheduledMute)
                installedAudioCallback?(Data([1, 2, 3]))
            },
            checkpoint: {
                events.append("checkpoint:\($0)")
            },
            endTrace: {
                events.append("end:\($0)")
            }
        )

        let bufferedChunks = output.pendingChunks.withLock { $0 }

        #expect(recordedFile == expectedURL)
        #expect(output.recordingURL == expectedURL)
        #expect(bufferedChunks == [Data([1, 2, 3])])
        #expect(events == [
            "makeRecordingURL",
            "setRecordedFile",
            "checkpoint:toggleRecord_recording_file_prepared",
            "setAudioChunkCallback",
            "checkpoint:toggleRecord_audio_callback_set",
            "setRecordingStateStarting",
            "checkpoint:toggleRecord_state_set_starting",
            "scheduleSystemMute",
            "checkpoint:toggleRecord_before_startRecording",
            "startRecording",
            "end:recorder_startRecording_done",
        ])
    }

    @Test @MainActor func engineRecordingStartFlowSyntheticTriggerHandlesHundredStartsQuickly() async throws {
        let triggerCount = 100
        var recordedFileCount = 0
        var callbackCount = 0
        var startingStateCount = 0
        var muteCount = 0
        var startRecordingCount = 0
        var elapsedMilliseconds: [Double] = []
        elapsedMilliseconds.reserveCapacity(triggerCount)

        for iteration in 0..<triggerCount {
            var installedAudioCallback: ((Data) -> Void)?
            var recordingState: RecordingState = .idle
            let expectedURL = URL(fileURLWithPath: "/tmp/voco-engine-start-flow-\(iteration).wav")

            let scheduledAt = DispatchTime.now().uptimeNanoseconds
            let elapsedMs: Double = try await withCheckedThrowingContinuation { continuation in
                Task { @MainActor in
                    do {
                        let output = try await EngineRecordingStartFlow.run(
                            makeRecordingURL: {
                                expectedURL
                            },
                            setRecordedFile: {
                                #expect($0 == expectedURL)
                                recordedFileCount += 1
                            },
                            setAudioChunkCallback: {
                                callbackCount += 1
                                installedAudioCallback = $0
                            },
                            setRecordingStateStarting: {
                                startingStateCount += 1
                                recordingState = .starting
                            },
                            scheduleSystemMute: {
                                muteCount += 1
                            },
                            startRecording: {
                                #expect($0 == expectedURL)
                                #expect(recordingState == .starting)
                                startRecordingCount += 1
                                installedAudioCallback?(Data([UInt8(iteration % 255)]))
                            },
                            checkpoint: { _ in },
                            endTrace: { _ in }
                        )

                        let chunkCount = output.pendingChunks.withLock { $0.count }
                        #expect(chunkCount == 1)

                        let finishedAt = DispatchTime.now().uptimeNanoseconds
                        continuation.resume(returning: Double(finishedAt - scheduledAt) / 1_000_000)
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }

            elapsedMilliseconds.append(elapsedMs)
        }

        let totalMs = elapsedMilliseconds.reduce(0, +)
        let averageMs = totalMs / Double(triggerCount)
        let maxMs = elapsedMilliseconds.max() ?? 0

        print(
            String(
                format: "engineRecordingStartFlowSynthetic100 count=%d avgMs=%.3f maxMs=%.3f",
                triggerCount,
                averageMs,
                maxMs
            )
        )

        #expect(recordedFileCount == triggerCount)
        #expect(callbackCount == triggerCount)
        #expect(startingStateCount == triggerCount)
        #expect(muteCount == triggerCount)
        #expect(startRecordingCount == triggerCount)
        #expect(averageMs < 20)
        #expect(maxMs < 200)
    }

}

@MainActor
private final class TestAsyncGate {
    private var isOpen = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        guard !isOpen else { return }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func open() {
        isOpen = true
        let pending = waiters
        waiters.removeAll()
        pending.forEach { $0.resume() }
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

private func disabledAutoApplyModelService() -> VocoAutoApplyModelService {
    VocoAutoApplyModelService(
        modelURL: FileManager.default.temporaryDirectory
            .appendingPathComponent("disabled-auto-apply-\(UUID().uuidString).json")
    )
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
