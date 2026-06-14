import Foundation
import Testing
@testable import Voco

struct VocoVocabularyPhoneticCanonicalizationTests {
    private func requireLoadedPinyinDatabase() async throws {
        for _ in 0..<100 {
            if PinyinDatabase.shared.isLoaded { return }
            try await Task.sleep(nanoseconds: 50_000_000)
        }

        try #require(PinyinDatabase.shared.isLoaded)
    }

    @Test func vocabularyPhoneticRepairNormalizesExistingNameTerm() async throws {
        try await requireLoadedPinyinDatabase()

        let terms = VocoCanonicalizationService.vocabularyTerms(from: ["王小明"])
        let result = VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: disabledAutoApplyModelService()
        ).normalize(
            "汪曉鳴",
            activeContextIDs: [],
            additionalTerms: terms
        )

        #expect(result.normalizedText == "王小明")
        #expect(result.replacements.count == 1)
        #expect(result.replacements.first?.originalText == "汪曉鳴")
        #expect(result.replacements.first?.replacementText == "王小明")
        #expect(result.replacements.first?.reason == "vocabulary-phonetic-match")
    }

    @Test func standaloneVocabularyNameDoesNotKeepAutoAddedTerminalPeriod() async throws {
        try await requireLoadedPinyinDatabase()

        let terms = VocoCanonicalizationService.vocabularyTerms(from: ["王小明"])
        let service = VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: disabledAutoApplyModelService()
        )

        let phonetic = service.normalize(
            "汪曉鳴。",
            activeContextIDs: [],
            additionalTerms: terms
        )
        #expect(phonetic.normalizedText == "王小明")
        #expect(phonetic.replacements.first?.reason == "vocabulary-phonetic-match")

        let canonical = service.normalize(
            "王小明。",
            activeContextIDs: [],
            additionalTerms: terms
        )
        #expect(canonical.normalizedText == "王小明")
        #expect(canonical.replacements.isEmpty)
    }

    @Test func vocabularyPhoneticRepairDoesNotReplaceKnownCommonWords() async throws {
        try await requireLoadedPinyinDatabase()

        let terms = VocoCanonicalizationService.vocabularyTerms(from: ["尖瑞"])
        let result = VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: disabledAutoApplyModelService()
        ).normalize(
            "這個意見很尖銳",
            activeContextIDs: [],
            additionalTerms: terms
        )

        #expect(PinyinDatabase.shared.frequency(of: "尖銳") > 0)
        #expect(result.normalizedText == "這個意見很尖銳")
        #expect(result.replacements.isEmpty)
    }

    @Test func vocabularyTerminalPeriodCleanupDoesNotAffectLongerSentences() async throws {
        try await requireLoadedPinyinDatabase()

        let terms = VocoCanonicalizationService.vocabularyTerms(from: ["王小明"])
        let result = VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: disabledAutoApplyModelService()
        ).normalize(
            "我叫汪曉鳴。",
            activeContextIDs: [],
            additionalTerms: terms
        )

        #expect(result.normalizedText == "我叫王小明。")
    }

    private func disabledAutoApplyModelService() -> VocoAutoApplyModelService {
        VocoAutoApplyModelService(
            modelURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("disabled-auto-apply-\(UUID().uuidString).json")
        )
    }
}
