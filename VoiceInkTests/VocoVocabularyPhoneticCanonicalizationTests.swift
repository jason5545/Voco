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

        let terms = VocoCanonicalizationService.vocabularyTerms(from: ["簡瑞成"])
        let result = VocoCanonicalizationService(contextPacks: []).normalize(
            "尖銳唇",
            activeContextIDs: [],
            additionalTerms: terms
        )

        #expect(result.normalizedText == "簡瑞成")
        #expect(result.replacements.count == 1)
        #expect(result.replacements.first?.originalText == "尖銳唇")
        #expect(result.replacements.first?.replacementText == "簡瑞成")
        #expect(result.replacements.first?.reason == "vocabulary-phonetic-match")
    }

    @Test func vocabularyPhoneticRepairDoesNotReplaceKnownCommonWords() async throws {
        try await requireLoadedPinyinDatabase()

        let terms = VocoCanonicalizationService.vocabularyTerms(from: ["簡瑞"])
        let result = VocoCanonicalizationService(contextPacks: []).normalize(
            "這個意見很尖銳",
            activeContextIDs: [],
            additionalTerms: terms
        )

        #expect(PinyinDatabase.shared.frequency(of: "尖銳") > 0)
        #expect(result.normalizedText == "這個意見很尖銳")
        #expect(result.replacements.isEmpty)
    }
}
