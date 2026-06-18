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

    @Test func vocabularyDoesNotPhoneticallyRewriteNameTerm() async throws {
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

        #expect(result.normalizedText == "汪曉鳴")
        #expect(result.replacements.isEmpty)
        #expect(result.suggestions.isEmpty)
    }

    @Test func canonicalVocabularyNameDoesNotKeepAutoAddedTerminalPeriod() async throws {
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
        #expect(phonetic.normalizedText == "汪曉鳴。")
        #expect(phonetic.replacements.isEmpty)

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

    @Test func phoneticCorrectionTermsApplyExplicitPairsAgainstImportedVocabulary() async throws {
        try await requireLoadedPinyinDatabase()

        let terms = VocoCanonicalizationService.vocabularyTerms(from: [
            "簡瑞成",
            "李聖苓",
            "簡瑞彥",
            "李聖葒",
            "簡岳雄",
            "世紀風電",
        ])
        let service = stablePhoneticCorrectionCanonicalizationService()
        let result = service.normalize(
            "簡瑞成李勝林。李聖林。簡瑞燕。李信宏。李勝宏。簡越雄。四季風電。",
            activeContextIDs: [],
            additionalTerms: terms
        )

        #expect(result.normalizedText == "簡瑞成李聖苓。李聖苓。簡瑞彥。李信宏。李聖葒。簡岳雄。世紀風電。")
        #expect(result.replacements.map(\.originalText) == ["李勝林", "李聖林", "簡瑞燕", "李勝宏", "簡越雄", "四季風電"])
        #expect(result.replacements.map(\.replacementText) == ["李聖苓", "李聖苓", "簡瑞彥", "李聖葒", "簡岳雄", "世紀風電"])
        #expect(result.replacements.allSatisfy { $0.reason == VocoPhoneticCorrectionService.reason })
        #expect(result.replacements.contains { $0.originalText == "李信宏" } == false)
    }

    @Test func phoneticCorrectionTermsRequireTargetVocabularyByDefault() async throws {
        try await requireLoadedPinyinDatabase()

        let terms = VocoCanonicalizationService.vocabularyTerms(from: ["簡瑞成"])
        let service = stablePhoneticCorrectionCanonicalizationService()
        let result = service.normalize(
            "李勝林。簡瑞燕。四季風電。",
            activeContextIDs: [],
            additionalTerms: terms
        )

        #expect(result.normalizedText == "李勝林。簡瑞燕。四季風電。")
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

        #expect(result.normalizedText == "我叫汪曉鳴。")
        #expect(result.replacements.isEmpty)
    }

    private func disabledAutoApplyModelService() -> VocoAutoApplyModelService {
        VocoAutoApplyModelService(
            modelURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("disabled-auto-apply-\(UUID().uuidString).json")
        )
    }

    private func stablePhoneticCorrectionCanonicalizationService() -> VocoCanonicalizationService {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("VocoPhoneticCorrectionTests-\(UUID().uuidString)", isDirectory: true)
        let defaults = UserDefaults(suiteName: "VocoPhoneticCorrectionTests-\(UUID().uuidString)") ?? .standard

        return VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: VocoAutoApplyModelService(
                modelURL: root.appendingPathComponent("missing-auto-apply-model.json"),
                defaults: defaults
            ),
            runtimeCorrectionModelService: VocoRuntimeCorrectionModelService(
                artifactURL: root.appendingPathComponent("missing-runtime-correction-artifact.json"),
                eventLogURL: nil,
                defaults: defaults
            ),
            phoneticCorrectionService: VocoPhoneticCorrectionService(
                rulesURL: root
                    .appendingPathComponent("PhoneticCorrections", isDirectory: true)
                    .appendingPathComponent("phonetic-correction-rules.json")
            )
        )
    }
}
