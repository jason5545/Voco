import Foundation
import Testing
@testable import Voco

struct PhoneticFeatureExtractorTests {
    private func requireLoadedPinyinDatabase() async throws {
        for _ in 0..<100 {
            if PinyinDatabase.shared.isLoaded { return }
            try await Task.sleep(nanoseconds: 50_000_000)
        }

        try #require(PinyinDatabase.shared.isLoaded)
    }

    @Test func classifiesBasicScriptLengthAndCommandSignals() async throws {
        try await requireLoadedPinyinDatabase()

        let chinese = PhoneticFeatureExtractor.extract("修正")
        #expect(chinese.scriptMode == .zhOnly)
        #expect(chinese.languageMode == .mandarin)
        #expect(chinese.lengthBucket == .oneToFour)
        #expect(chinese.isCommandLike == true)
        #expect(chinese.phones.isEmpty == false)

        let english = PhoneticFeatureExtractor.extract("Load Fail")
        #expect(english.scriptMode == .enOnly)
        #expect(english.languageMode == .english)
        #expect(english.lengthBucket == .oneToFour)

        let mixed = PhoneticFeatureExtractor.extract("Workaround 與實作")
        #expect(mixed.scriptMode == .mixedZhEn)
        #expect(mixed.languageMode == .codeSwitch)
        #expect(mixed.isTechnicalTermCandidate == true)
    }

    @Test func extractsMandarinConfusionPairsForKnownExamples() async throws {
        try await requireLoadedPinyinDatabase()

        let examples = [
            ("修正", "小振"),
            ("失重", "實作"),
            ("智商", "諮商"),
            ("專欄", "專案"),
            ("拍板", "排版"),
            ("變吃", "辨識"),
        ]

        for (raw, target) in examples {
            let comparison = PhoneticFeatureExtractor.compare(raw: raw, target: target)
            #expect(comparison.raw.scriptMode == .zhOnly)
            #expect(comparison.target.scriptMode == .zhOnly)
            #expect(comparison.languageMode == .mandarin)
            #expect(comparison.weightedPhoneticEditDistance != nil)
            #expect(comparison.pinyinSimilarity != nil)
            #expect(comparison.confusionPairs.isEmpty == false)
        }
    }

    @Test func marksCrossScriptTechnicalExamplesWithoutApplyingThem() async throws {
        try await requireLoadedPinyinDatabase()

        let examples = [
            ("凹頭", "auto"),
            ("西成", "SESSION"),
            ("Load Fail", "Cloudflare"),
            ("fly demo up", "Flight envelope"),
            ("work on the resources", "Workaround 與實作"),
            ("藝術", "Issues"),
            ("often", "Orphan"),
        ]

        for (raw, target) in examples {
            let comparison = PhoneticFeatureExtractor.compare(raw: raw, target: target)
            #expect(comparison.languageMode == .crossScript || comparison.target.isTechnicalTermCandidate || comparison.raw.scriptMode == .enOnly)
            #expect(comparison.weightedPhoneticEditDistance != nil)
            #expect(comparison.confusionPairs.isEmpty == false)
        }
    }

    @Test func negativeControlSixtyNineRoundsDoesNotLookLikeCorrection() async throws {
        try await requireLoadedPinyinDatabase()

        let comparison = PhoneticFeatureExtractor.compare(raw: "69 輪", target: "69 輪")
        #expect(comparison.raw.scriptMode == .zhOnly)
        #expect(comparison.raw.lengthBucket == .oneToFour)
        #expect(comparison.isPurePhoneticCandidate == false)
        #expect(comparison.confusionPairs.isEmpty)
        #expect(comparison.weightedPhoneticEditDistance == 0)
    }
}
