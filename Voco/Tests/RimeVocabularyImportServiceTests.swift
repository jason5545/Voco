import Foundation
import SwiftData
import Testing
@testable import Voco

struct RimeVocabularyImportServiceTests {
    private let service = RimeVocabularyImportService()

    @Test func parserKeepsSectionCodeWeightAndCategory() throws {
        let text = """
        # Rime table
        # coding: utf-8
        # 人名
        王小明\twang xiao ming\t42
        小明\txiao ming
        """

        let candidates = service.parse(text, sourceFile: .personalDictionary)

        #expect(candidates.count == 2)
        #expect(candidates[0].term == "王小明")
        #expect(candidates[0].sourceFile == "personal_dict.txt")
        #expect(candidates[0].section == "人名")
        #expect(candidates[0].code == "wang xiao ming")
        #expect(candidates[0].weight == 42)
        #expect(candidates[0].categoryGuess == .personName)
        #expect(candidates[1].weight == nil)
    }

    @Test func sectionAndCategoryGuessesSeparatePeopleTechAndGeneralTerms() throws {
        let personal = service.parse(
            """
            # 人名
            王小明\twang xiao ming
            # 一般詞
            天空\ttian kong
            """,
            sourceFile: .personalDictionary
        )
        let custom = service.parse(
            """
            # ========== 技術縮寫 ==========
            API\tapi
            # ========== 常用英文單字 ==========
            okay\tokay
            """,
            sourceFile: .customPhrase
        )

        #expect(personal[0].categoryGuess == .personName)
        #expect(personal[1].categoryGuess == .generalTerm)
        #expect(custom[0].section == "技術縮寫")
        #expect(custom[0].categoryGuess == .technicalTerm)
        #expect(custom[1].categoryGuess == .generalTerm)
    }

    @Test func previewSkipsDuplicateRimeCandidates() throws {
        let candidates = service.parse(
            """
            # 技術縮寫
            API\tapi
            API\tapi2
            """,
            sourceFile: .customPhrase
        )

        let preview = service.makePreview(
            candidates: candidates,
            existingVocabularyWords: [],
            existingProtectedTerms: []
        )

        #expect(preview.items.count == 2)
        #expect(preview.items[0].isImportable)
        #expect(preview.items[1].isSkipped)
        #expect(preview.items[1].duplicateInPreview)
        #expect(preview.items[1].skipReasons.contains("duplicate in RIME preview"))
    }

    @Test func previewSkipsUnsafeCandidates() throws {
        let candidates = service.parse(
            """
            # 常用英文單字
            okay\tokay
            x\tx
            updated.\tupdated
            person@example.com\temail
            這是一個很長很長很長的句子。\tlong
            """,
            sourceFile: .customPhrase
        )

        let preview = service.makePreview(
            candidates: candidates,
            existingVocabularyWords: [],
            existingProtectedTerms: []
        )

        #expect(preview.items.allSatisfy { $0.isSkipped })
        #expect(preview.items.contains { $0.candidate.term == "okay" && $0.skipReasons.contains("common term") })
        #expect(preview.items.contains { $0.candidate.term == "x" && $0.skipReasons.contains("too short") })
        #expect(preview.items.contains { $0.candidate.term == "updated." && $0.skipReasons.contains("too much punctuation") })
        #expect(preview.items.contains { $0.candidate.term == "person@example.com" && $0.skipReasons.contains("email-like") })
        #expect(preview.items.contains { $0.candidate.term.hasPrefix("這是一個") && $0.skipReasons.contains("sentence-like") })
    }

    @Test @MainActor func previewDoesNotMutateSwiftDataOrProtectionList() throws {
        let context = try makeRimeDictionaryContext()
        context.insert(VocabularyWord(word: "API"))
        try context.save()

        let directory = try temporaryRimeDirectory(
            personal: """
            # 人名
            王小明\twang xiao ming
            """,
            custom: """
            # 技術縮寫
            API\tapi
            NDT\tndt
            """
        )
        let protectionList = TestProtectionList(words: ["NDT"])
        let beforeWords = try fetchVocabularyWords(from: context)

        let preview = service.makePreview(
            rimeDirectory: directory,
            context: context,
            protectionList: protectionList
        )
        let afterWords = try fetchVocabularyWords(from: context)

        #expect(beforeWords == afterWords)
        #expect(preview.summary.existingCount == 2)
        #expect(!protectionList.words.contains("王小明"))
    }

    @Test @MainActor func importSelectedCandidatesOnly() throws {
        let context = try makeRimeDictionaryContext()
        let protectionList = TestProtectionList()

        let personCandidates = service.parse(
            """
            # 人名
            王小明\twang xiao ming
            小明\txiao ming
            """,
            sourceFile: .personalDictionary
        )
        let technicalCandidates = service.parse(
            """
            # 技術縮寫
            API\tapi
            """,
            sourceFile: .customPhrase
        )
        let preview = service.makePreview(
            candidates: personCandidates + technicalCandidates,
            existingVocabularyWords: [],
            existingProtectedTerms: []
        )
        let selected = preview.items.filter { $0.candidate.term == "王小明" }

        let result = try service.importSelectedItems(
            selected,
            context: context,
            protectionList: protectionList
        )
        let words = try fetchVocabularyWords(from: context)

        #expect(result.insertedVocabularyCount == 1)
        #expect(result.insertedProtectedTermCount == 1)
        #expect(words == ["王小明"])
        #expect(protectionList.words == ["王小明"])
    }
}

private final class TestProtectionList: CorrectionProtectionManaging {
    var words: Set<String>

    init(words: Set<String> = []) {
        self.words = words
    }

    func allWords() -> [String] {
        Array(words)
    }

    func addSynchronously(_ word: String) {
        words.insert(word)
    }
}

@MainActor
private func makeRimeDictionaryContext() throws -> ModelContext {
    let schema = Schema([VocabularyWord.self])
    let config = ModelConfiguration(
        "rime-dictionary-test-\(UUID().uuidString)",
        schema: schema,
        isStoredInMemoryOnly: true
    )
    let container = try ModelContainer(for: schema, configurations: [config])
    return ModelContext(container)
}

@MainActor
private func fetchVocabularyWords(from context: ModelContext) throws -> [String] {
    let descriptor = FetchDescriptor<VocabularyWord>(sortBy: [SortDescriptor(\.word)])
    return try context.fetch(descriptor).map(\.word)
}

private func temporaryRimeDirectory(personal: String, custom: String) throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("RimeVocabularyImportServiceTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    try personal.write(
        to: directory.appendingPathComponent("personal_dict.txt"),
        atomically: true,
        encoding: .utf8
    )
    try custom.write(
        to: directory.appendingPathComponent("custom_phrase.txt"),
        atomically: true,
        encoding: .utf8
    )
    return directory
}
