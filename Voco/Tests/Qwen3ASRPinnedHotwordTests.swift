import Foundation
import Testing
@testable import Voco

struct Qwen3ASRPinnedHotwordTests {
    @Test func decodeAcceptsValidFileWithNumericBoost() throws {
        let pinned = try Qwen3ASRPinnedHotwords.decode(
            from: pinnedData(terms: ["富貴一路三十巷十六號一樓"], boost: 6.5)
        )

        #expect(pinned.terms == ["富貴一路三十巷十六號一樓"])
        #expect(pinned.boost == 6.5)
    }

    @Test func decodeAcceptsNullBoost() throws {
        let pinned = try Qwen3ASRPinnedHotwords.decode(
            from: pinnedData(terms: ["富貴一路三十巷十六號一樓"], boost: nil)
        )

        #expect(pinned.terms == ["富貴一路三十巷十六號一樓"])
        #expect(pinned.boost == nil)
    }

    @Test func decodeAcceptsAbsentBoostAndToleratesUnknownFields() throws {
        let json = """
        {
          "schema": "\(Qwen3ASRPinnedHotwords.supportedSchema)",
          "terms": ["富貴一路三十巷十六號一樓"],
          "futureField": { "nested": [1, 2, 3] }
        }
        """
        let pinned = try Qwen3ASRPinnedHotwords.decode(from: Data(json.utf8))

        #expect(pinned.terms == ["富貴一路三十巷十六號一樓"])
        #expect(pinned.boost == nil)
    }

    @Test func decodeTrimsAndDeduplicatesTerms() throws {
        let pinned = try Qwen3ASRPinnedHotwords.decode(
            from: pinnedData(
                terms: [" 富貴一路三十巷十六號一樓 ", "富貴一路三十巷十六號一樓", "GitHub ", " GitHub", " "],
                boost: nil
            )
        )

        #expect(pinned.terms == ["富貴一路三十巷十六號一樓", "GitHub"])
    }

    @Test func decodeRejectsWrongSchema() {
        assertPinnedDecodeError(
            pinnedData(schema: "bad.schema", terms: ["富貴一路"], boost: nil),
            equals: .unsupportedSchema("bad.schema")
        )
    }

    @Test func decodeRejectsInvalidJSON() {
        do {
            _ = try Qwen3ASRPinnedHotwords.decode(from: Data("not json".utf8))
            Issue.record("Expected decode to throw")
        } catch let error as Qwen3ASRPinnedHotwordsError {
            guard case .invalidJSON = error else {
                Issue.record("Expected invalidJSON, got \(error)")
                return
            }
        } catch {
            Issue.record("Unexpected error \(error)")
        }
    }

    @Test func decodeRejectsMoreThanEightTerms() {
        let terms = (1...9).map { "term\($0)" }
        assertPinnedDecodeError(
            pinnedData(terms: terms, boost: nil),
            equals: .tooManyTerms(9)
        )
    }

    @Test func decodeRejectsTermShorterThanTwoCharacters() {
        assertPinnedDecodeError(
            pinnedData(terms: ["甲"], boost: nil),
            equals: .invalidTermLength("甲")
        )
    }

    @Test func decodeRejectsTermLongerThanFortyCharacters() {
        let term = String(repeating: "富", count: 41)
        assertPinnedDecodeError(
            pinnedData(terms: [term], boost: nil),
            equals: .invalidTermLength(term)
        )
    }

    @Test func decodeRejectsZeroBoost() {
        assertPinnedDecodeError(
            pinnedData(terms: ["富貴一路"], boost: 0),
            equals: .invalidBoost(0)
        )
    }

    @Test func decodeRejectsBoostAboveSixteen() {
        assertPinnedDecodeError(
            pinnedData(terms: ["富貴一路"], boost: 17),
            equals: .invalidBoost(17)
        )
    }

    @Test func decodeRejectsNegativeBoost() {
        assertPinnedDecodeError(
            pinnedData(terms: ["富貴一路"], boost: -1),
            equals: .invalidBoost(-1)
        )
    }

    @Test func decodeRejectsEmptyTerms() {
        assertPinnedDecodeError(
            pinnedData(terms: [], boost: nil),
            equals: .emptyTerms
        )
        assertPinnedDecodeError(
            pinnedData(terms: ["", "  "], boost: nil),
            equals: .emptyTerms
        )
    }

    @Test @MainActor func storePicksUpPinnedFileFromInjectedDirectory() throws {
        let directory = try makeTemporaryDirectory()
        try pinnedData(terms: ["富貴一路三十巷十六號一樓"], boost: 6)
            .write(to: directory.appendingPathComponent(Qwen3ASRContextBiasStore.pinnedHotwordsFileName))
        let store = Qwen3ASRContextBiasStore(
            fileURL: directory.appendingPathComponent(Qwen3ASRContextBiasStore.profileFileName),
            defaults: try temporaryDefaults()
        )

        #expect(store.pinnedHotwords?.terms == ["富貴一路三十巷十六號一樓"])
        #expect(store.pinnedHotwords?.boost == 6)
        #expect(store.activePinnedHotwords()?.terms == ["富貴一路三十巷十六號一樓"])
    }

    @Test @MainActor func storeReturnsNilPinnedHotwordsWhenFileIsMissing() throws {
        let directory = try makeTemporaryDirectory()
        let store = Qwen3ASRContextBiasStore(
            fileURL: directory.appendingPathComponent(Qwen3ASRContextBiasStore.profileFileName),
            defaults: try temporaryDefaults()
        )

        #expect(store.pinnedHotwords == nil)
        #expect(store.activePinnedHotwords() == nil)
    }

    @Test @MainActor func storeIgnoresInvalidPinnedFile() throws {
        let directory = try makeTemporaryDirectory()
        try Data("not json".utf8)
            .write(to: directory.appendingPathComponent(Qwen3ASRContextBiasStore.pinnedHotwordsFileName))
        let store = Qwen3ASRContextBiasStore(
            fileURL: directory.appendingPathComponent(Qwen3ASRContextBiasStore.profileFileName),
            defaults: try temporaryDefaults()
        )

        #expect(store.pinnedHotwords == nil)
        #expect(store.activePinnedHotwords() == nil)
        #expect(store.activeProfile().sourceKind == .builtin)
    }

    @Test func mergedTermsKeepsAllPinnedAndCapsOnlyContextTerms() {
        let pinned = (1...8).map { "pinned\($0)" }
        let context = ["context1", "context2", "context3"]

        let merged = Qwen3ContextHotwordBias.mergedTerms(
            pinned: pinned,
            contextSelected: context,
            maxTermsPerDecode: 8
        )

        #expect(merged == pinned)
    }

    @Test func mergedTermsPutsPinnedFirstAndDeduplicatesContext() {
        let merged = Qwen3ContextHotwordBias.mergedTerms(
            pinned: ["B", "A"],
            contextSelected: ["A", "C", "B"],
            maxTermsPerDecode: 8
        )

        #expect(merged == ["B", "A", "C"])
    }

    @Test func mergedTermsCapLeavesRoomForContextAfterPinned() {
        let merged = Qwen3ContextHotwordBias.mergedTerms(
            pinned: ["P1"],
            contextSelected: ["C1", "C2", "C3"],
            maxTermsPerDecode: 2
        )

        #expect(merged == ["P1", "C1"])
    }

    @Test func needsSecondPassIsFalseWhenSelectedIsSubsetOfPinned() {
        #expect(Qwen3ContextHotwordBias.needsSecondPass(
            pinned: ["A", "B"],
            contextSelected: ["A", "B"]
        ) == false)
        #expect(Qwen3ContextHotwordBias.needsSecondPass(
            pinned: ["A", "B"],
            contextSelected: ["A"]
        ) == false)
        #expect(Qwen3ContextHotwordBias.needsSecondPass(
            pinned: ["A", "B"],
            contextSelected: []
        ) == false)
    }

    @Test func needsSecondPassIsTrueWhenSelectedHasTermOutsidePinned() {
        #expect(Qwen3ContextHotwordBias.needsSecondPass(
            pinned: ["A"],
            contextSelected: ["A", "C"]
        ) == true)
    }

    @Test func needsSecondPassIsFalseWhenBothAreEmpty() {
        #expect(Qwen3ContextHotwordBias.needsSecondPass(pinned: [], contextSelected: []) == false)
    }

    @Test func needsSecondPassIsTrueWhenPinnedIsEmptyAndSelectedIsNot() {
        #expect(Qwen3ContextHotwordBias.needsSecondPass(
            pinned: [],
            contextSelected: ["repo"]
        ) == true)
    }
}

private func assertPinnedDecodeError(
    _ data: Data,
    equals expected: Qwen3ASRPinnedHotwordsError
) {
    do {
        _ = try Qwen3ASRPinnedHotwords.decode(from: data)
        Issue.record("Expected decode error \(expected)")
    } catch let error as Qwen3ASRPinnedHotwordsError {
        #expect(error == expected)
    } catch {
        Issue.record("Unexpected error \(error)")
    }
}

private func pinnedData(
    schema: String = Qwen3ASRPinnedHotwords.supportedSchema,
    terms: [String],
    boost: Double?
) -> Data {
    let termsData = try! JSONSerialization.data(withJSONObject: terms)
    let termsJSON = String(data: termsData, encoding: .utf8)!
    let boostJSON = boost.map { String($0) } ?? "null"
    let json = """
    {
      "schema": "\(schema)",
      "terms": \(termsJSON),
      "boost": \(boostJSON)
    }
    """
    return Data(json.utf8)
}

private func makeTemporaryDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("Qwen3ASRPinnedHotwordTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
}

private func temporaryDefaults() throws -> UserDefaults {
    let suiteName = "Qwen3ASRPinnedHotwordTests-\(UUID().uuidString)"
    return try #require(UserDefaults(suiteName: suiteName))
}
