import Foundation
import SQLite3
import SwiftData
import Testing
@testable import Voco

@Suite(.serialized)
struct RowCorrectionMarkingsLabelTests {
    private func marking(_ state: String, applied: Bool, source: String? = "voco") -> [String: Any] {
        var raw: [String: Any] = ["eventId": "evt1", "state": state, "applied": applied]
        if let source { raw["correctionSource"] = source }
        return raw
    }

    private func json(_ markings: [[String: Any]]) -> String {
        let data = try! JSONSerialization.data(withJSONObject: markings)
        return String(data: data, encoding: .utf8)!
    }

    @Test func appliedIsGreen() {
        let label = RowCorrectionMarkings.label(json([marking("applied", applied: true)]))
        #expect(label?.tone == .green)
        #expect(label?.text.contains(String(localized: "Correction applied (Worker)")) == true)
        #expect(label?.text.contains("voco") == true)
    }

    @Test func reconciledCountsAsApplied() {
        let label = RowCorrectionMarkings.label(json([marking("reconciled", applied: true)]))
        #expect(label?.tone == .green)
    }

    @Test func pendingIsOrange() {
        let label = RowCorrectionMarkings.label(json([marking("pending", applied: false)]))
        #expect(label?.tone == .orange)
        #expect(label?.text.contains(String(localized: "Correction pending")) == true)
    }

    @Test func alreadyPresentIsSecondary() {
        let label = RowCorrectionMarkings.label(json([marking("alreadyPresent", applied: false)]))
        #expect(label?.tone == .secondary)
        #expect(label?.text.contains(String(localized: "Rule already exists")) == true)
    }

    @Test func otherStatesAreRecordedNotConfirmed() {
        let label = RowCorrectionMarkings.label(json([marking("unknownState", applied: false)]))
        #expect(label?.tone == .secondary)
        #expect(label?.text.contains(String(localized: "Correction recorded, not confirmed")) == true)
    }

    @Test func missingSourceReadsUnknownSource() {
        let label = RowCorrectionMarkings.label(json([marking("pending", applied: false, source: nil)]))
        #expect(label?.text.contains(String(localized: "unknown source")) == true)
    }

    @Test func sourcesAreDeduplicated() {
        let label = RowCorrectionMarkings.label(json([
            marking("pending", applied: false, source: "voco"),
            marking("pending", applied: false, source: "voco"),
        ]))
        #expect(label?.text.hasSuffix("voco") == true)
    }

    @Test func emptyAndMalformedCachesHaveNoLabel() {
        #expect(RowCorrectionMarkings.label(nil) == nil)
        #expect(RowCorrectionMarkings.label("not json") == nil)
        #expect(RowCorrectionMarkings.label("[]") == nil)
    }
}

@Suite(.serialized)
struct RowCorrectionMarkingsIdentityTests {
    private let context = RuleAssistantContext(
        rowPk: 42,
        timestampMs: 1_700_000_000_000,
        recordId: "11111111-2222-3333-4444-555555555555"
    )

    private func result(
        rowPk: Int64,
        platform: String = "voco",
        recordId: String? = "11111111-2222-3333-4444-555555555555",
        corrections: [[String: Any]] = []
    ) -> [String: Any] {
        var row: [String: Any] = ["platform": platform, "rowPk": rowPk]
        if let recordId { row["recordId"] = recordId }
        return [
            "schema": "voco.row-corrections.v1",
            "rows": [["correctionRow": row, "corrections": corrections]],
        ]
    }

    @Test func happyPathReturnsSerializedCorrections() throws {
        let corrections = [["eventId": "e1", "state": "applied", "applied": true, "correctionSource": "voco"]]
        let value = try RowCorrectionMarkings.forRow(result(rowPk: 42, corrections: corrections), context: context)
        let parsed = RowCorrectionMarkings.parse(value)
        #expect(parsed.count == 1)
        #expect(parsed[0].eventId == "e1")
    }

    @Test func wrongSchemaThrows() {
        var bad = result(rowPk: 42)
        bad["schema"] = "other"
        #expect(throws: RowCorrectionMarkingError.unknownSchema) {
            _ = try RowCorrectionMarkings.forRow(bad, context: context)
        }
    }

    @Test func androidPlatformRowDoesNotMatch() {
        let bad = result(rowPk: 42, platform: "vocotype")
        #expect(throws: RowCorrectionMarkingError.identityMismatch) {
            _ = try RowCorrectionMarkings.forRow(bad, context: context)
        }
    }

    @Test func wrongRowPkDoesNotMatch() {
        let bad = result(rowPk: 43)
        #expect(throws: RowCorrectionMarkingError.identityMismatch) {
            _ = try RowCorrectionMarkings.forRow(bad, context: context)
        }
    }

    @Test func recordIdMatchIsCaseInsensitive() throws {
        let value = try RowCorrectionMarkings.forRow(
            result(rowPk: 42, recordId: "11111111-2222-3333-4444-555555555555".uppercased()),
            context: context
        )
        #expect(value == "[]")
    }

    @Test func wrongRecordIdDoesNotMatch() {
        let bad = result(rowPk: 42, recordId: "99999999-2222-3333-4444-555555555555")
        #expect(throws: RowCorrectionMarkingError.identityMismatch) {
            _ = try RowCorrectionMarkings.forRow(bad, context: context)
        }
    }

    @Test func zeroOrTwoMatchingRowsThrow() {
        let empty: [String: Any] = ["schema": "voco.row-corrections.v1", "rows": [Any]()]
        #expect(throws: RowCorrectionMarkingError.identityMismatch) {
            _ = try RowCorrectionMarkings.forRow(empty, context: context)
        }
        var row: [String: Any] = ["platform": "voco", "rowPk": 42]
        row["recordId"] = "11111111-2222-3333-4444-555555555555"
        let doubled: [String: Any] = [
            "schema": "voco.row-corrections.v1",
            "rows": [
                ["correctionRow": row, "corrections": [Any]()],
                ["correctionRow": row, "corrections": [Any]()],
            ],
        ]
        #expect(throws: RowCorrectionMarkingError.identityMismatch) {
            _ = try RowCorrectionMarkings.forRow(doubled, context: context)
        }
    }
}

@Suite(.serialized)
struct TranscriptionRowIdentityTests {
    /// Proves sqliteRowPK is the real SQLite Z_PK: a temporary on-disk store is created,
    /// rows are inserted and saved, then Z_PK/ZID are read back through the SQLite C API.
    @MainActor
    @Test func sqliteRowPKMatchesZPK() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("voco-rowpk-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let storeURL = directory.appendingPathComponent("test.store")

        let config = ModelConfiguration(url: storeURL)
        let container = try ModelContainer(for: Transcription.self, configurations: config)
        let context = ModelContext(container)

        // Unsaved rows must not report a rowPk.
        let unsaved = Transcription(text: "尚未存檔", duration: 0)
        #expect(unsaved.sqliteRowPK == nil)

        var expected: [UUID: Int64] = [:]
        for index in 0..<3 {
            let row = Transcription(text: "第 \(index) 筆", duration: 0)
            context.insert(row)
            try context.save()
            guard let rowPk = row.sqliteRowPK else {
                Issue.record("saved row has no sqliteRowPK")
                return
            }
            expected[row.id] = rowPk
        }

        // Read Z_PK/ZID straight from the SQLite file and match by UUID.
        var db: OpaquePointer?
        guard sqlite3_open(storeURL.path, &db) == SQLITE_OK else {
            Issue.record("cannot open store")
            return
        }
        defer { sqlite3_close(db) }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT Z_PK, ZID FROM ZTRANSCRIPTION", -1, &statement, nil) == SQLITE_OK else {
            Issue.record("cannot prepare Z_PK query")
            return
        }
        defer { sqlite3_finalize(statement) }
        var actual: [UUID: Int64] = [:]
        while sqlite3_step(statement) == SQLITE_ROW {
            let pk = sqlite3_column_int64(statement, 0)
            guard let blob = sqlite3_column_blob(statement, 1) else { continue }
            let length = sqlite3_column_bytes(statement, 1)
            let data = Data(bytes: blob, count: Int(length))
            let uuid = UUID(uuid: (
                data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
                data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15]
            ))
            actual[uuid] = pk
        }
        #expect(actual == expected)
    }
}
