import Foundation
import SwiftData

/// One Worker correction event receipt, cached verbatim so extra ledger fields survive a round trip.
struct RowCorrectionMarking: Equatable {
    let raw: [String: Any]

    var eventId: String? { raw.raString("eventId") }
    var correctionSource: String? { raw.raString("correctionSource") }
    var state: String? { raw.raString("state") }
    var applied: Bool { raw.raBool("applied") }

    func toJSONObject() -> [String: Any] { raw }

    static func == (lhs: RowCorrectionMarking, rhs: RowCorrectionMarking) -> Bool {
        NSDictionary(dictionary: lhs.raw).isEqual(NSDictionary(dictionary: rhs.raw))
    }
}

enum RowCorrectionMarkingTone: Equatable {
    case green
    case orange
    case secondary
}

enum RowCorrectionMarkingError: Error, Equatable {
    case unknownSchema
    case identityMismatch
}

/// Worker event receipts, separate from ASR output and from Mac model-sync status.
enum RowCorrectionMarkings {
    static let tool = "get_auto_apply_row_corrections"
    static let schema = "voco.row-corrections.v1"
    static let batchSize = 20

    /// Decodes a cached corrections JSON array; a malformed cache reads as empty and is never rewritten here.
    static func parse(_ value: String?) -> [RowCorrectionMarking] {
        guard let value, let array = RAJSON.parseArray(value) else { return [] }
        return array.compactMap { $0 as? [String: Any] }.map { RowCorrectionMarking(raw: $0) }
    }

    static func label(_ value: String?) -> (text: String, tone: RowCorrectionMarkingTone)? {
        label(for: parse(value))
    }

    /// SF Symbol for history badges: applied / pending / anything else.
    static func badgeIcon(for markings: [RowCorrectionMarking]) -> String {
        if markings.contains(where: { $0.applied && ($0.state == "applied" || $0.state == "reconciled") }) {
            return "checkmark.seal.fill"
        }
        if markings.contains(where: { $0.state == "pending" }) {
            return "clock.badge"
        }
        return "seal"
    }

    static func label(for markings: [RowCorrectionMarking]) -> (text: String, tone: RowCorrectionMarkingTone)? {
        guard !markings.isEmpty else { return nil }
        let applied = markings.filter { $0.applied && ($0.state == "applied" || $0.state == "reconciled") }
        let relevant = applied.isEmpty ? markings : applied
        var seenSources: Set<String> = []
        let sources = relevant
            .map { marking -> String in
                if let source = marking.correctionSource, !source.isEmpty { return source }
                return String(localized: "unknown source")
            }
            .filter { seenSources.insert($0).inserted }
        let stateText: String
        let tone: RowCorrectionMarkingTone
        if !applied.isEmpty {
            stateText = String(localized: "Correction applied (Worker)")
            tone = .green
        } else if markings.contains(where: { $0.state == "pending" }) {
            stateText = String(localized: "Correction pending")
            tone = .orange
        } else if markings.contains(where: { $0.state == "alreadyPresent" }) {
            stateText = String(localized: "Rule already exists")
            tone = .secondary
        } else {
            stateText = String(localized: "Correction recorded, not confirmed")
            tone = .secondary
        }
        return ("\(stateText) · \(sources.joined(separator: "\u{3001}"))", tone)
    }

    /// Fail closed on mismatched rows or malformed replies; never erase a cache on a failed query.
    static func forRow(_ result: [String: Any], context: RuleAssistantContext) throws -> String {
        guard result.raString("schema") == schema else {
            throw RowCorrectionMarkingError.unknownSchema
        }
        let matching = result.raDictArray("rows").filter { row in
            guard let correctionRow = row.raDict("correctionRow") else { return false }
            guard correctionRow.raString("platform") == "voco",
                  correctionRow.raInt64("rowPk") == context.rowPk
            else { return false }
            if let recordId = context.recordId {
                guard let rowRecordId = correctionRow.raString("recordId"),
                      rowRecordId.caseInsensitiveCompare(recordId) == .orderedSame
                else { return false }
            }
            return true
        }
        guard matching.count == 1, let corrections = matching[0].raArray("corrections") else {
            throw RowCorrectionMarkingError.identityMismatch
        }
        return RAJSON.serializeArray(corrections)
    }

    static func fetch(worker: RuleAssistantMCP, contexts: [RuleAssistantContext]) async throws -> [Int64: String] {
        guard !contexts.isEmpty else { return [:] }
        var output: [Int64: String] = [:]
        var index = 0
        while index < contexts.count {
            let batch = Array(contexts[index..<Swift.min(index + batchSize, contexts.count)])
            let result = try await worker.call(tool, args: ["rows": batch.map { $0.correctionRow() }])
            for context in batch {
                output[context.rowPk] = try forRow(result, context: context)
            }
            index += batchSize
        }
        return output
    }
}

// MARK: - Transcription bridging

extension Transcription {
    /// Cached Worker correction receipts for this row (decode failure reads as empty).
    var corrections: [RowCorrectionMarking] {
        RowCorrectionMarkings.parse(correctionsJSON)
    }

    var correctionMarkingLabel: (text: String, tone: RowCorrectionMarkingTone)? {
        RowCorrectionMarkings.label(correctionsJSON)
    }

    /// The SQLite Z_PK of this record's row. PersistentIdentifier exposes no public
    /// isTemporary/uriRepresentation members, but its JSON encoding carries both:
    /// implementation.primaryKey is "p<N>" (N = Z_PK) once saved, and uriRepresentation is
    /// x-coredata://.../Transcription/p<N>. Matches the namespace used by
    /// tools/voco_auto_apply_control.py --row-pk and the retranscribe skill. nil for
    /// unsaved/temporary identifiers and unparseable forms; such records cannot use the
    /// rule assistant or correction markings.
    var sqliteRowPK: Int64? {
        guard let data = try? JSONEncoder().encode(persistentModelID),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let implementation = object["implementation"] as? [String: Any],
              (implementation["isTemporary"] as? Bool) == false
        else { return nil }
        if let primaryKey = implementation["primaryKey"] as? String,
           let value = Self.parseSQLitePrimaryKey(primaryKey) {
            return value
        }
        if let uri = implementation["uriRepresentation"] as? String,
           let last = uri.split(separator: "/").last {
            return Self.parseSQLitePrimaryKey(String(last))
        }
        return nil
    }

    private static func parseSQLitePrimaryKey(_ key: String) -> Int64? {
        guard key.hasPrefix("p") else { return nil }
        return Int64(key.dropFirst())
    }
}

// MARK: - History page refresh

/// Fetches Worker correction markings for visible history rows and caches them on each
/// Transcription's correctionsJSON. Never touches any other field, keeps the old value on
/// failure, and only runs when a Worker sync key exists and the Worker offers the tool.
@MainActor
final class RowCorrectionMarkingRefresher: ObservableObject {
    static let shared = RowCorrectionMarkingRefresher()

    /// Drives the low-key "Correction markings not refreshed; showing last result" status line.
    @Published private(set) var refreshFailed = false

    private var task: Task<Void, Never>?
    private var client: WorkerMCPClient?

    private init() {}

    func refresh(_ transcriptions: [Transcription], modelContext: ModelContext) {
        guard let syncKey = VocoAutoApplyModelService.defaultWorkerSyncKey(),
              !syncKey.isEmpty
        else { return }
        let rows: [(transcription: Transcription, context: RuleAssistantContext)] = transcriptions.compactMap { transcription in
            guard let rowPk = transcription.sqliteRowPK else { return nil }
            return (transcription, RuleAssistantContext(transcription: transcription, rowPk: rowPk))
        }
        guard !rows.isEmpty else { return }
        task?.cancel()
        client?.close()
        task = Task { @MainActor [weak self] in
            guard let self else { return }
            let client = WorkerMCPClient(
                baseURL: VocoAutoApplyWorkerSyncClient.defaultWorkerURL,
                syncKey: syncKey
            )
            self.client = client
            defer {
                client.close()
                if self.client === client { self.client = nil }
            }
            do {
                let tools = try await client.tools()
                guard tools.contains(where: { $0.name == RowCorrectionMarkings.tool }) else { return }
                let markings = try await RowCorrectionMarkings.fetch(
                    worker: client,
                    contexts: rows.map(\.context)
                )
                try Task.checkCancellation()
                for (transcription, context) in rows {
                    guard let value = markings[context.rowPk],
                          transcription.correctionsJSON != value
                    else { continue }
                    // Only correctionsJSON is ever written here.
                    transcription.correctionsJSON = value
                }
                try? modelContext.save()
                refreshFailed = false
            } catch is CancellationError {
                // Superseded by a newer refresh; keep the previous failure state.
            } catch {
                refreshFailed = true
            }
        }
    }
}
