import Foundation
import OSLog
import SwiftData

/// Applies a one-off dictionary repair through SwiftData so CloudKit exports it.
/// Direct SQLite edits leave no persistent history, never reach iCloud, and
/// come back after a reinstall.
///
/// Run once with:
/// `open /Applications/Voco.app --args -VocoDictionaryRepairPlanPath /path/plan.json`
/// The plan file is renamed to `*.applied` after a successful save.
enum VocoDictionaryRepairPlan {
    static let pathKey = "VocoDictionaryRepairPlanPath"

    private static let logger = Logger(subsystem: "com.jasonchien.Voco", category: "DictionaryRepairPlan")

    struct Plan: Decodable, Equatable {
        var updates: [Update] = []
        var inserts: [Insert] = []
        var deletes: [UUID] = []
    }

    struct Update: Decodable, Equatable {
        let id: UUID
        var originalText: String?
        var replacementText: String?
        var isEnabled: Bool?
        var source: String?
    }

    struct Insert: Decodable, Equatable {
        let id: UUID
        let originalText: String
        let replacementText: String
        let isEnabled: Bool
        let source: String
        let dateAdded: Date
        let hitCount: Int
        let lastSeenDate: Date
    }

    struct Result: Equatable {
        var updated = 0
        var inserted = 0
        var deleted = 0
    }

    static func decode(_ data: Data) throws -> Plan {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .secondsSince1970
        return try decoder.decode(Plan.self, from: data)
    }

    @MainActor
    static func applyIfRequested(context: ModelContext, defaults: UserDefaults = .standard) {
        guard let path = defaults.string(forKey: pathKey), !path.isEmpty else { return }
        let url = URL(fileURLWithPath: path)

        do {
            let plan = try decode(Data(contentsOf: url))
            let result = try apply(plan, context: context)
            logger.notice(
                "Applied dictionary repair plan: updated=\(result.updated, privacy: .public) inserted=\(result.inserted, privacy: .public) deleted=\(result.deleted, privacy: .public)"
            )
            try? FileManager.default.moveItem(at: url, to: url.appendingPathExtension("applied"))
        } catch {
            logger.error("Dictionary repair plan failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    /// Idempotent: matching values, existing inserts and missing deletes are skipped.
    @MainActor
    static func apply(_ plan: Plan, context: ModelContext) throws -> Result {
        var rowsByID: [UUID: WordReplacement] = [:]
        for row in try context.fetch(FetchDescriptor<WordReplacement>()) where rowsByID[row.id] == nil {
            rowsByID[row.id] = row
        }

        var result = Result()

        for update in plan.updates {
            guard let row = rowsByID[update.id] else { continue }
            var changed = false
            if let value = update.originalText, row.originalText != value {
                row.originalText = value
                changed = true
            }
            if let value = update.replacementText, row.replacementText != value {
                row.replacementText = value
                changed = true
            }
            if let value = update.isEnabled, row.isEnabled != value {
                row.isEnabled = value
                changed = true
            }
            if let value = update.source, row.source != value {
                row.source = value
                changed = true
            }
            if changed { result.updated += 1 }
        }

        for insert in plan.inserts where rowsByID[insert.id] == nil {
            let row = WordReplacement(
                originalText: insert.originalText,
                replacementText: insert.replacementText,
                dateAdded: insert.dateAdded,
                isEnabled: insert.isEnabled,
                source: insert.source
            )
            row.id = insert.id
            row.hitCount = insert.hitCount
            row.lastSeenDate = insert.lastSeenDate
            context.insert(row)
            rowsByID[insert.id] = row
            result.inserted += 1
        }

        for id in plan.deletes {
            guard let row = rowsByID.removeValue(forKey: id) else { continue }
            context.delete(row)
            result.deleted += 1
        }

        if result != Result() {
            do {
                try context.save()
            } catch {
                context.rollback()
                throw error
            }
        }
        return result
    }
}
