import Foundation
import SwiftData

/// Owns the single active rule-assistant conversation. Closing the panel or switching
/// views must not cancel an in-flight request, so sessions live here rather than in a
/// view; opening the assistant for a different record closes the previous conversation.
@MainActor
final class RuleAssistantSessionRegistry: ObservableObject {
    static let shared = RuleAssistantSessionRegistry()

    @Published private(set) var activeRecordId: UUID?

    private var session: RuleAssistantSession?

    private init() {}

    /// Returns the live session for this record, creating and wiring one on first use.
    /// nil when the record has no SQLite row identity (never saved), which blocks the assistant.
    @discardableResult
    func session(for transcription: Transcription, modelContext: ModelContext) -> RuleAssistantSession? {
        if let session, activeRecordId == transcription.id {
            syncConfig(session)
            return session
        }
        guard let rowPk = transcription.sqliteRowPK else { return nil }
        closeSession()

        let recordId = transcription.id
        let timestamp = transcription.timestamp
        let session = RuleAssistantSession(
            context: RuleAssistantContext(transcription: transcription, rowPk: rowPk),
            providerFactory: {
                guard let key = RuleAssistantKeyStore.shared.apiKey else { return nil }
                return OpenCodeGoClient(apiKey: key)
            },
            mcpFactory: {
                guard let syncKey = VocoAutoApplyModelService.defaultWorkerSyncKey(),
                      !syncKey.isEmpty
                else { return nil }
                return WorkerMCPClient(
                    baseURL: VocoAutoApplyWorkerSyncClient.defaultWorkerURL,
                    syncKey: syncKey
                )
            },
            syncNow: {
                RuleAssistantSyncResult(workerSync: await VocoAutoApplyModelService.shared.syncFromWorker())
            },
            neighborLoader: { [weak modelContext] before, after in
                guard let modelContext else { return [] }
                return Self.loadNeighbors(
                    recordId: recordId,
                    timestamp: timestamp,
                    before: before,
                    after: after,
                    modelContext: modelContext
                )
            },
            onCorrections: { [weak modelContext] _, json in
                guard let modelContext else { return }
                Self.writeCorrections(json, recordId: recordId, modelContext: modelContext)
            },
            guardSuggester: { source in
                RuleAssistantGuardSuggester.guards(for: source) { source in
                    VocoWordFrequencyLexicon.shared.words(
                        containing: source,
                        minFrequency: RuleAssistantGuardSuggester.minFrequency,
                        limit: RuleAssistantGuardSuggester.lookupLimit
                    )
                }
            },
            lexiconProbe: { source in
                let trimmed = source.trimmingCharacters(in: .whitespacesAndNewlines)
                let lexicon = VocoWordFrequencyLexicon.shared
                return lexicon.frequency(of: trimmed) == 0 && lexicon.words(
                    containing: trimmed,
                    minFrequency: RuleAssistantGuardSuggester.minFrequency,
                    limit: RuleAssistantGuardSuggester.lookupLimit
                ).isEmpty
            }
        )
        self.session = session
        activeRecordId = recordId
        syncConfig(session)
        return session
    }

    func closeSession() {
        session?.closeClients()
        session = nil
        activeRecordId = nil
    }

    /// The Go key was saved or removed: make the active session re-authenticate on its next request.
    func providerKeyChanged() {
        guard let session else { return }
        session.onProviderKeyChanged()
        syncConfig(session)
    }

    private func syncConfig(_ session: RuleAssistantSession) {
        session.setConfig(
            goKeyConfigured: RuleAssistantKeyStore.shared.apiKey != nil,
            syncConfigured: VocoAutoApplyModelService.defaultWorkerSyncKey()?.isEmpty == false
        )
    }

    /// Text-only contexts of the records around the selected one (by timestamp), excluding it.
    private static func loadNeighbors(
        recordId: UUID,
        timestamp: Date,
        before: Int,
        after: Int,
        modelContext: ModelContext
    ) -> [RuleAssistantContext] {
        var contexts: [RuleAssistantContext] = []
        if before > 0 {
            var descriptor = FetchDescriptor<Transcription>(
                predicate: #Predicate { $0.timestamp < timestamp },
                sortBy: [SortDescriptor(\.timestamp, order: .reverse)]
            )
            descriptor.fetchLimit = before
            contexts.append(contentsOf: makeContexts(for: (try? modelContext.fetch(descriptor)) ?? [], excluding: recordId))
        }
        if after > 0 {
            var descriptor = FetchDescriptor<Transcription>(
                predicate: #Predicate { $0.timestamp > timestamp },
                sortBy: [SortDescriptor(\.timestamp, order: .forward)]
            )
            descriptor.fetchLimit = after
            contexts.append(contentsOf: makeContexts(for: (try? modelContext.fetch(descriptor)) ?? [], excluding: recordId))
        }
        return contexts
    }

    private static func makeContexts(for rows: [Transcription], excluding recordId: UUID) -> [RuleAssistantContext] {
        rows.compactMap { row in
            guard row.id != recordId, let rowPk = row.sqliteRowPK else { return nil }
            // Neighbors are context, not the record under review: no runtime replay on the wire.
            return RuleAssistantContext(transcription: row, rowPk: rowPk, runtimeReplay: { _ in nil })
        }
    }

    /// Only correctionsJSON is ever written back from the assistant.
    private static func writeCorrections(_ json: String, recordId: UUID, modelContext: ModelContext) {
        var descriptor = FetchDescriptor<Transcription>(predicate: #Predicate { $0.id == recordId })
        descriptor.fetchLimit = 1
        guard let transcription = try? modelContext.fetch(descriptor).first else { return }
        guard transcription.correctionsJSON != json else { return }
        transcription.correctionsJSON = json
        try? modelContext.save()
    }
}
