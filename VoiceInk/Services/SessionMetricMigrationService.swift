import Foundation
import SwiftData
import OSLog

@MainActor
final class SessionMetricMigrationService {
    static let shared = SessionMetricMigrationService()

    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "SessionMetricMigrationService")
    private let completionKey = "HasCompletedStatsMigration"
    private let backfillVersionKey = "SessionMetricBackfillVersion"
    private let currentBackfillVersion = 4
    private(set) var isRunning = false

    private init() {}

    @discardableResult
    func runIfNeeded(modelContainer: ModelContainer) -> Task<Void, Never>? {
        let needsInitialMigration = !UserDefaults.standard.bool(forKey: completionKey)
        let needsCurrentBackfill = UserDefaults.standard.integer(forKey: backfillVersionKey) < currentBackfillVersion

        guard (needsInitialMigration || needsCurrentBackfill), !isRunning else { return nil }
        isRunning = true

        let logger = self.logger
        let completionKey = self.completionKey
        let backfillVersionKey = self.backfillVersionKey
        let currentBackfillVersion = self.currentBackfillVersion

        return Task.detached(priority: .utility) {
            let backgroundContext = ModelContext(modelContainer)
            var insertedCount = 0
            var updatedCount = 0

            do {
                // Fetch once instead of checking per record.
                let existingMetrics = try backgroundContext.fetch(FetchDescriptor<SessionMetric>())
                let existingMetricsByTranscriptionID = Dictionary(
                    existingMetrics.map { ($0.transcriptionId, $0) },
                    uniquingKeysWith: { first, _ in first }
                )

                let descriptor = FetchDescriptor<Transcription>(
                    predicate: #Predicate<Transcription> { $0.transcriptionStatus == "completed" }
                )
                let transcriptions = try backgroundContext.fetch(descriptor)

                for transcription in transcriptions {
                    if let existingMetric = existingMetricsByTranscriptionID[transcription.id] {
                        existingMetric.recordDictationMetadata(from: transcription)
                        updatedCount += 1
                        continue
                    }

                    let enhancementDuration = transcription.enhancementDuration.flatMap { $0 > 0 ? $0 : nil }
                    let audioDuration = max(transcription.duration, 0)
                    let transcriptionDuration = transcription.transcriptionDuration.flatMap { $0 > 0 ? $0 : nil }
                    let speedFactor = transcriptionDuration.flatMap { d in
                        audioDuration > 0 ? audioDuration / d : nil
                    }
                    let finalPastedText = transcription.finalPastedText ?? ""
                    let finalPastedTextForCounting = finalPastedText.trimmingCharacters(in: .whitespacesAndNewlines)
                    let finalPastedWordCount = finalPastedTextForCounting.isEmpty
                        ? 0
                        : WordCounter.count(in: finalPastedTextForCounting)
                    let textForCounting: String = {
                        if let finalPasted = transcription.finalPastedText,
                           !finalPasted.isEmpty { return finalPasted }
                        if let enhanced = transcription.enhancedText,
                           transcription.enhancementDuration != nil,
                           !enhanced.isEmpty { return enhanced }
                        return transcription.text
                    }()

                    let metric = SessionMetric(
                        transcriptionId: transcription.id,
                        timestamp: transcription.timestamp,
                        source: "recorder",
                        wordCount: WordCounter.count(in: textForCounting),
                        audioDuration: audioDuration,
                        transcriptionModelName: transcription.transcriptionModelName,
                        transcriptionDuration: transcriptionDuration,
                        speedFactor: speedFactor,
                        powerModeName: transcription.powerModeName,
                        aiEnhancementModelName: transcription.aiEnhancementModelName,
                        enhancementDuration: enhancementDuration,
                        asrEngineID: transcription.asrEngineID,
                        languageMode: transcription.languageMode,
                        activeContextIDs: transcription.activeContextIDs,
                        canonicalizationReplacementCount: transcription.canonicalizationReplacements.count,
                        canonicalizationSuggestionCount: transcription.canonicalizationSuggestions.count,
                        confidenceScore: transcription.confidenceScore,
                        confidenceRoute: transcription.confidenceRoute,
                        confidenceReasons: transcription.confidenceReasons,
                        candidateCount: transcription.hypotheses.count,
                        selectedCandidate: transcription.selectedCandidate,
                        candidateSelectionSource: transcription.candidateSelectionSource,
                        userCorrectionDistance: transcription.userCorrectionDistance,
                        finalPastedCharacterCount: finalPastedText.count,
                        finalPastedWordCount: finalPastedWordCount,
                        pasteCommandPosted: transcription.pasteCommandPosted
                    )
                    backgroundContext.insert(metric)
                    insertedCount += 1
                }

                if insertedCount > 0 || updatedCount > 0 {
                    try backgroundContext.save()
                }

                UserDefaults.standard.set(true, forKey: completionKey)
                UserDefaults.standard.set(currentBackfillVersion, forKey: backfillVersionKey)
                logger.notice("Completed stats migration/backfill with \(insertedCount, privacy: .public) inserted and \(updatedCount, privacy: .public) updated session metric(s)")
            } catch {
                logger.error("Stats migration failed: \(error.localizedDescription, privacy: .public)")
            }

            await MainActor.run {
                SessionMetricMigrationService.shared.isRunning = false
                NotificationCenter.default.post(name: .sessionMetricsDidChange, object: nil)
            }
        }
    }
}
