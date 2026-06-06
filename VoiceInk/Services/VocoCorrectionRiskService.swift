import Foundation
import SwiftData

@MainActor
enum VocoCorrectionRiskService {
    nonisolated static let defaultLookbackDays = 14
    nonisolated static let defaultMinimumSampleCount = 3
    nonisolated static let defaultSampleLimit = 30

    static func profile(
        in modelContext: ModelContext,
        excluding excludedTranscriptionID: UUID? = nil,
        now: Date = Date(),
        lookbackDays: Int = defaultLookbackDays,
        minimumSampleCount: Int = defaultMinimumSampleCount,
        sampleLimit: Int = defaultSampleLimit
    ) -> VocoCorrectionRiskProfile {
        let cutoff = Calendar.current.date(
            byAdding: .day,
            value: -max(1, lookbackDays),
            to: now
        ) ?? now

        var descriptor = FetchDescriptor<Transcription>(
            sortBy: [SortDescriptor(\.timestamp, order: .reverse)]
        )
        descriptor.fetchLimit = max(sampleLimit * 3, minimumSampleCount)

        let fetched = (try? modelContext.fetch(descriptor)) ?? []
        let recent = fetched
            .filter { transcription in
                if let excludedTranscriptionID, transcription.id == excludedTranscriptionID {
                    return false
                }

                return transcription.timestamp >= cutoff &&
                    transcription.transcriptionStatus != TranscriptionStatus.canceled.rawValue &&
                    transcription.transcriptionStatus != TranscriptionStatus.failed.rawValue
            }
            .prefix(sampleLimit)

        let recentSessions = Array(recent)
        let correctedSessions = recentSessions.filter(hasCorrectionSignal)

        return VocoCorrectionRiskProfile(
            recentSessionCount: recentSessions.count,
            correctedSessionCount: correctedSessions.count,
            recentCorrectionRate: correctionRate(
                correctedCount: correctedSessions.count,
                totalCount: recentSessions.count
            ),
            highRiskTermIDs: highRiskTermIDs(from: correctedSessions),
            lookbackDays: lookbackDays,
            minimumSampleCount: minimumSampleCount
        )
    }

    private static func hasCorrectionSignal(_ transcription: Transcription) -> Bool {
        if !transcription.correctionFeedback.isEmpty {
            return transcription.correctionFeedback.contains(where: \.isCorrectiveSignal)
        }

        if let distance = transcription.userCorrectionDistance, distance >= 0.08 {
            return true
        }

        if let analysis = transcription.retranscriptionAnalysis,
           analysis.changeCategory != .unchanged {
            return true
        }

        return false
    }

    private static func correctionRate(correctedCount: Int, totalCount: Int) -> Double {
        guard totalCount > 0 else { return 0 }
        return Double(correctedCount) / Double(totalCount)
    }

    private static func highRiskTermIDs(from transcriptions: [Transcription]) -> [String] {
        var counts: [String: Int] = [:]
        for transcription in transcriptions {
            let ids = Set(
                transcription.correctionFeedback
                    .filter(\.isCorrectiveSignal)
                    .flatMap(\.termIDs)
            )
            for id in ids where !id.isEmpty {
                counts[id, default: 0] += 1
            }
        }

        return counts
            .filter { $0.value >= 2 }
            .map(\.key)
            .sorted()
    }
}
