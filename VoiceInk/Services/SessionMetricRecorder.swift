import Foundation
import SwiftData
import OSLog

enum SessionMetricRecorder {
    private static let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "SessionMetricRecorder")
    private static let source = "recorder"

    @discardableResult
    static func recordRecorderSession(
        transcription: Transcription,
        model: (any TranscriptionModel)?,
        in modelContext: ModelContext,
        timestamp: Date = Date()
    ) throws -> Bool {
        guard transcription.transcriptionStatus == TranscriptionStatus.completed.rawValue else {
            return false
        }

        let transcriptionId = transcription.id
        let descriptor = FetchDescriptor<SessionMetric>(
            predicate: #Predicate<SessionMetric> { metric in
                metric.transcriptionId == transcriptionId
            }
        )

        if try modelContext.fetchCount(descriptor) > 0 {
            return false
        }

        let textForCounting = finalTextForCounting(from: transcription)
        let wordCount = WordCounter.count(in: textForCounting)
        let audioDuration = max(transcription.duration, 0)
        let transcriptionDuration = transcription.transcriptionDuration.flatMap { $0 > 0 ? $0 : nil }
        let speedFactor = transcriptionDuration.flatMap { duration in
            audioDuration > 0 ? audioDuration / duration : nil
        }

        let enhancementDuration = transcription.enhancementDuration.flatMap { $0 > 0 ? $0 : nil }

        let metric = SessionMetric(
            transcriptionId: transcription.id,
            timestamp: timestamp,
            source: source,
            wordCount: wordCount,
            audioDuration: audioDuration,
            transcriptionModelName: transcription.transcriptionModelName ?? model?.displayName,
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
            candidateSourceCounts: SessionMetric.candidateSourceCounts(from: transcription.hypothesisDetails),
            reviewRequiredCandidateCount: SessionMetric.reviewRequiredCandidateCount(in: transcription.hypothesisDetails),
            candidateDivergenceRatio: SessionMetric.candidateDivergenceRatio(in: transcription.hypothesisDetails),
            selectedCandidateHypothesisSource: SessionMetric.selectedCandidateHypothesisSource(
                in: transcription.hypothesisDetails,
                selectedCandidate: transcription.selectedCandidate
            ),
            selectedCandidate: transcription.selectedCandidate,
            candidateSelectionSource: transcription.candidateSelectionSource,
            userCorrectionDistance: transcription.userCorrectionDistance,
            sourceTranscriptionID: transcription.sourceTranscriptionID,
            retranscriptionChangeCategory: transcription.retranscriptionAnalysis?.changeCategory.rawValue,
            retranscriptionChangeRatio: transcription.retranscriptionAnalysis?.changeRatio,
            retranscriptionEditDistance: transcription.retranscriptionAnalysis?.editDistance,
            retranscriptionConfidenceDelta: transcription.retranscriptionAnalysis?.confidenceDelta,
            finalPastedCharacterCount: finalPastedCharacterCount(from: transcription),
            finalPastedWordCount: finalPastedWordCount(from: transcription),
            pasteCommandPosted: transcription.pasteCommandPosted
        )

        modelContext.insert(metric)
        logger.notice("Recorded session metric for transcription \(transcriptionId.uuidString, privacy: .public)")
        return true
    }

    private static func finalTextForCounting(from transcription: Transcription) -> String {
        if let finalPastedText = transcription.finalPastedText,
           !finalPastedText.isEmpty {
            return finalPastedText
        }

        if let enhancedText = transcription.enhancedText,
           transcription.enhancementDuration != nil,
           !enhancedText.isEmpty {
            return enhancedText
        }

        return transcription.text
    }

    private static func finalPastedCharacterCount(from transcription: Transcription) -> Int {
        transcription.finalPastedText?.count ?? 0
    }

    private static func finalPastedWordCount(from transcription: Transcription) -> Int {
        guard let pastedText = transcription.finalPastedText?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !pastedText.isEmpty
        else {
            return 0
        }
        return WordCounter.count(in: pastedText)
    }
}
