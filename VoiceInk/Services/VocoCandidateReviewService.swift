import Foundation

enum VocoCandidateReviewService {
    @discardableResult
    static func acceptCandidate(
        _ candidate: String,
        for transcription: Transcription,
        normalizationResult: VocoNormalizationResult,
        confidenceAssessment: VocoConfidenceAssessment,
        rawTranscript: String?
    ) -> CorrectionFeedbackSignal? {
        let accepted = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !accepted.isEmpty else { return nil }

        let signal = CorrectionFeedbackService.candidateSelectionSignal(
            normalizationResult: normalizationResult,
            assessment: confidenceAssessment,
            selectedCandidate: accepted,
            rawTranscript: rawTranscript
        )

        transcription.text = accepted
        transcription.normalizedTranscript = accepted
        transcription.selectedCandidate = accepted

        guard let signal else { return nil }
        if hasCandidateFeedback(for: accepted, in: transcription) {
            if transcription.userCorrectionDistance == nil {
                transcription.userCorrectionDistance = signal.changeRatio
            }
            return nil
        }

        transcription.recordCorrectionFeedback(signal)
        transcription.userCorrectionDistance = signal.changeRatio
        return signal
    }

    @discardableResult
    static func acceptPersistedCandidate(
        _ candidate: String,
        for transcription: Transcription
    ) -> CorrectionFeedbackSignal? {
        let normalizedText = firstNonEmpty(
            transcription.normalizedTranscript,
            transcription.selectedCandidate,
            transcription.text
        )
        let rawTranscript = firstNonEmpty(transcription.rawTranscript, normalizedText)

        let normalizationResult = VocoNormalizationResult(
            originalText: rawTranscript,
            normalizedText: normalizedText,
            activeContextIDs: transcription.activeContextIDs,
            replacements: transcription.canonicalizationReplacements,
            suggestions: transcription.canonicalizationSuggestions
        )

        let assessment = VocoConfidenceAssessment(
            score: transcription.confidenceScore ?? 1,
            route: VocoConfidenceRoute(rawValue: transcription.confidenceRoute ?? "") ?? .directInsertion,
            reasons: transcription.confidenceReasons,
            candidates: transcription.hypotheses,
            candidateLabels: transcription.hypothesisLabels,
            hypothesisDetails: transcription.hypothesisDetails,
            correctionRiskProfile: correctionRiskProfile(from: transcription),
            selectedCandidate: firstNonEmpty(transcription.selectedCandidate, normalizedText)
        )

        return acceptCandidate(
            candidate,
            for: transcription,
            normalizationResult: normalizationResult,
            confidenceAssessment: assessment,
            rawTranscript: rawTranscript
        )
    }

    private static func hasCandidateFeedback(for acceptedText: String, in transcription: Transcription) -> Bool {
        transcription.correctionFeedback.contains { signal in
            signal.kind == .candidateSelection && signal.acceptedText == acceptedText
        }
    }

    private static func correctionRiskProfile(from transcription: Transcription) -> VocoCorrectionRiskProfile? {
        guard let recentSessionCount = transcription.correctionRiskSampleCount,
              let correctedSessionCount = transcription.correctionRiskCorrectedCount,
              let recentCorrectionRate = transcription.correctionRiskRate
        else {
            return nil
        }

        return VocoCorrectionRiskProfile(
            recentSessionCount: recentSessionCount,
            correctedSessionCount: correctedSessionCount,
            recentCorrectionRate: recentCorrectionRate,
            highRiskTermIDs: transcription.correctionRiskTermIDs,
            lookbackDays: VocoCorrectionRiskService.defaultLookbackDays,
            minimumSampleCount: VocoCorrectionRiskService.defaultMinimumSampleCount
        )
    }

    private static func firstNonEmpty(_ values: String?...) -> String {
        values
            .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
            .first { !$0.isEmpty } ?? ""
    }
}
