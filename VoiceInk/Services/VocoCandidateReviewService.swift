import Foundation

enum VocoCandidateReviewService {
    private static let maxPersistedCandidateCount = 5

    static func review(for assessment: VocoConfidenceAssessment) -> VocoCandidateReview? {
        guard assessment.route == .reviewSuggested else { return nil }

        let selectedCandidate = assessment.selectedCandidate.trimmingCharacters(in: .whitespacesAndNewlines)
        var entries: [(candidate: String, label: String, hypothesis: VocoHypothesis)] = []
        var seen: Set<String> = []

        if !selectedCandidate.isEmpty {
            let selectedIndex = assessment.candidates.firstIndex {
                $0.trimmingCharacters(in: .whitespacesAndNewlines) == selectedCandidate
            }
            let selectedHypothesis = selectedIndex
                .flatMap { assessment.hypothesisForCandidate(at: $0) }
                ?? fallbackHypothesis(
                    text: selectedCandidate,
                    label: selectedIndex.map { labelForCandidate(at: $0, assessment: assessment, hypothesis: nil) } ?? "Recommended",
                    assessment: assessment
                )
            entries.append((selectedCandidate, selectedHypothesis.label, selectedHypothesis))
            seen.insert(selectedCandidate)
        }

        for (index, rawCandidate) in assessment.candidates.enumerated() {
            let candidate = rawCandidate.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !candidate.isEmpty,
                  seen.insert(candidate).inserted
            else { continue }

            let hypothesis = assessment.hypothesisForCandidate(at: index)
            let label = labelForCandidate(at: index, assessment: assessment, hypothesis: hypothesis)
            entries.append((
                candidate,
                label,
                hypothesis ?? fallbackHypothesis(text: candidate, label: label, assessment: assessment)
            ))
        }

        let hasActionableAlternative: Bool
        if selectedCandidate.isEmpty {
            hasActionableAlternative = entries.count > 1
        } else {
            hasActionableAlternative = entries.contains { $0.candidate != selectedCandidate }
        }

        guard entries.count > 1,
              hasActionableAlternative
        else { return nil }

        let visibleEntries = Array(entries.prefix(5))
        return VocoCandidateReview(
            candidates: visibleEntries.map(\.candidate),
            candidateLabels: visibleEntries.map(\.label),
            hypotheses: visibleEntries.map(\.hypothesis),
            confidenceScore: assessment.score,
            reasons: assessment.reasons
        )
    }

    @discardableResult
    static func acceptCandidate(
        _ candidate: String,
        for transcription: Transcription,
        normalizationResult: VocoNormalizationResult,
        confidenceAssessment: VocoConfidenceAssessment,
        rawTranscript: String?,
        selectionSource: VocoCandidateSelectionSource = .userSelection
    ) -> CorrectionFeedbackSignal? {
        let accepted = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !accepted.isEmpty else { return nil }

        let signal = CorrectionFeedbackService.candidateSelectionSignal(
            normalizationResult: normalizationResult,
            assessment: confidenceAssessment,
            selectedCandidate: accepted,
            rawTranscript: rawTranscript,
            selectionSource: selectionSource
        )

        transcription.text = accepted
        transcription.normalizedTranscript = accepted
        transcription.selectedCandidate = accepted
        transcription.recordCandidateSelectionSource(selectionSource)
        recordAcceptedCandidateMetadata(
            accepted,
            for: transcription,
            assessment: confidenceAssessment
        )

        guard let signal else { return nil }
        if hasCandidateFeedback(for: accepted, in: transcription) {
            if signal.isCorrectiveSignal, transcription.userCorrectionDistance == nil {
                transcription.userCorrectionDistance = signal.changeRatio
            }
            return nil
        }

        transcription.recordCorrectionFeedback(signal)
        if signal.isCorrectiveSignal {
            transcription.userCorrectionDistance = signal.changeRatio
        }
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

    private static func recordAcceptedCandidateMetadata(
        _ accepted: String,
        for transcription: Transcription,
        assessment: VocoConfidenceAssessment
    ) {
        let baseCandidates = transcription.hypotheses.isEmpty ? assessment.candidates : transcription.hypotheses
        var candidates = normalizedCandidateList(baseCandidates)
        var labels = alignedLabels(
            transcription.hypothesisLabels.isEmpty ? assessment.candidateLabels : transcription.hypothesisLabels,
            count: candidates.count
        )
        var details = alignedHypotheses(
            transcription.hypothesisDetails.isEmpty ? assessment.hypothesisDetails : transcription.hypothesisDetails,
            candidates: candidates,
            labels: labels,
            assessment: assessment
        )

        if candidates.contains(where: { isSameCandidate($0, accepted) }) {
            transcription.hypotheses = candidates
            transcription.hypothesisLabels = labels
            transcription.hypothesisDetails = details
            return
        }

        if candidates.count >= maxPersistedCandidateCount {
            candidates.removeLast()
            if !labels.isEmpty { labels.removeLast() }
            if !details.isEmpty { details.removeLast() }
        }

        candidates.append(accepted)
        labels.append("Typed correction")
        details.append(
            VocoHypothesis(
                id: "custom-rescue",
                text: accepted,
                label: "Typed correction",
                source: .customRescue,
                confidenceScore: assessment.score,
                reasons: ["candidate-custom"] + assessment.reasons,
                activeContextIDs: assessment.hypothesisDetails.first?.activeContextIDs ?? [],
                appliedTermIDs: [],
                requiresReview: false
            )
        )

        transcription.hypotheses = candidates
        transcription.hypothesisLabels = labels
        transcription.hypothesisDetails = details
    }

    private static func normalizedCandidateList(_ candidates: [String]) -> [String] {
        var seen: Set<String> = []
        return candidates
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .filter { seen.insert(candidateKey($0)).inserted }
            .prefix(maxPersistedCandidateCount)
            .map { $0 }
    }

    private static func alignedLabels(_ labels: [String], count: Int) -> [String] {
        (0..<count).map { index in
            guard labels.indices.contains(index),
                  !labels[index].trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                return "Candidate"
            }
            return labels[index]
        }
    }

    private static func alignedHypotheses(
        _ hypotheses: [VocoHypothesis],
        candidates: [String],
        labels: [String],
        assessment: VocoConfidenceAssessment
    ) -> [VocoHypothesis] {
        candidates.enumerated().map { index, candidate in
            if hypotheses.indices.contains(index),
               isSameCandidate(hypotheses[index].text, candidate) {
                return hypotheses[index]
            }

            return fallbackHypothesis(
                text: candidate,
                label: labels.indices.contains(index) ? labels[index] : "Candidate",
                assessment: assessment
            )
        }
    }

    private static func hasCandidateFeedback(for acceptedText: String, in transcription: Transcription) -> Bool {
        transcription.correctionFeedback.contains { signal in
            signal.kind == .candidateSelection && signal.acceptedText == acceptedText
        }
    }

    private static func labelForCandidate(
        at index: Int,
        assessment: VocoConfidenceAssessment,
        hypothesis: VocoHypothesis?
    ) -> String {
        if assessment.candidateLabels.indices.contains(index) {
            return assessment.candidateLabels[index]
        }

        if let hypothesis {
            return hypothesis.label
        }

        return "Candidate"
    }

    private static func fallbackHypothesis(
        text: String,
        label: String,
        assessment: VocoConfidenceAssessment
    ) -> VocoHypothesis {
        VocoHypothesis(
            id: "review.\(label.lowercased().replacingOccurrences(of: " ", with: "-"))",
            text: text,
            label: label,
            source: .autoContext,
            confidenceScore: assessment.score,
            reasons: assessment.reasons,
            activeContextIDs: [],
            appliedTermIDs: [],
            requiresReview: true
        )
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

    private static func isSameCandidate(_ lhs: String, _ rhs: String) -> Bool {
        candidateKey(lhs) == candidateKey(rhs)
    }

    private static func candidateKey(_ candidate: String) -> String {
        candidate
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current)
    }
}
