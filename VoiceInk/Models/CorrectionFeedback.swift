import Foundation

enum CorrectionFeedbackKind: String, Codable, Equatable {
    case candidateSelection
    case retranscriptionChange
    case userSubstitution

    var displayName: String {
        switch self {
        case .candidateSelection:
            return "Candidate selection"
        case .retranscriptionChange:
            return "Retranscription change"
        case .userSubstitution:
            return "User substitution"
        }
    }
}

struct CorrectionFeedbackSignal: Codable, Equatable {
    let kind: CorrectionFeedbackKind
    let sourceText: String
    let proposedText: String?
    let acceptedText: String
    let confidenceScore: Double?
    let changeRatio: Double?
    let reason: String
    let termIDs: [String]
    let createdAt: Date

    init(
        kind: CorrectionFeedbackKind,
        sourceText: String,
        proposedText: String? = nil,
        acceptedText: String,
        confidenceScore: Double? = nil,
        changeRatio: Double? = nil,
        reason: String,
        termIDs: [String] = [],
        createdAt: Date = Date()
    ) {
        self.kind = kind
        self.sourceText = sourceText
        self.proposedText = proposedText
        self.acceptedText = acceptedText
        self.confidenceScore = confidenceScore
        self.changeRatio = changeRatio
        self.reason = reason
        self.termIDs = termIDs
        self.createdAt = createdAt
    }
}

enum CorrectionFeedbackService {
    static func candidateSelectionSignal(
        normalizationResult: VocoNormalizationResult,
        assessment: VocoConfidenceAssessment,
        selectedCandidate: String,
        rawTranscript: String? = nil
    ) -> CorrectionFeedbackSignal? {
        let accepted = selectedCandidate.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !accepted.isEmpty else { return nil }

        let source = firstNonEmpty(rawTranscript, normalizationResult.originalText)
        let proposed = normalizationResult.normalizedText
        let comparisonBase = accepted == proposed ? source : proposed
        let analysis = RetranscriptionAnalyticsService.analyze(
            sourceText: comparisonBase,
            retranscribedText: accepted,
            sourceConfidenceScore: nil,
            retranscribedConfidenceScore: nil
        )

        let reason: String
        if accepted == assessment.selectedCandidate {
            reason = "candidate-confirmed"
        } else if assessment.candidates.contains(accepted) {
            reason = "candidate-override"
        } else {
            reason = "candidate-custom"
        }

        return CorrectionFeedbackSignal(
            kind: .candidateSelection,
            sourceText: source,
            proposedText: proposed,
            acceptedText: accepted,
            confidenceScore: assessment.score,
            changeRatio: analysis.changeRatio,
            reason: reason,
            termIDs: termIDs(from: normalizationResult)
        )
    }

    static func retranscriptionSignal(
        sourceText: String,
        retranscribedText: String,
        analysis: RetranscriptionAnalysis,
        confidenceScore: Double? = nil
    ) -> CorrectionFeedbackSignal? {
        let source = sourceText.trimmingCharacters(in: .whitespacesAndNewlines)
        let accepted = retranscribedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !source.isEmpty, !accepted.isEmpty else { return nil }
        guard analysis.changeCategory != .unchanged else { return nil }

        return CorrectionFeedbackSignal(
            kind: .retranscriptionChange,
            sourceText: source,
            acceptedText: accepted,
            confidenceScore: confidenceScore,
            changeRatio: analysis.changeRatio,
            reason: "retranscription-\(analysis.changeCategory.rawValue)"
        )
    }

    static func userSubstitutionSignal(
        _ substitution: WordSubstitution,
        confidenceScore: Double? = nil,
        reason: String = "user-substitution"
    ) -> CorrectionFeedbackSignal? {
        let source = substitution.original.trimmingCharacters(in: .whitespacesAndNewlines)
        let accepted = substitution.replacement.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !source.isEmpty, !accepted.isEmpty, source != accepted else { return nil }

        let analysis = RetranscriptionAnalyticsService.analyze(
            sourceText: source,
            retranscribedText: accepted,
            sourceConfidenceScore: nil,
            retranscribedConfidenceScore: nil
        )

        return CorrectionFeedbackSignal(
            kind: .userSubstitution,
            sourceText: source,
            acceptedText: accepted,
            confidenceScore: confidenceScore,
            changeRatio: analysis.changeRatio,
            reason: reason
        )
    }

    private static func termIDs(from result: VocoNormalizationResult) -> [String] {
        var seen: Set<String> = []
        return (result.replacements + result.suggestions)
            .map(\.termID)
            .filter { seen.insert($0).inserted }
    }

    private static func firstNonEmpty(_ values: String?...) -> String {
        values
            .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
            .first { !$0.isEmpty } ?? ""
    }
}
