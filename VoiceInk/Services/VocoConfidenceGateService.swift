import Foundation

final class VocoConfidenceGateService {
    static let shared = VocoConfidenceGateService()

    func assess(
        normalizationResult: VocoNormalizationResult,
        rawTranscript: String? = nil
    ) -> VocoConfidenceAssessment {
        var score = 1.0
        var reasons: [String] = []

        let replacementCount = normalizationResult.replacements.count
        let suggestionCount = normalizationResult.suggestions.count

        if replacementCount > 0 {
            score -= min(Double(replacementCount) * 0.03, 0.12)
        }

        if suggestionCount > 0 {
            score -= min(Double(suggestionCount) * 0.18, 0.36)
            reasons.append("unresolved-suggestions")
        }

        if normalizationResult.replacements.contains(where: { $0.confidence < 0.92 }) {
            score -= 0.12
            reasons.append("low-confidence-replacement")
        }

        if hasHighRiskAcceptedTerm(in: normalizationResult.replacements) {
            score -= 0.08
            reasons.append("high-risk-term")
        }

        if isHeavyNormalization(normalizationResult) {
            score -= 0.15
            reasons.append("heavy-normalization")
        }

        if let rawTranscript,
           !rawTranscript.isEmpty,
           rawTranscript != normalizationResult.originalText {
            score -= 0.02
            reasons.append("raw-cleanup-drift")
        }

        let boundedScore = max(0.0, min(1.0, score))
        if reasons.isEmpty {
            reasons.append("canonicalization-clean")
        }

        let route: VocoConfidenceRoute = shouldSuggestReview(
            score: boundedScore,
            normalizationResult: normalizationResult,
            reasons: reasons
        ) ? .reviewSuggested : .directInsertion

        let hypothesisDetails = VocoHypothesisManagerService.buildHypotheses(
            normalizationResult: normalizationResult,
            rawTranscript: rawTranscript,
            confidenceScore: boundedScore,
            route: route,
            reasons: reasons
        )

        return VocoConfidenceAssessment(
            score: boundedScore,
            route: route,
            reasons: reasons,
            candidates: hypothesisDetails.map(\.text),
            candidateLabels: hypothesisDetails.map(\.label),
            hypothesisDetails: hypothesisDetails,
            selectedCandidate: normalizationResult.normalizedText
        )
    }

    private func shouldSuggestReview(
        score: Double,
        normalizationResult: VocoNormalizationResult,
        reasons: [String]
    ) -> Bool {
        if score < 0.78 { return true }
        if !normalizationResult.suggestions.isEmpty { return true }
        if reasons.contains("heavy-normalization") { return true }
        if reasons.contains("low-confidence-replacement") { return true }
        return false
    }

    private func hasHighRiskAcceptedTerm(in replacements: [VocoReplacement]) -> Bool {
        replacements.contains { replacement in
            replacement.termID == "song.homura" ||
            (replacement.termID.hasPrefix("song.") && replacement.replacementText.count == 1)
        }
    }

    private func isHeavyNormalization(_ result: VocoNormalizationResult) -> Bool {
        if result.replacements.count >= 4 { return true }

        let originalCount = max(result.originalText.count, 1)
        let delta = abs(result.normalizedText.count - result.originalText.count)
        return Double(delta) / Double(originalCount) > 0.35
    }
}
