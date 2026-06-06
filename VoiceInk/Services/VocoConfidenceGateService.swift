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

        let candidateOptions = candidateOptions(from: normalizationResult, rawTranscript: rawTranscript)

        return VocoConfidenceAssessment(
            score: boundedScore,
            route: route,
            reasons: reasons,
            candidates: candidateOptions.map(\.text),
            candidateLabels: candidateOptions.map(\.label),
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

    private func candidateOptions(
        from result: VocoNormalizationResult,
        rawTranscript: String?
    ) -> [(text: String, label: String)] {
        var candidates = [
            (result.normalizedText, "Recommended"),
            (applying(result.replacements + result.suggestions, to: result.originalText), "With suggestions"),
            (result.originalText, "Original"),
        ]

        if let rawTranscript, !rawTranscript.isEmpty {
            candidates.append((rawTranscript, "Raw ASR"))
        }

        var seen: Set<String> = []
        return candidates
            .map { option in
                (
                    text: option.0.trimmingCharacters(in: .whitespacesAndNewlines),
                    label: option.1
                )
            }
            .filter { !$0.text.isEmpty }
            .filter { seen.insert($0.text).inserted }
            .prefix(5)
            .map { $0 }
    }

    private func applying(_ replacements: [VocoReplacement], to text: String) -> String {
        guard !replacements.isEmpty else { return text }

        var result = text
        let sortedReplacements = replacements
            .compactMap { replacement -> (replacement: VocoReplacement, start: Int, length: Int)? in
                guard let start = replacement.rangeStart,
                      let length = replacement.rangeLength
                else {
                    return nil
                }
                return (replacement, start, length)
            }
            .sorted { $0.start > $1.start }

        var occupied: [Range<Int>] = []
        for item in sortedReplacements {
            let range = item.start..<(item.start + item.length)
            if occupied.contains(where: { $0.overlaps(range) }) {
                continue
            }
            guard let startIndex = result.index(result.startIndex, offsetBy: item.start, limitedBy: result.endIndex),
                  let endIndex = result.index(startIndex, offsetBy: item.length, limitedBy: result.endIndex)
            else {
                continue
            }

            result.replaceSubrange(startIndex..<endIndex, with: item.replacement.replacementText)
            occupied.append(range)
        }

        return result
    }
}
