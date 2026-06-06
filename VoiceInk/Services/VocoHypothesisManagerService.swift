import Foundation

enum VocoHypothesisManagerService {
    static func buildHypotheses(
        normalizationResult: VocoNormalizationResult,
        rawTranscript: String?,
        confidenceScore: Double?,
        route: VocoConfidenceRoute,
        reasons: [String]
    ) -> [VocoHypothesis] {
        let normalized = normalizationResult.normalizedText
        let suggested = applying(
            normalizationResult.replacements + normalizationResult.suggestions,
            to: normalizationResult.originalText
        )

        var drafts: [(text: String, label: String, source: VocoHypothesisSource, termIDs: [String], requiresReview: Bool)] = [
            (
                normalized,
                "Recommended",
                .autoContext,
                normalizationResult.replacements.map(\.termID),
                route == .reviewSuggested
            ),
            (
                suggested,
                "With suggestions",
                .suggestedRepair,
                (normalizationResult.replacements + normalizationResult.suggestions).map(\.termID),
                true
            ),
            (
                normalizationResult.originalText,
                "Original",
                .originalCleaned,
                [],
                false
            ),
        ]

        if let rawTranscript, !rawTranscript.isEmpty {
            drafts.append((rawTranscript, "Raw ASR", .rawASR, [], false))
        }

        var seen: Set<String> = []
        return drafts
            .map { draft in
                (
                    text: draft.text.trimmingCharacters(in: .whitespacesAndNewlines),
                    label: draft.label,
                    source: draft.source,
                    termIDs: draft.termIDs,
                    requiresReview: draft.requiresReview
                )
            }
            .filter { !$0.text.isEmpty }
            .filter { seen.insert($0.text).inserted }
            .prefix(5)
            .map { draft in
                VocoHypothesis(
                    id: draft.source.rawValue,
                    text: draft.text,
                    label: draft.label,
                    source: draft.source,
                    confidenceScore: confidenceScore,
                    reasons: reasons,
                    activeContextIDs: normalizationResult.activeContextIDs,
                    appliedTermIDs: deduplicated(draft.termIDs),
                    requiresReview: draft.requiresReview
                )
            }
    }

    private static func applying(_ replacements: [VocoReplacement], to text: String) -> String {
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

    private static func deduplicated(_ values: [String]) -> [String] {
        var seen: Set<String> = []
        return values.filter { seen.insert($0).inserted }
    }
}
