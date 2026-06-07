import Foundation

enum VocoHypothesisManagerService {
    static func buildHypotheses(
        normalizationResult: VocoNormalizationResult,
        rawTranscript: String?,
        rawCleanupRescueCandidate: String? = nil,
        confidenceScore: Double?,
        route: VocoConfidenceRoute,
        reasons: [String]
    ) -> [VocoHypothesis] {
        let normalized = normalizationResult.normalizedText
        let suggested = applying(
            normalizationResult.replacements + normalizationResult.suggestions,
            to: normalizationResult.originalText
        )
        let segmentRescue = segmentRescueCandidate(
            normalizationResult: normalizationResult,
            rawTranscript: rawTranscript,
            route: route,
            reasons: reasons
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
        ]

        if let segmentRescue {
            drafts.append((
                segmentRescue.text,
                "Segment rescue",
                .segmentRescue,
                segmentRescue.termIDs,
                true
            ))
        }

        if let rawCleanupRescueCandidate = rawCleanupRescueCandidate?.trimmingCharacters(in: .whitespacesAndNewlines),
           !rawCleanupRescueCandidate.isEmpty,
           rawCleanupRescueCandidate != normalized {
            drafts.append((
                rawCleanupRescueCandidate,
                "Raw cleanup rescue",
                .customRescue,
                [],
                true
            ))
        }

        drafts.append((
            normalizationResult.originalText,
            "Original",
            .originalCleaned,
            [],
            false
        ))

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
                    divergenceFromRecommended: divergenceFromRecommended(
                        for: draft.text,
                        recommended: normalized
                    ),
                    reasons: draft.source == .segmentRescue ? ["segment-rescue"] + reasons : reasons,
                    activeContextIDs: normalizationResult.activeContextIDs,
                    appliedTermIDs: deduplicated(draft.termIDs),
                    requiresReview: draft.requiresReview
                )
            }
    }

    private static func segmentRescueCandidate(
        normalizationResult: VocoNormalizationResult,
        rawTranscript: String?,
        route: VocoConfidenceRoute,
        reasons: [String]
    ) -> (text: String, termIDs: [String])? {
        guard route == .reviewSuggested,
              reasons.contains("raw-cleanup-significant"),
              let rawTranscript = rawTranscript?.trimmingCharacters(in: .whitespacesAndNewlines),
              !rawTranscript.isEmpty,
              rawTranscript != normalizationResult.originalText
        else {
            return nil
        }

        let replacements = normalizationResult.replacements
            .filter { $0.confidence >= 0.95 }
            .filter { !$0.originalText.isEmpty && $0.originalText != $0.replacementText }
        guard !replacements.isEmpty else { return nil }

        var result = rawTranscript
        var appliedTermIDs: [String] = []
        for replacement in replacements {
            guard let range = result.range(
                of: replacement.originalText,
                options: [.caseInsensitive, .diacriticInsensitive]
            ) else {
                continue
            }
            result.replaceSubrange(range, with: replacement.replacementText)
            appliedTermIDs.append(replacement.termID)
        }

        let trimmed = result.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !appliedTermIDs.isEmpty,
              !trimmed.isEmpty,
              trimmed != rawTranscript,
              trimmed != normalizationResult.normalizedText,
              trimmed != normalizationResult.originalText
        else {
            return nil
        }

        return (trimmed, deduplicated(appliedTermIDs))
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

    private static func divergenceFromRecommended(
        for candidate: String,
        recommended: String
    ) -> Double? {
        let candidate = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        let recommended = recommended.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !candidate.isEmpty,
              !recommended.isEmpty,
              candidate.localizedCaseInsensitiveCompare(recommended) != .orderedSame
        else {
            return nil
        }

        return RetranscriptionAnalyticsService.analyze(
            sourceText: recommended,
            retranscribedText: candidate,
            sourceConfidenceScore: nil,
            retranscribedConfidenceScore: nil
        ).changeRatio
    }

    private static func deduplicated(_ values: [String]) -> [String] {
        var seen: Set<String> = []
        return values.filter { seen.insert($0).inserted }
    }
}
