import Foundation

final class VocoConfidenceGateService {
    static let shared = VocoConfidenceGateService()

    func assess(
        normalizationResult: VocoNormalizationResult,
        rawTranscript: String? = nil,
        correctionRiskProfile: VocoCorrectionRiskProfile? = nil
    ) -> VocoConfidenceAssessment {
        var score = 1.0
        var reasons: [String] = []
        let protectedTermGuarded = hasAutoApplyProtectedTermGuard(normalizationResult.suggestions)
        let selectedCandidate = selectedCandidate(
            normalizationResult: normalizationResult,
            rawTranscript: rawTranscript,
            protectedTermGuarded: protectedTermGuarded
        )

        let replacementCount = normalizationResult.replacements.count
        let suggestionCount = normalizationResult.suggestions.count
        let affectedTermIDs = termIDs(from: normalizationResult)

        if replacementCount > 0 {
            score -= min(Double(replacementCount) * 0.03, 0.12)
        }

        if suggestionCount > 0 {
            score -= min(Double(suggestionCount) * 0.18, 0.36)
            reasons.append("unresolved-suggestions")
        }

        if protectedTermGuarded {
            score -= 0.24
            reasons.append(VocoAutoApplyModelService.protectedTermGuardReason)
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

        if hasProtectedTermReplacement(normalizationResult.replacements) {
            score -= 0.22
            reasons.append("protected-term-replacement")
        }

        if let rawTranscript,
           !rawTranscript.isEmpty,
           rawTranscript != normalizationResult.originalText {
            let drift = rawCleanupDrift(rawTranscript: rawTranscript, cleanedText: normalizationResult.originalText)
            if drift.isSignificant {
                score -= min(0.20, 0.06 + drift.changeRatio * 0.5)
                reasons.append("raw-cleanup-significant")
            } else {
                score -= 0.02
                reasons.append("raw-cleanup-drift")
            }

            if drift.prefersRawFallback {
                score -= 0.16
                reasons.append("raw-cleanup-local-regression")
            }
        }

        if let correctionRiskProfile,
           correctionRiskProfile.hasElevatedCorrectionRate,
           !affectedTermIDs.isEmpty {
            score -= min(0.16, 0.04 + correctionRiskProfile.recentCorrectionRate * 0.16)
            reasons.append("recent-correction-rate")
        }

        if let correctionRiskProfile,
           correctionRiskProfile.hasHighRiskOverlap(with: affectedTermIDs) {
            score -= 0.14
            reasons.append("recent-term-corrections")
        }

        let boundedScore = max(0.0, min(1.0, score))
        if reasons.isEmpty {
            reasons.append("canonicalization-clean")
        }

        let reviewTriggers = reviewTriggers(
            score: boundedScore,
            normalizationResult: normalizationResult,
            reasons: reasons,
            affectedTermIDs: affectedTermIDs,
            correctionRiskProfile: correctionRiskProfile
        )
        let route: VocoConfidenceRoute = reviewTriggers.isEmpty ? .directInsertion : .reviewSuggested

        let hypothesisDetails = VocoHypothesisManagerService.buildHypotheses(
            normalizationResult: normalizationResult,
            rawTranscript: rawTranscript,
            confidenceScore: boundedScore,
            route: route,
            reasons: reasons,
            selectedCandidate: selectedCandidate
        )

        return VocoConfidenceAssessment(
            score: boundedScore,
            route: route,
            reasons: reasons,
            reviewTriggers: reviewTriggers,
            candidates: hypothesisDetails.map(\.text),
            candidateLabels: hypothesisDetails.map(\.label),
            hypothesisDetails: hypothesisDetails,
            correctionRiskProfile: correctionRiskProfile,
            selectedCandidate: selectedCandidate
        )
    }

    private func selectedCandidate(
        normalizationResult: VocoNormalizationResult,
        rawTranscript: String?,
        protectedTermGuarded: Bool
    ) -> String {
        guard protectedTermGuarded else {
            return normalizationResult.normalizedText
        }

        if let rawTranscript = rawTranscript?.trimmingCharacters(in: .whitespacesAndNewlines),
           !rawTranscript.isEmpty {
            return rawTranscript
        }

        return normalizationResult.originalText
    }

    private func reviewTriggers(
        score: Double,
        normalizationResult: VocoNormalizationResult,
        reasons: [String],
        affectedTermIDs: [String],
        correctionRiskProfile: VocoCorrectionRiskProfile?
    ) -> [VocoReviewTrigger] {
        var triggers: [VocoReviewTrigger] = []

        if score < 0.78 {
            triggers.append(
                VocoReviewTrigger(
                    id: "low-confidence-score",
                    reason: "low-confidence-score",
                    detail: "Score \(percent(score)) below \(percent(0.78))"
                )
            )
        }

        if !normalizationResult.suggestions.isEmpty {
            triggers.append(
                VocoReviewTrigger(
                    id: "unresolved-suggestions",
                    reason: "unresolved-suggestions",
                    detail: countDetail(normalizationResult.suggestions.count, singular: "suggestion")
                )
            )
        }

        if reasons.contains("heavy-normalization") {
            triggers.append(
                VocoReviewTrigger(
                    id: "heavy-normalization",
                    reason: "heavy-normalization",
                    detail: countDetail(normalizationResult.replacements.count, singular: "replacement")
                )
            )
        }

        if reasons.contains("low-confidence-replacement") {
            triggers.append(
                VocoReviewTrigger(
                    id: "low-confidence-replacement",
                    reason: "low-confidence-replacement",
                    detail: "Replacement confidence below 92%"
                )
            )
        }

        if reasons.contains("raw-cleanup-significant") {
            triggers.append(
                VocoReviewTrigger(
                    id: "raw-cleanup-significant",
                    reason: "raw-cleanup-significant",
                    detail: "Raw cleanup changed text"
                )
            )
        }

        if reasons.contains("raw-cleanup-local-regression") {
            triggers.append(
                VocoReviewTrigger(
                    id: "raw-cleanup-local-regression",
                    reason: "raw-cleanup-local-regression",
                    detail: "Raw cleanup changed a higher-confidence local phrase"
                )
            )
        }

        if reasons.contains("protected-term-replacement") {
            triggers.append(
                VocoReviewTrigger(
                    id: "protected-term-replacement",
                    reason: "protected-term-replacement",
                    detail: "Replacement changed a protected term"
                )
            )
        }

        if reasons.contains(VocoAutoApplyModelService.protectedTermGuardReason) {
            triggers.append(
                VocoReviewTrigger(
                    id: VocoAutoApplyModelService.protectedTermGuardReason,
                    reason: VocoAutoApplyModelService.protectedTermGuardReason,
                    detail: "Protected term outside allowlist"
                )
            )
        }

        if reasons.contains("recent-term-corrections") {
            let riskIDs = correctionRiskProfile?.highRiskTermIDs ?? []
            let overlappingIDs = affectedTermIDs.filter { riskIDs.contains($0) }
            triggers.append(
                VocoReviewTrigger(
                    id: "recent-term-corrections",
                    reason: "recent-term-corrections",
                    detail: overlappingIDs.isEmpty ? "Recent term corrections" : overlappingIDs.joined(separator: ", ")
                )
            )
        }

        if reasons.contains("recent-correction-rate"), !normalizationResult.replacements.isEmpty {
            triggers.append(
                VocoReviewTrigger(
                    id: "recent-correction-rate",
                    reason: "recent-correction-rate",
                    detail: correctionRiskProfile.map { "\(percent($0.recentCorrectionRate)) recent correction rate" }
                )
            )
        }

        return triggers
    }

    private func rawCleanupDrift(rawTranscript: String, cleanedText: String) -> RawCleanupDrift {
        let analysis = RetranscriptionAnalyticsService.analyze(
            sourceText: rawTranscript,
            retranscribedText: cleanedText,
            sourceConfidenceScore: nil,
            retranscribedConfidenceScore: nil
        )
        return RawCleanupDrift(
            changeRatio: analysis.changeRatio,
            isSignificant: analysis.changeCategory == .meaningfulChange,
            prefersRawFallback: rawCleanupPrefersRawFallback(
                rawTranscript: rawTranscript,
                cleanedText: cleanedText
            )
        )
    }

    private func rawCleanupPrefersRawFallback(rawTranscript: String, cleanedText: String) -> Bool {
        let raw = normalizedForCleanupRisk(rawTranscript)
        let cleaned = normalizedForCleanupRisk(cleanedText)
        guard raw != cleaned else { return false }

        let rawChars = Array(raw)
        let cleanedChars = Array(cleaned)

        var prefix = 0
        while prefix < rawChars.count,
              prefix < cleanedChars.count,
              rawChars[prefix] == cleanedChars[prefix] {
            prefix += 1
        }

        var rawEnd = rawChars.count
        var cleanedEnd = cleanedChars.count
        while rawEnd > prefix,
              cleanedEnd > prefix,
              rawChars[rawEnd - 1] == cleanedChars[cleanedEnd - 1] {
            rawEnd -= 1
            cleanedEnd -= 1
        }

        let rawMiddle = String(rawChars[prefix..<rawEnd])
        let cleanedMiddle = String(cleanedChars[prefix..<cleanedEnd])
        guard !rawMiddle.isEmpty,
              !cleanedMiddle.isEmpty,
              rawMiddle.count <= 6,
              cleanedMiddle.count <= 6,
              rawMiddle.contains(where: \.isCJK),
              cleanedMiddle.contains(where: \.isCJK)
        else {
            return false
        }

        return phraseLooksSafer(rawMiddle, than: cleanedMiddle)
    }

    private func phraseLooksSafer(_ rawPhrase: String, than cleanedPhrase: String) -> Bool {
        let rawFrequency = phraseFrequency(rawPhrase)
        let cleanedFrequency = phraseFrequency(cleanedPhrase)
        guard rawFrequency >= 100 else { return false }

        if cleanedFrequency == 0 {
            return true
        }

        return rawFrequency >= max(cleanedFrequency * 2, cleanedFrequency + 500)
    }

    private func phraseFrequency(_ phrase: String) -> Int {
        let converted = OpenCCConverter.shared.convert(phrase)
        guard PinyinDatabase.shared.isLoaded else { return 0 }

        let wordFrequency = PinyinDatabase.shared.frequency(of: converted)
        if wordFrequency > 0 { return wordFrequency }

        if converted.count == 2 {
            return PinyinDatabase.shared.bigramFrequency(of: converted)
        }

        return 0
    }

    private func normalizedForCleanupRisk(_ text: String) -> String {
        OpenCCConverter.shared.convert(text)
            .lowercased()
            .filter { $0.isLetter || $0.isNumber }
    }

    private func hasHighRiskAcceptedTerm(in replacements: [VocoReplacement]) -> Bool {
        replacements.contains { replacement in
            replacement.termID == "song.homura" ||
            (replacement.termID.hasPrefix("song.") && replacement.replacementText.count == 1)
        }
    }

    private func hasProtectedTermReplacement(_ replacements: [VocoReplacement]) -> Bool {
        replacements.contains { replacement in
            containsProtectedTerm(replacement.originalText)
        }
    }

    private func hasAutoApplyProtectedTermGuard(_ suggestions: [VocoReplacement]) -> Bool {
        suggestions.contains { suggestion in
            suggestion.reason == VocoAutoApplyModelService.protectedTermGuardReason
        }
    }

    private func containsProtectedTerm(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }

        let protection = CorrectionProtectionList.shared
        if protection.containsSubstring(in: trimmed) { return true }

        let converted = OpenCCConverter.shared.convert(trimmed)
        return converted != trimmed && protection.containsSubstring(in: converted)
    }

    private func isHeavyNormalization(_ result: VocoNormalizationResult) -> Bool {
        if result.replacements.count >= 4 { return true }

        let originalCount = max(result.originalText.count, 1)
        let delta = abs(result.normalizedText.count - result.originalText.count)
        return Double(delta) / Double(originalCount) > 0.35
    }

    private func termIDs(from result: VocoNormalizationResult) -> [String] {
        var seen: Set<String> = []
        return (result.replacements + result.suggestions)
            .map(\.termID)
            .filter { seen.insert($0).inserted }
    }

    private func countDetail(_ count: Int, singular: String) -> String {
        count == 1 ? "1 \(singular)" : "\(count) \(singular)s"
    }

    private func percent(_ value: Double) -> String {
        "\(Int((max(0, min(1, value)) * 100).rounded()))%"
    }

    private struct RawCleanupDrift {
        let changeRatio: Double
        let isSignificant: Bool
        let prefersRawFallback: Bool
    }
}
