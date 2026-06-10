import Foundation
import SwiftData

@MainActor
enum CorrectionFeedbackLearningService {
    private static let maximumBroadLearningChangeRatio = 0.45
    private static let broadLearningCharacterCount = 6
    private static let shortRetranscriptionUnitLimit = 10

    @discardableResult
    static func stageLearningCandidates(
        from signal: CorrectionFeedbackSignal?,
        in modelContext: ModelContext
    ) -> [WordSubstitution] {
        guard let signal else { return [] }

        let substitutions = learningSubstitutions(from: signal)
        for substitution in substitutions {
            AutoCorrectionStagingService.shared.stageCorrection(
                substitution,
                in: modelContext,
                source: WordReplacement.sourceCorrectionFeedback
            )
        }
        return substitutions
    }

    static func learningSubstitutions(from signal: CorrectionFeedbackSignal) -> [WordSubstitution] {
        guard isEligibleForAutoLearning(signal) else { return [] }

        switch signal.kind {
        case .candidateSelection:
            guard signal.isCorrectiveSignal else { return [] }
            return extractedSubstitutions(from: signal)
        case .retranscriptionChange:
            return extractedSubstitutions(from: signal)
        case .userSubstitution:
            return directSubstitution(from: signal).map { [$0] } ?? []
        }
    }

    private static func extractedSubstitutions(from signal: CorrectionFeedbackSignal) -> [WordSubstitution] {
        let accepted = signal.acceptedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !accepted.isEmpty else { return [] }
        guard !isBroadLearningRewrite(signal) else { return [] }

        let sources = uniqueNonEmpty([signal.sourceText, signal.proposedText])
        for source in sources where source != accepted {
            if let substitution = AutoCorrectionStagingService.shared.extractSubstitution(
                original: source,
                edited: accepted
            ) {
                return [substitution]
            }

            if let substitution = tokenSpanSubstitution(original: source, edited: accepted) {
                return [substitution]
            }

            if let substitution = characterSpanSubstitution(original: source, edited: accepted) {
                return [substitution]
            }
        }

        return []
    }

    private static func isEligibleForAutoLearning(_ signal: CorrectionFeedbackSignal) -> Bool {
        guard signal.isCorrectiveSignal else { return false }
        guard !hasUntrustedLearningReason(signal.reason) else { return false }

        let classification = CorrectionEvidenceClassifier.classify(signal: signal)
        switch classification.evidenceTier {
        case .t0Untrusted, .negativeEvidence, .none:
            return false
        case .t1WeakInteraction, .t2ConfirmedSpan, .t3ConfirmedRepeated, .t4Gold:
            break
        }

        guard !hasBlockedLearningNoise(classification.noiseFlags) else { return false }

        if signal.kind == .retranscriptionChange,
           hasShortNonTechnicalRetranscriptionRisk(signal, classification: classification) {
            return false
        }

        return true
    }

    private static func hasUntrustedLearningReason(_ reason: String) -> Bool {
        let normalized = reason.lowercased()
        return normalized.contains("llm")
            || normalized.contains("enhancement")
            || normalized.contains("ztext-enhanced")
            || normalized.contains("automatic-correction")
            || normalized.contains("candidate-not-selected")
    }

    private static func hasBlockedLearningNoise(_ flags: [CorrectionEvidenceNoiseFlag]) -> Bool {
        flags.contains(.llmOnly)
            || flags.contains(.stalePendingTranscriptSuspected)
            || flags.contains(.targetLengthExpansionRatioHigh)
            || flags.contains(.fullSentenceRewriteSuspected)
            || flags.contains(.crossLanguageReconstruction)
    }

    private static func hasShortNonTechnicalRetranscriptionRisk(
        _ signal: CorrectionFeedbackSignal,
        classification: CorrectionEvidenceClassification
    ) -> Bool {
        guard let comparison = classification.phoneticComparison else { return false }
        guard !comparison.raw.isTechnicalTermCandidate,
              !comparison.target.isTechnicalTermCandidate
        else {
            return false
        }

        let rawUnits = comparison.raw.unitCount
        let targetUnits = comparison.target.unitCount
        return comparison.raw.lengthBucket == .oneToFour
            || comparison.target.lengthBucket == .oneToFour
            || min(rawUnits, targetUnits) <= shortRetranscriptionUnitLimit
    }

    private static func directSubstitution(from signal: CorrectionFeedbackSignal) -> WordSubstitution? {
        let source = signal.sourceText.trimmingCharacters(in: .whitespacesAndNewlines)
        let accepted = signal.acceptedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !source.isEmpty, !accepted.isEmpty, source != accepted else { return nil }
        return WordSubstitution(original: source, replacement: accepted)
    }

    private static func isBroadLearningRewrite(_ signal: CorrectionFeedbackSignal) -> Bool {
        guard let changeRatio = signal.changeRatio,
              changeRatio > maximumBroadLearningChangeRatio
        else {
            return false
        }

        let source = signal.sourceText.trimmingCharacters(in: .whitespacesAndNewlines)
        let accepted = signal.acceptedText.trimmingCharacters(in: .whitespacesAndNewlines)
        return max(source.count, accepted.count) >= broadLearningCharacterCount
    }

    private static func tokenSpanSubstitution(original: String, edited: String) -> WordSubstitution? {
        let originalTokens = whitespaceTokens(original)
        let editedTokens = whitespaceTokens(edited)
        guard !originalTokens.isEmpty, !editedTokens.isEmpty else { return nil }

        var prefix = 0
        while prefix < originalTokens.count,
              prefix < editedTokens.count,
              tokensMatch(originalTokens[prefix], editedTokens[prefix]) {
            prefix += 1
        }

        var originalSuffix = originalTokens.count
        var editedSuffix = editedTokens.count
        while originalSuffix > prefix,
              editedSuffix > prefix,
              tokensMatch(originalTokens[originalSuffix - 1], editedTokens[editedSuffix - 1]) {
            originalSuffix -= 1
            editedSuffix -= 1
        }

        let originalSegment = originalTokens[prefix..<originalSuffix].joined(separator: " ")
        let editedSegment = editedTokens[prefix..<editedSuffix].joined(separator: " ")
        guard isSafeReplacementSegment(originalSegment, editedSegment) else { return nil }

        return WordSubstitution(original: originalSegment, replacement: editedSegment)
    }

    private static func characterSpanSubstitution(original: String, edited: String) -> WordSubstitution? {
        let originalChars = Array(original)
        let editedChars = Array(edited)
        guard !originalChars.isEmpty, !editedChars.isEmpty else { return nil }

        var prefix = 0
        while prefix < originalChars.count,
              prefix < editedChars.count,
              originalChars[prefix] == editedChars[prefix] {
            prefix += 1
        }

        var originalSuffix = originalChars.count
        var editedSuffix = editedChars.count
        while originalSuffix > prefix,
              editedSuffix > prefix,
              originalChars[originalSuffix - 1] == editedChars[editedSuffix - 1] {
            originalSuffix -= 1
            editedSuffix -= 1
        }

        let rawSegments = anchoredCharacterSegments(
            originalChars: originalChars,
            editedChars: editedChars,
            prefix: prefix,
            originalSuffix: originalSuffix,
            editedSuffix: editedSuffix
        )
        let originalSegment = cleanedCharacterSegment(rawSegments.original)
        let editedSegment = cleanedCharacterSegment(rawSegments.edited)
        let sharedContextCount = rawSegments.sharedContextCount
        guard isSafeCharacterReplacementSegment(
            originalSegment,
            editedSegment,
            sharedContextCount: sharedContextCount
        ) else {
            return nil
        }

        return WordSubstitution(original: originalSegment, replacement: editedSegment)
    }

    private static func anchoredCharacterSegments(
        originalChars: [Character],
        editedChars: [Character],
        prefix: Int,
        originalSuffix: Int,
        editedSuffix: Int
    ) -> (original: String, edited: String, sharedContextCount: Int) {
        let originalSegment = String(originalChars[prefix..<originalSuffix])
        let editedSegment = String(editedChars[prefix..<editedSuffix])
        let sharedContextCount = prefix + (originalChars.count - originalSuffix)

        if originalSegment.isEmpty,
           prefix < originalChars.count,
           editedSuffix < editedChars.count,
           originalChars[prefix] == editedChars[editedSuffix] {
            return (
                String(originalChars[prefix...prefix]),
                String(editedChars[prefix...editedSuffix]),
                max(0, sharedContextCount - 1)
            )
        }

        if editedSegment.isEmpty,
           prefix < editedChars.count,
           originalSuffix < originalChars.count,
           originalChars[originalSuffix] == editedChars[prefix] {
            return (
                String(originalChars[prefix...originalSuffix]),
                String(editedChars[prefix...prefix]),
                max(0, sharedContextCount - 1)
            )
        }

        return (originalSegment, editedSegment, sharedContextCount)
    }

    private static func whitespaceTokens(_ text: String) -> [String] {
        text.components(separatedBy: .whitespacesAndNewlines)
            .map { $0.trimmingCharacters(in: .punctuationCharacters) }
            .filter { !$0.isEmpty }
    }

    private static func tokensMatch(_ lhs: String, _ rhs: String) -> Bool {
        lhs.localizedCaseInsensitiveCompare(rhs) == .orderedSame
    }

    private static func isSafeReplacementSegment(_ original: String, _ replacement: String) -> Bool {
        guard !original.isEmpty, !replacement.isEmpty else { return false }
        guard original.localizedCaseInsensitiveCompare(replacement) != .orderedSame else { return false }
        guard original.count <= 40, replacement.count <= 40 else { return false }

        let originalTokenCount = whitespaceTokens(original).count
        let replacementTokenCount = whitespaceTokens(replacement).count
        guard originalTokenCount <= 4, replacementTokenCount <= 4 else { return false }

        let hasLatinOrDigit = (original + replacement).contains {
            ($0.isLetter && !$0.isCJK && !$0.isKana) || $0.isNumber
        }
        return hasLatinOrDigit
    }

    private static func cleanedCharacterSegment(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters))
    }

    private static func isSafeCharacterReplacementSegment(
        _ original: String,
        _ replacement: String,
        sharedContextCount: Int
    ) -> Bool {
        guard !original.isEmpty, !replacement.isEmpty else { return false }
        guard original.localizedCaseInsensitiveCompare(replacement) != .orderedSame else { return false }
        guard original.count <= 18, replacement.count <= 18 else { return false }

        let combined = original + replacement
        let hasKana = combined.contains(where: \.isKana)
        let hasLatinOrDigit = combined.contains {
            ($0.isLetter && !$0.isCJK && !$0.isKana) || $0.isNumber
        }
        let hasCJK = combined.contains(where: \.isCJK)
        guard hasCJK || hasKana || hasLatinOrDigit else { return false }
        guard sharedContextCount >= 2 else { return false }

        if original.count == 1, let char = original.first, correctionSkipChars.contains(char) {
            return false
        }

        let cleanedOriginal = original.filter { !$0.isPunctuation && !$0.isWhitespace }
        let cleanedReplacement = replacement.filter { !$0.isPunctuation && !$0.isWhitespace }
        return !cleanedOriginal.isEmpty && !cleanedReplacement.isEmpty && cleanedOriginal != cleanedReplacement
    }

    private static func uniqueNonEmpty(_ values: [String?]) -> [String] {
        var seen: Set<String> = []
        return values
            .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .filter { seen.insert($0).inserted }
    }
}

private extension Character {
    var isKana: Bool {
        unicodeScalars.contains { scalar in
            (0x3040...0x309F).contains(scalar.value) ||
                (0x30A0...0x30FF).contains(scalar.value)
        }
    }
}
