import Foundation
import SwiftData

@MainActor
enum CorrectionFeedbackLearningService {
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
        }

        return []
    }

    private static func directSubstitution(from signal: CorrectionFeedbackSignal) -> WordSubstitution? {
        let source = signal.sourceText.trimmingCharacters(in: .whitespacesAndNewlines)
        let accepted = signal.acceptedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !source.isEmpty, !accepted.isEmpty, source != accepted else { return nil }
        return WordSubstitution(original: source, replacement: accepted)
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

        let hasLatinOrDigit = (original + replacement).contains { $0.isLetter && !$0.isCJK || $0.isNumber }
        return hasLatinOrDigit
    }

    private static func uniqueNonEmpty(_ values: [String?]) -> [String] {
        var seen: Set<String> = []
        return values
            .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .filter { seen.insert($0).inserted }
    }
}
