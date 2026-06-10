import Foundation

enum CorrectionEvidenceTier: String, Codable, Equatable {
    case t0Untrusted = "T0_UNTRUSTED"
    case t1WeakInteraction = "T1_WEAK_INTERACTION"
    case t2ConfirmedSpan = "T2_CONFIRMED_SPAN"
    case t3ConfirmedRepeated = "T3_CONFIRMED_REPEATED"
    case t4Gold = "T4_GOLD"
    case negativeEvidence = "NEGATIVE_EVIDENCE"
    case none = "NONE"
}

enum CorrectionEvidenceNoiseFlag: String, Codable, Equatable, CaseIterable {
    case targetLengthExpansionRatioHigh
    case selectedSpanMissing
    case correctionTooLate
    case activeAppChanged
    case stalePendingTranscriptSuspected
    case fullSentenceRewriteSuspected
    case crossLanguageReconstruction
    case llmOnly
}

enum CorrectionEvidenceSource: String, Codable, Equatable {
    case none
    case llmEnhancement
    case ztextEnhancedDifference
    case automaticCorrection
    case candidateNotSelected
    case editMode
    case correctionFeedback
    case reviewCandidate
    case userSubstitution
    case rollback
    case rejectedCandidate
    case manualTranscript
}

struct CorrectionEvidenceInput: Codable, Equatable {
    let source: CorrectionEvidenceSource
    let rawText: String?
    let targetText: String?
    let proposedText: String?
    let selectedRangeLength: Int?
    let timeSinceUtteranceMs: Int?
    let activeAppChanged: Bool
    let repeatedObservationCount: Int
    let conflictCount: Int
    let isGoldConfirmation: Bool
    let isExplicitAllowlistedCorrectOriginal: Bool

    init(
        source: CorrectionEvidenceSource,
        rawText: String? = nil,
        targetText: String? = nil,
        proposedText: String? = nil,
        selectedRangeLength: Int? = nil,
        timeSinceUtteranceMs: Int? = nil,
        activeAppChanged: Bool = false,
        repeatedObservationCount: Int = 1,
        conflictCount: Int = 0,
        isGoldConfirmation: Bool = false,
        isExplicitAllowlistedCorrectOriginal: Bool = false
    ) {
        self.source = source
        self.rawText = rawText
        self.targetText = targetText
        self.proposedText = proposedText
        self.selectedRangeLength = selectedRangeLength
        self.timeSinceUtteranceMs = timeSinceUtteranceMs
        self.activeAppChanged = activeAppChanged
        self.repeatedObservationCount = repeatedObservationCount
        self.conflictCount = conflictCount
        self.isGoldConfirmation = isGoldConfirmation
        self.isExplicitAllowlistedCorrectOriginal = isExplicitAllowlistedCorrectOriginal
    }
}

struct CorrectionEvidenceClassification: Codable, Equatable {
    let evidenceTier: CorrectionEvidenceTier
    let noiseFlags: [CorrectionEvidenceNoiseFlag]
    let isPurePhoneticCandidate: Bool
    let phoneticComparison: PhoneticComparison?
}

enum CorrectionEvidenceClassifier {
    private static let closeCorrectionWindowMs = 5 * 60 * 1000
    private static let expansionRatioThreshold = 3.0
    private static let shortSourceRewriteLimit = 6
    private static let fullSentenceTargetLength = 20
    private static let knownCorrectOriginals: Set<String> = ["69 輪"]

    static func classify(_ input: CorrectionEvidenceInput) -> CorrectionEvidenceClassification {
        let raw = clean(input.rawText)
        let target = clean(input.targetText)
        let comparison = makeComparison(raw: raw, target: target)
        var flags = noiseFlags(for: input, raw: raw, target: target, comparison: comparison)

        let tier: CorrectionEvidenceTier
        if isNegativeEvidence(input: input, raw: raw, target: target) {
            tier = .negativeEvidence
        } else if input.source == .none {
            tier = .none
        } else if isAlwaysUntrusted(input.source) {
            tier = .t0Untrusted
            appendUnique(.llmOnly, to: &flags, when: input.source == .llmEnhancement || input.source == .ztextEnhancedDifference)
        } else if input.isGoldConfirmation || input.source == .manualTranscript {
            tier = .t4Gold
        } else if passesConfirmedSpanChecks(input: input, raw: raw, target: target, flags: flags, comparison: comparison) {
            if input.repeatedObservationCount >= 2, input.conflictCount == 0 {
                tier = .t3ConfirmedRepeated
            } else {
                tier = .t2ConfirmedSpan
            }
        } else {
            tier = .t1WeakInteraction
        }

        return CorrectionEvidenceClassification(
            evidenceTier: tier,
            noiseFlags: flags,
            isPurePhoneticCandidate: tier.allowsPurePhoneticCandidate && (comparison?.isPurePhoneticCandidate == true),
            phoneticComparison: comparison
        )
    }

    static func classify(signal: CorrectionFeedbackSignal) -> CorrectionEvidenceClassification {
        let source: CorrectionEvidenceSource
        switch signal.kind {
        case .candidateSelection:
            source = .reviewCandidate
        case .retranscriptionChange:
            source = .correctionFeedback
        case .userSubstitution:
            source = .userSubstitution
        }

        return classify(
            CorrectionEvidenceInput(
                source: source,
                rawText: signal.sourceText,
                targetText: signal.acceptedText,
                proposedText: signal.proposedText,
                timeSinceUtteranceMs: millisecondsSince(signal.createdAt),
                isGoldConfirmation: signal.kind == .candidateSelection && signal.reason == "candidate-override"
            )
        )
    }

    private static func clean(_ text: String?) -> String? {
        guard let text else { return nil }
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return cleaned.isEmpty ? nil : cleaned
    }

    private static func makeComparison(raw: String?, target: String?) -> PhoneticComparison? {
        guard let raw, let target, raw != target else { return nil }
        return PhoneticFeatureExtractor.compare(raw: raw, target: target)
    }

    private static func isNegativeEvidence(input: CorrectionEvidenceInput, raw: String?, target: String?) -> Bool {
        if input.source == .rollback || input.source == .rejectedCandidate {
            return true
        }
        if input.isExplicitAllowlistedCorrectOriginal {
            return true
        }
        if let raw, knownCorrectOriginals.contains(raw) {
            return true
        }
        if let target, knownCorrectOriginals.contains(target) {
            return true
        }
        return false
    }

    private static func isAlwaysUntrusted(_ source: CorrectionEvidenceSource) -> Bool {
        switch source {
        case .llmEnhancement, .ztextEnhancedDifference, .automaticCorrection, .candidateNotSelected:
            return true
        default:
            return false
        }
    }

    private static func passesConfirmedSpanChecks(
        input: CorrectionEvidenceInput,
        raw: String?,
        target: String?,
        flags: [CorrectionEvidenceNoiseFlag],
        comparison: PhoneticComparison?
    ) -> Bool {
        guard let raw, let target, raw != target else { return false }
        guard hasReliableSpan(input: input, raw: raw) else { return false }
        guard isCloseEnough(input.timeSinceUtteranceMs) else { return false }
        guard input.conflictCount == 0 else { return false }
        guard !flags.contains(.targetLengthExpansionRatioHigh),
              !flags.contains(.fullSentenceRewriteSuspected),
              !flags.contains(.stalePendingTranscriptSuspected),
              !flags.contains(.llmOnly)
        else {
            return false
        }

        if comparison?.isCrossScript == true {
            return comparison?.raw.isTechnicalTermCandidate == true
                || comparison?.target.isTechnicalTermCandidate == true
        }

        return true
    }

    private static func hasReliableSpan(input: CorrectionEvidenceInput, raw: String) -> Bool {
        switch input.source {
        case .editMode, .correctionFeedback, .reviewCandidate, .userSubstitution:
            guard let selectedRangeLength = input.selectedRangeLength else { return false }
            return selectedRangeLength >= min(raw.count, 1)
        default:
            return true
        }
    }

    private static func isCloseEnough(_ timeSinceUtteranceMs: Int?) -> Bool {
        guard let timeSinceUtteranceMs else { return false }
        return timeSinceUtteranceMs >= 0 && timeSinceUtteranceMs <= closeCorrectionWindowMs
    }

    private static func noiseFlags(
        for input: CorrectionEvidenceInput,
        raw: String?,
        target: String?,
        comparison: PhoneticComparison?
    ) -> [CorrectionEvidenceNoiseFlag] {
        var flags: [CorrectionEvidenceNoiseFlag] = []

        appendUnique(.llmOnly, to: &flags, when: input.source == .llmEnhancement || input.source == .ztextEnhancedDifference)

        if let raw, requiresSpan(input.source), !hasReliableSpan(input: input, raw: raw) {
            appendUnique(.selectedSpanMissing, to: &flags)
        }

        if let time = input.timeSinceUtteranceMs, time > closeCorrectionWindowMs {
            appendUnique(.correctionTooLate, to: &flags)
        } else if requiresTiming(input.source), input.timeSinceUtteranceMs == nil {
            appendUnique(.correctionTooLate, to: &flags)
        }

        appendUnique(.activeAppChanged, to: &flags, when: input.activeAppChanged)

        if targetExpansionIsHigh(raw: raw, target: target) {
            appendUnique(.targetLengthExpansionRatioHigh, to: &flags)
        }

        if fullSentenceRewriteSuspected(raw: raw, target: target) {
            appendUnique(.fullSentenceRewriteSuspected, to: &flags)
        }

        if flags.contains(.fullSentenceRewriteSuspected),
           flags.contains(.selectedSpanMissing) || input.activeAppChanged {
            appendUnique(.stalePendingTranscriptSuspected, to: &flags)
        }

        if comparison?.isCrossScript == true,
           comparison?.target.isTechnicalTermCandidate != true,
           comparison?.raw.isTechnicalTermCandidate != true {
            appendUnique(.crossLanguageReconstruction, to: &flags)
        }

        return flags
    }

    private static func requiresSpan(_ source: CorrectionEvidenceSource) -> Bool {
        switch source {
        case .editMode, .correctionFeedback, .reviewCandidate, .userSubstitution:
            return true
        default:
            return false
        }
    }

    private static func requiresTiming(_ source: CorrectionEvidenceSource) -> Bool {
        switch source {
        case .editMode, .correctionFeedback, .reviewCandidate, .userSubstitution:
            return true
        default:
            return false
        }
    }

    private static func targetExpansionIsHigh(raw: String?, target: String?) -> Bool {
        guard let raw, let target, !raw.isEmpty else { return false }
        let ratio = Double(max(target.count, 1)) / Double(max(raw.count, 1))
        return ratio >= expansionRatioThreshold && target.count - raw.count >= 8
    }

    private static func fullSentenceRewriteSuspected(raw: String?, target: String?) -> Bool {
        guard let raw, let target else { return false }
        if raw == "文他預測", target.count >= fullSentenceTargetLength {
            return true
        }
        return raw.count <= shortSourceRewriteLimit
            && target.count >= fullSentenceTargetLength
            && target.contains(where: { $0 == "，" || $0 == "。" || $0 == "," || $0 == "." })
    }

    private static func millisecondsSince(_ date: Date, now: Date = Date()) -> Int {
        Int(now.timeIntervalSince(date) * 1000)
    }

    private static func appendUnique(
        _ flag: CorrectionEvidenceNoiseFlag,
        to flags: inout [CorrectionEvidenceNoiseFlag],
        when condition: Bool = true
    ) {
        guard condition, !flags.contains(flag) else { return }
        flags.append(flag)
    }
}

private extension CorrectionEvidenceTier {
    var allowsPurePhoneticCandidate: Bool {
        switch self {
        case .t2ConfirmedSpan, .t3ConfirmedRepeated, .t4Gold:
            return true
        case .t0Untrusted, .t1WeakInteraction, .negativeEvidence, .none:
            return false
        }
    }
}
