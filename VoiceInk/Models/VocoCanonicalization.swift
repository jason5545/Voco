import Foundation

struct VocoCanonicalTerm: Codable, Equatable, Identifiable {
    let id: String
    let canonical: String
    let aliases: [String]
    let type: String
    let contexts: [String]
    let caseSensitive: Bool
    let autoReplaceThreshold: Double
    let requiresContextForAutoReplace: Bool

    init(
        id: String,
        canonical: String,
        aliases: [String],
        type: String,
        contexts: [String] = [],
        caseSensitive: Bool = false,
        autoReplaceThreshold: Double = 0.9,
        requiresContextForAutoReplace: Bool = false
    ) {
        self.id = id
        self.canonical = canonical
        self.aliases = aliases
        self.type = type
        self.contexts = contexts
        self.caseSensitive = caseSensitive
        self.autoReplaceThreshold = autoReplaceThreshold
        self.requiresContextForAutoReplace = requiresContextForAutoReplace
    }
}

struct VocoContextPack: Codable, Equatable, Identifiable {
    let id: String
    let displayName: String
    let terms: [VocoCanonicalTerm]

    var aliasCount: Int {
        terms.reduce(0) { $0 + $1.aliases.count }
    }

    var contextRequiredTermCount: Int {
        terms.filter(\.requiresContextForAutoReplace).count
    }

    var canonicalPreview: String {
        terms
            .prefix(6)
            .map(\.canonical)
            .joined(separator: ", ")
    }
}

struct VocoReplacement: Codable, Equatable {
    let originalText: String
    let replacementText: String
    let termID: String
    let confidence: Double
    let reason: String
    let rangeStart: Int?
    let rangeLength: Int?
}

struct VocoNormalizationResult: Codable, Equatable {
    let originalText: String
    let normalizedText: String
    let activeContextIDs: [String]
    let replacements: [VocoReplacement]
    let suggestions: [VocoReplacement]
}

enum VocoConfidenceRoute: String, Codable, Equatable {
    case directInsertion
    case reviewSuggested
}

struct VocoReviewTrigger: Codable, Equatable, Identifiable {
    let id: String
    let reason: String
    let detail: String?

    init(id: String, reason: String, detail: String? = nil) {
        self.id = id
        self.reason = reason
        self.detail = detail
    }

    var displayName: String {
        VocoSignalDisplayFormatter.displayReason(for: reason)
    }
}

enum VocoHypothesisSource: String, Codable, Equatable {
    case autoContext
    case autoApplyModel
    case suggestedRepair
    case originalCleaned
    case rawASR
    case segmentRescue
    case customRescue

    var displayName: String {
        switch self {
        case .autoContext:
            return "AUTO + context"
        case .autoApplyModel:
            return String(localized: "Auto-apply model")
        case .suggestedRepair:
            return "Suggestion pass"
        case .originalCleaned:
            return "Cleaned ASR"
        case .rawASR:
            return "Raw ASR"
        case .segmentRescue:
            return "Segment rescue"
        case .customRescue:
            return "Custom rescue"
        }
    }

    var analyticsSortPriority: Int {
        switch self {
        case .autoContext:
            return 0
        case .autoApplyModel:
            return 1
        case .suggestedRepair:
            return 2
        case .segmentRescue:
            return 3
        case .customRescue:
            return 4
        case .originalCleaned:
            return 5
        case .rawASR:
            return 6
        }
    }
}

struct VocoHypothesis: Codable, Equatable, Identifiable {
    let id: String
    let text: String
    let label: String
    let source: VocoHypothesisSource
    let confidenceScore: Double?
    let divergenceFromRecommended: Double?
    let reasons: [String]
    let activeContextIDs: [String]
    let appliedTermIDs: [String]
    let requiresReview: Bool

    init(
        id: String,
        text: String,
        label: String,
        source: VocoHypothesisSource,
        confidenceScore: Double?,
        divergenceFromRecommended: Double? = nil,
        reasons: [String],
        activeContextIDs: [String],
        appliedTermIDs: [String],
        requiresReview: Bool
    ) {
        self.id = id
        self.text = text
        self.label = label
        self.source = source
        self.confidenceScore = confidenceScore
        self.divergenceFromRecommended = divergenceFromRecommended
        self.reasons = reasons
        self.activeContextIDs = activeContextIDs
        self.appliedTermIDs = appliedTermIDs
        self.requiresReview = requiresReview
    }

    var sourceDisplayName: String {
        source.displayName
    }
}

struct VocoCorrectionRiskProfile: Codable, Equatable {
    static let empty = VocoCorrectionRiskProfile(
        recentSessionCount: 0,
        correctedSessionCount: 0,
        recentCorrectionRate: 0,
        highRiskTermIDs: [],
        lookbackDays: 14,
        minimumSampleCount: 3
    )

    let recentSessionCount: Int
    let correctedSessionCount: Int
    let recentCorrectionRate: Double
    let highRiskTermIDs: [String]
    let lookbackDays: Int
    let minimumSampleCount: Int

    var hasEnoughSamples: Bool {
        recentSessionCount >= minimumSampleCount
    }

    var hasElevatedCorrectionRate: Bool {
        hasEnoughSamples && recentCorrectionRate >= 0.35
    }

    func hasHighRiskOverlap(with termIDs: [String]) -> Bool {
        let riskIDs = Set(highRiskTermIDs)
        return termIDs.contains { riskIDs.contains($0) }
    }
}

struct VocoConfidenceAssessment: Codable, Equatable {
    let score: Double
    let route: VocoConfidenceRoute
    let reasons: [String]
    let reviewTriggers: [VocoReviewTrigger]
    let candidates: [String]
    let candidateLabels: [String]
    let hypothesisDetails: [VocoHypothesis]
    let correctionRiskProfile: VocoCorrectionRiskProfile?
    let selectedCandidate: String

    init(
        score: Double,
        route: VocoConfidenceRoute,
        reasons: [String],
        reviewTriggers: [VocoReviewTrigger] = [],
        candidates: [String],
        candidateLabels: [String] = [],
        hypothesisDetails: [VocoHypothesis] = [],
        correctionRiskProfile: VocoCorrectionRiskProfile? = nil,
        selectedCandidate: String
    ) {
        self.score = score
        self.route = route
        self.reasons = reasons
        self.reviewTriggers = reviewTriggers
        self.candidates = candidates
        self.candidateLabels = candidateLabels
        self.hypothesisDetails = hypothesisDetails
        self.correctionRiskProfile = correctionRiskProfile
        self.selectedCandidate = selectedCandidate
    }

    enum CodingKeys: String, CodingKey {
        case score
        case route
        case reasons
        case reviewTriggers
        case candidates
        case candidateLabels
        case hypothesisDetails
        case correctionRiskProfile
        case selectedCandidate
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.init(
            score: try container.decode(Double.self, forKey: .score),
            route: try container.decode(VocoConfidenceRoute.self, forKey: .route),
            reasons: try container.decode([String].self, forKey: .reasons),
            reviewTriggers: try container.decodeIfPresent([VocoReviewTrigger].self, forKey: .reviewTriggers) ?? [],
            candidates: try container.decode([String].self, forKey: .candidates),
            candidateLabels: try container.decodeIfPresent([String].self, forKey: .candidateLabels) ?? [],
            hypothesisDetails: try container.decodeIfPresent([VocoHypothesis].self, forKey: .hypothesisDetails) ?? [],
            correctionRiskProfile: try container.decodeIfPresent(
                VocoCorrectionRiskProfile.self,
                forKey: .correctionRiskProfile
            ),
            selectedCandidate: try container.decode(String.self, forKey: .selectedCandidate)
        )
    }

    func labelForCandidate(at index: Int) -> String {
        guard candidateLabels.indices.contains(index) else {
            return VocoCandidateLabelDisplayFormatter.displayName(for: "Candidate")
        }
        return VocoCandidateLabelDisplayFormatter.displayName(for: candidateLabels[index])
    }

    func hypothesisForCandidate(at index: Int) -> VocoHypothesis? {
        guard hypothesisDetails.indices.contains(index) else { return nil }
        return hypothesisDetails[index]
    }
}

enum VocoCandidateSelectionSource: String, Codable, Equatable {
    case userSelection
    case dismissedFallback
    case timeoutFallback
    case automaticFallback
    case finalPaste

    var displayName: String {
        switch self {
        case .userSelection:
            return "User selection"
        case .dismissedFallback:
            return "Dismissed fallback"
        case .timeoutFallback:
            return "Timeout fallback"
        case .automaticFallback:
            return "Automatic fallback"
        case .finalPaste:
            return "Final pasted text"
        }
    }
}

enum VocoCandidateLabelDisplayFormatter {
    static func displayName(for label: String) -> String {
        switch label {
        case "Recommended":
            return "Recommended"
        case "With suggestions":
            return "With suggestions"
        case "Segment rescue":
            return "Segment rescue"
        case "Raw cleanup rescue":
            return "Raw cleanup rescue"
        case "Original":
            return "Original"
        case "Raw ASR":
            return "Raw ASR"
        case "Typed correction":
            return "Typed correction"
        case "Auto-apply model":
            return String(localized: "Auto-apply model")
        case "Candidate":
            return "Candidate"
        default:
            return label
        }
    }
}

enum VocoSignalDisplayFormatter {
    static func displayReasons(for reasons: [String]) -> [String] {
        var seen: Set<String> = []
        return reasons
            .map(displayReason(for:))
            .filter { seen.insert($0).inserted }
    }

    static func displayStyleGuardReasons(for reasons: [String]) -> [String] {
        var seen: Set<String> = []
        return reasons
            .map(displayStyleGuardReason(for:))
            .filter { seen.insert($0).inserted }
    }

    static func displayReason(for reason: String) -> String {
        switch reason {
        case "alias-match":
            return "Alias match"
        case "canonical-match":
            return "Already canonical"
        case "canonicalization-clean":
            return "Clean"
        case "candidate-confirmed":
            return "Candidate confirmed"
        case "candidate-custom":
            return "Custom candidate"
        case "candidate-dismissed-fallback":
            return "Dismissed fallback"
        case "candidate-override":
            return "Candidate changed"
        case "candidate-timeout-fallback":
            return "Timeout fallback"
        case "candidate-auto-fallback":
            return "Automatic fallback"
        case "case-normalization":
            return "Case normalization"
        case "context-required":
            return "Needs context"
        case "contextual-alias-match":
            return "Context match"
        case "heavy-normalization":
            return "Heavy normalization"
        case "high-risk-term":
            return "High-risk term"
        case "inactive-context-suggestion":
            return "Inactive context"
        case "low-confidence-replacement":
            return "Low confidence"
        case "low-confidence-score":
            return "Low score"
        case "auto-apply-model":
            return String(localized: "Auto-apply model")
        case "auto-apply-model-suggestion":
            return String(localized: "Auto-apply suggestion")
        case "auto-apply-model-protected-term-guard":
            return String(localized: "Protected term guard")
        case "phonetic-correction-term":
            return String(localized: "Phonetic correction")
        case "protected-term-replacement":
            return "Protected term changed"
        case "raw-cleanup-drift":
            return "Cleanup drift"
        case "raw-cleanup-local-regression":
            return "Cleanup local regression"
        case "raw-cleanup-significant":
            return "Cleanup changed text"
        case "recent-correction-rate":
            return "Recent corrections"
        case "recent-term-corrections":
            return "Term was corrected"
        case "segment-rescue":
            return "Segment rescue"
        case "unresolved-suggestions":
            return "Needs choice"
        case "user-substitution":
            return "User substitution"
        default:
            if let retranscriptionReason = retranscriptionDisplayReason(for: reason) {
                return retranscriptionReason
            }
            return fallbackDisplayReason(for: reason)
        }
    }

    static func displayStyleGuardReason(for reason: String) -> String {
        styleGuardReasonDisplayComponents(for: reason).detail
    }

    static func displayStyleGuardReasonCategory(for reason: String) -> String {
        styleGuardReasonDisplayComponents(for: reason).category
    }

    private static func styleGuardReasonDisplayComponents(for reason: String) -> (category: String, detail: String) {
        let trimmed = reason.trimmingCharacters(in: .whitespacesAndNewlines)
        let parts = trimmed.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
        let categoryID = parts.first.map(String.init) ?? trimmed
        let category = styleGuardCategoryDisplayName(for: categoryID)
        guard parts.count > 1 else {
            return (category, category)
        }

        let payload = String(parts[1]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !payload.isEmpty else {
            return (category, category)
        }

        return (category, "\(category) (\(payload))")
    }

    private static func styleGuardCategoryDisplayName(for category: String) -> String {
        switch category {
        case "assistant-opener":
            return "Assistant opener"
        case "dropped-mixed-language-term":
            return "Dropped mixed language term"
        case "introduced-structured-format":
            return "Structured formatting"
        case "style-expansion":
            return "Style expansion"
        default:
            return displayReason(for: category)
        }
    }

    private static func retranscriptionDisplayReason(for reason: String) -> String? {
        guard reason.hasPrefix("retranscription-") else { return nil }

        let rawCategory = String(reason.dropFirst("retranscription-".count))
        guard let category = RetranscriptionChangeCategory(rawValue: rawCategory) else {
            return "Retranscription change"
        }
        switch category {
        case .unchanged:
            return "Retranscription unchanged"
        case .minorChange:
            return "Retranscription minor"
        case .meaningfulChange:
            return "Retranscription meaningful"
        }
    }

    private static func fallbackDisplayReason(for reason: String) -> String {
        let words = reason
            .replacingOccurrences(of: "_", with: "-")
            .split(separator: "-")
            .map { String($0) }

        guard !words.isEmpty else { return reason }

        return words.enumerated()
            .map { index, word in
                index == 0 ? word.capitalized : word
            }
            .joined(separator: " ")
    }
}

enum VocoReviewTriggerDisplayFormatter {
    static func summaries(for triggers: [VocoReviewTrigger]) -> [String] {
        var seen: Set<String> = []
        return triggers
            .filter { seen.insert($0.id).inserted }
            .map(summary(for:))
    }

    static func summary(for trigger: VocoReviewTrigger) -> String {
        let detail = trigger.detail?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let detail, !detail.isEmpty else {
            return trigger.displayName
        }

        return "\(trigger.displayName) (\(detail))"
    }
}

enum VocoHypothesisDisplayFormatter {
    static func summary(for hypothesis: VocoHypothesis) -> String? {
        var parts: [String] = []

        if let confidenceScore = hypothesis.confidenceScore {
            parts.append("Confidence \(percent(confidenceScore))")
        }

        if let divergence = hypothesis.divergenceFromRecommended,
           divergence > 0 {
            parts.append("Delta \(percent(divergence))")
        }

        let reasons = VocoSignalDisplayFormatter.displayReasons(for: hypothesis.reasons)
        if !reasons.isEmpty {
            parts.append(reasons.joined(separator: ", "))
        }

        let termIDs = uniqueNonEmpty(hypothesis.appliedTermIDs)
        if !termIDs.isEmpty {
            parts.append("Terms \(termIDs.joined(separator: ", "))")
        }

        let contexts = uniqueNonEmpty(
            VocoCanonicalizationService.contextDisplayNames(for: hypothesis.activeContextIDs)
        )
        if !contexts.isEmpty {
            parts.append("Contexts \(contexts.joined(separator: ", "))")
        }

        if hypothesis.requiresReview {
            parts.append("Review required")
        }

        guard !parts.isEmpty else { return nil }
        return parts.joined(separator: " · ")
    }

    private static func percent(_ value: Double) -> String {
        let bounded = max(0, min(1, value))
        return "\(Int((bounded * 100).rounded()))%"
    }

    private static func uniqueNonEmpty(_ values: [String]) -> [String] {
        var seen: Set<String> = []
        return values
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .filter { seen.insert($0).inserted }
    }
}
