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
    let autoApplyModelVersion: String?
    let autoApplyModelGeneratedAt: String?
    let autoApplyPolicyHitIDs: [String]

    init(
        originalText: String,
        normalizedText: String,
        activeContextIDs: [String],
        replacements: [VocoReplacement],
        suggestions: [VocoReplacement],
        autoApplyModelVersion: String? = nil,
        autoApplyModelGeneratedAt: String? = nil,
        autoApplyPolicyHitIDs: [String] = []
    ) {
        self.originalText = originalText
        self.normalizedText = normalizedText
        self.activeContextIDs = activeContextIDs
        self.replacements = replacements
        self.suggestions = suggestions
        self.autoApplyModelVersion = autoApplyModelVersion
        self.autoApplyModelGeneratedAt = autoApplyModelGeneratedAt
        self.autoApplyPolicyHitIDs = autoApplyPolicyHitIDs
    }

    private enum CodingKeys: String, CodingKey {
        case originalText
        case normalizedText
        case activeContextIDs
        case replacements
        case suggestions
        case autoApplyModelVersion
        case autoApplyModelGeneratedAt
        case autoApplyPolicyHitIDs
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        originalText = try container.decode(String.self, forKey: .originalText)
        normalizedText = try container.decode(String.self, forKey: .normalizedText)
        activeContextIDs = try container.decode([String].self, forKey: .activeContextIDs)
        replacements = try container.decode([VocoReplacement].self, forKey: .replacements)
        suggestions = try container.decode([VocoReplacement].self, forKey: .suggestions)
        autoApplyModelVersion = try container.decodeIfPresent(String.self, forKey: .autoApplyModelVersion)
        autoApplyModelGeneratedAt = try container.decodeIfPresent(String.self, forKey: .autoApplyModelGeneratedAt)
        autoApplyPolicyHitIDs = try container.decodeIfPresent([String].self, forKey: .autoApplyPolicyHitIDs) ?? []
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(originalText, forKey: .originalText)
        try container.encode(normalizedText, forKey: .normalizedText)
        try container.encode(activeContextIDs, forKey: .activeContextIDs)
        try container.encode(replacements, forKey: .replacements)
        try container.encode(suggestions, forKey: .suggestions)
        try container.encodeIfPresent(autoApplyModelVersion, forKey: .autoApplyModelVersion)
        try container.encodeIfPresent(autoApplyModelGeneratedAt, forKey: .autoApplyModelGeneratedAt)
        if !autoApplyPolicyHitIDs.isEmpty {
            try container.encode(autoApplyPolicyHitIDs, forKey: .autoApplyPolicyHitIDs)
        }
    }
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

    /// English on purpose: only the CSV export prints these source names, and an exported file
    /// should read the same whatever the system language is.
    var displayName: String {
        switch self {
        case .autoContext:
            return "AUTO + context"
        case .autoApplyModel:
            return "Auto-apply model"
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

    /// English on purpose: only the CSV export prints these, see `VocoHypothesisSource.displayName`.
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
            return String(localized: "Recommended")
        case "With suggestions":
            return String(localized: "With suggestions")
        case "Segment rescue":
            return String(localized: "Segment rescue")
        case "Raw cleanup rescue":
            return String(localized: "Raw cleanup rescue")
        case "Original":
            return String(localized: "Original")
        case "Raw ASR":
            return String(localized: "Raw ASR")
        case "Typed correction":
            return String(localized: "Typed correction")
        case "Auto-apply model":
            return String(localized: "Auto-apply model")
        case "Candidate":
            return String(localized: "Candidate")
        default:
            return label
        }
    }
}

enum VocoSignalDisplayFormatter {
    static func displayReasons(for reasons: [String], localized: Bool = true) -> [String] {
        var seen: Set<String> = []
        return reasons
            .map { displayReason(for: $0, localized: localized) }
            .filter { seen.insert($0).inserted }
    }

    static func displayStyleGuardReasons(for reasons: [String], localized: Bool = true) -> [String] {
        var seen: Set<String> = []
        return reasons
            .map { displayStyleGuardReason(for: $0, localized: localized) }
            .filter { seen.insert($0).inserted }
    }

    /// `localized: false` returns the English source text, which the CSV export asks for.
    static func displayReason(for reason: String, localized: Bool = true) -> String {
        switch reason {
        case "alias-match":
            return text("Alias match", localized)
        case "canonical-match":
            return text("Already canonical", localized)
        case "canonicalization-clean":
            return text("Clean", localized)
        case "candidate-confirmed":
            return text("Candidate confirmed", localized)
        case "candidate-custom":
            return text("Custom candidate", localized)
        case "candidate-dismissed-fallback":
            return text("Dismissed fallback", localized)
        case "candidate-override":
            return text("Candidate changed", localized)
        case "candidate-timeout-fallback":
            return text("Timeout fallback", localized)
        case "candidate-auto-fallback":
            return text("Automatic fallback", localized)
        case "case-normalization":
            return text("Case normalization", localized)
        case "context-required":
            return text("Needs context", localized)
        case "contextual-alias-match":
            return text("Context match", localized)
        case "heavy-normalization":
            return text("Heavy normalization", localized)
        case "high-risk-term":
            return text("High-risk term", localized)
        case "inactive-context-suggestion":
            return text("Inactive context", localized)
        case "low-confidence-replacement":
            return text("Low confidence", localized)
        case "low-confidence-score":
            return text("Low score", localized)
        case "auto-apply-model":
            return String(localized: "Auto-apply model")
        case "auto-apply-model-suggestion":
            return String(localized: "Auto-apply suggestion")
        case "auto-apply-model-protected-term-guard":
            return String(localized: "Protected term guard")
        case "phonetic-correction-term":
            return String(localized: "Phonetic correction")
        case "protected-term-replacement":
            return text("Protected term changed", localized)
        case "raw-cleanup-drift":
            return text("Cleanup drift", localized)
        case "raw-cleanup-local-regression":
            return text("Cleanup local regression", localized)
        case "raw-cleanup-significant":
            return text("Cleanup changed text", localized)
        case "recent-correction-rate":
            return text("Recent corrections", localized)
        case "recent-term-corrections":
            return text("Term was corrected", localized)
        case "segment-rescue":
            return text("Segment rescue", localized)
        case "unresolved-suggestions":
            return text("Needs choice", localized)
        case "user-substitution":
            return text("User substitution", localized)
        default:
            if let retranscriptionReason = retranscriptionDisplayReason(for: reason, localized: localized) {
                return retranscriptionReason
            }
            return fallbackDisplayReason(for: reason)
        }
    }

    static func displayStyleGuardReason(for reason: String, localized: Bool = true) -> String {
        styleGuardReasonDisplayComponents(for: reason, localized: localized).detail
    }

    static func displayStyleGuardReasonCategory(for reason: String, localized: Bool = true) -> String {
        styleGuardReasonDisplayComponents(for: reason, localized: localized).category
    }

    private static func styleGuardReasonDisplayComponents(
        for reason: String,
        localized: Bool
    ) -> (category: String, detail: String) {
        let trimmed = reason.trimmingCharacters(in: .whitespacesAndNewlines)
        let parts = trimmed.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
        let categoryID = parts.first.map(String.init) ?? trimmed
        let category = styleGuardCategoryDisplayName(for: categoryID, localized: localized)
        guard parts.count > 1 else {
            return (category, category)
        }

        let payload = String(parts[1]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !payload.isEmpty else {
            return (category, category)
        }

        return (category, "\(category) (\(payload))")
    }

    private static func styleGuardCategoryDisplayName(for category: String, localized: Bool) -> String {
        switch category {
        case "assistant-opener":
            return text("Assistant opener", localized)
        case "dropped-mixed-language-term":
            return text("Dropped mixed language term", localized)
        case "introduced-structured-format":
            return text("Structured formatting", localized)
        case "style-expansion":
            return text("Style expansion", localized)
        default:
            return displayReason(for: category, localized: localized)
        }
    }

    private static func retranscriptionDisplayReason(for reason: String, localized: Bool) -> String? {
        guard reason.hasPrefix("retranscription-") else { return nil }

        let rawCategory = String(reason.dropFirst("retranscription-".count))
        guard let category = RetranscriptionChangeCategory(rawValue: rawCategory) else {
            return text("Retranscription change", localized)
        }
        switch category {
        case .unchanged:
            return text("Retranscription unchanged", localized)
        case .minorChange:
            return text("Retranscription minor", localized)
        case .meaningfulChange:
            return text("Retranscription meaningful", localized)
        }
    }

    /// `localized: false` returns the English source text. (The CSV export uses that path.)
    private static func text(_ english: String, _ localized: Bool) -> String {
        localized ? String(localized: String.LocalizationValue(english)) : english
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
    static func summaries(for triggers: [VocoReviewTrigger], localized: Bool = true) -> [String] {
        var seen: Set<String> = []
        return triggers
            .filter { seen.insert($0.id).inserted }
            .map { summary(for: $0, localized: localized) }
    }

    static func summary(for trigger: VocoReviewTrigger, localized: Bool = true) -> String {
        let detail = trigger.detail?.trimmingCharacters(in: .whitespacesAndNewlines)
        let name = VocoSignalDisplayFormatter.displayReason(for: trigger.reason, localized: localized)
        guard let detail, !detail.isEmpty else {
            return name
        }

        return "\(name) (\(detail))"
    }
}

enum VocoHypothesisDisplayFormatter {
    /// English on purpose: only the CSV export prints this summary, see
    /// `VocoHypothesisSource.displayName`.
    static func summary(for hypothesis: VocoHypothesis) -> String? {
        var parts: [String] = []

        if let confidenceScore = hypothesis.confidenceScore {
            parts.append("Confidence \(percent(confidenceScore))")
        }

        if let divergence = hypothesis.divergenceFromRecommended,
           divergence > 0 {
            parts.append("Delta \(percent(divergence))")
        }

        let reasons = VocoSignalDisplayFormatter.displayReasons(for: hypothesis.reasons, localized: false)
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
