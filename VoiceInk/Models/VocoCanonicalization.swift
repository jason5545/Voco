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

enum VocoHypothesisSource: String, Codable, Equatable {
    case autoContext
    case suggestedRepair
    case originalCleaned
    case rawASR
    case segmentRescue
    case customRescue

    var displayName: String {
        switch self {
        case .autoContext:
            return "AUTO + context"
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
}

struct VocoHypothesis: Codable, Equatable, Identifiable {
    let id: String
    let text: String
    let label: String
    let source: VocoHypothesisSource
    let confidenceScore: Double?
    let reasons: [String]
    let activeContextIDs: [String]
    let appliedTermIDs: [String]
    let requiresReview: Bool

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
    let candidates: [String]
    let candidateLabels: [String]
    let hypothesisDetails: [VocoHypothesis]
    let correctionRiskProfile: VocoCorrectionRiskProfile?
    let selectedCandidate: String

    init(
        score: Double,
        route: VocoConfidenceRoute,
        reasons: [String],
        candidates: [String],
        candidateLabels: [String] = [],
        hypothesisDetails: [VocoHypothesis] = [],
        correctionRiskProfile: VocoCorrectionRiskProfile? = nil,
        selectedCandidate: String
    ) {
        self.score = score
        self.route = route
        self.reasons = reasons
        self.candidates = candidates
        self.candidateLabels = candidateLabels
        self.hypothesisDetails = hypothesisDetails
        self.correctionRiskProfile = correctionRiskProfile
        self.selectedCandidate = selectedCandidate
    }

    func labelForCandidate(at index: Int) -> String {
        guard candidateLabels.indices.contains(index) else { return "Candidate" }
        return candidateLabels[index]
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
        }
    }
}

struct VocoCandidateSelection: Codable, Equatable {
    let candidate: String
    let source: VocoCandidateSelectionSource

    init(candidate: String, source: VocoCandidateSelectionSource = .userSelection) {
        self.candidate = candidate
        self.source = source
    }
}

enum VocoSignalDisplayFormatter {
    static func displayReasons(for reasons: [String]) -> [String] {
        var seen: Set<String> = []
        return reasons
            .map(displayReason(for:))
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
        case "raw-cleanup-drift":
            return "Cleanup drift"
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

    private static func retranscriptionDisplayReason(for reason: String) -> String? {
        guard reason.hasPrefix("retranscription-") else { return nil }

        let rawCategory = String(reason.dropFirst("retranscription-".count))
        guard let category = RetranscriptionChangeCategory(rawValue: rawCategory) else {
            return "Retranscription change"
        }
        return "Retranscription \(category.displayName.lowercased())"
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

enum VocoHypothesisDisplayFormatter {
    static func summary(for hypothesis: VocoHypothesis) -> String? {
        var parts: [String] = []

        if let confidenceScore = hypothesis.confidenceScore {
            parts.append("Confidence \(percent(confidenceScore))")
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
