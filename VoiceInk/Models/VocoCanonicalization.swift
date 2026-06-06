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
