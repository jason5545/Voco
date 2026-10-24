import Foundation
import SwiftData

@Model
final class SessionMetric {
    var id: UUID = UUID()
    var transcriptionId: UUID = UUID()
    var timestamp: Date = Date()
    var source: String?
    var wordCount: Int = 0
    var audioDuration: TimeInterval = 0
    var transcriptionModelName: String?
    var transcriptionDuration: TimeInterval?
    var speedFactor: Double?
    @Attribute(originalName: "powerModeName")
    var modeName: String?
    var aiEnhancementModelName: String?
    var enhancementDuration: TimeInterval?
    var asrEngineID: String?
    var languageMode: String?
    var activeContextIDsJSON: String?
    var canonicalizationReplacementCount: Int = 0
    var canonicalizationSuggestionCount: Int = 0
    var confidenceScore: Double?
    var confidenceRoute: String?
    var confidenceReasonsJSON: String?
    var reviewTriggerCount: Int = 0
    var reviewTriggerIDsJSON: String?
    var reviewTriggerSummariesJSON: String?
    var candidateCount: Int = 0
    var candidateSourceCountsJSON: String?
    var reviewRequiredCandidateCount: Int = 0
    var candidateDivergenceRatio: Double?
    var selectedCandidateHypothesisSource: String?
    var selectedCandidate: String?
    var candidateSelectionSource: String?
    var userCorrectionDistance: Double?
    var correctionFeedbackCount: Int = 0
    var correctiveFeedbackCount: Int = 0
    var correctionFeedbackReasonsJSON: String?
    var styleGuardReasonCount: Int = 0
    var styleGuardReasonsJSON: String?
    var styleGuardRejectedCharacterCount: Int = 0
    var sourceTranscriptionID: UUID?
    var retranscriptionChangeCategory: String?
    var retranscriptionChangeRatio: Double?
    var retranscriptionEditDistance: Int?
    var retranscriptionConfidenceDelta: Double?
    var finalPastedCharacterCount: Int = 0
    var finalPastedWordCount: Int = 0
    var pasteCommandPosted: Bool?

    init(
        transcriptionId: UUID,
        timestamp: Date = Date(),
        source: String? = "recorder",
        wordCount: Int,
        audioDuration: TimeInterval,
        transcriptionModelName: String?,
        transcriptionDuration: TimeInterval?,
        speedFactor: Double?,
        modeName: String?,
        aiEnhancementModelName: String?,
        enhancementDuration: TimeInterval?,
        asrEngineID: String? = nil,
        languageMode: String? = nil,
        activeContextIDs: [String] = [],
        canonicalizationReplacementCount: Int = 0,
        canonicalizationSuggestionCount: Int = 0,
        confidenceScore: Double? = nil,
        confidenceRoute: String? = nil,
        confidenceReasons: [String] = [],
        reviewTriggerCount: Int = 0,
        reviewTriggerIDs: [String] = [],
        reviewTriggerSummaries: [String] = [],
        candidateCount: Int = 0,
        candidateSourceCounts: [String: Int] = [:],
        reviewRequiredCandidateCount: Int = 0,
        candidateDivergenceRatio: Double? = nil,
        selectedCandidateHypothesisSource: String? = nil,
        selectedCandidate: String? = nil,
        candidateSelectionSource: String? = nil,
        userCorrectionDistance: Double? = nil,
        correctionFeedbackCount: Int = 0,
        correctiveFeedbackCount: Int = 0,
        correctionFeedbackReasons: [String] = [],
        styleGuardReasonCount: Int = 0,
        styleGuardReasons: [String] = [],
        styleGuardRejectedCharacterCount: Int = 0,
        sourceTranscriptionID: UUID? = nil,
        retranscriptionChangeCategory: String? = nil,
        retranscriptionChangeRatio: Double? = nil,
        retranscriptionEditDistance: Int? = nil,
        retranscriptionConfidenceDelta: Double? = nil,
        finalPastedCharacterCount: Int = 0,
        finalPastedWordCount: Int = 0,
        pasteCommandPosted: Bool? = nil
    ) {
        self.id = UUID()
        self.transcriptionId = transcriptionId
        self.timestamp = timestamp
        self.source = source
        self.wordCount = wordCount
        self.audioDuration = audioDuration
        self.transcriptionModelName = transcriptionModelName
        self.transcriptionDuration = transcriptionDuration
        self.speedFactor = speedFactor
        self.modeName = modeName
        self.aiEnhancementModelName = aiEnhancementModelName
        self.enhancementDuration = enhancementDuration
        self.asrEngineID = asrEngineID
        self.languageMode = languageMode
        self.activeContextIDsJSON = Self.encodeJSON(activeContextIDs)
        self.canonicalizationReplacementCount = canonicalizationReplacementCount
        self.canonicalizationSuggestionCount = canonicalizationSuggestionCount
        self.confidenceScore = confidenceScore
        self.confidenceRoute = confidenceRoute
        self.confidenceReasonsJSON = Self.encodeJSON(confidenceReasons)
        self.reviewTriggerCount = reviewTriggerCount
        self.reviewTriggerIDsJSON = Self.encodeJSON(reviewTriggerIDs)
        self.reviewTriggerSummariesJSON = Self.encodeJSON(reviewTriggerSummaries)
        self.candidateCount = candidateCount
        self.candidateSourceCountsJSON = Self.encodeJSON(candidateSourceCounts)
        self.reviewRequiredCandidateCount = reviewRequiredCandidateCount
        self.candidateDivergenceRatio = candidateDivergenceRatio
        self.selectedCandidateHypothesisSource = selectedCandidateHypothesisSource
        self.selectedCandidate = selectedCandidate
        self.candidateSelectionSource = candidateSelectionSource
        self.userCorrectionDistance = userCorrectionDistance
        self.correctionFeedbackCount = correctionFeedbackCount
        self.correctiveFeedbackCount = correctiveFeedbackCount
        self.correctionFeedbackReasonsJSON = Self.encodeJSON(correctionFeedbackReasons)
        self.styleGuardReasonCount = styleGuardReasonCount
        self.styleGuardReasonsJSON = Self.encodeJSON(styleGuardReasons)
        self.styleGuardRejectedCharacterCount = styleGuardRejectedCharacterCount
        self.sourceTranscriptionID = sourceTranscriptionID
        self.retranscriptionChangeCategory = retranscriptionChangeCategory
        self.retranscriptionChangeRatio = retranscriptionChangeRatio
        self.retranscriptionEditDistance = retranscriptionEditDistance
        self.retranscriptionConfidenceDelta = retranscriptionConfidenceDelta
        self.finalPastedCharacterCount = finalPastedCharacterCount
        self.finalPastedWordCount = finalPastedWordCount
        self.pasteCommandPosted = pasteCommandPosted
    }
}
