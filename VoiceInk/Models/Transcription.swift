import Foundation
import SwiftData

enum TranscriptionStatus: String, Codable {
    case pending
    case completed
    case failed
    case canceled
}

@Model
final class Transcription {
    static let canceledTranscriptionText = "The transcription was canceled."

    var id: UUID = UUID()
    var text: String = ""
    var enhancedText: String?
    var timestamp: Date = Date()
    var duration: TimeInterval = 0
    var audioFileURL: String?
    var transcriptionModelName: String?
    var aiEnhancementModelName: String?
    var promptName: String?
    var transcriptionDuration: TimeInterval?
    var enhancementDuration: TimeInterval?
    var aiRequestSystemMessage: String?
    var aiRequestUserMessage: String?
    @Attribute(originalName: "powerModeName")
    var modeName: String?
    @Attribute(originalName: "powerModeEmoji")
    var modeEmoji: String?
    var transcriptionStatus: String?
    var rawTranscript: String?
    var normalizedTranscript: String?
    var finalPastedText: String?
    var pasteCommandPosted: Bool?
    var activeContextIDsJSON: String?
    var canonicalizationReplacementsJSON: String?
    var canonicalizationSuggestionsJSON: String?
    var asrEngineID: String?
    var languageMode: String?
    var confidenceScore: Double?
    var confidenceRoute: String?
    var confidenceReasonsJSON: String?
    var reviewTriggersJSON: String?
    var hypothesesJSON: String?
    var hypothesisLabelsJSON: String?
    var hypothesisDetailsJSON: String?
    var correctionRiskRate: Double?
    var correctionRiskSampleCount: Int?
    var correctionRiskCorrectedCount: Int?
    var correctionRiskTermIDsJSON: String?
    var selectedCandidate: String?
    var candidateSelectionSource: String?
    var userCorrectionDistance: Double?
    var styleGuardReasonsJSON: String?
    var styleGuardRejectedText: String?
    var sourceTranscriptionID: UUID?
    var retranscriptionSourceText: String?
    var retranscriptionAnalysisJSON: String?
    var correctionFeedbackJSON: String?
    var autoApplyModelVersion: String?
    var autoApplyModelGeneratedAt: String?
    var autoApplyPolicyHitIDsJSON: String?
    var qwen3AdapterMetadataJSON: String?
    var qwen3SpecialistRoutingMetadataJSON: String?
    /// Cached Worker correction receipts for this row (voco.row-corrections.v1 payload).
    /// Written only by RowCorrectionMarkingRefresher / the rule assistant; Core Data name ZCORRECTIONSJSON.
    var correctionsJSON: String?

    init(text: String,
         duration: TimeInterval,
         enhancedText: String? = nil,
         audioFileURL: String? = nil,
         transcriptionModelName: String? = nil,
         aiEnhancementModelName: String? = nil,
         promptName: String? = nil,
         transcriptionDuration: TimeInterval? = nil,
         enhancementDuration: TimeInterval? = nil,
         aiRequestSystemMessage: String? = nil,
         aiRequestUserMessage: String? = nil,
         modeName: String? = nil,
         modeEmoji: String? = nil,
         powerModeName: String? = nil,
         powerModeEmoji: String? = nil,
         rawTranscript: String? = nil,
         normalizedTranscript: String? = nil,
         finalPastedText: String? = nil,
         pasteCommandPosted: Bool? = nil,
         activeContextIDs: [String] = [],
         canonicalizationReplacements: [VocoReplacement] = [],
         canonicalizationSuggestions: [VocoReplacement] = [],
         asrEngineID: String? = nil,
         languageMode: String? = nil,
         confidenceScore: Double? = nil,
         confidenceAssessment: VocoConfidenceAssessment? = nil,
         reviewTriggers: [VocoReviewTrigger] = [],
         hypotheses: [String] = [],
         hypothesisLabels: [String] = [],
         hypothesisDetails: [VocoHypothesis] = [],
         correctionRiskRate: Double? = nil,
         correctionRiskSampleCount: Int? = nil,
         correctionRiskCorrectedCount: Int? = nil,
         correctionRiskTermIDs: [String] = [],
         selectedCandidate: String? = nil,
         candidateSelectionSource: VocoCandidateSelectionSource? = nil,
         userCorrectionDistance: Double? = nil,
         styleGuardReasons: [String] = [],
         styleGuardRejectedText: String? = nil,
         sourceTranscriptionID: UUID? = nil,
         retranscriptionSourceText: String? = nil,
         retranscriptionAnalysis: RetranscriptionAnalysis? = nil,
         correctionFeedback: [CorrectionFeedbackSignal] = [],
         transcriptionStatus: TranscriptionStatus = .pending) {
        self.id = UUID()
        self.text = text
        self.enhancedText = enhancedText
        self.timestamp = Date()
        self.duration = duration
        self.audioFileURL = audioFileURL
        self.transcriptionModelName = transcriptionModelName
        self.aiEnhancementModelName = aiEnhancementModelName
        self.promptName = promptName
        self.transcriptionDuration = transcriptionDuration
        self.enhancementDuration = enhancementDuration
        self.aiRequestSystemMessage = aiRequestSystemMessage
        self.aiRequestUserMessage = aiRequestUserMessage
        self.modeName = modeName ?? powerModeName
        self.modeEmoji = modeEmoji ?? powerModeEmoji
        self.rawTranscript = rawTranscript
        self.normalizedTranscript = normalizedTranscript
        self.finalPastedText = finalPastedText
        self.pasteCommandPosted = pasteCommandPosted
        self.activeContextIDsJSON = Self.encodeJSON(activeContextIDs)
        self.canonicalizationReplacementsJSON = Self.encodeJSON(canonicalizationReplacements)
        self.canonicalizationSuggestionsJSON = Self.encodeJSON(canonicalizationSuggestions)
        self.asrEngineID = asrEngineID
        self.languageMode = languageMode
        self.confidenceScore = confidenceAssessment?.score ?? confidenceScore
        self.confidenceRoute = confidenceAssessment?.route.rawValue
        self.confidenceReasonsJSON = Self.encodeJSON(confidenceAssessment?.reasons ?? [])
        self.reviewTriggersJSON = Self.encodeJSON(confidenceAssessment?.reviewTriggers ?? reviewTriggers)
        self.hypothesesJSON = Self.encodeJSON(confidenceAssessment?.candidates ?? hypotheses)
        self.hypothesisLabelsJSON = Self.encodeJSON(confidenceAssessment?.candidateLabels ?? hypothesisLabels)
        self.hypothesisDetailsJSON = Self.encodeJSON(confidenceAssessment?.hypothesisDetails ?? hypothesisDetails)
        if let riskProfile = confidenceAssessment?.correctionRiskProfile {
            self.correctionRiskRate = riskProfile.recentCorrectionRate
            self.correctionRiskSampleCount = riskProfile.recentSessionCount
            self.correctionRiskCorrectedCount = riskProfile.correctedSessionCount
            self.correctionRiskTermIDsJSON = Self.encodeJSON(riskProfile.highRiskTermIDs)
        } else {
            self.correctionRiskRate = correctionRiskRate
            self.correctionRiskSampleCount = correctionRiskSampleCount
            self.correctionRiskCorrectedCount = correctionRiskCorrectedCount
            self.correctionRiskTermIDsJSON = Self.encodeJSON(correctionRiskTermIDs)
        }
        self.selectedCandidate = confidenceAssessment?.selectedCandidate ?? selectedCandidate
        self.candidateSelectionSource = candidateSelectionSource?.rawValue
        self.userCorrectionDistance = userCorrectionDistance
        self.styleGuardReasonsJSON = Self.encodeJSON(styleGuardReasons)
        self.styleGuardRejectedText = styleGuardRejectedText
        self.sourceTranscriptionID = sourceTranscriptionID
        self.retranscriptionSourceText = retranscriptionSourceText
        self.retranscriptionAnalysisJSON = Self.encodeJSON(retranscriptionAnalysis)
        self.correctionFeedbackJSON = Self.encodeJSON(correctionFeedback)
        self.transcriptionStatus = transcriptionStatus.rawValue
    }
}
