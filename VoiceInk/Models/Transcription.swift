import Foundation
import SwiftData

enum TranscriptionStatus: String, Codable {
    case pending
    case completed
    case failed
    case canceled
}

struct TranscriptionDetailText: Equatable, Identifiable {
    let id: String
    let label: String
    let text: String
    let isEnhanced: Bool
}

@Model
final class Transcription {
    static let canceledTranscriptionText = "The transcription was canceled."

    var id: UUID
    var text: String
    var enhancedText: String?
    var timestamp: Date
    var duration: TimeInterval
    var audioFileURL: String?
    var transcriptionModelName: String?
    var aiEnhancementModelName: String?
    var promptName: String?
    var transcriptionDuration: TimeInterval?
    var enhancementDuration: TimeInterval?
    var aiRequestSystemMessage: String?
    var aiRequestUserMessage: String?
    var powerModeName: String?
    var powerModeEmoji: String?
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

    var activeContextIDs: [String] {
        get { Self.decodeStringArray(activeContextIDsJSON) }
        set { activeContextIDsJSON = Self.encodeJSON(newValue) }
    }

    var canonicalizationReplacements: [VocoReplacement] {
        get { Self.decodeReplacements(canonicalizationReplacementsJSON) }
        set { canonicalizationReplacementsJSON = Self.encodeJSON(newValue) }
    }

    var canonicalizationSuggestions: [VocoReplacement] {
        get { Self.decodeReplacements(canonicalizationSuggestionsJSON) }
        set { canonicalizationSuggestionsJSON = Self.encodeJSON(newValue) }
    }

    var hypotheses: [String] {
        get { Self.decodeStringArray(hypothesesJSON) }
        set { hypothesesJSON = Self.encodeJSON(newValue) }
    }

    var hypothesisLabels: [String] {
        get { Self.decodeStringArray(hypothesisLabelsJSON) }
        set { hypothesisLabelsJSON = Self.encodeJSON(newValue) }
    }

    var hypothesisDetails: [VocoHypothesis] {
        get { Self.decodeHypotheses(hypothesisDetailsJSON) }
        set { hypothesisDetailsJSON = Self.encodeJSON(newValue) }
    }

    var confidenceReasons: [String] {
        get { Self.decodeStringArray(confidenceReasonsJSON) }
        set { confidenceReasonsJSON = Self.encodeJSON(newValue) }
    }

    var reviewTriggers: [VocoReviewTrigger] {
        get { Self.decodeReviewTriggers(reviewTriggersJSON) }
        set { reviewTriggersJSON = Self.encodeJSON(newValue) }
    }

    var correctionRiskTermIDs: [String] {
        get { Self.decodeStringArray(correctionRiskTermIDsJSON) }
        set { correctionRiskTermIDsJSON = Self.encodeJSON(newValue) }
    }

    var styleGuardReasons: [String] {
        get { Self.decodeStringArray(styleGuardReasonsJSON) }
        set { styleGuardReasonsJSON = Self.encodeJSON(newValue) }
    }

    var retranscriptionAnalysis: RetranscriptionAnalysis? {
        get { Self.decodeRetranscriptionAnalysis(retranscriptionAnalysisJSON) }
        set { retranscriptionAnalysisJSON = Self.encodeJSON(newValue) }
    }

    var correctionFeedback: [CorrectionFeedbackSignal] {
        get { Self.decodeCorrectionFeedback(correctionFeedbackJSON) }
        set { correctionFeedbackJSON = Self.encodeJSON(newValue) }
    }

    var hasDictationMetadata: Bool {
        Self.hasText(rawTranscript) ||
        Self.hasText(normalizedTranscript) ||
        Self.hasText(finalPastedText) ||
        pasteCommandPosted != nil ||
        Self.hasText(asrEngineID) ||
        Self.hasText(languageMode) ||
        confidenceScore != nil ||
        Self.hasText(confidenceRoute) ||
        !confidenceReasons.isEmpty ||
        !reviewTriggers.isEmpty ||
        correctionRiskRate != nil ||
        !correctionRiskTermIDs.isEmpty ||
        Self.hasText(selectedCandidate) ||
        Self.hasText(candidateSelectionSource) ||
        userCorrectionDistance != nil ||
        !activeContextIDs.isEmpty ||
        !canonicalizationReplacements.isEmpty ||
        !canonicalizationSuggestions.isEmpty ||
        !hypotheses.isEmpty ||
        !hypothesisDetails.isEmpty
    }

    var historyDisplayText: String {
        Self.firstNonEmpty(
            finalPastedText,
            enhancedText,
            selectedCandidate,
            normalizedTranscript,
            text
        ) ?? text
    }

    var detailDisplayTexts: [TranscriptionDetailText] {
        let normalizedText = Self.firstNonEmpty(normalizedTranscript, text) ?? text
        let selectedText = Self.firstNonEmpty(selectedCandidate)
        var items: [TranscriptionDetailText] = []
        var seenTexts: Set<String> = []

        if let rawText = Self.firstNonEmpty(rawTranscript),
           !Self.isSameText(rawText, normalizedText),
           !Self.isSameText(rawText, selectedText) {
            Self.appendDetailText(
                id: "raw-asr",
                label: "Raw ASR",
                text: rawText,
                isEnhanced: false,
                to: &items,
                seenTexts: &seenTexts
            )
        } else if !Self.hasText(rawTranscript),
                  Self.hasText(normalizedTranscript),
                  !Self.isSameText(text, normalizedText) {
            Self.appendDetailText(
                id: "original",
                label: "Original",
                text: text,
                isEnhanced: false,
                to: &items,
                seenTexts: &seenTexts
            )
        }

        Self.appendDetailText(
            id: "normalized",
            label: Self.hasText(normalizedTranscript) ? "Normalized" : "Original",
            text: normalizedText,
            isEnhanced: false,
            to: &items,
            seenTexts: &seenTexts
        )

        if let selectedText,
           !Self.isSameText(selectedText, normalizedText) {
            Self.appendDetailText(
                id: "selected",
                label: "Selected",
                text: selectedText,
                isEnhanced: false,
                to: &items,
                seenTexts: &seenTexts
            )
        }

        if let enhancedText = Self.firstNonEmpty(enhancedText) {
            Self.appendDetailText(
                id: "enhanced",
                label: "Enhanced",
                text: enhancedText,
                isEnhanced: true,
                to: &items,
                seenTexts: &seenTexts
            )
        }

        if let finalPastedText = Self.firstNonEmpty(finalPastedText) {
            Self.appendDetailText(
                id: "pasted",
                label: "Pasted",
                text: finalPastedText,
                isEnhanced: true,
                to: &items,
                seenTexts: &seenTexts
            )
        }

        return items
    }

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
        self.powerModeName = powerModeName
        self.powerModeEmoji = powerModeEmoji
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

    func recordASRMetadata(
        rawTranscript: String?,
        normalizationResult: VocoNormalizationResult,
        confidenceAssessment: VocoConfidenceAssessment,
        asrEngineID: String?,
        languageMode: String?
    ) {
        self.rawTranscript = rawTranscript
        self.normalizedTranscript = normalizationResult.normalizedText
        self.activeContextIDs = normalizationResult.activeContextIDs
        self.canonicalizationReplacements = normalizationResult.replacements
        self.canonicalizationSuggestions = normalizationResult.suggestions
        self.asrEngineID = asrEngineID
        self.languageMode = languageMode
        self.confidenceScore = confidenceAssessment.score
        self.confidenceRoute = confidenceAssessment.route.rawValue
        self.confidenceReasons = confidenceAssessment.reasons
        self.reviewTriggers = confidenceAssessment.reviewTriggers
        self.hypotheses = confidenceAssessment.candidates
        self.hypothesisLabels = confidenceAssessment.candidateLabels
        self.hypothesisDetails = confidenceAssessment.hypothesisDetails
        recordCorrectionRisk(confidenceAssessment.correctionRiskProfile)
        self.selectedCandidate = confidenceAssessment.selectedCandidate
        self.candidateSelectionSource = nil
    }

    func recordCandidateSelectionSource(_ source: VocoCandidateSelectionSource) {
        candidateSelectionSource = source.rawValue
    }

    @discardableResult
    func recordCandidateReviewFeedback(
        normalizationResult: VocoNormalizationResult,
        confidenceAssessment: VocoConfidenceAssessment,
        selectedCandidate: String,
        rawTranscript: String?
    ) -> CorrectionFeedbackSignal? {
        let signal = CorrectionFeedbackService.candidateSelectionSignal(
            normalizationResult: normalizationResult,
            assessment: confidenceAssessment,
            selectedCandidate: selectedCandidate,
            rawTranscript: rawTranscript
        )
        recordCorrectionFeedback(signal)
        return signal
    }

    func recordStyleGuardRejection(response: String, reasons: [String]) {
        styleGuardRejectedText = response
        styleGuardReasons = reasons
    }

    func recordPasteAttempt(text: String, didPostCommand: Bool) {
        finalPastedText = text
        pasteCommandPosted = didPostCommand
    }

    @discardableResult
    func recordRetranscriptionAnalysis(source: Transcription) -> CorrectionFeedbackSignal? {
        let sourceText = source.enhancedText?.isEmpty == false ? source.enhancedText! : source.text
        let newText = enhancedText?.isEmpty == false ? enhancedText! : text
        let analysis = RetranscriptionAnalyticsService.analyze(
            sourceText: sourceText,
            retranscribedText: newText,
            sourceConfidenceScore: source.confidenceScore,
            retranscribedConfidenceScore: confidenceScore
        )

        sourceTranscriptionID = source.id
        retranscriptionSourceText = sourceText
        retranscriptionAnalysis = analysis
        userCorrectionDistance = analysis.changeRatio
        let signal = CorrectionFeedbackService.retranscriptionSignal(
            sourceText: sourceText,
            retranscribedText: newText,
            analysis: analysis,
            confidenceScore: confidenceScore
        )
        recordCorrectionFeedback(signal)
        return signal
    }

    func recordCorrectionFeedback(_ signal: CorrectionFeedbackSignal?) {
        guard let signal else { return }
        var signals = correctionFeedback
        signals.append(signal)
        correctionFeedback = signals
    }

    func markAsCanceledTranscription(
        duration: TimeInterval? = nil,
        modelName: String? = nil
    ) {
        text = Self.canceledTranscriptionText
        enhancedText = nil
        transcriptionStatus = TranscriptionStatus.canceled.rawValue
        if let duration {
            self.duration = duration
        }
        if let modelName {
            transcriptionModelName = modelName
        }
        transcriptionDuration = nil
        enhancementDuration = nil
        aiEnhancementModelName = nil
        promptName = nil
        aiRequestSystemMessage = nil
        aiRequestUserMessage = nil
        normalizedTranscript = nil
        finalPastedText = nil
        pasteCommandPosted = nil
        activeContextIDs = []
        canonicalizationReplacements = []
        canonicalizationSuggestions = []
        asrEngineID = nil
        languageMode = nil
        confidenceScore = nil
        confidenceRoute = nil
        confidenceReasons = []
        reviewTriggers = []
        hypotheses = []
        hypothesisLabels = []
        hypothesisDetails = []
        correctionRiskRate = nil
        correctionRiskSampleCount = nil
        correctionRiskCorrectedCount = nil
        correctionRiskTermIDs = []
        selectedCandidate = nil
        candidateSelectionSource = nil
        userCorrectionDistance = nil
        styleGuardReasons = []
        styleGuardRejectedText = nil
        sourceTranscriptionID = nil
        retranscriptionSourceText = nil
        retranscriptionAnalysis = nil
        correctionFeedback = []
    }

    private func recordCorrectionRisk(_ profile: VocoCorrectionRiskProfile?) {
        guard let profile else {
            correctionRiskRate = nil
            correctionRiskSampleCount = nil
            correctionRiskCorrectedCount = nil
            correctionRiskTermIDs = []
            return
        }

        correctionRiskRate = profile.recentCorrectionRate
        correctionRiskSampleCount = profile.recentSessionCount
        correctionRiskCorrectedCount = profile.correctedSessionCount
        correctionRiskTermIDs = profile.highRiskTermIDs
    }

    private static func encodeJSON<T: Encodable>(_ value: T) -> String? {
        guard let data = try? JSONEncoder().encode(value) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private static func hasText(_ value: String?) -> Bool {
        value?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
    }

    private static func firstNonEmpty(_ values: String?...) -> String? {
        values
            .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
            .first { !$0.isEmpty }
    }

    private static func isSameText(_ lhs: String?, _ rhs: String?) -> Bool {
        guard let lhs = lhs?.trimmingCharacters(in: .whitespacesAndNewlines),
              let rhs = rhs?.trimmingCharacters(in: .whitespacesAndNewlines),
              !lhs.isEmpty,
              !rhs.isEmpty
        else { return false }
        return lhs == rhs
    }

    private static func appendDetailText(
        id: String,
        label: String,
        text: String,
        isEnhanced: Bool,
        to items: inout [TranscriptionDetailText],
        seenTexts: inout Set<String>
    ) {
        let displayText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !displayText.isEmpty,
              seenTexts.insert(displayText).inserted
        else { return }

        items.append(
            TranscriptionDetailText(
                id: id,
                label: label,
                text: displayText,
                isEnhanced: isEnhanced
            )
        )
    }

    private static func decodeStringArray(_ json: String?) -> [String] {
        guard let json,
              let data = json.data(using: .utf8),
              let values = try? JSONDecoder().decode([String].self, from: data)
        else {
            return []
        }
        return values
    }

    private static func decodeReplacements(_ json: String?) -> [VocoReplacement] {
        guard let json,
              let data = json.data(using: .utf8),
              let values = try? JSONDecoder().decode([VocoReplacement].self, from: data)
        else {
            return []
        }
        return values
    }

    private static func decodeReviewTriggers(_ json: String?) -> [VocoReviewTrigger] {
        guard let json,
              let data = json.data(using: .utf8),
              let values = try? JSONDecoder().decode([VocoReviewTrigger].self, from: data)
        else {
            return []
        }
        return values
    }

    private static func decodeHypotheses(_ json: String?) -> [VocoHypothesis] {
        guard let json,
              let data = json.data(using: .utf8),
              let values = try? JSONDecoder().decode([VocoHypothesis].self, from: data)
        else {
            return []
        }
        return values
    }

    private static func decodeRetranscriptionAnalysis(_ json: String?) -> RetranscriptionAnalysis? {
        guard let json,
              let data = json.data(using: .utf8)
        else {
            return nil
        }
        return try? JSONDecoder().decode(RetranscriptionAnalysis.self, from: data)
    }

    private static func decodeCorrectionFeedback(_ json: String?) -> [CorrectionFeedbackSignal] {
        guard let json,
              let data = json.data(using: .utf8),
              let values = try? JSONDecoder().decode([CorrectionFeedbackSignal].self, from: data)
        else {
            return []
        }
        return values
    }
}
