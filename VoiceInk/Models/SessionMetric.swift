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
    var powerModeName: String?
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
    var candidateCount: Int = 0
    var candidateSourceCountsJSON: String?
    var reviewRequiredCandidateCount: Int = 0
    var selectedCandidateHypothesisSource: String?
    var selectedCandidate: String?
    var candidateSelectionSource: String?
    var userCorrectionDistance: Double?
    var sourceTranscriptionID: UUID?
    var retranscriptionChangeCategory: String?
    var retranscriptionChangeRatio: Double?
    var retranscriptionEditDistance: Int?
    var retranscriptionConfidenceDelta: Double?
    var finalPastedCharacterCount: Int = 0
    var finalPastedWordCount: Int = 0
    var pasteCommandPosted: Bool?

    var activeContextIDs: [String] {
        get { Self.decodeStringArray(activeContextIDsJSON) }
        set { activeContextIDsJSON = Self.encodeJSON(newValue) }
    }

    var confidenceReasons: [String] {
        get { Self.decodeStringArray(confidenceReasonsJSON) }
        set { confidenceReasonsJSON = Self.encodeJSON(newValue) }
    }

    var candidateSourceCounts: [String: Int] {
        get { Self.decodeStringIntDictionary(candidateSourceCountsJSON) }
        set { candidateSourceCountsJSON = Self.encodeJSON(newValue) }
    }

    init(
        transcriptionId: UUID,
        timestamp: Date = Date(),
        source: String? = "recorder",
        wordCount: Int,
        audioDuration: TimeInterval,
        transcriptionModelName: String?,
        transcriptionDuration: TimeInterval?,
        speedFactor: Double?,
        powerModeName: String?,
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
        candidateCount: Int = 0,
        candidateSourceCounts: [String: Int] = [:],
        reviewRequiredCandidateCount: Int = 0,
        selectedCandidateHypothesisSource: String? = nil,
        selectedCandidate: String? = nil,
        candidateSelectionSource: String? = nil,
        userCorrectionDistance: Double? = nil,
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
        self.powerModeName = powerModeName
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
        self.candidateCount = candidateCount
        self.candidateSourceCountsJSON = Self.encodeJSON(candidateSourceCounts)
        self.reviewRequiredCandidateCount = reviewRequiredCandidateCount
        self.selectedCandidateHypothesisSource = selectedCandidateHypothesisSource
        self.selectedCandidate = selectedCandidate
        self.candidateSelectionSource = candidateSelectionSource
        self.userCorrectionDistance = userCorrectionDistance
        self.sourceTranscriptionID = sourceTranscriptionID
        self.retranscriptionChangeCategory = retranscriptionChangeCategory
        self.retranscriptionChangeRatio = retranscriptionChangeRatio
        self.retranscriptionEditDistance = retranscriptionEditDistance
        self.retranscriptionConfidenceDelta = retranscriptionConfidenceDelta
        self.finalPastedCharacterCount = finalPastedCharacterCount
        self.finalPastedWordCount = finalPastedWordCount
        self.pasteCommandPosted = pasteCommandPosted
    }

    func recordDictationMetadata(from transcription: Transcription) {
        asrEngineID = transcription.asrEngineID
        languageMode = transcription.languageMode
        activeContextIDs = transcription.activeContextIDs
        canonicalizationReplacementCount = transcription.canonicalizationReplacements.count
        canonicalizationSuggestionCount = transcription.canonicalizationSuggestions.count
        confidenceScore = transcription.confidenceScore
        confidenceRoute = transcription.confidenceRoute
        confidenceReasons = transcription.confidenceReasons
        candidateCount = transcription.hypotheses.count
        candidateSourceCounts = Self.candidateSourceCounts(from: transcription.hypothesisDetails)
        reviewRequiredCandidateCount = Self.reviewRequiredCandidateCount(in: transcription.hypothesisDetails)
        selectedCandidateHypothesisSource = Self.selectedCandidateHypothesisSource(
            in: transcription.hypothesisDetails,
            selectedCandidate: transcription.selectedCandidate
        )
        selectedCandidate = transcription.selectedCandidate
        candidateSelectionSource = transcription.candidateSelectionSource
        userCorrectionDistance = transcription.userCorrectionDistance
        recordRetranscriptionMetadata(from: transcription)
        recordFinalPasteMetadata(from: transcription)
    }

    func recordRetranscriptionMetadata(from transcription: Transcription) {
        sourceTranscriptionID = transcription.sourceTranscriptionID
        retranscriptionChangeCategory = transcription.retranscriptionAnalysis?.changeCategory.rawValue
        retranscriptionChangeRatio = transcription.retranscriptionAnalysis?.changeRatio
        retranscriptionEditDistance = transcription.retranscriptionAnalysis?.editDistance
        retranscriptionConfidenceDelta = transcription.retranscriptionAnalysis?.confidenceDelta
    }

    func recordFinalPasteMetadata(from transcription: Transcription) {
        let pastedText = transcription.finalPastedText ?? ""
        let textForCounting = pastedText.trimmingCharacters(in: .whitespacesAndNewlines)
        finalPastedCharacterCount = pastedText.count
        finalPastedWordCount = textForCounting.isEmpty ? 0 : WordCounter.count(in: textForCounting)
        if finalPastedWordCount > 0 {
            wordCount = finalPastedWordCount
        }
        pasteCommandPosted = transcription.pasteCommandPosted
    }

    static func candidateSourceCounts(from hypotheses: [VocoHypothesis]) -> [String: Int] {
        hypotheses.reduce(into: [:]) { counts, hypothesis in
            counts[hypothesis.source.rawValue, default: 0] += 1
        }
    }

    static func reviewRequiredCandidateCount(in hypotheses: [VocoHypothesis]) -> Int {
        hypotheses.filter(\.requiresReview).count
    }

    static func selectedCandidateHypothesisSource(
        in hypotheses: [VocoHypothesis],
        selectedCandidate: String?
    ) -> String? {
        guard let selectedCandidate = selectedCandidate?.trimmingCharacters(in: .whitespacesAndNewlines),
              !selectedCandidate.isEmpty
        else {
            return nil
        }

        return hypotheses.first { hypothesis in
            hypothesis.text.trimmingCharacters(in: .whitespacesAndNewlines) == selectedCandidate
        }?.source.rawValue
    }

    private static func encodeJSON<T: Encodable>(_ value: T) -> String? {
        guard let data = try? JSONEncoder().encode(value) else { return nil }
        return String(data: data, encoding: .utf8)
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

    private static func decodeStringIntDictionary(_ json: String?) -> [String: Int] {
        guard let json,
              let data = json.data(using: .utf8),
              let values = try? JSONDecoder().decode([String: Int].self, from: data)
        else {
            return [:]
        }
        return values
    }
}
