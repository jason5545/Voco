import Foundation

extension SessionMetric {
    var activeContextIDs: [String] {
        get { Self.decodeStringArray(activeContextIDsJSON) }
        set { activeContextIDsJSON = Self.encodeJSON(newValue) }
    }

    var confidenceReasons: [String] {
        get { Self.decodeStringArray(confidenceReasonsJSON) }
        set { confidenceReasonsJSON = Self.encodeJSON(newValue) }
    }

    var reviewTriggerIDs: [String] {
        get { Self.decodeStringArray(reviewTriggerIDsJSON) }
        set { reviewTriggerIDsJSON = Self.encodeJSON(newValue) }
    }

    var reviewTriggerSummaries: [String] {
        get { Self.decodeStringArray(reviewTriggerSummariesJSON) }
        set { reviewTriggerSummariesJSON = Self.encodeJSON(newValue) }
    }

    var candidateSourceCounts: [String: Int] {
        get { Self.decodeStringIntDictionary(candidateSourceCountsJSON) }
        set { candidateSourceCountsJSON = Self.encodeJSON(newValue) }
    }

    var correctionFeedbackReasons: [String] {
        get { Self.decodeStringArray(correctionFeedbackReasonsJSON) }
        set { correctionFeedbackReasonsJSON = Self.encodeJSON(newValue) }
    }

    var styleGuardReasons: [String] {
        get { Self.decodeStringArray(styleGuardReasonsJSON) }
        set { styleGuardReasonsJSON = Self.encodeJSON(newValue) }
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
        reviewTriggerCount = Self.reviewTriggerCount(from: transcription.reviewTriggers)
        reviewTriggerIDs = Self.reviewTriggerIDs(from: transcription.reviewTriggers)
        reviewTriggerSummaries = Self.reviewTriggerSummaries(from: transcription.reviewTriggers)
        candidateCount = transcription.hypotheses.count
        candidateSourceCounts = Self.candidateSourceCounts(from: transcription.hypothesisDetails)
        reviewRequiredCandidateCount = Self.reviewRequiredCandidateCount(in: transcription.hypothesisDetails)
        candidateDivergenceRatio = Self.candidateDivergenceRatio(in: transcription.hypothesisDetails)
        selectedCandidateHypothesisSource = Self.selectedCandidateHypothesisSource(
            in: transcription.hypothesisDetails,
            selectedCandidate: transcription.selectedCandidate
        )
        selectedCandidate = transcription.selectedCandidate
        candidateSelectionSource = transcription.candidateSelectionSource
        userCorrectionDistance = transcription.userCorrectionDistance
        recordFeedbackAndStyleGuardMetadata(from: transcription)
        recordRetranscriptionMetadata(from: transcription)
        recordFinalPasteMetadata(from: transcription)
    }

    func recordFeedbackAndStyleGuardMetadata(from transcription: Transcription) {
        let styleReasons = Self.styleGuardReasons(from: transcription.styleGuardReasons)

        correctionFeedbackCount = transcription.correctionFeedback.count
        correctiveFeedbackCount = transcription.correctionFeedback.filter(\.isCorrectiveSignal).count
        correctionFeedbackReasons = Self.correctionFeedbackReasons(from: transcription.correctionFeedback)
        styleGuardReasonCount = styleReasons.count
        styleGuardReasons = styleReasons
        styleGuardRejectedCharacterCount = transcription.styleGuardRejectedText?.count ?? 0
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

    static func candidateDivergenceRatio(in hypotheses: [VocoHypothesis]) -> Double? {
        let storedRatios = hypotheses.compactMap(\.divergenceFromRecommended)
        if let maxStoredRatio = storedRatios.max() {
            return maxStoredRatio
        }

        let candidates = uniqueCandidateTexts(from: hypotheses)
        guard candidates.count > 1 else { return nil }

        let base = hypotheses.first { $0.source == .autoContext }?.text
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let comparisonBase: String
        if let base, !base.isEmpty {
            comparisonBase = base
        } else {
            comparisonBase = candidates[0]
        }

        let ratios = candidates
            .filter { $0.localizedCaseInsensitiveCompare(comparisonBase) != .orderedSame }
            .map {
                RetranscriptionAnalyticsService.analyze(
                    sourceText: comparisonBase,
                    retranscribedText: $0,
                    sourceConfidenceScore: nil,
                    retranscribedConfidenceScore: nil
                ).changeRatio
            }

        guard let maxRatio = ratios.max() else { return nil }
        return maxRatio
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

        let selectedKey = candidateKey(selectedCandidate)
        return hypotheses.first { hypothesis in
            candidateKey(hypothesis.text) == selectedKey
        }?.source.rawValue
    }

    static func reviewTriggerCount(from triggers: [VocoReviewTrigger]) -> Int {
        reviewTriggerIDs(from: triggers).count
    }

    static func reviewTriggerIDs(from triggers: [VocoReviewTrigger]) -> [String] {
        uniqueReviewTriggers(from: triggers).map(\.id)
    }

    static func reviewTriggerSummaries(from triggers: [VocoReviewTrigger]) -> [String] {
        VocoReviewTriggerDisplayFormatter.summaries(for: uniqueReviewTriggers(from: triggers))
    }

    static func correctionFeedbackReasons(from signals: [CorrectionFeedbackSignal]) -> [String] {
        uniqueNonEmpty(signals.map(\.reason))
    }

    static func styleGuardReasons(from reasons: [String]) -> [String] {
        uniqueNonEmpty(reasons)
    }

    private static func uniqueReviewTriggers(from triggers: [VocoReviewTrigger]) -> [VocoReviewTrigger] {
        var seen: Set<String> = []
        return triggers.filter { seen.insert($0.id).inserted }
    }

    private static func uniqueNonEmpty(_ values: [String]) -> [String] {
        var seen: Set<String> = []
        return values
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .filter { seen.insert($0).inserted }
    }

    private static func uniqueCandidateTexts(from hypotheses: [VocoHypothesis]) -> [String] {
        var seen: Set<String> = []
        return hypotheses
            .map { $0.text.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .filter { seen.insert(candidateKey($0)).inserted }
    }

    private static func candidateKey(_ candidate: String) -> String {
        candidate
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current)
    }

    static func encodeJSON<T: Encodable>(_ value: T) -> String? {
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
