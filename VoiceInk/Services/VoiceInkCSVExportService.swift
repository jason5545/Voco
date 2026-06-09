
import Foundation
import AppKit
import SwiftData

class VoiceInkCSVExportService {

    func exportTranscriptionsToCSV(transcriptions: [Transcription]) {
        let csvString = generateCSV(for: transcriptions)

        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.commaSeparatedText]
        savePanel.nameFieldStringValue = "Voco-transcription.csv"

        savePanel.begin { result in
            if result == .OK, let url = savePanel.url {
                do {
                    try csvString.write(to: url, atomically: true, encoding: .utf8)
                } catch {
                    print("Error writing CSV file: \(error)")
                }
            }
        }
    }

    func generateCSV(for transcriptions: [Transcription]) -> String {
        var csvString = csvRow(Self.headers) + "\n"

        for transcription in transcriptions {
            csvString.append(csvRow(values(for: transcription)))
            csvString.append("\n")
        }

        return csvString
    }

    private static let headers = [
        "Original Transcript",
        "Raw Transcript",
        "Normalized Transcript",
        "Enhanced Transcript",
        "Final Pasted Text",
        "Paste Command Posted",
        "Enhancement Model",
        "Prompt Name",
        "Transcription Model",
        "ASR Engine ID",
        "Language Mode",
        "Power Mode",
        "Active Context IDs",
        "Active Contexts",
        "Canonicalization Replacements",
        "Canonicalization Suggestions",
        "Confidence Score",
        "Confidence Route",
        "Confidence Reasons",
        "Review Triggers",
        "Candidate Labels",
        "Candidates",
        "Candidate Details",
        "Candidate Source Counts",
        "Review Required Candidates",
        "Candidate Divergence Ratio",
        "Selected Candidate",
        "Selected Candidate Source",
        "Candidate Selection Source",
        "User Correction Distance",
        "Correction Feedback",
        "Style Guard Reasons",
        "Style Guard Rejected Text",
        "Retranscription Source ID",
        "Retranscription Source Text",
        "Retranscription Change Category",
        "Retranscription Change Ratio",
        "Retranscription Edit Distance",
        "Retranscription Confidence Delta",
        "Enhancement Time",
        "Transcription Time",
        "Timestamp",
        "Duration",
    ]

    private func values(for transcription: Transcription) -> [String] {
        let retranscriptionAnalysis = transcription.retranscriptionAnalysis

        return [
            transcription.text,
            transcription.rawTranscript ?? "",
            transcription.normalizedTranscript ?? "",
            transcription.enhancedText ?? "",
            transcription.finalPastedText ?? "",
            boolString(transcription.pasteCommandPosted),
            transcription.aiEnhancementModelName ?? "",
            transcription.promptName ?? "",
            transcription.transcriptionModelName ?? "",
            transcription.asrEngineID ?? "",
            transcription.languageMode ?? "",
            modeDisplay(name: transcription.modeName, emoji: transcription.modeEmoji),
            joined(transcription.activeContextIDs),
            joined(VocoCanonicalizationService.contextDisplayNames(for: transcription.activeContextIDs)),
            replacementSummary(transcription.canonicalizationReplacements),
            replacementSummary(transcription.canonicalizationSuggestions),
            percent(transcription.confidenceScore),
            transcription.confidenceRoute ?? "",
            joined(VocoSignalDisplayFormatter.displayReasons(for: transcription.confidenceReasons)),
            reviewTriggerSummary(transcription.reviewTriggers),
            joined(transcription.hypothesisLabels),
            candidateSummary(labels: transcription.hypothesisLabels, candidates: transcription.hypotheses),
            candidateDetailSummary(labels: transcription.hypothesisLabels, hypotheses: transcription.hypothesisDetails),
            candidateSourceSummary(transcription.hypothesisDetails),
            reviewRequiredCandidateCount(transcription.hypothesisDetails),
            decimal(SessionMetric.candidateDivergenceRatio(in: transcription.hypothesisDetails)),
            transcription.selectedCandidate ?? "",
            selectedCandidateSource(
                hypotheses: transcription.hypothesisDetails,
                selectedCandidate: transcription.selectedCandidate
            ),
            selectionSourceDisplay(transcription.candidateSelectionSource),
            decimal(transcription.userCorrectionDistance),
            correctionFeedbackSummary(transcription.correctionFeedback),
            joined(transcription.styleGuardReasons),
            transcription.styleGuardRejectedText ?? "",
            transcription.sourceTranscriptionID?.uuidString ?? "",
            transcription.retranscriptionSourceText ?? "",
            retranscriptionAnalysis?.changeCategory.rawValue ?? "",
            decimal(retranscriptionAnalysis?.changeRatio),
            retranscriptionAnalysis.map { "\($0.editDistance)" } ?? "",
            decimal(retranscriptionAnalysis?.confidenceDelta),
            decimal(transcription.enhancementDuration),
            decimal(transcription.transcriptionDuration),
            transcription.timestamp.ISO8601Format(),
            decimal(transcription.duration),
        ]
    }

    private func csvRow(_ values: [String]) -> String {
        values
            .map(escapeCSVString)
            .joined(separator: ",")
    }

    private func escapeCSVString(_ string: String) -> String {
        let escapedString = string.replacingOccurrences(of: "\"", with: "\"\"")
        if escapedString.contains(",") || escapedString.contains("\n") || escapedString.contains("\"") {
            return "\"\(escapedString)\""
        }
        return escapedString
    }

    private func modeDisplay(name: String?, emoji: String?) -> String {
        switch (emoji?.trimmingCharacters(in: .whitespacesAndNewlines), name?.trimmingCharacters(in: .whitespacesAndNewlines)) {
        case let (.some(emojiValue), .some(nameValue)) where !emojiValue.isEmpty && !nameValue.isEmpty:
            return "\(emojiValue) \(nameValue)"
        case let (.some(emojiValue), _) where !emojiValue.isEmpty:
            return emojiValue
        case let (_, .some(nameValue)) where !nameValue.isEmpty:
            return nameValue
        default:
            return ""
        }
    }

    private func replacementSummary(_ replacements: [VocoReplacement]) -> String {
        replacements
            .map { replacement in
                "\(replacement.originalText) -> \(replacement.replacementText) [\(replacement.termID), \(percent(replacement.confidence)), \(replacement.reason)]"
            }
            .joined(separator: " | ")
    }

    private func candidateSummary(labels: [String], candidates: [String]) -> String {
        candidates.enumerated()
            .map { index, candidate in
                let label = labels.indices.contains(index) ? labels[index] : "Candidate"
                return "\(label): \(candidate)"
            }
            .joined(separator: " | ")
    }

    private func reviewTriggerSummary(_ triggers: [VocoReviewTrigger]) -> String {
        joined(VocoReviewTriggerDisplayFormatter.summaries(for: triggers))
    }

    private func candidateDetailSummary(labels: [String], hypotheses: [VocoHypothesis]) -> String {
        hypotheses.enumerated()
            .map { index, hypothesis in
                let label = labels.indices.contains(index) ? labels[index] : hypothesis.label
                let summary = VocoHypothesisDisplayFormatter.summary(for: hypothesis)
                let prefix = "\(label) / \(hypothesis.sourceDisplayName)"
                guard let summary, !summary.isEmpty else { return prefix }
                return "\(prefix): \(summary)"
            }
            .joined(separator: " | ")
    }

    private func correctionFeedbackSummary(_ signals: [CorrectionFeedbackSignal]) -> String {
        signals
            .map { signal in
                var metadata = [
                    signal.kind.displayName,
                    VocoSignalDisplayFormatter.displayReason(for: signal.reason),
                ]

                if let confidenceScore = signal.confidenceScore {
                    metadata.append(percent(confidenceScore))
                }
                if let changeRatio = signal.changeRatio {
                    metadata.append("change \(percent(changeRatio))")
                }
                if !signal.termIDs.isEmpty {
                    metadata.append("Terms \(signal.termIDs.joined(separator: ", "))")
                }
                metadata.append(signal.createdAt.ISO8601Format())

                var textParts = ["Source: \(signal.sourceText)"]
                if let proposedText = signal.proposedText, !proposedText.isEmpty {
                    textParts.append("Proposed: \(proposedText)")
                }
                textParts.append("Accepted: \(signal.acceptedText)")

                return "\(metadata.joined(separator: " · ")) - \(textParts.joined(separator: "; "))"
            }
            .joined(separator: " | ")
    }

    private func candidateSourceSummary(_ hypotheses: [VocoHypothesis]) -> String {
        let counts = SessionMetric.candidateSourceCounts(from: hypotheses)
        return sortedSourceCounts(counts)
            .map { "\(sourceDisplayName($0.key)): \($0.value)" }
            .joined(separator: " | ")
    }

    private func reviewRequiredCandidateCount(_ hypotheses: [VocoHypothesis]) -> String {
        let count = SessionMetric.reviewRequiredCandidateCount(in: hypotheses)
        return count > 0 ? "\(count)" : ""
    }

    private func selectedCandidateSource(
        hypotheses: [VocoHypothesis],
        selectedCandidate: String?
    ) -> String {
        guard let source = SessionMetric.selectedCandidateHypothesisSource(
            in: hypotheses,
            selectedCandidate: selectedCandidate
        ) else {
            return ""
        }
        return sourceDisplayName(source)
    }

    private func sortedSourceCounts(_ counts: [String: Int]) -> [(key: String, value: Int)] {
        counts
            .filter { $0.value > 0 }
            .sorted { lhs, rhs in
                if lhs.value != rhs.value {
                    return lhs.value > rhs.value
                }

                let lhsOrder = sourceSortOrder(lhs.key)
                let rhsOrder = sourceSortOrder(rhs.key)
                if lhsOrder != rhsOrder {
                    return lhsOrder < rhsOrder
                }

                return sourceDisplayName(lhs.key) < sourceDisplayName(rhs.key)
            }
    }

    private func sourceDisplayName(_ source: String) -> String {
        VocoHypothesisSource(rawValue: source)?.displayName ?? source
    }

    private func sourceSortOrder(_ source: String) -> Int {
        VocoHypothesisSource(rawValue: source)?.analyticsSortPriority ?? 99
    }

    private func joined(_ values: [String]) -> String {
        values.joined(separator: " | ")
    }

    private func percent(_ value: Double?) -> String {
        guard let value else { return "" }
        return "\(Int((value * 100).rounded()))%"
    }

    private func decimal(_ value: Double?) -> String {
        guard let value else { return "" }
        return decimal(value)
    }

    private func decimal(_ value: Double) -> String {
        String(format: "%.3f", value)
    }

    private func boolString(_ value: Bool?) -> String {
        guard let value else { return "" }
        return value ? "true" : "false"
    }

    private func selectionSourceDisplay(_ rawValue: String?) -> String {
        guard let rawValue, !rawValue.isEmpty else { return "" }
        return VocoCandidateSelectionSource(rawValue: rawValue)?.displayName ?? rawValue
    }
}
