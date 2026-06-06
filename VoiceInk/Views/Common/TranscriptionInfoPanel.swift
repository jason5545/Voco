import SwiftUI
import SwiftData

/// Reusable component that displays transcription Details and AI Request sections.
/// Used in both the inline history sliding panel and the separate history window's metadata view.
struct TranscriptionInfoPanel: View {
    @Environment(\.modelContext) private var modelContext
    @State private var candidateReviewError: String?

    let transcription: Transcription

    var body: some View {
        Form {
            detailsSection
            dictationSection
            correctionFeedbackSection
            retranscriptionSection
            styleGuardSection
            aiRequestSection
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
    }

    // MARK: - Details Section

    private var detailsSection: some View {
        Section {
            metadataRow(
                icon: "calendar",
                label: "Date",
                value: transcription.timestamp.formatted(date: .abbreviated, time: .shortened)
            )

            metadataRow(
                icon: "hourglass",
                label: "Duration",
                value: transcription.duration.formatTiming()
            )

            if let modelName = transcription.transcriptionModelName {
                metadataRow(
                    icon: "cpu.fill",
                    label: "Transcription Model",
                    value: modelName
                )

                if let duration = transcription.transcriptionDuration {
                    metadataRow(
                        icon: "clock.fill",
                        label: "Transcription Time",
                        value: duration.formatTiming()
                    )
                }
            }

            if let aiModel = transcription.aiEnhancementModelName {
                metadataRow(
                    icon: "sparkles",
                    label: "Enhancement Model",
                    value: aiModel
                )

                if let duration = transcription.enhancementDuration {
                    metadataRow(
                        icon: "clock.fill",
                        label: "Enhancement Time",
                        value: duration.formatTiming()
                    )
                }
            }

            if let promptName = transcription.promptName {
                metadataRow(
                    icon: "text.bubble.fill",
                    label: "Prompt",
                    value: promptName
                )
            }

            if let powerModeValue = powerModeDisplay(
                name: transcription.powerModeName,
                emoji: transcription.powerModeEmoji
            ) {
                metadataRow(
                    icon: "bolt.fill",
                    label: "Power Mode",
                    value: powerModeValue
                )
            }
        } header: {
            Text("Details")
        }
    }

    // MARK: - Dictation Section

    @ViewBuilder
    private var dictationSection: some View {
        if hasDictationMetadata {
            Section {
                if let asrEngineID = transcription.asrEngineID, !asrEngineID.isEmpty {
                    metadataRow(
                        icon: "waveform",
                        label: "ASR Engine",
                        value: asrEngineID
                    )
                }

                if let languageMode = transcription.languageMode, !languageMode.isEmpty {
                    metadataRow(
                        icon: "globe.asia.australia.fill",
                        label: "Language",
                        value: languageMode
                    )
                }

                if let finalPastedText = transcription.finalPastedText, !finalPastedText.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Final Pasted Text")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundColor(.secondary)
                        Text(finalPastedText)
                            .font(.system(size: 12, weight: .regular))
                            .lineSpacing(2)
                            .textSelection(.enabled)
                            .foregroundColor(.primary)
                    }
                }

                if let pasteCommandPosted = transcription.pasteCommandPosted {
                    metadataRow(
                        icon: "doc.on.clipboard",
                        label: "Paste Command",
                        value: pasteCommandPosted ? "Posted" : "Not posted"
                    )
                }

                if let confidenceScore = transcription.confidenceScore {
                    metadataRow(
                        icon: "gauge.with.dots.needle.bottom.50percent",
                        label: "Confidence",
                        value: confidenceDisplay(score: confidenceScore)
                    )
                }

                if let route = transcription.confidenceRoute, !route.isEmpty {
                    metadataRow(
                        icon: "arrow.triangle.branch",
                        label: "Route",
                        value: routeDisplay(route)
                    )
                }

                if let source = transcription.candidateSelectionSource, !source.isEmpty {
                    metadataRow(
                        icon: "cursorarrow.click.2",
                        label: "Selection Source",
                        value: selectionSourceDisplay(source)
                    )
                }

                if let correctionRiskRate = transcription.correctionRiskRate {
                    metadataRow(
                        icon: "clock.arrow.circlepath",
                        label: "Recent Correction Risk",
                        value: correctionRiskDisplay(
                            rate: correctionRiskRate,
                            sampleCount: transcription.correctionRiskSampleCount,
                            correctedCount: transcription.correctionRiskCorrectedCount
                        )
                    )
                }

                if !transcription.correctionRiskTermIDs.isEmpty {
                    metadataRow(
                        icon: "tag.fill",
                        label: "Risk Terms",
                        value: transcription.correctionRiskTermIDs.joined(separator: ", ")
                    )
                }

                if !transcription.confidenceReasons.isEmpty {
                    metadataRow(
                        icon: "exclamationmark.triangle.fill",
                        label: "Signals",
                        value: VocoSignalDisplayFormatter
                            .displayReasons(for: transcription.confidenceReasons)
                            .joined(separator: ", ")
                    )
                }

                if !transcription.activeContextIDs.isEmpty {
                    metadataRow(
                        icon: "square.stack.3d.up.fill",
                        label: "Contexts",
                        value: VocoCanonicalizationService.contextDisplayNames(for: transcription.activeContextIDs).joined(separator: ", ")
                    )
                }

                if !transcription.canonicalizationReplacements.isEmpty {
                    replacementList(
                        title: "Replacements",
                        replacements: transcription.canonicalizationReplacements
                    )
                }

                if !transcription.canonicalizationSuggestions.isEmpty {
                    replacementList(
                        title: "Suggestions",
                        replacements: transcription.canonicalizationSuggestions
                    )
                }

                if !transcription.hypotheses.isEmpty {
                    candidateList(
                        candidates: transcription.hypotheses,
                        labels: transcription.hypothesisLabels,
                        hypotheses: transcription.hypothesisDetails
                    )
                }
            } header: {
                Text("Dictation")
            }
        }
    }

    // MARK: - AI Request Section

    @ViewBuilder
    private var correctionFeedbackSection: some View {
        if !transcription.correctionFeedback.isEmpty {
            Section {
                ForEach(Array(transcription.correctionFeedback.enumerated()), id: \.offset) { _, signal in
                    feedbackItem(signal)
                }
            } header: {
                Text("Correction Feedback")
            }
        }
    }

    @ViewBuilder
    private var retranscriptionSection: some View {
        if transcription.sourceTranscriptionID != nil || transcription.retranscriptionAnalysis != nil {
            Section {
                if let sourceID = transcription.sourceTranscriptionID {
                    metadataRow(
                        icon: "arrow.triangle.2.circlepath",
                        label: "Source",
                        value: shortID(sourceID)
                    )
                }

                if let analysis = transcription.retranscriptionAnalysis {
                    metadataRow(
                        icon: "ruler",
                        label: "Change",
                        value: "\(Int((analysis.changeRatio * 100).rounded()))% (\(analysis.changeCategory.displayName))"
                    )

                    metadataRow(
                        icon: "number",
                        label: "Edit Distance",
                        value: "\(analysis.editDistance)"
                    )

                    if let confidenceDelta = analysis.confidenceDelta {
                        metadataRow(
                            icon: "gauge.with.dots.needle.bottom.50percent",
                            label: "Confidence Delta",
                            value: signedPercent(confidenceDelta)
                        )
                    }
                }

                if let sourceText = transcription.retranscriptionSourceText, !sourceText.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Source Text")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundColor(.secondary)
                        Text(sourceText)
                            .font(.system(size: 12, weight: .regular))
                            .lineSpacing(2)
                            .textSelection(.enabled)
                            .foregroundColor(.primary)
                    }
                }
            } header: {
                Text("Retranscription")
            }
        }
    }

    @ViewBuilder
    private var styleGuardSection: some View {
        if !transcription.styleGuardReasons.isEmpty || transcription.styleGuardRejectedText != nil {
            Section {
                if !transcription.styleGuardReasons.isEmpty {
                    metadataRow(
                        icon: "shield.lefthalf.filled",
                        label: "Reason",
                        value: transcription.styleGuardReasons.joined(separator: ", ")
                    )
                }

                if let rejectedText = transcription.styleGuardRejectedText, !rejectedText.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Rejected Output")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundColor(.secondary)
                        Text(rejectedText)
                            .font(.system(size: 12, weight: .regular))
                            .lineSpacing(2)
                            .textSelection(.enabled)
                            .foregroundColor(.primary)
                    }
                }
            } header: {
                Text("Style Guard")
            }
        }
    }

    @ViewBuilder
    private var aiRequestSection: some View {
        if transcription.aiRequestSystemMessage != nil || transcription.aiRequestUserMessage != nil {
            Section {
                if let systemMsg = transcription.aiRequestSystemMessage, !systemMsg.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("System Prompt")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundColor(.secondary)
                        Text(systemMsg)
                            .font(.system(size: 11, weight: .regular, design: .monospaced))
                            .lineSpacing(2)
                            .textSelection(.enabled)
                            .foregroundColor(.primary)
                    }
                }

                if let userMsg = transcription.aiRequestUserMessage, !userMsg.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("User Message")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundColor(.secondary)
                        Text(userMsg)
                            .font(.system(size: 11, weight: .regular, design: .monospaced))
                            .lineSpacing(2)
                            .textSelection(.enabled)
                            .foregroundColor(.primary)
                    }
                }
            } header: {
                HStack {
                    Text("AI Request")
                    Spacer()
                    CopyIconButton(textToCopy: fullRequestText)
                }
            }
        }
    }

    // MARK: - Helpers

    private var fullRequestText: String {
        var parts: [String] = []
        if let sys = transcription.aiRequestSystemMessage, !sys.isEmpty {
            parts.append("System Prompt:\n\(sys)")
        }
        if let user = transcription.aiRequestUserMessage, !user.isEmpty {
            parts.append("User Message:\n\(user)")
        }
        return parts.joined(separator: "\n\n")
    }

    private var hasDictationMetadata: Bool {
        transcription.rawTranscript != nil ||
        transcription.normalizedTranscript != nil ||
        transcription.finalPastedText != nil ||
        transcription.pasteCommandPosted != nil ||
        transcription.asrEngineID != nil ||
        transcription.languageMode != nil ||
        transcription.confidenceScore != nil ||
        transcription.correctionRiskRate != nil ||
        !transcription.correctionRiskTermIDs.isEmpty ||
        transcription.candidateSelectionSource != nil ||
        !transcription.activeContextIDs.isEmpty ||
        !transcription.canonicalizationReplacements.isEmpty ||
        !transcription.canonicalizationSuggestions.isEmpty ||
        !transcription.hypotheses.isEmpty
    }

    private func metadataRow(icon: String, label: String, value: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.secondary)
                .frame(width: 20, height: 20)

            Text(label)
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.secondary)

            Spacer(minLength: 0)

            Text(value)
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(.primary)
                .lineLimit(2)
                .multilineTextAlignment(.trailing)
        }
    }

    private func replacementList(title: String, replacements: [VocoReplacement]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(.secondary)

            ForEach(Array(replacements.enumerated()), id: \.offset) { _, replacement in
                VStack(alignment: .leading, spacing: 3) {
                    Text("\(replacement.originalText) -> \(replacement.replacementText)")
                        .font(.system(size: 12, weight: .semibold))
                        .textSelection(.enabled)

                    Text(
                        "\(confidenceDisplay(score: replacement.confidence)) · " +
                            VocoSignalDisplayFormatter.displayReason(for: replacement.reason)
                    )
                        .font(.system(size: 11, weight: .regular))
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding(.vertical, 4)
    }

    private func candidateList(candidates: [String], labels: [String], hypotheses: [VocoHypothesis]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Candidates")
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(.secondary)

            ForEach(Array(candidates.enumerated()), id: \.offset) { index, candidate in
                HStack(alignment: .top, spacing: 8) {
                    Text("\(index + 1)")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(.secondary)
                        .frame(width: 18, height: 18)
                        .background(Circle().fill(Color.secondary.opacity(0.12)))

                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 5) {
                            Text(candidateLabel(at: index, labels: labels))
                                .font(.system(size: 10, weight: .semibold))
                                .foregroundColor(.secondary)

                            if let source = hypothesisSource(at: index, hypotheses: hypotheses) {
                                Text(source)
                                    .font(.system(size: 10, weight: .regular))
                                    .foregroundColor(.secondary.opacity(0.75))
                                    .lineLimit(1)
                            }
                        }

                        Text(candidate)
                            .font(.system(size: 12, weight: .regular))
                            .textSelection(.enabled)

                        if let summary = hypothesisSummary(at: index, hypotheses: hypotheses) {
                            Text(summary)
                                .font(.system(size: 10, weight: .regular))
                                .foregroundColor(.secondary)
                                .lineLimit(2)
                                .textSelection(.enabled)
                        }
                    }

                    Spacer(minLength: 0)

                    candidateActionButton(candidate)
                }
            }

            if let candidateReviewError {
                Text(candidateReviewError)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.red)
            }
        }
        .padding(.vertical, 4)
    }

    private func candidateActionButton(_ candidate: String) -> some View {
        Button {
            acceptCandidate(candidate)
        } label: {
            Image(systemName: isSelectedCandidate(candidate) ? "checkmark.circle.fill" : "checkmark.circle")
                .font(.system(size: 15, weight: .semibold))
                .foregroundColor(isSelectedCandidate(candidate) ? .green : .accentColor)
                .frame(width: 24, height: 24)
                .contentShape(Rectangle())
        }
        .buttonStyle(.borderless)
        .help(isSelectedCandidate(candidate) ? "Selected candidate" : "Accept candidate")
    }

    private func acceptCandidate(_ candidate: String) {
        candidateReviewError = nil
        let feedbackSignal = VocoCandidateReviewService.acceptPersistedCandidate(
            candidate,
            for: transcription
        )
        CorrectionFeedbackLearningService.stageLearningCandidates(
            from: feedbackSignal,
            in: modelContext
        )

        do {
            try modelContext.save()
        } catch {
            modelContext.rollback()
            candidateReviewError = error.localizedDescription
        }
    }

    private func isSelectedCandidate(_ candidate: String) -> Bool {
        transcription.selectedCandidate == candidate
    }

    private func candidateLabel(at index: Int, labels: [String]) -> String {
        guard labels.indices.contains(index) else { return "Candidate" }
        return labels[index]
    }

    private func hypothesisSource(at index: Int, hypotheses: [VocoHypothesis]) -> String? {
        guard hypotheses.indices.contains(index) else { return nil }
        return hypotheses[index].sourceDisplayName
    }

    private func hypothesisSummary(at index: Int, hypotheses: [VocoHypothesis]) -> String? {
        guard hypotheses.indices.contains(index) else { return nil }
        return VocoHypothesisDisplayFormatter.summary(for: hypotheses[index])
    }

    private func feedbackItem(_ signal: CorrectionFeedbackSignal) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            metadataRow(
                icon: feedbackIcon(for: signal.kind),
                label: signal.kind.displayName,
                value: feedbackSummary(signal)
            )

            feedbackText(label: "Source", text: signal.sourceText)

            if let proposedText = signal.proposedText, !proposedText.isEmpty {
                feedbackText(label: "Proposed", text: proposedText)
            }

            feedbackText(label: "Accepted", text: signal.acceptedText)

            if !signal.termIDs.isEmpty {
                metadataRow(
                    icon: "tag.fill",
                    label: "Terms",
                    value: signal.termIDs.joined(separator: ", ")
                )
            }
        }
        .padding(.vertical, 4)
    }

    private func feedbackText(label: String, text: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(.secondary)
            Text(text)
                .font(.system(size: 12, weight: .regular))
                .lineSpacing(2)
                .textSelection(.enabled)
                .foregroundColor(.primary)
        }
    }

    private func feedbackIcon(for kind: CorrectionFeedbackKind) -> String {
        switch kind {
        case .candidateSelection:
            return "checklist.checked"
        case .retranscriptionChange:
            return "arrow.triangle.2.circlepath"
        case .userSubstitution:
            return "text.badge.checkmark"
        }
    }

    private func feedbackSummary(_ signal: CorrectionFeedbackSignal) -> String {
        var parts = [VocoSignalDisplayFormatter.displayReason(for: signal.reason)]
        if let confidenceScore = signal.confidenceScore {
            parts.append(confidenceDisplay(score: confidenceScore))
        }
        if let changeRatio = signal.changeRatio {
            parts.append("change \(Int((changeRatio * 100).rounded()))%")
        }
        return parts.joined(separator: " · ")
    }

    private func confidenceDisplay(score: Double) -> String {
        "\(Int((score * 100).rounded()))%"
    }

    private func correctionRiskDisplay(rate: Double, sampleCount: Int?, correctedCount: Int?) -> String {
        let rateText = confidenceDisplay(score: rate)
        guard let sampleCount, let correctedCount else { return rateText }
        return "\(rateText) · \(correctedCount)/\(sampleCount) corrected"
    }

    private func routeDisplay(_ route: String) -> String {
        switch VocoConfidenceRoute(rawValue: route) {
        case .directInsertion:
            return "Direct insertion"
        case .reviewSuggested:
            return "Review suggested"
        case .none:
            return route
        }
    }

    private func selectionSourceDisplay(_ source: String) -> String {
        VocoCandidateSelectionSource(rawValue: source)?.displayName ?? source
    }

    private func shortID(_ id: UUID) -> String {
        String(id.uuidString.prefix(8))
    }

    private func signedPercent(_ value: Double) -> String {
        let percent = Int((value * 100).rounded())
        return percent > 0 ? "+\(percent)%" : "\(percent)%"
    }

    private func powerModeDisplay(name: String?, emoji: String?) -> String? {
        guard name != nil || emoji != nil else { return nil }

        switch (emoji?.trimmingCharacters(in: .whitespacesAndNewlines), name?.trimmingCharacters(in: .whitespacesAndNewlines)) {
        case let (.some(emojiValue), .some(nameValue)) where !emojiValue.isEmpty && !nameValue.isEmpty:
            return "\(emojiValue) \(nameValue)"
        case let (.some(emojiValue), _) where !emojiValue.isEmpty:
            return emojiValue
        case let (_, .some(nameValue)) where !nameValue.isEmpty:
            return nameValue
        default:
            return nil
        }
    }
}
