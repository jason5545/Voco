import SwiftUI

struct TranscriptionListItem: View {
    let transcription: Transcription
    let isSelected: Bool
    let isChecked: Bool
    let onSelect: () -> Void
    let onToggleCheck: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Toggle("", isOn: Binding(
                get: { isChecked },
                set: { _ in onToggleCheck() }
            ))
            .toggleStyle(CircularCheckboxStyle())
            .labelsHidden()

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(transcription.timestamp, format: .dateTime.month(.abbreviated).day().hour().minute())
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.secondary)
                    Spacer()
                    if transcription.duration > 0 {
                        Text(transcription.duration.formatTiming())
                            .font(.system(size: 10, weight: .medium))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 3)
                            .background(
                                RoundedRectangle(cornerRadius: 4, style: .continuous)
                                    .fill(AppTheme.Surface.card)
                            )
                            .foregroundColor(.secondary)
                    }
                }

                Text(transcription.historyDisplayText)
                    .font(.system(size: 12, weight: .regular))
                    .lineLimit(2)
                    .foregroundColor(.primary)

                TranscriptionAssistiveBadgeRow(transcription: transcription)
            }
        }
        .padding(10)
        .background {
            if isSelected {
                RoundedRectangle(cornerRadius: AppTheme.Radius.card, style: .continuous)
                    .fill(AppTheme.Selection.fill)
                    .overlay {
                        RoundedRectangle(cornerRadius: AppTheme.Radius.card, style: .continuous)
                            .strokeBorder(AppTheme.Selection.border, lineWidth: 1)
                    }
            } else {
                RoundedRectangle(cornerRadius: AppTheme.Radius.card, style: .continuous)
                    .fill(AppTheme.Surface.subtle)
                    .overlay {
                        RoundedRectangle(cornerRadius: AppTheme.Radius.card, style: .continuous)
                            .strokeBorder(AppTheme.Border.tint, lineWidth: 1)
                    }
            }
        }
        .contentShape(Rectangle())
        .onTapGesture { onSelect() }
    }
}

struct TranscriptionAssistiveBadge: Equatable, Identifiable {
    enum Tone: Equatable {
        case accent
        case green
        case orange
        case purple
        case secondary

        var color: Color {
            switch self {
            case .accent:
                return .accentColor
            case .green:
                return .green
            case .orange:
                return .orange
            case .purple:
                return .purple
            case .secondary:
                return .secondary
            }
        }
    }

    let id: String
    let icon: String
    let title: String
    let tone: Tone

    static func badges(for transcription: Transcription, limit: Int = 3) -> [TranscriptionAssistiveBadge] {
        var badges: [TranscriptionAssistiveBadge] = []

        if let reviewBadge = reviewBadge(for: transcription) {
            badges.append(reviewBadge)
        }

        if let selectionSource = VocoCandidateSelectionSource(rawValue: transcription.candidateSelectionSource ?? "") {
            badges.append(selectionBadge(for: selectionSource))
        }

        if let retranscriptionBadge = retranscriptionBadge(for: transcription.retranscriptionAnalysis) {
            badges.append(retranscriptionBadge)
        }

        if let feedbackBadge = correctionFeedbackBadge(for: transcription.correctionFeedback) {
            badges.append(feedbackBadge)
        }

        if let styleGuardBadge = styleGuardBadge(for: transcription) {
            badges.append(styleGuardBadge)
        }

        let replacementCount = transcription.canonicalizationReplacements.count
        if replacementCount > 0 {
            badges.append(
                TranscriptionAssistiveBadge(
                    id: "canonicalization-replacements",
                    icon: "text.badge.checkmark",
                    title: countLabel(replacementCount, singular: "fix", plural: "fixes"),
                    tone: .accent
                )
            )
        }

        let suggestionCount = transcription.canonicalizationSuggestions.count
        if suggestionCount > 0 {
            badges.append(
                TranscriptionAssistiveBadge(
                    id: "canonicalization-suggestions",
                    icon: "questionmark.bubble.fill",
                    title: countLabel(suggestionCount, singular: "choice", plural: "choices"),
                    tone: .orange
                )
            )
        }

        if !transcription.activeContextIDs.isEmpty {
            badges.append(
                TranscriptionAssistiveBadge(
                    id: "contexts",
                    icon: "square.stack.3d.up.fill",
                    title: countLabel(transcription.activeContextIDs.count, singular: "context", plural: "contexts"),
                    tone: .secondary
                )
            )
        }

        guard limit > 0 else { return [] }
        return Array(badges.prefix(limit))
    }

    private static func selectionBadge(for source: VocoCandidateSelectionSource) -> TranscriptionAssistiveBadge {
        switch source {
        case .userSelection:
            return TranscriptionAssistiveBadge(
                id: "candidate-user-selection",
                icon: "checkmark.circle.fill",
                title: "Selected",
                tone: .green
            )
        case .dismissedFallback:
            return TranscriptionAssistiveBadge(
                id: "candidate-dismissed-fallback",
                icon: "xmark.circle.fill",
                title: "Dismissed",
                tone: .secondary
            )
        case .timeoutFallback:
            return TranscriptionAssistiveBadge(
                id: "candidate-timeout-fallback",
                icon: "clock.arrow.circlepath",
                title: "Timeout",
                tone: .orange
            )
        case .automaticFallback:
            return TranscriptionAssistiveBadge(
                id: "candidate-automatic-fallback",
                icon: "arrow.uturn.backward.circle.fill",
                title: "Auto",
                tone: .secondary
            )
        case .finalPaste:
            return TranscriptionAssistiveBadge(
                id: "candidate-final-paste",
                icon: "doc.on.clipboard.fill",
                title: "Pasted",
                tone: .accent
            )
        }
    }

    private static func correctionFeedbackBadge(for signals: [CorrectionFeedbackSignal]) -> TranscriptionAssistiveBadge? {
        guard !signals.isEmpty else { return nil }

        let correctiveCount = signals.filter(\.isCorrectiveSignal).count
        if correctiveCount > 0 {
            return TranscriptionAssistiveBadge(
                id: "correction-feedback",
                icon: "checklist.checked",
                title: countLabel(correctiveCount, singular: "correction", plural: "corrections"),
                tone: .green
            )
        }

        return TranscriptionAssistiveBadge(
            id: "correction-feedback-passive",
            icon: "checklist",
            title: countLabel(signals.count, singular: "feedback signal", plural: "feedback signals"),
            tone: .secondary
        )
    }

    private static func styleGuardBadge(for transcription: Transcription) -> TranscriptionAssistiveBadge? {
        let reasonCount = transcription.styleGuardReasons.count
        guard reasonCount > 0 || transcription.styleGuardRejectedText?.isEmpty == false else {
            return nil
        }

        return TranscriptionAssistiveBadge(
            id: "style-guard",
            icon: "shield.lefthalf.filled",
            title: reasonCount > 0 ? countLabel(reasonCount, singular: "style flag", plural: "style flags") : "Style guard",
            tone: .purple
        )
    }

    private static func reviewBadge(for transcription: Transcription) -> TranscriptionAssistiveBadge? {
        guard transcription.confidenceRoute == VocoConfidenceRoute.reviewSuggested.rawValue else {
            return nil
        }

        return TranscriptionAssistiveBadge(
            id: "review",
            icon: "exclamationmark.triangle.fill",
            title: reviewBadgeTitle(for: transcription.reviewTriggers),
            tone: .orange
        )
    }

    private static func reviewBadgeTitle(for triggers: [VocoReviewTrigger]) -> String {
        let displayNames = uniqueReviewTriggerDisplayNames(for: triggers)
        switch displayNames.count {
        case 0:
            return "Review"
        case 1:
            return displayNames[0]
        default:
            return countLabel(displayNames.count, singular: "signal", plural: "signals")
        }
    }

    private static func uniqueReviewTriggerDisplayNames(for triggers: [VocoReviewTrigger]) -> [String] {
        var seenIDs: Set<String> = []
        var seenNames: Set<String> = []
        return triggers
            .filter { seenIDs.insert($0.id).inserted }
            .map(\.displayName)
            .filter { seenNames.insert($0).inserted }
    }

    private static func retranscriptionBadge(for analysis: RetranscriptionAnalysis?) -> TranscriptionAssistiveBadge? {
        guard let analysis else { return nil }

        switch analysis.changeCategory {
        case .unchanged:
            return TranscriptionAssistiveBadge(
                id: "retranscription-unchanged",
                icon: "arrow.triangle.2.circlepath",
                title: "Re-run same",
                tone: .secondary
            )
        case .minorChange:
            return TranscriptionAssistiveBadge(
                id: "retranscription-minor",
                icon: "arrow.triangle.2.circlepath",
                title: "Minor \(percent(analysis.changeRatio))",
                tone: .secondary
            )
        case .meaningfulChange:
            return TranscriptionAssistiveBadge(
                id: "retranscription-meaningful",
                icon: "arrow.triangle.2.circlepath",
                title: "Re-run \(percent(analysis.changeRatio))",
                tone: .purple
            )
        }
    }

    private static func countLabel(_ count: Int, singular: String, plural: String) -> String {
        "\(count) \(count == 1 ? singular : plural)"
    }

    private static func percent(_ value: Double) -> String {
        "\(Int((value * 100).rounded()))%"
    }
}

struct TranscriptionAssistiveBadgeRow: View {
    private let badges: [TranscriptionAssistiveBadge]

    init(transcription: Transcription) {
        badges = TranscriptionAssistiveBadge.badges(for: transcription)
    }

    var body: some View {
        if !badges.isEmpty {
            HStack(spacing: 6) {
                ForEach(badges) { badge in
                    Label {
                        Text(badge.title)
                            .font(.system(size: 10, weight: .medium))
                            .lineLimit(1)
                            .minimumScaleFactor(0.85)
                    } icon: {
                        Image(systemName: badge.icon)
                            .font(.system(size: 9, weight: .semibold))
                    }
                    .foregroundStyle(badge.tone.color)
                    .labelStyle(.titleAndIcon)
                    .help(badge.title)
                }
            }
        }
    }
}

struct CircularCheckboxStyle: ToggleStyle {
    func makeBody(configuration: Configuration) -> some View {
        Button(action: {
            configuration.isOn.toggle()
        }) {
            Image(systemName: configuration.isOn ? "checkmark.circle.fill" : "circle")
                .symbolRenderingMode(.hierarchical)
                .foregroundColor(configuration.isOn ? AppTheme.Selection.foreground : .secondary)
                .font(.system(size: 18))
        }
        .buttonStyle(.plain)
    }
}
