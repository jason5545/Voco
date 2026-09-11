import SwiftUI
import AppKit

/// Rule-assistant side panel for one transcription record. The conversation session is owned
/// by RuleAssistantSessionRegistry: closing this panel or switching views never cancels
/// in-flight work, and reopening the same record reattaches to the live conversation.
struct RuleAssistantPanelView: View {
    let transcription: Transcription

    @Environment(\.modelContext) private var modelContext
    @ObservedObject private var registry = RuleAssistantSessionRegistry.shared
    @AppStorage(RuleAssistantSettingsView.autoScanOnOpenKey) private var autoScanOnOpen = true
    @State private var session: RuleAssistantSession?

    var body: some View {
        Group {
            if let session {
                RuleAssistantSessionView(session: session, transcription: transcription)
            } else {
                unavailableView
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear(perform: attach)
        .onChange(of: transcription.id) { _, _ in attach() }
    }

    private var unavailableView: some View {
        VStack(spacing: 10) {
            Spacer()
            Image(systemName: "text.badge.checkmark")
                .font(.system(size: 28))
                .foregroundColor(.secondary)
            Text("This record cannot use the rule assistant")
                .font(.system(size: 13, weight: .medium))
            Text("It has no database row identity yet, so corrections cannot be attributed to it.")
                .font(.system(size: 11))
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            Spacer()
        }
        .padding(24)
    }

    private func attach() {
        let attached = registry.session(for: transcription, modelContext: modelContext)
        session = attached
        // Opening the assistant already states the intent; run the find-issues turn once per session.
        guard autoScanOnOpen, let attached else { return }
        Task { @MainActor in await attached.autoScanIfNeeded() }
    }
}

// MARK: - Live session content

private struct RuleAssistantSessionView: View {
    @ObservedObject var session: RuleAssistantSession
    let transcription: Transcription

    private var state: RuleAssistantUIState { session.state }

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    explanationLine
                    recordCard
                    configurationWarnings
                    conversationSection
                    toolStatusSection
                    draftsSection
                    phaseSection

                    if state.neighborsShared > 0 {
                        Text("Shared \(state.neighborsShared) nearby records with the model")
                            .font(.footnote)
                            .foregroundColor(.secondary)
                    }
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            Divider()
            inputArea
        }
    }

    // MARK: Explanation & record

    private var explanationLine: some View {
        Text("Only this record's texts and this conversation are sent — never audio. When one record is not enough, the model may read up to 5 records before and after.")
            .font(.footnote)
            .foregroundColor(.secondary)
    }

    private var recordCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "doc.text")
                Text("voco:row:\(state.context.rowPk)")
                    .monospaced()
                if let model = state.context.transcriptionModelName {
                    Text("· \(model)")
                        .lineLimit(1)
                }
            }
            .font(.system(size: 11, weight: .medium))
            .foregroundColor(.secondary)

            ForEach(transcription.detailDisplayTexts) { item in
                VStack(alignment: .leading, spacing: 2) {
                    Text(item.label)
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.secondary)
                    Text(item.text)
                        .font(.system(size: 12))
                        .foregroundColor(.primary)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            if let marking = transcription.correctionMarkingLabel {
                Label {
                    Text(marking.text)
                        .font(.system(size: 10, weight: .medium))
                } icon: {
                    Image(systemName: RowCorrectionMarkings.badgeIcon(for: transcription.corrections))
                        .font(.system(size: 9, weight: .semibold))
                }
                .foregroundStyle(marking.tone.color)
            }

            if let version = state.context.autoApplyModelVersion {
                Label("Auto-apply \(version)", systemImage: "checkmark.shield")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(.secondary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background { AppCardBackground() }
    }

    // MARK: Configuration warnings

    @ViewBuilder
    private var configurationWarnings: some View {
        if !state.goKeyConfigured {
            HStack(spacing: 8) {
                Image(systemName: "key.fill")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(AppTheme.Status.warningStrong)
                Text("Set the OpenCode Go API key to use the rule assistant.")
                    .font(.system(size: 12))
                    .foregroundColor(.primary)
                Spacer()
                Button("Open Settings") {
                    NotificationCenter.default.post(
                        name: .navigateToDestination,
                        object: nil,
                        userInfo: ["destination": "Settings"]
                    )
                }
                .font(.system(size: 12, weight: .medium))
            }
            .padding(10)
            .background { AppCardBackground() }
        }

        if !state.syncConfigured {
            Label(
                "The Worker sync key is not configured on this Mac; rules cannot be published.",
                systemImage: "exclamationmark.triangle.fill"
            )
            .font(.system(size: 12))
            .foregroundColor(AppTheme.Status.error)
        }
    }

    // MARK: Conversation

    @ViewBuilder
    private var conversationSection: some View {
        if !state.transcript.isEmpty || !state.answer.isEmpty || !state.reasoning.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(Array(state.transcript.enumerated()), id: \.offset) { _, turn in
                    if turn.role == "user" {
                        userBubble(turn.text)
                    } else {
                        assistantBubble(turn.text)
                        if let question = turn.question {
                            questionCard(question, turn: turn)
                        }
                    }
                }

                if !state.answer.isEmpty {
                    assistantBubble(state.answer)
                }

                if !state.reasoning.isEmpty {
                    DisclosureGroup("Show thinking") {
                        Text(state.reasoning)
                            .font(.footnote)
                            .foregroundColor(.secondary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.top, 4)
                    }
                    .font(.footnote)
                    .foregroundColor(.secondary)
                }
            }
        }
    }

    private func userBubble(_ text: String) -> some View {
        HStack {
            Spacer(minLength: 40)
            Text(text)
                .font(.system(size: 13))
                .foregroundColor(.primary)
                .textSelection(.enabled)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .background {
                    RoundedRectangle(cornerRadius: AppTheme.Radius.card, style: .continuous)
                        .fill(AppTheme.Surface.materialCard)
                        .overlay {
                            RoundedRectangle(cornerRadius: AppTheme.Radius.card, style: .continuous)
                                .strokeBorder(AppTheme.Border.subtle, lineWidth: 1)
                        }
                }
        }
    }

    private func assistantBubble(_ text: String) -> some View {
        HStack {
            MarkdownContentView(
                text,
                fontSize: 13,
                foregroundColor: AppTheme.Text.primary
            )
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background {
                RoundedRectangle(cornerRadius: AppTheme.Radius.card, style: .continuous)
                    .fill(AppTheme.Surface.subtle)
                    .overlay {
                        RoundedRectangle(cornerRadius: AppTheme.Radius.card, style: .continuous)
                            .strokeBorder(AppTheme.Border.tint, lineWidth: 1)
                    }
            }
            Spacer(minLength: 40)
        }
    }

    // MARK: Question cards

    /// Interactive while it is the pending question; read-only (showing what was chosen) afterwards.
    private func questionCard(_ question: RuleAssistantQuestion, turn: RuleAssistantTurn) -> some View {
        let interactive = state.pendingQuestion?.id == question.id && turn.answeredOptionIds.isEmpty
        let selected = interactive ? state.selectedOptionIds : turn.answeredOptionIds
        return VStack(alignment: .leading, spacing: 8) {
            Text(question.prompt)
                .font(.system(size: 13, weight: .semibold))
                .fixedSize(horizontal: false, vertical: true)
            Text(question.multiSelect ? "Pick every option that applies" : "Pick one")
                .font(.footnote)
                .foregroundColor(.secondary)

            ForEach(Array(question.options.enumerated()), id: \.element.id) { index, option in
                let isSelected = selected.contains(option.id)
                VStack(alignment: .leading, spacing: 6) {
                    optionButton(option, index: index, isSelected: isSelected, interactive: interactive)
                    if isSelected, option.isCandidate {
                        scopePicker(for: option, turn: turn, interactive: interactive)
                    }
                }
            }

            if interactive {
                HStack(spacing: 10) {
                    Button("Send choices") {
                        run { await session.submitChoice() }
                    }
                    .buttonStyle(.borderedProminent)
                    .font(.system(size: 12, weight: .semibold))
                    .disabled(!state.canSubmitChoice || !configured)
                    Text("⌘1–⌘9 toggle options · a note below is sent with them")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }
            } else if !turn.answeredOptionIds.isEmpty {
                Label("Answered", systemImage: "checkmark.circle")
                    .font(.footnote)
                    .foregroundColor(.secondary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            RoundedRectangle(cornerRadius: AppTheme.Radius.card, style: .continuous)
                .fill(AppTheme.Surface.materialCard)
                .overlay {
                    RoundedRectangle(cornerRadius: AppTheme.Radius.card, style: .continuous)
                        .strokeBorder(interactive ? AppTheme.Border.tint : AppTheme.Border.subtle, lineWidth: 1)
                }
        }
    }

    private func optionButton(_ option: RuleAssistantQuestionOption, index: Int, isSelected: Bool, interactive: Bool) -> some View {
        let button = Button {
            session.toggleOption(option.id)
        } label: {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .font(.system(size: 14))
                    .foregroundColor(isSelected ? AppTheme.Status.positive : .secondary)
                VStack(alignment: .leading, spacing: 2) {
                    Text(option.label)
                        .font(.system(size: 13, weight: isSelected ? .semibold : .regular))
                        .foregroundColor(.primary)
                        .fixedSize(horizontal: false, vertical: true)
                    if let detail = option.detail {
                        Text(detail)
                            .font(.footnote)
                            .foregroundColor(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer(minLength: 0)
                if index < 9 {
                    Text("⌘\(index + 1)")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.secondary)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!interactive)

        return Group {
            if index < 9, interactive {
                button.keyboardShortcut(KeyEquivalent(Character(String(index + 1))), modifiers: .command)
            } else {
                button
            }
        }
    }

    private func scopePicker(for option: RuleAssistantQuestionOption, turn: RuleAssistantTurn, interactive: Bool) -> some View {
        let binding = Binding<RuleAssistantScope>(
            get: {
                interactive
                    ? (state.optionScopes[option.id] ?? .sentence)
                    : (turn.answeredScopes[option.id] ?? .sentence)
            },
            set: { session.setScope($0, for: option.id) }
        )
        return VStack(alignment: .leading, spacing: 4) {
            Picker("Scope", selection: binding) {
                Text("This sentence only").tag(RuleAssistantScope.sentence)
                Text("Context-locked").tag(RuleAssistantScope.context)
                Text("Any context").tag(RuleAssistantScope.broad)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .controlSize(.small)
            .disabled(!interactive)
            if binding.wrappedValue == .broad {
                Text("Any context creates a broad rule: the source will be replaced everywhere.")
                    .font(.footnote)
                    .foregroundColor(AppTheme.Status.warningStrong)
                if interactive, let guards = state.optionGuardPreviews[option.id], !guards.isEmpty {
                    Text("Will auto-protect longer words: \(guards.joined(separator: "\u{3001}"))")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(.leading, 22)
    }

    // MARK: Tool status

    @ViewBuilder
    private var toolStatusSection: some View {
        if !state.toolStatus.isEmpty {
            VStack(alignment: .leading, spacing: 2) {
                ForEach(Array(state.toolStatus.suffix(30).enumerated()), id: \.offset) { _, line in
                    Text(line)
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: Draft cards

    @ViewBuilder
    private var draftsSection: some View {
        ForEach(Array(state.drafts.enumerated()), id: \.element.draft.nonce) { index, entry in
            draftCard(entry: entry, index: index, total: state.drafts.count)
        }
    }

    private func draftCard(entry: RuleAssistantDraftEntry, index: Int, total: Int) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Rule \(index + 1) / \(total)")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                Text(typeLabel(entry.draft.eventType))
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Capsule().fill(AppTheme.Surface.controlActive))
            }

            if entry.draft.isBroad {
                Label(
                    "This is a broad replacement rule; it applies in every context. Check the source boundary before confirming.",
                    systemImage: "exclamationmark.triangle.fill"
                )
                .font(.system(size: 11))
                .foregroundColor(AppTheme.Status.error)
            }

            if entry.draft.isTransaction {
                Label(
                    "This change moves or merges whole families and cannot be duplicate-checked; confirm only when you are sure.",
                    systemImage: "exclamationmark.triangle.fill"
                )
                .font(.system(size: 11))
                .foregroundColor(AppTheme.Status.error)
            }

            draftFields(entry.draft)

            if !entry.autoGuards.isEmpty {
                autoGuardsSection(entry)
            }

            Divider()

            checkSection(entry.check)

            Divider()

            if let outcome = entry.outcome {
                Text(outcome)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.secondary)
            }

            HStack(spacing: 10) {
                if entry.consumed {
                    Label("Sent", systemImage: "checkmark.circle.fill")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(AppTheme.Status.positive)
                } else {
                    Button(action: { confirm(entry) }) {
                        Text("Confirm & Publish")
                            .font(.system(size: 12, weight: .semibold))
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!state.canConfirm(entry))
                }
                copyForReviewButton
            }
            if !entry.consumed {
                Text("Confirming re-runs the preview and duplicate check before writing.")
                    .font(.footnote)
                    .foregroundColor(.secondary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            RoundedRectangle(cornerRadius: AppTheme.Radius.card, style: .continuous)
                .fill(entry.consumed ? AppTheme.Surface.card : AppTheme.Accent.fillSubtle)
                .overlay {
                    RoundedRectangle(cornerRadius: AppTheme.Radius.card, style: .continuous)
                        .strokeBorder(AppTheme.Border.tint, lineWidth: 1)
                }
        }
    }

    private func typeLabel(_ eventType: String) -> String {
        switch eventType {
        case "correction": return String(localized: "Exact correction")
        case "contextLockedRule": return String(localized: "Context-locked replacement")
        case "replacementRule": return String(localized: "Broad replacement")
        case "replacementFamily": return String(localized: "Replacement family")
        case "tombstone": return String(localized: "Retire rule")
        case "moveAliasToFamily": return String(localized: "Move alias to family")
        case "mergeReplacementFamilies": return String(localized: "Merge families")
        case "replaceContextLockedRule": return String(localized: "Replace context lock")
        default: return eventType
        }
    }

    @ViewBuilder
    private func draftFields(_ draft: RuleAssistantDraft) -> some View {
        fieldRow("Source", draft.sourceText)
        fieldRow("Pattern", draft.sourcePattern)
        fieldRow("Target", draft.targetText)
        fieldRow("Family", draft.familyId)
        if !draft.aliases.isEmpty {
            fieldRow("Aliases", draft.aliases.joined(separator: ", "))
        }
        if !draft.contextTokensAny.isEmpty {
            fieldRow("Context tokens", draft.contextTokensAny.joined(separator: ", "))
        }
        if !draft.contextAliasesAny.isEmpty {
            fieldRow("Context aliases", draft.contextAliasesAny.joined(separator: ", "))
        }
        fieldRow("Policy", draft.policyId)
        fieldRow("Disposition", draft.disposition)
        fieldRow("From", draft.fromFamilyId)
        fieldRow("To", draft.toFamilyId)
        fieldRow("Reason", draft.reason)
        fieldRow("Note", draft.note)
        if !draft.positiveExamples.isEmpty {
            examplesSection("Positive examples", draft.positiveExamples)
        }
        if !draft.negativeExamples.isEmpty {
            examplesSection("Negative examples", draft.negativeExamples)
        }
    }

    /// Longer lexicon words that contain the broad source; on by default, off re-runs the Worker check.
    private func autoGuardsSection(_ entry: RuleAssistantDraftEntry) -> some View {
        let binding = Binding<Bool>(
            get: { entry.autoGuardsEnabled },
            set: { enabled in run { await session.setAutoGuards(nonce: entry.draft.nonce, enabled: enabled) } }
        )
        return VStack(alignment: .leading, spacing: 4) {
            Toggle(isOn: binding) {
                Text("Protect longer words automatically")
                    .font(.system(size: 11, weight: .medium))
            }
            .toggleStyle(.switch)
            .controlSize(.mini)
            .disabled(entry.consumed || state.phase.isBusy)
            Text(entry.autoGuards.joined(separator: "\u{3001}"))
                .font(.system(size: 11))
                .foregroundColor(entry.autoGuardsEnabled ? .primary : .secondary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            Text("A broad rule also fires inside longer words; these lexicon words are added as negative examples so the rule skips any sentence containing them.")
                .font(.footnote)
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private func fieldRow(_ label: LocalizedStringKey, _ value: String?) -> some View {
        if let value, !value.isEmpty {
            HStack(alignment: .top, spacing: 8) {
                Text(label)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundColor(.secondary)
                    .frame(width: 92, alignment: .leading)
                Text(value)
                    .font(.system(size: 12))
                    .foregroundColor(.primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private func examplesSection(_ label: LocalizedStringKey, _ examples: [RuleAssistantExample]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 10, weight: .semibold))
                .foregroundColor(.secondary)
            ForEach(Array(examples.enumerated()), id: \.offset) { _, example in
                VStack(alignment: .leading, spacing: 1) {
                    Text(example.text)
                        .font(.system(size: 11))
                        .foregroundColor(.primary)
                        .textSelection(.enabled)
                    if !example.context.isEmpty || !example.expectedText.isEmpty {
                        Text("\(example.context) → \(example.expectedText)")
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
                            .textSelection(.enabled)
                    }
                }
            }
        }
    }

    // MARK: Draft check results

    @ViewBuilder
    private func checkSection(_ check: RuleAssistantDraftCheck?) -> some View {
        if let check {
            if let preview = check.preview {
                Text(preview.wouldPublish
                     ? String(localized: "Preview: would publish immediately")
                     : String(localized: "Preview: would not publish"))
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(preview.wouldPublish ? AppTheme.Status.positive : .secondary)
                Text("conflicts \(preview.conflicts) · skipped \(preview.skipped) · unsupported \(preview.unsupported)")
                    .font(.footnote)
                    .foregroundColor(.secondary)
                if let reason = preview.reason {
                    Text(reason)
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }
                if let base = preview.baseModelSha256 {
                    Text("Base SHA \(base.prefix(12))")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }
            }

            if let duplicate = check.duplicate {
                if duplicate.found {
                    Text(duplicate.alreadyApplied
                         ? String(localized: "Duplicate check: already applied")
                         : String(localized: "Duplicate check: possible duplicate (policies \(duplicate.duplicatePolicies), events \(duplicate.duplicateEvents))"))
                        .font(.footnote)
                        .foregroundColor(AppTheme.Status.warningStrong)
                } else {
                    Text("Duplicate check: none found")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }
                if let next = duplicate.suggestedNextAction {
                    Text(next)
                        .font(.footnote)
                        .foregroundColor(.secondary)
                }
            }

            if let blocked = check.blockedReason {
                Text(blocked)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(AppTheme.Status.error)
            }
        } else {
            Text("Checking with the Worker…")
                .font(.footnote)
                .foregroundColor(.secondary)
        }
    }

    // MARK: Phase

    @ViewBuilder
    private var phaseSection: some View {
        switch state.phase {
        case .failed(let message):
            VStack(alignment: .leading, spacing: 6) {
                Text(message)
                    .font(.system(size: 12))
                    .foregroundColor(AppTheme.Status.error)
                    .textSelection(.enabled)
                if state.publishedSha256 != nil {
                    Button("Resync Mac model") {
                        run { await session.resync() }
                    }
                    .font(.system(size: 12, weight: .medium))
                }
            }
        case .published:
            if let message = state.publishMessage {
                Label(message, systemImage: "checkmark.circle.fill")
                    .font(.system(size: 12))
                    .foregroundColor(AppTheme.Status.positive)
            }
        case .syncing:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(state.publishMessage ?? String(localized: "Syncing Mac model…"))
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)
            }
        case .loadingTools, .thinking, .checking, .confirming:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(busyLabel)
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)
            }
        case .idle, .draftReady:
            EmptyView()
        }
    }

    private var busyLabel: String {
        switch state.phase {
        case .loadingTools: return String(localized: "Loading Worker tools…")
        case .thinking: return String(localized: "Waiting for the AI…")
        case .checking: return String(localized: "Checking the draft with the Worker…")
        case .confirming: return String(localized: "Publishing…")
        default: return String(localized: "Working…")
        }
    }

    // MARK: Input

    private var userTextBinding: Binding<String> {
        Binding(
            get: { state.userText },
            set: { session.setUserText($0) }
        )
    }

    private var configured: Bool {
        state.goKeyConfigured && state.syncConfigured
    }

    private var hasText: Bool {
        !state.userText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// Return / Send: a choice with selected options, or typed text. Typed text may also
    /// interrupt a running find-issues turn.
    private var canSend: Bool {
        guard configured else { return false }
        if state.phase.isBusy { return state.isInterruptible && hasText }
        return state.canSubmitChoice || hasText
    }

    private var canScan: Bool {
        configured && !state.phase.isBusy && !hasText
    }

    private var composerPlaceholder: String {
        if state.pendingQuestion != nil {
            return String(localized: "Optional note to send with your choice, or answer in your own words")
        }
        return state.transcript.isEmpty
            ? String(localized: "Describe what you actually said, or press Find issues")
            : String(localized: "Add detail or ask a follow-up")
    }

    private var inputArea: some View {
        VStack(spacing: 8) {
            ComposerEditor(
                text: userTextBinding,
                placeholder: composerPlaceholder,
                isEnabled: !state.phase.isBusy || state.isInterruptible,
                onSend: { if canSend { send() } }
            )
            .frame(minHeight: 68, maxHeight: 120)
            .padding(8)
            .background {
                RoundedRectangle(cornerRadius: AppTheme.Radius.card, style: .continuous)
                    .fill(AppTheme.Surface.subtle)
                    .overlay {
                        RoundedRectangle(cornerRadius: AppTheme.Radius.card, style: .continuous)
                            .strokeBorder(AppTheme.Border.tint, lineWidth: 1)
                    }
            }
            .disabled(state.phase.isBusy && !state.isInterruptible)

            HStack(spacing: 10) {
                if state.phase.isBusy {
                    Button("Stop") {
                        session.cancel()
                    }
                    .font(.system(size: 12, weight: .medium))
                    if state.isInterruptible {
                        Button("Send to AI") {
                            send()
                        }
                        .font(.system(size: 12, weight: .medium))
                        .disabled(!canSend)
                        .keyboardShortcut(.return, modifiers: .command)
                    }
                } else {
                    Button(state.canSubmitChoice ? "Send choices" : "Send to AI") {
                        send()
                    }
                    .buttonStyle(.borderedProminent)
                    .font(.system(size: 12, weight: .semibold))
                    .disabled(!canSend)
                    .keyboardShortcut(.return, modifiers: .command)

                    Button("Find issues") {
                        run { await session.submitScan() }
                    }
                    .font(.system(size: 12, weight: .medium))
                    .disabled(!canScan)
                    .help("The AI lists the places it suspects; tick the ones that are wrong.")
                }
                Spacer()
                if !state.transcript.isEmpty {
                    copyForReviewButton
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    // MARK: Review export

    /// Copies the whole conversation, with a client marker and a note for Claude Code / Codex,
    /// so Jason can paste it into a review session without retyping anything.
    private var copyForReviewButton: some View {
        Button {
            let client = "voco \(RuleAssistantConstants.appVersion) · macOS \(ProcessInfo.processInfo.operatingSystemVersionString)"
            let text = session.exportForReview(client: client)
            let pasteboard = NSPasteboard.general
            pasteboard.clearContents()
            pasteboard.setString(text, forType: .string)
            session.noteCopiedForReview()
        } label: {
            Label("Copy for review", systemImage: "doc.on.clipboard")
                .font(.system(size: 12, weight: .medium))
        }
        .help("Copies this whole conversation (record, replies, choices, drafts, Worker checks, tool calls) with a note for Claude Code or Codex.")
    }

    // MARK: Actions

    private func send() {
        // Typing during a find-issues turn replaces it with Jason's own words.
        if state.phase.isBusy {
            guard state.isInterruptible, hasText else { return }
            session.cancel()
        }
        if state.canSubmitChoice {
            run { await session.submitChoice() }
            return
        }
        let text = state.userText
        run { await session.submit(text) }
    }

    private func confirm(_ entry: RuleAssistantDraftEntry) {
        run { await session.confirm(nonce: entry.draft.nonce) }
    }

    /// Unstructured on purpose: panel dismissal must not cancel in-flight work.
    private func run(_ operation: @escaping @MainActor () async -> Void) {
        Task { @MainActor in await operation() }
    }
}

// MARK: - Composer editor

/// AppKit-backed composer. Return sends, Shift+Return inserts a newline, and keys consumed
/// by an IME composition never send. SwiftUI's TextField cannot offer this: without an
/// onSubmit, AppKit answers Return with select-all, and onKeyPress fires while composing.
private struct ComposerEditor: NSViewRepresentable {
    @Binding var text: String
    var placeholder: String
    var isEnabled: Bool
    var onSend: () -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.drawsBackground = false
        scrollView.borderType = .noBorder
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true

        let textView = ComposerTextView()
        textView.onSendReturn = onSend
        textView.placeholder = placeholder
        textView.delegate = context.coordinator
        textView.isRichText = false
        textView.importsGraphics = false
        textView.font = .systemFont(ofSize: 13)
        textView.textColor = .labelColor
        textView.drawsBackground = false
        textView.allowsUndo = true
        textView.textContainerInset = NSSize(width: 2, height: 2)
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.textContainer?.widthTracksTextView = true
        textView.autoresizingMask = [.width]

        scrollView.documentView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? ComposerTextView else { return }
        context.coordinator.parent = self
        textView.onSendReturn = onSend
        if textView.placeholder != placeholder {
            textView.placeholder = placeholder
        }
        if textView.string != text {
            textView.string = text
            textView.didChangeText()
        }
        textView.isEditable = isEnabled
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: ComposerEditor
        init(_ parent: ComposerEditor) { self.parent = parent }
        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            parent.text = textView.string
        }
    }
}

private final class ComposerTextView: NSTextView {
    var onSendReturn: (() -> Void)?
    var placeholder: String = "" {
        didSet { needsDisplay = true }
    }

    override var intrinsicContentSize: NSSize {
        guard let container = textContainer, let manager = layoutManager else {
            return super.intrinsicContentSize
        }
        manager.ensureLayout(for: container)
        let used = manager.usedRect(for: container)
        return NSSize(width: NSView.noIntrinsicMetric,
                      height: used.height + textContainerInset.height * 2)
    }

    override func didChangeText() {
        super.didChangeText()
        invalidateIntrinsicContentSize()
        needsDisplay = true
    }

    override func keyDown(with event: NSEvent) {
        let isReturn = event.keyCode == 36 || event.keyCode == 76
        if isReturn && !hasMarkedText() {
            if event.modifierFlags.intersection(.deviceIndependentFlagsMask).contains(.shift) {
                insertLineBreak(nil)
            } else {
                onSendReturn?()
            }
            return
        }
        super.keyDown(with: event)
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard string.isEmpty, !placeholder.isEmpty else { return }
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font ?? NSFont.systemFont(ofSize: 13),
            .foregroundColor: NSColor.placeholderTextColor,
        ]
        let padding = textContainer?.lineFragmentPadding ?? 0
        let rect = NSRect(x: textContainerInset.width + padding,
                          y: textContainerInset.height,
                          width: bounds.width - textContainerInset.width - padding - 4,
                          height: bounds.height - textContainerInset.height)
        placeholder.draw(in: rect, withAttributes: attributes)
    }
}
