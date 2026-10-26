import SwiftUI

/// Rule assistant inside the Quick History panel. Replaces the detail page instead of opening a
/// side panel, so the floating panel stays the only window and paste still targets the app the
/// panel was opened from. The panel grows to `panelHeight` while this view is shown.
struct VocoQuickHistoryRuleAssistantView: View {
    static let panelHeight: CGFloat = 640

    let transcription: Transcription
    let onBack: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            RuleAssistantPanelView(transcription: transcription)
                .id(transcription.id)
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            Button(action: onBack) {
                Image(systemName: "chevron.left")
                    .font(.system(size: 14, weight: .semibold))
            }
            .buttonStyle(.plain)
            .keyboardShortcut("[", modifiers: .command)
            .help("Back to details")
            .accessibilityLabel("Back to details")

            Text("Rule Assistant")
                .font(.system(size: 14, weight: .semibold))

            HistoryWindowDragArea()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .padding(.horizontal, HistoryLayout.headerInset)
        .frame(height: 52)
    }
}
