import SwiftUI

struct VocoHistoryPanel: View {
    let transcription: Transcription
    let onClose: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            AppPanelHeader(title: "Rule Assistant", onClose: onClose)
            RuleAssistantPanelView(transcription: transcription)
                .id(transcription.id)
        }
    }
}
