import SwiftUI

struct VocoEphemeralRecorderBadge: View {
    let privacyNotice: String?

    var body: some View {
        HStack(spacing: 3) {
            Image(systemName: privacyNotice == nil ? "eye.slash" : "exclamationmark.triangle")
            Text("Ephemeral")
        }
        .font(.system(size: 10, weight: .semibold))
        .foregroundStyle(privacyNotice == nil ? Color.white.opacity(0.72) : Color.orange)
        .lineLimit(1)
        .help(privacyNotice ?? String(localized: "Ephemeral dictation"))
        .accessibilityLabel(privacyNotice ?? String(localized: "Ephemeral dictation"))
    }
}
