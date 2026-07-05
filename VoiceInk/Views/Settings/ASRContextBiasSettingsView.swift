import Foundation
import SwiftUI

struct ASRContextBiasSettingsView: View {
    @ObservedObject private var store = Qwen3ASRContextBiasStore.shared
    @AppStorage(Qwen3ASRContextBiasStore.enabledKey) private var isEnabled = true

    private var status: Qwen3ASRContextBiasStatus {
        store.status
    }

    private var availabilityMessage: String {
        switch status.sourceKind {
        case .downloaded:
            return "Downloaded profile active"
        case .builtin:
            return "Built-in profile active"
        case .unavailable:
            return "No usable profile"
        }
    }

    private var availabilityIconName: String {
        switch status.sourceKind {
        case .downloaded:
            return "checkmark.icloud.fill"
        case .builtin:
            return "shippingbox.fill"
        case .unavailable:
            return "xmark.circle"
        }
    }

    private var availabilityColor: Color {
        switch status.sourceKind {
        case .downloaded:
            return .green
        case .builtin:
            return .secondary
        case .unavailable:
            return .orange
        }
    }

    private var remoteMessage: String {
        if let lastMessage = status.lastMessage, !lastMessage.isEmpty {
            return lastMessage
        }
        return "Remote profile not synced"
    }

    private var sourceLabel: String {
        switch status.sourceKind {
        case .downloaded:
            return "Downloaded"
        case .builtin:
            return "Built-in"
        case .unavailable:
            return "Unavailable"
        }
    }

    private var boostLabel: String {
        guard let boost = status.boost else { return "-" }
        return String(format: "%.1f", Double(boost))
    }

    private var guardLabel: String {
        guard let repeatNgramSize = status.repeatNgramSize,
              let repeatNgramMaxCount = status.repeatNgramMaxCount else {
            return "-"
        }
        return "\(repeatNgramSize)/\(repeatNgramMaxCount)"
    }

    private var shortSHA: String {
        guard let sha = status.sha256, !sha.isEmpty else { return "-" }
        return String(sha.prefix(12))
    }

    var body: some View {
        Section {
            Toggle("ASR Context Bias", isOn: $isEnabled)
                .onChange(of: isEnabled) { _, _ in
                    store.reload()
                }

            HStack(spacing: 6) {
                Image(systemName: availabilityIconName)
                    .foregroundColor(availabilityColor)
                Text(availabilityMessage)
                    .foregroundColor(.secondary)
                Spacer()
                Button {
                    store.reload()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.plain)
                .help("Reload profile")
            }
            .font(.footnote)

            HStack(spacing: 6) {
                Image(systemName: "icloud.and.arrow.down")
                    .foregroundColor(store.isDownloading ? .secondary : .accentColor)
                Text(remoteMessage)
                    .foregroundColor(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                Button {
                    Task {
                        await store.downloadLatest()
                    }
                } label: {
                    if store.isDownloading {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        Image(systemName: "arrow.triangle.2.circlepath.icloud")
                    }
                }
                .buttonStyle(.plain)
                .disabled(store.isDownloading)
                .help("Sync remote profile")
            }
            .font(.footnote)

            LabeledContent("Source", value: sourceLabel)
            LabeledContent("Artifact", value: status.artifactId ?? "-")
            LabeledContent("Terms", value: "\(status.termCount)")
            LabeledContent("Boost", value: boostLabel)
            LabeledContent("Guard", value: guardLabel)
            LabeledContent("SHA", value: shortSHA)
        } header: {
            Text("ASR Context Bias")
        }
    }
}
