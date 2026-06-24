import SwiftUI

struct AutoApplyModelSettingsView: View {
    @ObservedObject private var service = VocoAutoApplyModelService.shared
    @State private var isSyncingRemoteModel = false

    private var toggleBinding: Binding<Bool> {
        Binding(
            get: { service.settingsToggleIsOn },
            set: { newValue in
                guard service.settingsToggleIsEnabled else { return }
                service.isUserEnabled = newValue
            }
        )
    }

    private var remoteStatusMessage: String? {
        if service.status.remoteIsInSync == true {
            return String(localized: "Remote latest matches local")
        }
        if service.status.remoteIsInSync == false {
            return String(localized: "Remote latest differs from local")
        }
        return service.status.remoteMessage
    }

    private var remoteStatusIconName: String {
        if service.status.remoteIsInSync == true { return "checkmark.icloud.fill" }
        if service.status.remoteIsInSync == false { return "arrow.down.circle.fill" }
        return "icloud"
    }

    private var remoteStatusColor: Color {
        if service.status.remoteIsInSync == true { return .green }
        if service.status.remoteIsInSync == false { return .orange }
        return .secondary
    }

    var body: some View {
        Section {
            Toggle("Auto-Apply Model", isOn: toggleBinding)
                .disabled(!service.settingsToggleIsEnabled)

            HStack(spacing: 6) {
                Image(systemName: service.status.isAvailable ? "checkmark.circle.fill" : "xmark.circle")
                    .foregroundColor(service.status.isAvailable ? .green : .secondary)
                Text(service.status.message)
                    .foregroundColor(.secondary)
                Spacer()
                Button {
                    service.reload()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.plain)
                .help("Reload model")
            }
            .font(.footnote)

            if let remoteStatusMessage {
                HStack(spacing: 6) {
                    Image(systemName: remoteStatusIconName)
                        .foregroundColor(remoteStatusColor)
                    Text(remoteStatusMessage)
                        .foregroundColor(.secondary)
                    if let version = service.status.remoteLatestVersion {
                        Text(version)
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    Spacer()
                    Button {
                        syncRemoteModel()
                    } label: {
                        if isSyncingRemoteModel {
                            ProgressView()
                                .controlSize(.small)
                        } else {
                            Image(systemName: "arrow.triangle.2.circlepath.icloud")
                        }
                    }
                    .buttonStyle(.plain)
                    .disabled(isSyncingRemoteModel)
                    .help("Sync remote model")
                }
                .font(.footnote)
            } else {
                HStack(spacing: 6) {
                    Image(systemName: "icloud")
                        .foregroundColor(.secondary)
                    Text("Remote status not checked")
                        .foregroundColor(.secondary)
                    Spacer()
                    Button {
                        syncRemoteModel()
                    } label: {
                        if isSyncingRemoteModel {
                            ProgressView()
                                .controlSize(.small)
                        } else {
                            Image(systemName: "arrow.triangle.2.circlepath.icloud")
                        }
                    }
                    .buttonStyle(.plain)
                    .disabled(isSyncingRemoteModel)
                    .help("Sync remote model")
                }
                .font(.footnote)
            }
        } header: {
            Text("Auto-Apply Model")
        }
    }

    private func syncRemoteModel() {
        guard !isSyncingRemoteModel else { return }
        isSyncingRemoteModel = true
        Task {
            _ = await service.syncFromWorker()
            await MainActor.run {
                isSyncingRemoteModel = false
            }
        }
    }
}
