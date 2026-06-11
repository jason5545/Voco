import SwiftUI

struct AutoApplyModelSettingsView: View {
    @ObservedObject private var service = VocoAutoApplyModelService.shared

    private var toggleBinding: Binding<Bool> {
        Binding(
            get: { service.settingsToggleIsOn },
            set: { newValue in
                guard service.settingsToggleIsEnabled else { return }
                service.isUserEnabled = newValue
            }
        )
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
        } header: {
            Text("Auto-Apply Model")
        }
    }
}
