import SwiftUI

struct DictionarySettingsPanel: View {
    let onDismiss: () -> Void
    @State private var enabledContextPackIDs = Set(VocoCanonicalizationService.enabledContextPackIDs())

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack(spacing: 12) {
                Text("Dictionary Settings")
                    .font(.headline)
                    .fontWeight(.semibold)
                    .foregroundColor(.primary)

                Spacer()

                Button(action: onDismiss) {
                    Image(systemName: "xmark")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.secondary)
                        .padding(6)
                        .background(Color.secondary.opacity(0.1))
                        .clipShape(Circle())
                }
                .buttonStyle(.plain)
                .help("Close")
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 16)
            .background(Color(NSColor.windowBackgroundColor))
            .overlay(
                Divider().opacity(0.5), alignment: .bottom
            )

            // Content
            Form {
                Section {
                    LabeledContent("Quick Add to Dictionary") {
                        ShortcutRecorder(action: .quickAddToDictionary)
                            .controlSize(.small)
                    }
                } header: {
                    Text("Shortcuts")
                }

                Section {
                    ForEach(VocoCanonicalizationService.builtInContextPacks) { pack in
                        Toggle(isOn: contextPackBinding(for: pack.id)) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(pack.displayName)
                                    .font(.system(size: 13, weight: .medium))
                                Text("\(pack.terms.count) canonical terms")
                                    .font(.system(size: 11))
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .toggleStyle(.switch)
                    }
                } header: {
                    Text("Context Packs")
                }

            }
            .formStyle(.grouped)
            .scrollContentBackground(.hidden)
        }
    }

    private func contextPackBinding(for id: String) -> Binding<Bool> {
        Binding(
            get: {
                enabledContextPackIDs.contains(id)
            },
            set: { isEnabled in
                if isEnabled {
                    enabledContextPackIDs.insert(id)
                } else {
                    enabledContextPackIDs.remove(id)
                }
                VocoCanonicalizationService.setEnabledContextPackIDs(
                    VocoCanonicalizationService.builtInContextPacks
                        .map(\.id)
                        .filter { enabledContextPackIDs.contains($0) }
                )
            }
        )
    }
}
