import SwiftUI

struct DictionarySettingsPanel: View {
    let onDismiss: () -> Void
    @State private var enabledContextPackIDs = Set(VocoCanonicalizationService.enabledContextPackIDs())
    @State private var expandedContextPackIDs: Set<String> = []

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
                        VStack(alignment: .leading, spacing: 10) {
                            Toggle(isOn: contextPackBinding(for: pack.id)) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(pack.displayName)
                                        .font(.system(size: 13, weight: .medium))

                                    HStack(spacing: 10) {
                                        Label("\(pack.terms.count) terms", systemImage: "tag.fill")
                                        Label("\(pack.aliasCount) aliases", systemImage: "text.quote")
                                        if pack.contextRequiredTermCount > 0 {
                                            Label("\(pack.contextRequiredTermCount) contextual", systemImage: "scope")
                                        }
                                    }
                                    .font(.system(size: 11))
                                    .foregroundStyle(.secondary)
                                }
                            }
                            .toggleStyle(.switch)

                            DisclosureGroup(isExpanded: expandedBinding(for: pack.id)) {
                                VStack(alignment: .leading, spacing: 8) {
                                    ForEach(pack.terms) { term in
                                        ContextPackTermRow(term: term)
                                    }
                                }
                                .padding(.top, 4)
                            } label: {
                                Text(pack.canonicalPreview)
                                    .font(.system(size: 11, weight: .medium))
                                    .foregroundStyle(.secondary)
                                    .lineLimit(2)
                            }
                        }
                        .padding(.vertical, 4)
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

    private func expandedBinding(for id: String) -> Binding<Bool> {
        Binding(
            get: {
                expandedContextPackIDs.contains(id)
            },
            set: { isExpanded in
                if isExpanded {
                    expandedContextPackIDs.insert(id)
                } else {
                    expandedContextPackIDs.remove(id)
                }
            }
        )
    }
}

private struct ContextPackTermRow: View {
    let term: VocoCanonicalTerm

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text(term.canonical)
                    .font(.system(size: 12, weight: .semibold))
                    .textSelection(.enabled)

                Text(term.type)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)

                if term.requiresContextForAutoReplace {
                    Image(systemName: "scope")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.orange)
                        .help("Context required")
                }
            }

            if !term.aliases.isEmpty {
                Text(term.aliases.joined(separator: ", "))
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .textSelection(.enabled)
            }
        }
    }
}
