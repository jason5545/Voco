import SwiftUI

/// Settings section for the rule assistant: the OpenCode Go API key (this Mac's Keychain
/// only) and the read-only Worker sync key status. Key values are never displayed or logged.
struct RuleAssistantSettingsView: View {
    static let autoScanOnOpenKey = "RuleAssistantAutoScanOnOpen"

    @State private var apiKeyInput = ""
    @State private var hasStoredKey = RuleAssistantKeyStore.shared.apiKey != nil
    @AppStorage(Self.autoScanOnOpenKey) private var autoScanOnOpen = true

    var body: some View {
        Section {
            Toggle("Find issues automatically when the assistant opens", isOn: $autoScanOnOpen)
            Text("The AI scans the record once per conversation and lists suspected errors as options to tick. Turn this off to start every conversation by hand.")
                .font(.footnote)
                .foregroundColor(.secondary)

            HStack {
                Text("OpenCode Go API key")

                Spacer()

                Circle()
                    .fill(hasStoredKey ? AppTheme.Status.positive : AppTheme.Status.error)
                    .frame(width: 8, height: 8)
                Text(hasStoredKey ? "Set" : "Not set")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            HStack {
                SecureField("API key", text: $apiKeyInput)

                Button(hasStoredKey ? "Update" : "Save") {
                    save()
                }
                .disabled(apiKeyInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                if hasStoredKey {
                    Button("Remove") {
                        remove()
                    }
                }
            }

            Text("The key is stored only in this Mac's Keychain and is sent to OpenCode Go only. It never goes to the Worker, into model payloads, logs, or UserDefaults.")
                .font(.footnote)
                .foregroundColor(.secondary)

            HStack {
                Text("Worker sync key")

                Spacer()

                Circle()
                    .fill(workerSyncKeyIsSet ? AppTheme.Status.positive : AppTheme.Status.error)
                    .frame(width: 8, height: 8)
                Text(workerSyncKeyStatus)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        } header: {
            Text("Rule Assistant")
        }
    }

    private var workerSyncKeyIsSet: Bool {
        VocoAutoApplyModelService.defaultWorkerSyncKey()?.isEmpty == false
    }

    /// Source type only; the key value itself is never shown.
    private var workerSyncKeyStatus: String {
        let environmentKey = ProcessInfo.processInfo.environment["VOCO_SYNC_KEY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let environmentKey, !environmentKey.isEmpty {
            return String(localized: "Set (environment variable)")
        }
        if workerSyncKeyIsSet {
            return String(localized: "Set (key file)")
        }
        return String(localized: "Not set")
    }

    private func save() {
        let key = apiKeyInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { return }
        guard RuleAssistantKeyStore.shared.save(key) else { return }
        apiKeyInput = ""
        hasStoredKey = true
        notifyKeyChanged()
    }

    private func remove() {
        RuleAssistantKeyStore.shared.remove()
        hasStoredKey = false
        notifyKeyChanged()
    }

    private func notifyKeyChanged() {
        Task { @MainActor in
            RuleAssistantSessionRegistry.shared.providerKeyChanged()
        }
    }
}
