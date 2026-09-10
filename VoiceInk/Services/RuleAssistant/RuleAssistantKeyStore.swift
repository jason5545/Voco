import Foundation

/// Keychain storage for the OpenCode Go API key used by the rule assistant.
/// The key is sent to the OpenCode Go endpoint only: never to the Worker, never inside
/// model payloads, logs, UserDefaults, or git. Shared by the session registry and Settings.
struct RuleAssistantKeyStore {
    static let shared = RuleAssistantKeyStore()

    static let keychainKey = "RuleAssistantOpenCodeGoKey"

    private let keychain = KeychainService.shared

    private init() {}

    var apiKey: String? {
        guard let value = keychain.getString(forKey: Self.keychainKey, syncable: false)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !value.isEmpty
        else { return nil }
        return value
    }

    @discardableResult
    func save(_ key: String) -> Bool {
        keychain.save(
            key.trimmingCharacters(in: .whitespacesAndNewlines),
            forKey: Self.keychainKey,
            syncable: false
        )
    }

    @discardableResult
    func remove() -> Bool {
        keychain.delete(forKey: Self.keychainKey, syncable: false)
    }
}
