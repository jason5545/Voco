import Foundation

/// Voco language defaults kept out of the upstream language dictionary.
///
/// The upstream table still declares its own fallbacks; Voco injects "auto" as the
/// user-facing default so a fresh install follows the detected language.
extension TranscriptionLanguageSupport {
    static let selectedLanguageKey = "SelectedLanguage"
    static let defaultLanguageCode = "auto"

    static func selectedLanguage(in defaults: UserDefaults = .standard) -> String {
        defaults.string(forKey: selectedLanguageKey) ?? defaultLanguageCode
    }
}
