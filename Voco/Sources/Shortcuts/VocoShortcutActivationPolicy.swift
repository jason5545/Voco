import Foundation

/// When a modifier-only recording shortcut fires.
///
/// Upstream `ee6cf208` delays toggle-mode modifier-only shortcuts (Fn, Option…) until the key is
/// released, so a chord such as Fn+arrow never starts a recording. Voco starts on press instead:
/// Jason speaks right after pressing Fn, and waiting for the release pushed capture past the first
/// syllable (2026-10-25 merge). Chords are still handled by the interruption path, which cancels an
/// accidental start. Double tap needs the release to count taps, so it keeps upstream behavior.
enum VocoShortcutActivationPolicy {
    static func requiresStandaloneModifierRelease(_ mode: RecordingShortcutManager.Mode) -> Bool {
        mode == .doubleTap
    }
}
