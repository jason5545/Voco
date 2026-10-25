import Foundation

// Protocol for objects that provide live recorder state to the UI.
@MainActor
protocol RecorderStateProvider: AnyObject {
    var recordingState: RecordingState { get }
    var partialTranscript: String { get }
    var isEditMode: Bool { get }
    var isEphemeralDictation: Bool { get }
    var ephemeralPrivacyNotice: String? { get }
    var pendingDictionaryEntry: WordSubstitution? { get }

    func confirmDictionaryEntry()
    func dismissDictionaryEntry()
}

extension RecorderStateProvider {
    var isEphemeralDictation: Bool { false }
    var ephemeralPrivacyNotice: String? { nil }
}
