import Foundation

struct WordSubstitution {
    let original: String
    let replacement: String
}

struct EditModeSelectionSnapshot: Equatable {
    let text: String
    let pid: pid_t
}

enum EditModeSelectionSnapshotPolicy {
    static func validated(
        _ snapshot: EditModeSelectionSnapshot?,
        capturedAppPID: pid_t?
    ) -> EditModeSelectionSnapshot? {
        guard let snapshot, snapshot.pid == capturedAppPID else { return nil }
        return snapshot
    }
}

@MainActor
private final class EditModeDetectionWaitResolution {
    private var continuation: CheckedContinuation<Bool, Never>?

    init(_ continuation: CheckedContinuation<Bool, Never>) {
        self.continuation = continuation
    }

    @discardableResult
    func resolve(_ completed: Bool) -> Bool {
        guard let continuation else { return false }
        self.continuation = nil
        continuation.resume(returning: completed)
        return true
    }
}

@MainActor
enum EditModeDetectionWaiter {
    static func wait(
        for detectionTask: Task<Void, Never>,
        timeoutNanoseconds: UInt64
    ) async -> Bool {
        await withCheckedContinuation { continuation in
            let resolution = EditModeDetectionWaitResolution(continuation)

            Task { @MainActor in
                await detectionTask.value
                resolution.resolve(true)
            }

            Task { @MainActor in
                do {
                    try await Task.sleep(nanoseconds: timeoutNanoseconds)
                } catch {
                    if resolution.resolve(false) {
                        detectionTask.cancel()
                    }
                    return
                }

                if resolution.resolve(false) {
                    detectionTask.cancel()
                }
            }
        }
    }
}

extension VoiceInkEngine {
    var forkState: ForkEngineState { vocoCoordinator }

    var downloadProgress: [String: Double] {
        get { forkState.downloadProgress }
        set { forkState.downloadProgress = newValue }
    }

    var whisperMLXDownloadStates: [String: Bool] {
        get { forkState.whisperMLXDownloadStates }
        set { forkState.whisperMLXDownloadStates = newValue }
    }

    var qwen3DownloadStates: [String: Bool] {
        get { forkState.qwen3DownloadStates }
        set { forkState.qwen3DownloadStates = newValue }
    }

    var isEditMode: Bool { forkState.isEditMode }
    var editModeSelectedText: String? { forkState.editModeSelectedText }
    var isEphemeralDictation: Bool { dictationPrivacy.isEphemeral }
    var ephemeralPrivacyNotice: String? { dictationPrivacy.privacyNotice }

    var pendingDictionaryEntry: WordSubstitution? {
        get { forkState.pendingDictionaryEntry }
        set { forkState.pendingDictionaryEntry = newValue }
    }

    var capturedFrontAppPID: pid_t? {
        get { forkState.capturedFrontAppPID }
        set { forkState.capturedFrontAppPID = newValue }
    }
}
