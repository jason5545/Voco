// VoiceInkEngine+ForkProperties.swift
// Fork-specific properties and state management for Voco
// Properties needed by WhisperMLX, Qwen3, WhisperCoreML, and Edit Mode

import Foundation
import Combine

// MARK: - Word Substitution (Edit Mode dictionary suggestion)
struct WordSubstitution {
    let original: String
    let replacement: String
}

struct EditModeSelectionSnapshot: Equatable {
    let text: String
    let pid: pid_t
}

enum EditModeSelectionSnapshotPolicy {
    /// Detection and recording startup are separate async steps. Only carry an
    /// Edit Mode selection into the pipeline when both still refer to the same
    /// foreground process.
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
    /// Races an unstructured AX detection task against a real deadline. A task
    /// group cannot be used here because leaving its scope waits for the losing
    /// child and turns the nominal timeout into an unbounded wait.
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

// MARK: - Fork-specific stored properties via associated objects
// Since Swift extensions cannot add stored properties,
// we use a dedicated holder class initialized in VoiceInkEngine.

/// Container for fork-specific mutable state on VoiceInkEngine.
@MainActor
class ForkEngineState: ObservableObject {
    @Published var downloadProgress: [String: Double] = [:]
    @Published var whisperMLXDownloadStates: [String: Bool] = [:]
    @Published var qwen3DownloadStates: [String: Bool] = [:]
    @Published private(set) var editModeSelection: EditModeSelectionSnapshot?
    @Published var pendingDictionaryEntry: WordSubstitution?
    /// Tracks the deferred edit mode detection task so it can be cancelled on dismiss.
    var editModeDetectionTask: Task<Void, Never>?

    var isEditMode: Bool { editModeSelection != nil }
    var editModeSelectedText: String? { editModeSelection?.text }

    func armEditMode(selectedText: String, pid: pid_t) {
        editModeSelection = EditModeSelectionSnapshot(text: selectedText, pid: pid)
    }

    func clearEditMode() {
        editModeSelection = nil
    }
}

extension VoiceInkEngine {
    /// Access the fork state container. Lazily created via objc associated objects.
    var forkState: ForkEngineState {
        if let existing = objc_getAssociatedObject(self, &AssociatedKeys.forkState) as? ForkEngineState {
            return existing
        }
        let state = ForkEngineState()
        // Forward forkState changes to engine so SwiftUI views observing engine will re-render
        let cancellable = state.objectWillChange.sink { [weak self] _ in
            self?.objectWillChange.send()
        }
        objc_setAssociatedObject(self, &AssociatedKeys.forkStateSink, cancellable, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        objc_setAssociatedObject(self, &AssociatedKeys.forkState, state, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        return state
    }

    // Convenience accessors
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

    var isEditMode: Bool {
        forkState.isEditMode
    }

    var editModeSelectedText: String? {
        forkState.editModeSelectedText
    }

    var pendingDictionaryEntry: WordSubstitution? {
        get { forkState.pendingDictionaryEntry }
        set { forkState.pendingDictionaryEntry = newValue }
    }
}

private enum AssociatedKeys {
    nonisolated(unsafe) static var forkState = "forkState"
    nonisolated(unsafe) static var forkStateSink = "forkStateSink"
}
