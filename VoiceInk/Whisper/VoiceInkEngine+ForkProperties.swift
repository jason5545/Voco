// VoiceInkEngine+ForkProperties.swift
// Fork-specific properties and state management for Voco
// Properties needed by WhisperMLX, Qwen3, WhisperCoreML, and Edit Mode

import Foundation

// MARK: - Word Substitution (Edit Mode dictionary suggestion)
struct WordSubstitution {
    let original: String
    let replacement: String
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
    @Published var isEditMode: Bool = false
    @Published var editModeSelectedText: String?
    @Published var pendingDictionaryEntry: WordSubstitution?
}

extension VoiceInkEngine {
    /// Access the fork state container. Lazily created via objc associated objects.
    var forkState: ForkEngineState {
        if let existing = objc_getAssociatedObject(self, &AssociatedKeys.forkState) as? ForkEngineState {
            return existing
        }
        let state = ForkEngineState()
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
        get { forkState.isEditMode }
        set { forkState.isEditMode = newValue }
    }

    var editModeSelectedText: String? {
        get { forkState.editModeSelectedText }
        set { forkState.editModeSelectedText = newValue }
    }

    var pendingDictionaryEntry: WordSubstitution? {
        get { forkState.pendingDictionaryEntry }
        set { forkState.pendingDictionaryEntry = newValue }
    }
}

private enum AssociatedKeys {
    nonisolated(unsafe) static var forkState = "forkState"
}
