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

struct VocoCandidateReview: Identifiable {
    static let timeoutSeconds: TimeInterval = 20

    let id = UUID()
    let candidates: [String]
    let candidateLabels: [String]
    let hypotheses: [VocoHypothesis]
    let confidenceScore: Double
    let reasons: [String]
    let reviewTriggers: [VocoReviewTrigger]

    init(
        candidates: [String],
        candidateLabels: [String] = [],
        hypotheses: [VocoHypothesis] = [],
        confidenceScore: Double,
        reasons: [String],
        reviewTriggers: [VocoReviewTrigger] = []
    ) {
        self.candidates = candidates
        self.candidateLabels = candidateLabels
        self.hypotheses = hypotheses
        self.confidenceScore = confidenceScore
        self.reasons = reasons
        self.reviewTriggers = reviewTriggers
    }

    var defaultCandidate: String? {
        candidates.first
    }

    var timeoutFallbackCandidate: String? {
        defaultCandidate
    }

    func keyboardShortcutForCandidate(at index: Int) -> String? {
        guard candidates.indices.contains(index),
              index < 5
        else { return nil }
        return "\(index + 1)"
    }

    static func shouldRefreshTimeout(forTypedCandidate typedCandidate: String) -> Bool {
        !typedCandidate.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func labelForCandidate(at index: Int) -> String {
        guard candidateLabels.indices.contains(index) else { return "Candidate" }
        return candidateLabels[index]
    }

    func hypothesisForCandidate(at index: Int) -> VocoHypothesis? {
        guard hypotheses.indices.contains(index) else { return nil }
        return hypotheses[index]
    }

    func sourceDisplayNameForCandidate(at index: Int) -> String? {
        hypothesisForCandidate(at: index)?.sourceDisplayName
    }

    var displayReasons: [String] {
        VocoSignalDisplayFormatter.displayReasons(for: reasons)
    }

    var displayReviewSignals: [String] {
        let triggerSummaries = VocoReviewTriggerDisplayFormatter.summaries(for: reviewTriggers)
        return triggerSummaries.isEmpty ? displayReasons : triggerSummaries
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
    @Published var isEditMode: Bool = false
    @Published var editModeSelectedText: String?
    @Published var pendingDictionaryEntry: WordSubstitution?
    @Published var pendingCandidateReview: VocoCandidateReview?
    var pendingCandidateContinuation: CheckedContinuation<VocoCandidateSelection?, Never>?
    var pendingCandidateTimeoutTask: Task<Void, Never>?
    /// Tracks the deferred edit mode detection task so it can be cancelled on dismiss.
    var editModeDetectionTask: Task<Void, Never>?

    func clearEditMode() {
        isEditMode = false
        editModeSelectedText = nil
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

    var pendingCandidateReview: VocoCandidateReview? {
        get { forkState.pendingCandidateReview }
        set { forkState.pendingCandidateReview = newValue }
    }
}

private enum AssociatedKeys {
    nonisolated(unsafe) static var forkState = "forkState"
    nonisolated(unsafe) static var forkStateSink = "forkStateSink"
}
