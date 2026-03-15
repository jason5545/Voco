// VoiceInkEngine+ForkFeatures.swift
// Fork-specific features: Edit Mode pipeline dispatch, dictionary dismiss timer,
// deferred model cleanup, and EditModeCacheService integration.
// Isolated from VoiceInkEngine.swift to minimize upstream merge conflicts.

import Foundation
import SwiftData
import os

extension VoiceInkEngine {

    // MARK: - Edit Mode Pipeline Dispatch

    /// Captures edit mode state and runs the pipeline with fork-specific parameters.
    /// Call this instead of pipeline.run() directly to inject edit mode callbacks.
    func runPipelineWithForkFeatures(
        transcription: Transcription,
        audioURL: URL,
        model: any TranscriptionModel,
        session: TranscriptionSession?
    ) async {
        let editMode = forkState.isEditMode
        let editSelectedText = forkState.editModeSelectedText

        await pipeline.run(
            transcription: transcription,
            audioURL: audioURL,
            model: model,
            session: session,
            isEditMode: editMode,
            editModeSelectedText: editSelectedText,
            onStateChange: { [weak self] state in self?.recordingState = state },
            shouldCancel: { [weak self] in self?.shouldCancelRecording ?? false },
            onCleanup: { [weak self] in await self?.cleanupResources() },
            onDismiss: { [weak self] in
                self?.forkState.isEditMode = false
                self?.forkState.editModeSelectedText = nil
                await self?.recorderUIManager?.dismissMiniRecorder()
            },
            onEditModeComplete: { [weak self] substitution in
                self?.forkState.pendingDictionaryEntry = substitution
                self?.forkState.isEditMode = false
                self?.forkState.editModeSelectedText = nil
                self?.recordingState = .idle
                self?.startDictionaryDismissTimer()
            }
        )

        forkState.isEditMode = false
        forkState.editModeSelectedText = nil
    }

    // MARK: - Dictionary Dismiss Timer

    /// Confirms the pending dictionary entry and saves it as a WordReplacement.
    func confirmDictionaryEntry() {
        guard let entry = forkState.pendingDictionaryEntry else { return }
        dictionaryDismissTimer?.cancel()
        dictionaryDismissTimer = nil

        let replacement = WordReplacement(
            originalText: entry.original,
            replacementText: entry.replacement
        )
        modelContext.insert(replacement)
        try? modelContext.save()

        NotificationManager.shared.showNotification(
            title: "\(entry.original) → \(entry.replacement)",
            type: .success,
            duration: 2.0
        )
        forkState.pendingDictionaryEntry = nil
        Task { await recorderUIManager?.dismissMiniRecorder() }
    }

    /// Dismisses the pending dictionary entry without saving.
    func dismissDictionaryEntry() {
        dictionaryDismissTimer?.cancel()
        dictionaryDismissTimer = nil
        forkState.pendingDictionaryEntry = nil
        Task { await recorderUIManager?.dismissMiniRecorder() }
    }

    /// Starts a 15-second timer that auto-dismisses the dictionary entry suggestion.
    func startDictionaryDismissTimer() {
        dictionaryDismissTimer?.cancel()
        let work = DispatchWorkItem { [weak self] in
            Task { @MainActor in
                self?.forkState.pendingDictionaryEntry = nil
                await self?.recorderUIManager?.dismissMiniRecorder()
            }
        }
        dictionaryDismissTimer = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 15, execute: work)
    }

    // MARK: - Deferred Model Cleanup

    /// Cancels any pending model cleanup timer.
    func cancelScheduledModelCleanup() {
        deferredModelCleanupTask?.cancel()
        deferredModelCleanupTask = nil
    }

    /// Schedules model resource cleanup after the configured keep-alive period.
    func scheduleModelResourceCleanup() {
        cancelScheduledModelCleanup()

        let configuredKeepAlive = UserDefaults.standard.double(forKey: modelKeepAliveSecondsKey)
        let keepAliveSeconds = max(0, configuredKeepAlive)
        guard keepAliveSeconds > 0 else {
            Task { [weak self] in
                await self?.cleanupResources()
            }
            return
        }

        logger.notice("cleanupModelResources: scheduled in \(String(format: "%.0f", keepAliveSeconds))s")
        deferredModelCleanupTask = Task { [weak self] in
            guard let self else { return }
            do {
                try await Task.sleep(for: .seconds(keepAliveSeconds))
            } catch {
                return
            }
            guard !Task.isCancelled else { return }
            await self.cleanupResources()
        }
    }

    // MARK: - EditModeCacheService Integration

    /// Caches app context from EditModeCacheService into the enhancement service.
    /// Called during recording start, before the app loses frontmost status.
    func cacheEditModeAppContext(capturedAppName: String?) {
        guard let enhancementService else { return }
        let editCache = EditModeCacheService.shared
        enhancementService.cachedAppName = editCache.cachedAppName ?? capturedAppName
        enhancementService.cachedWindowTitle = editCache.cachedWindowTitle
        enhancementService.cachedSelectedText = editCache.cachedSelectedText
    }
}

// MARK: - Fork Feature Stored Properties

// These use associated objects since Swift extensions cannot add stored properties.
extension VoiceInkEngine {
    var dictionaryDismissTimer: DispatchWorkItem? {
        get { objc_getAssociatedObject(self, &ForkFeatureKeys.dictionaryDismissTimer) as? DispatchWorkItem }
        set { objc_setAssociatedObject(self, &ForkFeatureKeys.dictionaryDismissTimer, newValue, .OBJC_ASSOCIATION_RETAIN_NONATOMIC) }
    }

    var deferredModelCleanupTask: Task<Void, Never>? {
        get { objc_getAssociatedObject(self, &ForkFeatureKeys.deferredModelCleanupTask) as? Task<Void, Never> }
        set { objc_setAssociatedObject(self, &ForkFeatureKeys.deferredModelCleanupTask, newValue, .OBJC_ASSOCIATION_RETAIN_NONATOMIC) }
    }

    var modelKeepAliveSecondsKey: String { "ModelKeepAliveSeconds" }
}

private enum ForkFeatureKeys {
    nonisolated(unsafe) static var dictionaryDismissTimer = "dictionaryDismissTimer"
    nonisolated(unsafe) static var deferredModelCleanupTask = "deferredModelCleanupTask"
}
