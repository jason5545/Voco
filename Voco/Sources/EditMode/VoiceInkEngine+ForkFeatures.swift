// VoiceInkEngine+ForkFeatures.swift
// Fork-specific features: Edit Mode pipeline dispatch, dictionary dismiss timer,
// deferred model cleanup, and EditModeCacheService integration.
// Isolated from VoiceInkEngine.swift to minimize upstream merge conflicts.

import Foundation
import SwiftData
import os

extension VoiceInkEngine {

    // MARK: - Dictionary Dismiss Timer

    /// Confirms the pending dictionary entry and saves it as a WordReplacement.
    /// Promotes an existing staged entry if one exists, otherwise creates a new one.
    func confirmDictionaryEntry() {
        guard let entry = forkState.pendingDictionaryEntry else { return }
        dictionaryDismissTimer?.cancel()
        dictionaryDismissTimer = nil

        let orig = entry.original
        let repl = entry.replacement
        let descriptor = FetchDescriptor<WordReplacement>(
            predicate: #Predicate<WordReplacement> {
                $0.originalText == orig && $0.replacementText == repl
            }
        )

        if let existing = (try? modelContext.fetch(descriptor))?.first {
            existing.isEnabled = true
            existing.source = WordReplacement.sourceUser
            existing.lastSeenDate = Date()
        } else {
            let replacement = WordReplacement(
                originalText: entry.original,
                replacementText: entry.replacement,
                source: WordReplacement.sourceUser
            )
            modelContext.insert(replacement)
        }
        try? modelContext.save()

        NotificationManager.shared.showNotification(
            title: "\(entry.original) → \(entry.replacement)",
            type: .success,
            duration: 2.0
        )
        forkState.pendingDictionaryEntry = nil
        Task { await recorderUIManager?.dismissRecorderPanel() }
    }

    /// Dismisses the pending dictionary entry without saving.
    func dismissDictionaryEntry() {
        dictionaryDismissTimer?.cancel()
        dictionaryDismissTimer = nil
        forkState.pendingDictionaryEntry = nil
        Task { await recorderUIManager?.dismissRecorderPanel() }
    }

    /// Starts a 15-second timer that rejects the dictionary entry on expiry.
    func startDictionaryDismissTimer() {
        dictionaryDismissTimer?.cancel()
        let work = DispatchWorkItem { [weak self] in
            Task { @MainActor in
                guard let self else { return }
                self.forkState.pendingDictionaryEntry = nil
                await self.recorderUIManager?.dismissRecorderPanel()
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
    /// If `ModelPrewarmService.isKeepAliveActive` is true, cleanup is suppressed
    /// so the model stays resident for fast transcription.
    func scheduleModelResourceCleanup() {
        cancelScheduledModelCleanup()

        // Keep-alive ping keeps memory pages resident — don't unload the model
        if prewarmService?.isKeepAliveActive == true {
            logger.notice("cleanupModelResources: skipped (keep-alive active)")
            return
        }

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
