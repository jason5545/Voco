import Foundation
import SwiftData
import os
import AppKit

@MainActor
final class ModelPrewarmService: ObservableObject {
    private let transcriptionModelManager: TranscriptionModelManager
    private let whisperModelManager: WhisperModelManager
    private let modelContext: ModelContext
    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "ModelPrewarm")
    private lazy var serviceRegistry = TranscriptionServiceRegistry(
        modelProvider: whisperModelManager,
        modelsDirectory: whisperModelManager.modelsDirectory,
        modelContext: modelContext
    )
    private let prewarmAudioURL = Bundle.main.url(forResource: "esc", withExtension: "wav")
    private let prewarmEnabledKey = "PrewarmModelOnWake"

    /// Heartbeat task that periodically warms Metal shader cache
    private var heartbeatTask: Task<Void, Never>?
    /// Interval between heartbeat warmups (seconds)
    private let heartbeatInterval: TimeInterval = 300  // 5 minutes
    /// Track screen sleep state to pause heartbeat
    private var isScreenAsleep = false
    /// Track last successful prewarm to skip redundant warmups
    private var lastPrewarmTime: Date?

    init(transcriptionModelManager: TranscriptionModelManager, whisperModelManager: WhisperModelManager, modelContext: ModelContext) {
        self.transcriptionModelManager = transcriptionModelManager
        self.whisperModelManager = whisperModelManager
        self.modelContext = modelContext
        setupNotifications()
        schedulePrewarmOnAppLaunch()
    }

    // MARK: - Notification Setup

    private func setupNotifications() {
        let center = NSWorkspace.shared.notificationCenter

        // Trigger on wake from sleep
        center.addObserver(
            self,
            selector: #selector(handleSystemWake),
            name: NSWorkspace.didWakeNotification,
            object: nil
        )

        // Trigger on screen wake (covers screen lock/unlock without full system sleep —
        // very common on Apple Silicon Macs that rarely do full sleep)
        center.addObserver(
            self,
            selector: #selector(handleScreenWake),
            name: NSWorkspace.screensDidWakeNotification,
            object: nil
        )

        // Pause heartbeat when screen sleeps to save energy
        center.addObserver(
            self,
            selector: #selector(handleScreenSleep),
            name: NSWorkspace.screensDidSleepNotification,
            object: nil
        )

        logger.notice("ModelPrewarmService initialized - listening for system wake, screen wake/sleep, and app launch")
    }

    // MARK: - Trigger Handlers

    /// Trigger on app launch (cold start)
    private func schedulePrewarmOnAppLaunch() {
        logger.notice("App launched, scheduling prewarm")
        Task {
            try? await Task.sleep(for: .seconds(3))
            await performPrewarm(reason: "app launch")
            startHeartbeat()
        }
    }

    /// Trigger on full system wake from sleep
    @objc private func handleSystemWake() {
        isScreenAsleep = false
        logger.notice("System wake detected, scheduling prewarm")
        Task {
            try? await Task.sleep(for: .seconds(3))
            await performPrewarm(reason: "system wake")
            startHeartbeat()
        }
    }

    /// Trigger on screen wake (more frequent than full system wake on Apple Silicon)
    @objc private func handleScreenWake() {
        isScreenAsleep = false
        logger.notice("Screen wake detected, scheduling prewarm")
        Task {
            try? await Task.sleep(for: .seconds(2))
            await performPrewarm(reason: "screen wake")
            startHeartbeat()
        }
    }

    /// Pause heartbeat when screen sleeps to save energy
    @objc private func handleScreenSleep() {
        isScreenAsleep = true
        stopHeartbeat()
        logger.notice("Screen sleep detected, heartbeat paused")
    }

    // MARK: - Heartbeat (Metal Shader Keep-Alive)

    /// Starts a periodic heartbeat that runs a micro-inference to keep Metal shader cache warm.
    /// Metal compiled shaders are device-global, so warming any engine instance keeps shaders
    /// cached for all instances (including VoiceInkEngine's real transcription engine).
    private func startHeartbeat() {
        guard heartbeatTask == nil, shouldPrewarm() else { return }

        logger.notice("Starting shader heartbeat (interval: \(self.heartbeatInterval, privacy: .public)s)")
        heartbeatTask = Task { [weak self] in
            while !Task.isCancelled {
                do {
                    try await Task.sleep(for: .seconds(self?.heartbeatInterval ?? 300))
                } catch {
                    return  // Task cancelled
                }
                guard let self, !Task.isCancelled, !self.isScreenAsleep else { return }

                // Skip if a recent prewarm happened within 80% of heartbeat interval
                if let last = self.lastPrewarmTime,
                   Date().timeIntervalSince(last) < self.heartbeatInterval * 0.8 {
                    continue
                }

                await self.performPrewarm(reason: "heartbeat")
            }
        }
    }

    /// Stops the heartbeat timer.
    private func stopHeartbeat() {
        heartbeatTask?.cancel()
        heartbeatTask = nil
    }

    // MARK: - Core Prewarming Logic

    private func performPrewarm(reason: String) async {
        guard shouldPrewarm() else { return }

        guard let audioURL = prewarmAudioURL else {
            logger.error("❌ Prewarm audio file (esc.wav) not found")
            return
        }

        guard let currentModel = transcriptionModelManager.currentTranscriptionModel else {
            logger.notice("No model selected, skipping prewarm")
            return
        }

        logger.notice("Prewarming \(currentModel.displayName, privacy: .public) (\(reason, privacy: .public))")
        let startTime = Date()

        do {
            let _ = try await serviceRegistry.transcribe(audioURL: audioURL, model: currentModel)
            let duration = Date().timeIntervalSince(startTime)
            lastPrewarmTime = Date()

            logger.notice("Prewarm completed in \(String(format: "%.2f", duration), privacy: .public)s (\(reason, privacy: .public))")

        } catch {
            logger.error("❌ Prewarm failed (\(reason, privacy: .public)): \(error.localizedDescription, privacy: .public)")
        }
    }

    // MARK: - Validation

    private func shouldPrewarm() -> Bool {
        // Check if user has enabled prewarming
        let isEnabled = UserDefaults.standard.bool(forKey: prewarmEnabledKey)
        guard isEnabled else {
            logger.notice("Prewarm disabled by user")
            return false
        }

        // Only prewarm on-device models (cloud models don't need it)
        guard let model = transcriptionModelManager.currentTranscriptionModel else {
            return false
        }

        switch model.provider {
        case .local, .parakeet, .whisperMLX, .qwen3, .qwen3CoreML:
            return true
        default:
            logger.notice("Skipping prewarm - cloud models don't need it")
            return false
        }
    }

    deinit {
        NSWorkspace.shared.notificationCenter.removeObserver(self)
        heartbeatTask?.cancel()
        logger.notice("ModelPrewarmService deinitialized")
    }
}
