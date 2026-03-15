import AppKit
import Foundation
import os

@MainActor
protocol ModelPrewarmTranscribing: AnyObject {
    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String
}

extension TranscriptionServiceRegistry: ModelPrewarmTranscribing {}

@MainActor
final class ModelPrewarmService: ObservableObject {
    private let transcriptionModelManager: TranscriptionModelManager
    private let serviceRegistry: any ModelPrewarmTranscribing
    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "ModelPrewarm")
    private let prewarmEnabledKey = "PrewarmModelOnWake"
    private let prewarmAudioURL: URL?
    private let prewarmDelay: Duration

    private var scheduledPrewarmTask: Task<Void, Never>?
    private var activePrewarmTask: Task<Void, Never>?

    init(
        transcriptionModelManager: TranscriptionModelManager,
        serviceRegistry: any ModelPrewarmTranscribing,
        prewarmAudioURL: URL? = Bundle.main.url(forResource: "esc", withExtension: "wav"),
        prewarmDelay: Duration = .seconds(3),
        observeWorkspaceNotifications: Bool = true,
        scheduleInitialPrewarm: Bool = true
    ) {
        self.transcriptionModelManager = transcriptionModelManager
        self.serviceRegistry = serviceRegistry
        self.prewarmAudioURL = prewarmAudioURL
        self.prewarmDelay = prewarmDelay

        if observeWorkspaceNotifications {
            setupNotifications()
        }

        if scheduleInitialPrewarm {
            schedulePrewarm(trigger: "app launch")
        }
    }

    // MARK: - Notification Setup

    private func setupNotifications() {
        let center = NSWorkspace.shared.notificationCenter

        // Trigger on wake from sleep
        center.addObserver(
            self,
            selector: #selector(handleWakeNotification),
            name: NSWorkspace.didWakeNotification,
            object: nil
        )

        logger.notice("ModelPrewarmService initialized - listening for wake and app launch")
    }

    // MARK: - Trigger Handlers

    /// Trigger on wake from sleep or screen unlock
    @objc private func handleWakeNotification() {
        schedulePrewarm(trigger: "wake")
    }

    func schedulePrewarm(trigger: String) {
        if activePrewarmTask != nil {
            logger.notice("Prewarm already in progress, ignoring \(trigger, privacy: .public) trigger")
            return
        }

        scheduledPrewarmTask?.cancel()
        let delay = prewarmDelay
        let delayDescription = String(describing: delay)
        logger.notice("Scheduling prewarm for \(trigger, privacy: .public) in \(delayDescription, privacy: .public)")

        scheduledPrewarmTask = Task { [weak self] in
            guard let self else { return }

            do {
                try await Task.sleep(for: delay)
            } catch {
                await self.handleScheduledPrewarmCancellation(trigger: trigger)
                return
            }

            await self.startPrewarmIfNeeded(trigger: trigger)
        }
    }

    private func handleScheduledPrewarmCancellation(trigger: String) {
        logger.notice("Cancelled scheduled prewarm for \(trigger, privacy: .public)")
    }

    private func startPrewarmIfNeeded(trigger: String) {
        scheduledPrewarmTask = nil

        guard activePrewarmTask == nil else {
            logger.notice("Prewarm already in progress, skipping \(trigger, privacy: .public) trigger")
            return
        }

        let task = Task { [weak self] in
            guard let self else { return }
            await self.performPrewarm(trigger: trigger)
        }
        activePrewarmTask = task

        Task { [weak self] in
            await task.value
            await self?.clearActivePrewarmTask()
        }
    }

    private func clearActivePrewarmTask() {
        activePrewarmTask = nil
    }

    // MARK: - Core Prewarming Logic

    private func performPrewarm(trigger: String) async {
        guard shouldPrewarm() else { return }

        guard let audioURL = prewarmAudioURL else {
            logger.error("❌ Prewarm audio file (esc.wav) not found")
            return
        }

        guard let currentModel = transcriptionModelManager.currentTranscriptionModel else {
            logger.notice("No model selected, skipping prewarm")
            return
        }

        logger.notice("Prewarming \(currentModel.displayName, privacy: .public) from \(trigger, privacy: .public)")
        let startTime = Date()

        do {
            let _ = try await serviceRegistry.transcribe(audioURL: audioURL, model: currentModel)
            let duration = Date().timeIntervalSince(startTime)

            logger.notice("Prewarm completed in \(String(format: "%.2f", duration), privacy: .public)s")

        } catch {
            logger.error("❌ Prewarm failed: \(error.localizedDescription, privacy: .public)")
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
        scheduledPrewarmTask?.cancel()
        activePrewarmTask?.cancel()
        NSWorkspace.shared.notificationCenter.removeObserver(self)
        logger.notice("ModelPrewarmService deinitialized")
    }
}
