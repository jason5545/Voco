import AppKit
import Foundation
import IOKit.ps
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
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "ModelPrewarm")
    private let prewarmEnabledKey = "PrewarmModelOnWake"
    private let prewarmAudioURL: URL?
    private let prewarmDelay: Duration
    private let userDefaults: UserDefaults

    private var scheduledPrewarmTask: Task<Void, Never>?
    private var activePrewarmTask: Task<Void, Never>?

    // MARK: - Keep-Alive

    static let keepAliveEnabledKey = "KeepModelAlive"
    static let keepAliveOnBatteryKey = "KeepModelAliveOnBattery"
    private let keepAliveInterval: TimeInterval = 5 * 60  // 5 minutes
    private var keepAliveTask: Task<Void, Never>?

    /// Whether the keep-alive loop is currently running.
    /// Other subsystems (e.g. deferred model cleanup) can check this to avoid
    /// unloading a model that keep-alive is trying to keep resident.
    var isKeepAliveActive: Bool { keepAliveTask != nil }

    init(
        transcriptionModelManager: TranscriptionModelManager,
        serviceRegistry: any ModelPrewarmTranscribing,
        prewarmAudioURL: URL? = Bundle.main.url(forResource: "sound7", withExtension: "wav"),
        prewarmDelay: Duration = .seconds(3),
        userDefaults: UserDefaults = .standard,
        observeWorkspaceNotifications: Bool = true,
        scheduleInitialPrewarm: Bool = true
    ) {
        self.transcriptionModelManager = transcriptionModelManager
        self.serviceRegistry = serviceRegistry
        self.prewarmAudioURL = prewarmAudioURL
        self.prewarmDelay = prewarmDelay
        self.userDefaults = userDefaults

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
            guard let self else { return }
            self.clearActivePrewarmTask()
            // (Re)start keep-alive after prewarm completes from launch or wake
            self.startKeepAlive()
        }
    }

    private func clearActivePrewarmTask() {
        activePrewarmTask = nil
    }

    // MARK: - Core Prewarming Logic

    private func performPrewarm(trigger: String) async {
        guard shouldPrewarm() else { return }

        guard let audioURL = prewarmAudioURL else {
            logger.error("❌ Prewarm audio file (sound7.wav) not found")
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
        let isEnabled = userDefaults.bool(forKey: prewarmEnabledKey)
        guard isEnabled else {
            logger.notice("Prewarm disabled by user")
            return false
        }

        // Only prewarm on-device models (cloud models don't need it)
        guard let model = transcriptionModelManager.currentTranscriptionModel else {
            return false
        }

        switch model.provider {
        case .whisper, .fluidAudio, .whisperMLX, .qwen3, .qwen3CoreML:
            return true
        default:
            logger.notice("Skipping prewarm - cloud models don't need it")
            return false
        }
    }

    // MARK: - Keep-Alive Loop

    /// Starts or restarts the keep-alive loop.
    /// Called after prewarm completes on app launch and wake, and when settings change.
    func startKeepAlive() {
        keepAliveTask?.cancel()
        guard userDefaults.bool(forKey: Self.keepAliveEnabledKey) else {
            keepAliveTask = nil
            logger.notice("Keep-alive disabled")
            return
        }
        guard shouldPrewarm() else {
            keepAliveTask = nil
            return
        }

        logger.notice("Keep-alive started (interval: \(String(format: "%.0f", self.keepAliveInterval))s)")
        keepAliveTask = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                do {
                    try await Task.sleep(for: .seconds(self.keepAliveInterval))
                } catch {
                    break
                }
                guard !Task.isCancelled else { break }

                // Skip if on battery and user hasn't opted in
                if self.isOnBattery && !self.userDefaults.bool(forKey: Self.keepAliveOnBatteryKey) {
                    self.logger.notice("Keep-alive ping skipped (on battery)")
                    continue
                }

                await self.performPrewarm(trigger: "keep-alive")
            }
        }
    }

    /// Stops the keep-alive loop.
    func stopKeepAlive() {
        keepAliveTask?.cancel()
        keepAliveTask = nil
        logger.notice("Keep-alive stopped")
    }

    // MARK: - Battery Detection

    private var isOnBattery: Bool {
        guard let snapshot = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let sources = IOPSCopyPowerSourcesList(snapshot)?.takeRetainedValue() as? [CFTypeRef],
              !sources.isEmpty else {
            // Desktop Mac or unable to read — assume AC power
            return false
        }
        for source in sources {
            if let desc = IOPSGetPowerSourceDescription(snapshot, source)?.takeUnretainedValue() as? [String: Any],
               let state = desc[kIOPSPowerSourceStateKey] as? String {
                if state == kIOPSBatteryPowerValue {
                    return true
                }
            }
        }
        return false
    }

    deinit {
        scheduledPrewarmTask?.cancel()
        activePrewarmTask?.cancel()
        keepAliveTask?.cancel()
        NSWorkspace.shared.notificationCenter.removeObserver(self)
        logger.notice("ModelPrewarmService deinitialized")
    }
}
