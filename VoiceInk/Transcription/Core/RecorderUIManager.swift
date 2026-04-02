import Foundation
import SwiftUI
import os

@MainActor
class RecorderUIManager: ObservableObject {
    @Published var miniRecorderError: String?

    @Published var recorderType: String = UserDefaults.standard.string(forKey: "RecorderType") ?? "mini" {
        didSet {
            if isMiniRecorderVisible {
                if oldValue == "notch" {
                    notchWindowManager?.destroyWindow()
                    notchWindowManager = nil
                } else {
                    miniWindowManager?.destroyWindow()
                    miniWindowManager = nil
                }
                Task { @MainActor in
                    try? await Task.sleep(nanoseconds: 50_000_000)
                    showRecorderPanel()
                }
            }
            UserDefaults.standard.set(recorderType, forKey: "RecorderType")
        }
    }

    @Published var isMiniRecorderVisible = false {
        didSet {
            if isMiniRecorderVisible {
                showRecorderPanel()
            } else {
                hideRecorderPanel()
            }
        }
    }

    var notchWindowManager: NotchWindowManager?
    var miniWindowManager: MiniWindowManager?

    private weak var engine: VoiceInkEngine?
    private var recorder: Recorder?

    private let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "RecorderUIManager")

    // Double-tap hotkey cancel
    private var lastRecordingStopTime: Date?
    private let doublePressCancelThreshold: TimeInterval = 0.4
    private var doublePressStopTask: Task<Void, Never>?

    init() {}

    /// Call after VoiceInkEngine is created to break the circular init dependency.
    func configure(engine: VoiceInkEngine, recorder: Recorder) {
        self.engine = engine
        self.recorder = recorder
        setupNotifications()
        EditModeCacheService.shared.startPolling()
    }

    // MARK: - Recorder Panel Management

    func showRecorderPanel() {
        guard let engine = engine, let recorder = recorder else { return }
        StartupTracer.checkpoint("showRecorderPanel_enter(\(recorderType))")
        logger.notice("Showing \(self.recorderType, privacy: .public) recorder")

        if recorderType == "notch" {
            if notchWindowManager == nil {
                notchWindowManager = NotchWindowManager(engine: engine, recorder: recorder)
            }
            notchWindowManager?.show()
        } else {
            if miniWindowManager == nil {
                miniWindowManager = MiniWindowManager(engine: engine, recorder: recorder)
            }
            miniWindowManager?.show()
        }
    }

    func hideRecorderPanel() {
        if recorderType == "notch" {
            notchWindowManager?.hide()
        } else {
            miniWindowManager?.hide()
        }
    }

    // MARK: - Mini Recorder Management

    func toggleMiniRecorder(powerModeId: UUID? = nil) async {
        guard let engine = engine else { return }
        logger.notice("toggleMiniRecorder called – visible=\(self.isMiniRecorderVisible, privacy: .public), state=\(String(describing: engine.recordingState), privacy: .public)")

        if isMiniRecorderVisible {
            if engine.recordingState == .recording {
                if lastRecordingStopTime != nil {
                    // Second press while still recording — cancel, skip transcribing
                    logger.notice("toggleMiniRecorder: double-press cancel")
                    lastRecordingStopTime = nil
                    doublePressStopTask?.cancel()
                    doublePressStopTask = nil
                    await cancelRecording()
                } else {
                    // First press — wait briefly for possible second press
                    logger.notice("toggleMiniRecorder: first press, waiting for double-press")
                    lastRecordingStopTime = Date()
                    doublePressStopTask?.cancel()
                    doublePressStopTask = Task { @MainActor [weak self] in
                        try? await Task.sleep(nanoseconds: 400_000_000)
                        guard let self, !Task.isCancelled else { return }
                        self.lastRecordingStopTime = nil
                        self.doublePressStopTask = nil
                        self.logger.notice("toggleMiniRecorder: no double-press, stopping normally")
                        await engine.toggleRecord(powerModeId: powerModeId)
                    }
                }
            } else {
                lastRecordingStopTime = nil
            }
        } else {
            StartupTracer.begin("hotkey_press")
            lastRecordingStopTime = nil
            engine.cancelScheduledModelCleanup()
            StartupTracer.checkpoint("cancelModelCleanup_done")
            SoundManager.shared.playStartSound()
            StartupTracer.checkpoint("playStartSound_done")

            await detectEditMode(engine: engine)
            StartupTracer.checkpoint("detectEditMode_done")

            isMiniRecorderVisible = true
            StartupTracer.checkpoint("isMiniRecorderVisible_set")
            await engine.toggleRecord(powerModeId: powerModeId)
        }
    }

    func dismissMiniRecorder() async {
        guard let engine = engine, let recorder = recorder else { return }
        logger.notice("dismissMiniRecorder called – state=\(String(describing: engine.recordingState), privacy: .public)")

        if engine.recordingState == .busy {
            logger.notice("dismissMiniRecorder: early return, state is busy")
            return
        }

        let wasRecording = engine.recordingState == .recording

        await MainActor.run {
            engine.recordingState = .busy
        }

        // Cancel and release any active streaming session to prevent resource leaks.
        engine.currentSession?.cancel()
        engine.currentSession = nil

        if wasRecording {
            await recorder.stopRecording()
        }

        hideRecorderPanel()

        lastRecordingStopTime = nil

        // Clear captured context when the recorder is dismissed
        if let enhancementService = engine.enhancementService {
            await MainActor.run {
                enhancementService.clearCapturedContexts()
            }
        }

        await MainActor.run {
            engine.forkState.editModeDetectionTask?.cancel()
            engine.forkState.editModeDetectionTask = nil
            engine.forkState.clearEditMode()
            engine.forkState.pendingDictionaryEntry = nil
            isMiniRecorderVisible = false
        }

        engine.scheduleModelResourceCleanup()

        if UserDefaults.standard.bool(forKey: PowerModeDefaults.autoRestoreKey) {
            await PowerModeSessionManager.shared.endSession()
            await MainActor.run {
                PowerModeManager.shared.setActiveConfiguration(nil)
            }
        }

        await MainActor.run {
            engine.recordingState = .idle
        }

        // Restart edit mode cache polling so next recording gets fresh AX state
        EditModeCacheService.shared.startPolling()

        logger.notice("dismissMiniRecorder completed")
    }

    func resetOnLaunch() async {
        guard let engine = engine, let recorder = recorder else { return }
        logger.notice("Resetting recording state on launch")
        await recorder.stopRecording()
        hideRecorderPanel()
        await MainActor.run {
            isMiniRecorderVisible = false
            engine.shouldCancelRecording = false
            lastRecordingStopTime = nil
            doublePressStopTask?.cancel()
            doublePressStopTask = nil
            miniRecorderError = nil
            engine.forkState.editModeDetectionTask?.cancel()
            engine.forkState.editModeDetectionTask = nil
            engine.forkState.clearEditMode()
            engine.forkState.pendingDictionaryEntry = nil
            engine.recordingState = .idle
        }
        await engine.cleanupResources()
    }

    func cancelRecording() async {
        guard let engine = engine else { return }
        logger.notice("cancelRecording called")
        lastRecordingStopTime = nil
        doublePressStopTask?.cancel()
        doublePressStopTask = nil
        SoundManager.shared.playEscSound()
        engine.shouldCancelRecording = true
        await dismissMiniRecorder()
        NotificationManager.shared.showNotification(
            title: String(localized: "Recording Cancelled"),
            type: .info,
            duration: 1.5
        )
    }

    // MARK: - Edit Mode Detection (Fork-only)

    /// Detects edit mode state from the AX cache snapshot.
    /// Isolated from toggleMiniRecorder to minimize upstream merge conflicts.
    private func detectEditMode(engine: VoiceInkEngine) async {
        defer { EditModeCacheService.shared.stopPolling() }

        // Direct terminal check — safety net regardless of cache state.
        // Cache may be stale if the user switched apps after polling stopped.
        let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier
        if let bundleID, EditModeCacheService.terminalBundleIDs.contains(bundleID) {
            engine.forkState.clearEditMode()
            logger.notice("Edit mode skipped: terminal app (\(bundleID, privacy: .public))")
            return
        }

        let snapshot = EditModeCacheService.shared.snapshotEditModeState()

        if snapshot.isEditable, let selectedText = snapshot.selectedText, !selectedText.isEmpty {
            engine.forkState.isEditMode = true
            engine.forkState.editModeSelectedText = selectedText
        } else if snapshot.isEditable {
            // Editable field but AX couldn't get selected text (e.g. Chrome URL bar).
            // Fetch inline while the original app is still frontmost so menuAction (⌘C)
            // targets the correct app. The recorder panel hasn't appeared yet.
            if let selectedText = await SelectedTextService.fetchSelectedText(), !selectedText.isEmpty {
                engine.forkState.isEditMode = true
                engine.forkState.editModeSelectedText = selectedText
            } else {
                engine.forkState.clearEditMode()
            }
        } else if snapshot.focusedElementUnavailable {
            // AX focused element unavailable (e.g. Electron apps) — defer menuAction fallback.
            // Don't block recording start; fetch in background and set edit mode later.
            engine.forkState.clearEditMode()
            engine.forkState.editModeDetectionTask = Task { @MainActor [weak engine] in
                guard let engine else { return }
                if let selectedText = await SelectedTextService.fetchSelectedText(), !selectedText.isEmpty {
                    engine.forkState.isEditMode = true
                    engine.forkState.editModeSelectedText = selectedText
                }
            }
        } else {
            engine.forkState.clearEditMode()
        }
        logger.notice("Edit mode from cache: isEdit=\(engine.forkState.isEditMode), hasText=\(engine.forkState.editModeSelectedText != nil), cacheEditable=\(snapshot.isEditable), cacheUnavail=\(snapshot.focusedElementUnavailable)")
    }

    // MARK: - Notification Handling

    private func setupNotifications() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleToggleMiniRecorder),
            name: .toggleMiniRecorder,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleDismissMiniRecorder),
            name: .dismissMiniRecorder,
            object: nil
        )
    }

    @objc public func handleToggleMiniRecorder() {
        logger.notice("handleToggleMiniRecorder: .toggleMiniRecorder notification received")
        Task {
            await toggleMiniRecorder()
        }
    }

    @objc public func handleDismissMiniRecorder() {
        logger.notice("handleDismissMiniRecorder: .dismissMiniRecorder notification received")
        Task {
            await dismissMiniRecorder()
        }
    }
}
