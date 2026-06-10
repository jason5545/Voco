import Foundation
import SwiftUI
import AppKit
import os

enum RecorderPanelStyle: String, CaseIterable, Identifiable {
    case notch
    case mini

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .notch:
            return "Notch"
        case .mini:
            return "Mini"
        }
    }

    static var stored: RecorderPanelStyle {
        let rawValue = UserDefaults.standard.string(forKey: "RecorderType") ?? RecorderPanelStyle.mini.rawValue
        return RecorderPanelStyle(rawValue: rawValue) ?? .mini
    }
}

@MainActor
protocol RecorderPanelPresenting: AnyObject {
    var isRecorderPanelVisible: Bool { get }
    func dismissRecorderPanel() async
}

@MainActor
class RecorderUIManager: ObservableObject, RecorderPanelPresenting {
    @Published var recorderPanelStyle: RecorderPanelStyle = .stored {
        didSet {
            guard oldValue != recorderPanelStyle else { return }
            rebuildVisiblePanel(previousStyle: oldValue)
            UserDefaults.standard.set(recorderPanelStyle.rawValue, forKey: "RecorderType")
        }
    }

    var recorderType: String {
        get { recorderPanelStyle.rawValue }
        set { recorderPanelStyle = RecorderPanelStyle(rawValue: newValue) ?? .mini }
    }

    @Published var isRecorderPanelVisible = false {
        didSet {
            guard oldValue != isRecorderPanelVisible else { return }

            if isRecorderPanelVisible {
                showRecorderPanel()
            } else {
                hideRecorderPanel()
            }
        }
    }

    private var notchWindowManager: NotchWindowManager?
    private var miniWindowManager: MiniWindowManager?

    private weak var engine: VoiceInkEngine?
    private var recorder: Recorder?

    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "RecorderUIManager")
    private var lastRecordingStopTime: Date?
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

    private func showRecorderPanel() {
        guard let engine = engine, let recorder = recorder else { return }
        StartupTracer.checkpoint("showRecorderPanel_enter(\(recorderPanelStyle.rawValue))")

        switch recorderPanelStyle {
        case .notch:
            if notchWindowManager == nil {
                notchWindowManager = NotchWindowManager(
                    engine: engine,
                    recorder: recorder,
                    assistantSession: engine.assistantSession,
                    onRecordButtonTapped: { [weak self] in
                        Task { @MainActor in
                            await self?.toggleRecorderPanel()
                        }
                    },
                    onCloseTapped: { [weak self] in
                        Task { @MainActor in
                            await self?.dismissRecorderPanel()
                        }
                    },
                    onAssistantFollowUp: { [weak engine] text in
                        Task { @MainActor in
                            await engine?.sendAssistantFollowUp(text)
                        }
                    }
                )
            }
            notchWindowManager?.show()
        case .mini:
            if miniWindowManager == nil {
                miniWindowManager = MiniWindowManager(
                    engine: engine,
                    recorder: recorder,
                    assistantSession: engine.assistantSession,
                    onRecordButtonTapped: { [weak self] in
                        Task { @MainActor in
                            await self?.toggleRecorderPanel()
                        }
                    },
                    onCloseTapped: { [weak self] in
                        Task { @MainActor in
                            await self?.dismissRecorderPanel()
                        }
                    },
                    onAssistantFollowUp: { [weak engine] text in
                        Task { @MainActor in
                            await engine?.sendAssistantFollowUp(text)
                        }
                    }
                )
            }
            miniWindowManager?.show()
        }
    }

    private func hideRecorderPanel() {
        switch recorderPanelStyle {
        case .notch:
            notchWindowManager?.hide()
        case .mini:
            miniWindowManager?.hide()
        }
    }

    private func rebuildVisiblePanel(previousStyle: RecorderPanelStyle) {
        guard isRecorderPanelVisible else { return }

        switch previousStyle {
        case .notch:
            notchWindowManager?.destroyWindow()
            notchWindowManager = nil
        case .mini:
            miniWindowManager?.destroyWindow()
            miniWindowManager = nil
        }

        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 50_000_000)
            showRecorderPanel()
        }
    }

    // MARK: - Recorder Panel Management

    func toggleRecorderPanel(modeId: UUID? = nil) async {
        guard let engine = engine else { return }

        if isRecorderPanelVisible {
            switch engine.recordingState {
            case .recording:
                if lastRecordingStopTime != nil {
                    lastRecordingStopTime = nil
                    doublePressStopTask?.cancel()
                    doublePressStopTask = nil
                    await cancelRecording()
                } else {
                    lastRecordingStopTime = Date()
                    doublePressStopTask?.cancel()
                    doublePressStopTask = Task { @MainActor [weak self, weak engine] in
                        try? await Task.sleep(nanoseconds: 400_000_000)
                        guard let self, let engine, !Task.isCancelled else { return }
                        self.lastRecordingStopTime = nil
                        self.doublePressStopTask = nil
                        await engine.toggleRecord(modeId: modeId)
                    }
                }
            case .starting, .transcribing, .enhancing:
                await cancelRecording()
            case .idle:
                if engine.assistantSession.canSendFollowUp {
                    SoundManager.shared.playStartSound()
                    await engine.toggleRecord(
                        modeId: modeId,
                        isAssistantFollowUp: true
                    )
                } else {
                    await dismissRecorderPanel()
                }
            case .busy:
                await dismissRecorderPanel()
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
            isRecorderPanelVisible = true
            StartupTracer.checkpoint("isRecorderPanelVisible_set")
            await engine.toggleRecord(modeId: modeId)
        }
    }

    func dismissRecorderPanel() async {
        guard let engine = engine else { return }

        hideRecorderPanel()
        isRecorderPanelVisible = false
        engine.assistantSession.reset()
        lastRecordingStopTime = nil
        doublePressStopTask?.cancel()
        doublePressStopTask = nil
        engine.forkState.editModeDetectionTask?.cancel()
        engine.forkState.editModeDetectionTask = nil
        engine.forkState.clearEditMode()
        engine.forkState.pendingDictionaryEntry = nil
        engine.dismissCandidateReview()
        engine.scheduleModelResourceCleanup()
        EditModeCacheService.shared.startPolling()
    }

    func resetOnLaunch() async {
        guard let engine = engine else { return }
        logger.notice("Resetting recording state on launch")
        await engine.resetRecordingSession()
        hideRecorderPanel()
        isRecorderPanelVisible = false
        engine.assistantSession.reset()
        engine.forkState.editModeDetectionTask?.cancel()
        engine.forkState.editModeDetectionTask = nil
        engine.forkState.clearEditMode()
        engine.forkState.pendingDictionaryEntry = nil
        engine.dismissCandidateReview()
    }

    func cancelRecording() async {
        guard let engine = engine else { return }
        lastRecordingStopTime = nil
        doublePressStopTask?.cancel()
        doublePressStopTask = nil
        SoundManager.shared.playEscSound()
        await engine.cancelRecording()
        await dismissRecorderPanel()
        NotificationManager.shared.showNotification(
            title: String(localized: "Recording Cancelled"),
            type: .info,
            duration: 1.5
        )
    }

    // MARK: - Edit Mode Detection

    private func detectEditMode(engine: VoiceInkEngine) async {
        defer { EditModeCacheService.shared.stopPolling() }

        let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier
        if let bundleID, EditModeCacheService.terminalBundleIDs.contains(bundleID) {
            engine.forkState.clearEditMode()
            return
        }

        let snapshot = EditModeCacheService.shared.snapshotEditModeState()
        let currentPID = NSWorkspace.shared.frontmostApplication?.processIdentifier
        let cacheMatchesFrontmostApp = EditModeDetectionPolicy.cacheMatchesFrontmostApp(
            cachedPID: snapshot.pid,
            currentPID: currentPID
        )

        if cacheMatchesFrontmostApp,
           snapshot.isEditable,
           let selectedText = snapshot.selectedText,
           !selectedText.isEmpty {
            engine.forkState.isEditMode = true
            engine.forkState.editModeSelectedText = selectedText
        } else if cacheMatchesFrontmostApp, snapshot.isEditable {
            if let selectedText = await SelectedTextService.fetchSelectedText(), !selectedText.isEmpty {
                engine.forkState.isEditMode = true
                engine.forkState.editModeSelectedText = selectedText
            } else {
                engine.forkState.clearEditMode()
            }
        } else if !cacheMatchesFrontmostApp || snapshot.focusedElementUnavailable {
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
    }

    // MARK: - Notification Handling

    private func setupNotifications() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleToggleRecorderPanelNotification),
            name: .toggleRecorderPanel,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleDismissRecorderPanelNotification),
            name: .dismissRecorderPanel,
            object: nil
        )
    }

    @objc public func handleToggleRecorderPanelNotification() {
        Task {
            await toggleRecorderPanel()
        }
    }

    @objc public func handleDismissRecorderPanelNotification() {
        Task {
            switch engine?.recordingState {
            case .starting, .recording, .transcribing, .enhancing:
                await cancelRecording()
            case .idle, .busy, nil:
                await dismissRecorderPanel()
            }
        }
    }
}

enum EditModeDetectionPolicy {
    static func cacheMatchesFrontmostApp(cachedPID: pid_t?, currentPID: pid_t?) -> Bool {
        guard let cachedPID, let currentPID else { return false }
        return cachedPID == currentPID
    }
}
