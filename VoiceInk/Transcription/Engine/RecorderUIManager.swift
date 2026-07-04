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
            await RecorderPanelStartFlow.run(
                resetStopStateAndCancelModelCleanup: {
                    lastRecordingStopTime = nil
                    engine.cancelScheduledModelCleanup()
                },
                playStartSound: {
                    SoundManager.shared.playStartSound()
                },
                detectEditMode: {
                    await detectEditMode(engine: engine)
                },
                setRecorderPanelVisible: {
                    isRecorderPanelVisible = true
                },
                toggleRecord: {
                    await engine.toggleRecord(modeId: modeId)
                }
            )
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

        engine.forkState.editModeDetectionTask?.cancel()
        engine.forkState.editModeDetectionTask = nil

        let frontmostApplication = NSWorkspace.shared.frontmostApplication
        let bundleID = frontmostApplication?.bundleIdentifier
        let currentPID = frontmostApplication?.processIdentifier
        if let bundleID, EditModeCacheService.terminalBundleIDs.contains(bundleID) {
            engine.forkState.clearEditMode()
            return
        }

        guard let currentPID else {
            engine.forkState.clearEditMode()
            return
        }

        let snapshot = EditModeCacheService.shared.snapshotEditModeState()
        let decision = EditModeDetectionPolicy.initialDecision(
            bundleID: bundleID,
            currentPID: currentPID,
            cachedPID: snapshot.pid,
            cachedIsEditable: snapshot.isEditable,
            focusedElementUnavailable: snapshot.focusedElementUnavailable
        )

        switch decision {
        case .clear:
            engine.forkState.clearEditMode()
            return

        case .applyLive(let cacheMatchesFrontmostApp, let focusedElementUnavailable):
            await applyLiveEditModeDetection(
                engine: engine,
                pid: currentPID,
                bundleID: bundleID,
                cacheMatchesFrontmostApp: cacheMatchesFrontmostApp,
                focusedElementUnavailable: focusedElementUnavailable
            )
        }
    }

    private func applyLiveEditModeDetection(
        engine: VoiceInkEngine,
        pid: pid_t,
        bundleID: String?,
        cacheMatchesFrontmostApp: Bool,
        focusedElementUnavailable: Bool
    ) async {
        if let focusedTextInfo = SelectedTextService.focusedEditableTextInfo(for: pid) {
            let selectedText = await SelectedTextService.fetchSelectedTextForEditModeDetection()
            if let selectedText,
               EditModeDetectionPolicy.shouldRejectAXSelection(
                   role: focusedTextInfo.role,
                   selectedText: selectedText,
                   fieldValue: focusedTextInfo.fieldValue,
                   selectedRangeLength: focusedTextInfo.selectedRangeLength
               ) {
                engine.forkState.clearEditMode()
                return
            }
            applyEditModeResult(engine: engine, hasTrustedEditableSignal: true, selectedText: selectedText)
            return
        }

        guard EditModeDetectionPolicy.canUseElectronSelectionFallback(
            bundleID: bundleID,
            cacheMatchesFrontmostApp: cacheMatchesFrontmostApp,
            focusedElementUnavailable: focusedElementUnavailable
        ) else {
            engine.forkState.clearEditMode()
            return
        }

        engine.forkState.clearEditMode()
        engine.forkState.editModeDetectionTask = Task { @MainActor [weak self, weak engine] in
            guard let self, let engine else { return }

            let selectedText = await SelectedTextService.fetchSelectedTextForElectronFallback()
            guard !Task.isCancelled else { return }

            guard NSWorkspace.shared.frontmostApplication?.processIdentifier == pid else {
                engine.forkState.clearEditMode()
                return
            }

            self.applyEditModeResult(engine: engine, hasTrustedEditableSignal: true, selectedText: selectedText)
        }
    }

    private func applyEditModeResult(
        engine: VoiceInkEngine,
        hasTrustedEditableSignal: Bool,
        selectedText: String?
    ) {
        if EditModeDetectionPolicy.shouldEnterEditMode(
            hasTrustedEditableSignal: hasTrustedEditableSignal,
            selectedText: selectedText
        ) {
            engine.forkState.isEditMode = true
            engine.forkState.editModeSelectedText = selectedText
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

@MainActor
enum RecorderPanelStartFlow {
    static func run(
        resetStopStateAndCancelModelCleanup: @MainActor () -> Void,
        playStartSound: @MainActor () -> Void,
        detectEditMode: @MainActor () async -> Void,
        setRecorderPanelVisible: @MainActor () -> Void,
        toggleRecord: @MainActor () async -> Void,
        beginTrace: @MainActor (String) -> Void = { StartupTracer.begin($0) },
        checkpoint: @MainActor (String) -> Void = { StartupTracer.checkpoint($0) }
    ) async {
        beginTrace("hotkey_press")
        resetStopStateAndCancelModelCleanup()
        checkpoint("cancelModelCleanup_done")
        playStartSound()
        checkpoint("playStartSound_done")
        await detectEditMode()
        checkpoint("detectEditMode_done")
        setRecorderPanelVisible()
        checkpoint("isRecorderPanelVisible_set")
        await toggleRecord()
    }
}

enum EditModeDetectionPolicy {
    enum InitialDecision: Equatable {
        case clear
        case applyLive(cacheMatchesFrontmostApp: Bool, focusedElementUnavailable: Bool)
    }

    private static let electronSelectionFallbackBundleIDs: Set<String> = [
        "com.anthropic.claudefordesktop",
        "com.exafunction.windsurf",
        "com.microsoft.VSCode",
        "com.openai.chat",
        "com.openai.codex",
        "com.todesktop.230313mzl4w4u92",
        "dev.kiro.desktop",
    ]

    static func initialDecision(
        bundleID: String?,
        currentPID: pid_t?,
        cachedPID: pid_t?,
        cachedIsEditable: Bool,
        focusedElementUnavailable: Bool,
        terminalBundleIDs: Set<String> = EditModeCacheService.terminalBundleIDs
    ) -> InitialDecision {
        if let bundleID, terminalBundleIDs.contains(bundleID) {
            return .clear
        }

        guard let currentPID else {
            return .clear
        }

        let cacheMatchesFrontmostApp = cacheMatchesFrontmostApp(
            cachedPID: cachedPID,
            currentPID: currentPID
        )

        if cacheMatchesFrontmostApp,
           !cachedIsEditable,
           !canUseElectronSelectionFallback(
               bundleID: bundleID,
               cacheMatchesFrontmostApp: cacheMatchesFrontmostApp,
               focusedElementUnavailable: focusedElementUnavailable
           ) {
            return .clear
        }

        return .applyLive(
            cacheMatchesFrontmostApp: cacheMatchesFrontmostApp,
            focusedElementUnavailable: focusedElementUnavailable
        )
    }

    static func cacheMatchesFrontmostApp(cachedPID: pid_t?, currentPID: pid_t?) -> Bool {
        guard let cachedPID, let currentPID else { return false }
        return cachedPID == currentPID
    }

    static func isElectronSelectionFallbackBundleID(_ bundleID: String?) -> Bool {
        guard let bundleID else { return false }
        return electronSelectionFallbackBundleIDs.contains(bundleID)
    }

    static func canUseElectronSelectionFallback(
        bundleID: String?,
        cacheMatchesFrontmostApp: Bool,
        focusedElementUnavailable: Bool
    ) -> Bool {
        cacheMatchesFrontmostApp &&
            focusedElementUnavailable &&
            isElectronSelectionFallbackBundleID(bundleID)
    }

    static func shouldEnterEditMode(hasTrustedEditableSignal: Bool, selectedText: String?) -> Bool {
        guard hasTrustedEditableSignal,
              let selectedText,
              !selectedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return false
        }
        return true
    }

    static func isClipboardEcho(candidate: String?, clipboardBaseline: String?) -> Bool {
        guard let candidate, let clipboardBaseline else { return false }
        return candidate == clipboardBaseline
    }

    static func shouldRejectAXSelection(
        role: String,
        selectedText: String,
        fieldValue: String?,
        selectedRangeLength: Int?
    ) -> Bool {
        let singleLineEditableRoles: Set<String> = [
            kAXTextFieldRole as String,
            kAXComboBoxRole as String,
        ]

        if singleLineEditableRoles.contains(role),
           let fieldValue,
           selectedText == fieldValue {
            return true
        }

        if selectedRangeLength == 0, !selectedText.isEmpty {
            return true
        }

        return false
    }
}
