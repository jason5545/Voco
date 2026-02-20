import Foundation
import SwiftUI
import os

// MARK: - UI Management Extension
extension WhisperState {

    /// Terminal app bundle IDs where Edit Mode should be skipped
    /// (selecting text + voice instruction → LLM edit → paste back is unreliable and risky in terminals)
    private static let terminalBundleIDs: Set<String> = [
        "com.apple.Terminal",
        "com.googlecode.iterm2",
        "net.kovidgoyal.kitty",
        "com.mitchellh.ghostty",
        "io.alacritty",
        "dev.warp.Warp-Stable",
        "com.github.wez.wezterm",
        "co.zeit.hyper",
        "org.tabby",
    ]

    // MARK: - Recorder Panel Management
    
    func showRecorderPanel() {
        logger.notice("📱 Showing \(self.recorderType) recorder")
        if recorderType == "notch" {
            if notchWindowManager == nil {
                notchWindowManager = NotchWindowManager(whisperState: self, recorder: recorder)
            }
            notchWindowManager?.show()
        } else {
            if miniWindowManager == nil {
                miniWindowManager = MiniWindowManager(whisperState: self, recorder: recorder)
            }
            miniWindowManager?.show()
        }
    }
    
    func hideRecorderPanel() {
        if recorderType == "notch" {
            notchWindowManager?.hide()
            notchWindowManager = nil
        } else {
            miniWindowManager?.hide()
            miniWindowManager = nil
        }
    }
    
    // MARK: - Mini Recorder Management
    
    func toggleMiniRecorder(powerModeId: UUID? = nil) async {
        logger.notice("toggleMiniRecorder called – visible=\(self.isMiniRecorderVisible), state=\(String(describing: self.recordingState))")
        if isMiniRecorderVisible {
            if recordingState == .recording {
                logger.notice("toggleMiniRecorder: stopping recording (was recording)")
                await toggleRecord(powerModeId: powerModeId)
            } else {
                logger.notice("toggleMiniRecorder: cancelling (was not recording)")
                await cancelRecording()
            }
        } else {
            SoundManager.shared.playStartSound()

            // Show the recorder window immediately — before any async work
            await MainActor.run {
                isMiniRecorderVisible = true // This will call showRecorderPanel() via didSet
            }

            // Capture frontmost app info synchronously (fast, no AX)
            let frontApp = NSWorkspace.shared.frontmostApplication
            let frontBundleID = frontApp?.bundleIdentifier
            let isTerminal = frontBundleID.map { Self.terminalBundleIDs.contains($0) } ?? false
            let axTrusted = AXIsProcessTrusted()
            let frontPid = frontApp?.processIdentifier

            // Start recording IMMEDIATELY — zero delay
            await toggleRecord(powerModeId: powerModeId)

            // Edit Mode detection runs in parallel — does NOT block recording
            // AX queries on Chrome/Electron can take 100-2000ms each; we cap at 500ms.
            cancelEditModeDetectionTask()
            editModeDetectionTask = Task { [weak self] in
                guard let self = self else { return }

                guard axTrusted, !isTerminal, let pid = frontPid else {
                    await MainActor.run {
                        self.isEditMode = false
                        self.editModeSelectedText = nil
                    }
                    return
                }

                // Race AX detection against a 500ms timeout
                let result: (isEdit: Bool, selectedText: String?)? = await withTaskGroup(of: (Bool, String?)?.self) { group in
                    group.addTask {
                        // AX queries (potentially slow on Chrome/Electron)
                        guard SelectedTextService.isEditableTextFocused(for: pid) else {
                            return (false, nil)
                        }
                        let selectedText = await SelectedTextService.fetchSelectedTextForEditModeDetection()
                        if let selectedText, !selectedText.isEmpty {
                            return (true, selectedText)
                        }
                        return (false, nil)
                    }
                    group.addTask {
                        // Timeout sentinel
                        try? await Task.sleep(for: .milliseconds(500))
                        return nil // nil signals timeout
                    }

                    // First non-nil result wins; nil = timeout
                    for await value in group {
                        if let v = value {
                            group.cancelAll()
                            return v
                        }
                    }
                    group.cancelAll()
                    return nil // timeout
                }

                guard !Task.isCancelled else { return }

                await MainActor.run {
                    guard self.recordingState == .recording else { return }
                    if let result, result.isEdit {
                        self.isEditMode = true
                        self.editModeSelectedText = result.selectedText
                    } else {
                        self.isEditMode = false
                        self.editModeSelectedText = nil
                    }
                }
            }
        }
    }
    
    func dismissMiniRecorder() async {
        logger.notice("dismissMiniRecorder called – state=\(String(describing: self.recordingState))")
        if recordingState == .busy {
            logger.notice("dismissMiniRecorder: early return, state is busy")
            return
        }

        cancelStartupPreparationTask()
        cancelEditModeDetectionTask()

        let wasRecording = recordingState == .recording

        await MainActor.run {
            self.recordingState = .busy
        }

        // Cancel and release any active streaming session to prevent resource leaks.
        currentSession?.cancel()
        currentSession = nil

        if wasRecording {
            await recorder.stopRecording()
        }
        
        hideRecorderPanel()
        
        // Clear captured context when the recorder is dismissed
        if let enhancementService = enhancementService {
            await MainActor.run {
                enhancementService.clearCapturedContexts()
            }
        }
        
        await MainActor.run {
            isEditMode = false
            editModeSelectedText = nil
            pendingDictionaryEntry = nil
            isMiniRecorderVisible = false
        }

        await cleanupModelResources()
        
        if UserDefaults.standard.bool(forKey: PowerModeDefaults.autoRestoreKey) {
            await PowerModeSessionManager.shared.endSession()
            await MainActor.run {
                PowerModeManager.shared.setActiveConfiguration(nil)
            }
        }
        
        await MainActor.run {
            recordingState = .idle
        }
        logger.notice("dismissMiniRecorder completed")
    }

    func resetOnLaunch() async {
        logger.notice("🔄 Resetting recording state on launch")
        cancelStartupPreparationTask()
        cancelEditModeDetectionTask()
        await recorder.stopRecording()
        hideRecorderPanel()
        await MainActor.run {
            isMiniRecorderVisible = false
            shouldCancelRecording = false
            miniRecorderError = nil
            isEditMode = false
            editModeSelectedText = nil
            pendingDictionaryEntry = nil
            recordingState = .idle
        }
        await cleanupModelResources()
    }
    
    func cancelRecording() async {
        logger.notice("cancelRecording called")
        SoundManager.shared.playEscSound()
        shouldCancelRecording = true
        await dismissMiniRecorder()
    }
    
    // MARK: - Notification Handling
    
    func setupNotifications() {
        NotificationCenter.default.addObserver(self, selector: #selector(handleToggleMiniRecorder), name: .toggleMiniRecorder, object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(handleDismissMiniRecorder), name: .dismissMiniRecorder, object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(handlePromptChange), name: .promptDidChange, object: nil)
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
    
    @objc func handlePromptChange() {
        // Update the whisper context with the new prompt
        Task {
            await updateContextPrompt()
        }
    }
    
    private func updateContextPrompt() async {
        // Always reload the prompt from UserDefaults to ensure we have the latest
        let currentPrompt = UserDefaults.standard.string(forKey: "TranscriptionPrompt") ?? whisperPrompt.transcriptionPrompt
        
        if let context = whisperContext {
            await context.setPrompt(currentPrompt)
        }
    }
} 
