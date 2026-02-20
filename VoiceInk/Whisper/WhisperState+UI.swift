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

            // Detect selected text → decide whether to enter Edit Mode
            // (runs after window is visible; accessibility API can be slow)
            // Skip Edit Mode for terminal apps (hard safety valve — terminal inputs
            // are AX-editable but pasting back risks executing commands).
            let frontApp = NSWorkspace.shared.frontmostApplication
            let frontBundleID = frontApp?.bundleIdentifier
            let isTerminal = frontBundleID.map { Self.terminalBundleIDs.contains($0) } ?? false

            if AXIsProcessTrusted() && !isTerminal,
               let pid = frontApp?.processIdentifier,
               SelectedTextService.isEditableTextFocused(for: pid)
            {
                let selectedText = await SelectedTextService.fetchSelectedTextForEditModeDetection()
                if let selectedText, !selectedText.isEmpty {
                    isEditMode = true
                    editModeSelectedText = selectedText
                } else {
                    isEditMode = false
                    editModeSelectedText = nil
                }
            } else {
                isEditMode = false
                editModeSelectedText = nil
            }

            await toggleRecord(powerModeId: powerModeId)
        }
    }
    
    func dismissMiniRecorder() async {
        logger.notice("dismissMiniRecorder called – state=\(String(describing: self.recordingState))")
        if recordingState == .busy {
            logger.notice("dismissMiniRecorder: early return, state is busy")
            return
        }

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
