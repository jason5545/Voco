import Foundation
import SwiftUI
import os

// MARK: - UI Management Extension
extension WhisperState {

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

            // Stop background polling — not needed while recording
            EditModeCacheService.shared.stopPolling()

            // Show the recorder window immediately — before any async work
            await MainActor.run {
                isMiniRecorderVisible = true // This will call showRecorderPanel() via didSet
            }

            // Edit Mode: always read from cache (event-driven invalidation ensures safe defaults)
            let cache = EditModeCacheService.shared
            if cache.cachedIsEditable, let selectedText = cache.cachedSelectedText, !selectedText.isEmpty {
                self.isEditMode = true
                self.editModeSelectedText = selectedText
            } else if cache.cachedFocusedElementUnavailable {
                // AX focused element unavailable (e.g. Claude desktop Electron) — try menuAction fallback (~50ms)
                if let selectedText = await SelectedTextService.fetchSelectedText(), !selectedText.isEmpty {
                    self.isEditMode = true
                    self.editModeSelectedText = selectedText
                } else {
                    self.isEditMode = false
                    self.editModeSelectedText = nil
                }
            } else {
                self.isEditMode = false
                self.editModeSelectedText = nil
            }
            logger.notice("Edit mode from cache: isEdit=\(self.isEditMode), hasText=\(self.editModeSelectedText != nil)")

            // Start recording IMMEDIATELY — zero delay
            await toggleRecord(powerModeId: powerModeId)
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

        // Resume background polling after recording ends
        EditModeCacheService.shared.startPolling()

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

        // Resume background polling after reset
        EditModeCacheService.shared.startPolling()
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
