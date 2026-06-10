// TranscriptionPipeline+ForkFeatures.swift
// Fork-specific pipeline features: Edit Mode branch and Context-Aware Insertion.
// Isolated from TranscriptionPipeline.swift to minimize upstream merge conflicts.

import Foundation
import SwiftData
import AppKit
import os

extension TranscriptionPipeline {

    // MARK: - Edit Mode Branch

    /// Handles the Edit Mode pipeline: voice commands → LLM edit instruction → paste replacement.
    /// Returns true if edit mode was handled (caller should return early), false otherwise.
    func handleEditMode(
        text: String,
        selectedText: String,
        transcription: Transcription,
        enhancementService: AIEnhancementService?,
        onStateChange: @escaping (RecordingState) -> Void,
        shouldCancel: () -> Bool,
        onCleanup: @escaping () async -> Void,
        onDismiss: @escaping () async -> Void,
        onEditModeComplete: ((WordSubstitution?) -> Void)?
    ) async -> Bool {
        // 1. Direct edit commands (no LLM needed)
        if let editCommand = VoiceCommandService.shared.detectEditModeCommand(in: text) {
            logger.notice("🎤 Edit mode command detected: \(editCommand.rawValue, privacy: .private)")
            transcription.transcriptionStatus = TranscriptionStatus.completed.rawValue
            try? modelContext.save()
            NotificationCenter.default.post(name: .transcriptionCompleted, object: transcription)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                editCommand.execute()
            }
            await onDismiss()
            return true
        }

        // 2. LLM-based edit instruction
        guard let enhancementService, enhancementService.isConfigured else {
            logger.warning("⚠️ Edit mode: AI not configured, cannot process LLM instruction")
            transcription.transcriptionStatus = TranscriptionStatus.failed.rawValue
            transcription.text = text
            try? modelContext.save()
            await onDismiss()
            return true
        }

        if shouldCancel() { await onCleanup(); return true }

        onStateChange(.enhancing)

        do {
            let (editedText, editDuration, substitution) = try await enhancementService.enhanceForEditMode(
                instruction: text, selectedText: selectedText
            )
            logger.notice("📝 Edit mode result: \(editedText, privacy: .private)")
            transcription.enhancedText = editedText
            transcription.enhancementDuration = editDuration
            transcription.aiEnhancementModelName = enhancementService.getAIService()?.currentModel
            transcription.transcriptionStatus = TranscriptionStatus.completed.rawValue
            transcription.aiRequestSystemMessage = enhancementService.lastSystemMessageSent
            transcription.aiRequestUserMessage = enhancementService.lastUserMessageSent
            try? modelContext.save()
            NotificationCenter.default.post(name: .transcriptionCompleted, object: transcription)

            if shouldCancel() { await onCleanup(); return true }

            // Paste to replace selected text (no trailing space)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                CursorPaster.pasteAtCursor(editedText)
            }

            // If LLM identified a simple word substitution → show dictionary confirmation
            if let sub = substitution {
                let signal = CorrectionFeedbackService.userSubstitutionSignal(sub)
                transcription.recordCorrectionFeedback(signal)
                try? modelContext.save()
                logEditModeShadowCorrection(signal, transcription: transcription)
                onEditModeComplete?(sub)
                return true
            }

            // Fallback: diff-based extraction when LLM didn't return a substitution
            if let diffSub = AutoCorrectionStagingService.shared.extractSubstitution(
                original: selectedText, edited: editedText
            ) {
                let signal = CorrectionFeedbackService.userSubstitutionSignal(diffSub)
                transcription.recordCorrectionFeedback(signal)
                try? modelContext.save()
                logEditModeShadowCorrection(signal, transcription: transcription)
                // Show dictionary confirmation UI for diff-extracted pair too
                onEditModeComplete?(diffSub)
                return true
            }
        } catch {
            logger.error("❌ Edit mode enhancement failed: \(error.localizedDescription)")
            transcription.transcriptionStatus = TranscriptionStatus.failed.rawValue
            try? modelContext.save()
        }

        await onDismiss()
        return true
    }

    private func logEditModeShadowCorrection(
        _ signal: CorrectionFeedbackSignal?,
        transcription: Transcription
    ) {
        guard PhoneticShadowLogger.isShadowLoggingEnabled, let signal else { return }
        let event = PhoneticShadowEvent.userCorrection(
            signal: signal,
            eventType: .userCorrection,
            source: .userSubstitution,
            utteranceId: transcription.id.uuidString,
            transcriptionDbId: transcription.id.uuidString
        )
        PhoneticShadowLogger.shared.log(event)
    }

    // MARK: - Context-Aware Insertion

    /// Applies context-aware text insertion adjustments (spacing, LLM merge).
    /// Returns the adjusted text ready for pasting.
    func applyContextAwareInsertion(
        _ text: String,
        enhancementService: AIEnhancementService?,
        capturedAppPID: pid_t? = nil
    ) async -> String {
        let contextAwareEnabled = UserDefaults.standard.bool(forKey: "ContextAwareInsertionEnabled")
        let appendSpace = UserDefaults.standard.bool(forKey: "AppendTrailingSpace")

        guard contextAwareEnabled else {
            return text + (appendSpace ? " " : "")
        }

        // Query surrounding text via AX API — prefer PID captured at recording start
        let context: SurroundingTextContext?
        let pid = capturedAppPID ?? NSWorkspace.shared.frontmostApplication?.processIdentifier
        if let pid {
            context = SurroundingTextService.shared.querySurroundingText(for: pid)
        } else {
            context = nil
        }

        // LLM merge takes precedence for mid-text insertion — if it succeeds, skip rule-based entirely
        let llmMergeEnabled = UserDefaults.standard.bool(forKey: "ContextAwareLLMMergeEnabled")
        if llmMergeEnabled,
           let ctx = context, !ctx.textBefore.isEmpty, !ctx.textAfter.isEmpty,
           let enhancementService, enhancementService.isConfigured {
            do {
                let (merged, _) = try await enhancementService.enhanceMerge(
                    insertedText: text,
                    textBefore: ctx.textBefore,
                    textAfter: ctx.textAfter
                )
                if !merged.isEmpty {
                    return merged
                }
            } catch {
                logger.warning("⚠️ LLM merge failed, falling back to rule-based: \(error.localizedDescription)")
            }
        }

        // Rule-based adjustments (fallback, or when LLM merge not applicable)
        return ContextAwareInsertionService.shared.adjust(
            text, context: context, appendTrailingSpace: appendSpace)
    }
}
