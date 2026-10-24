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
        selection: EditModeSelectionSnapshot,
        transcription: Transcription,
        enhancementService: AIEnhancementService?,
        onStateChange: @escaping (RecordingState) -> Void,
        shouldCancel: @escaping () -> Bool,
        onCleanup: @escaping () async -> Void,
        onDismiss: @escaping () async -> Void,
        onEditModeComplete: ((WordSubstitution?) -> Void)?
    ) async -> Bool {
        let selectedText = selection.text

        // 1. Direct edit commands (no LLM needed)
        if let editCommand = VoiceCommandService.shared.detectEditModeCommand(in: text) {
            guard await editModeSelectionIsStillActive(selection) else {
                await cancelEditModeDelivery(
                    transcription: transcription,
                    reason: "Edit mode selection changed before direct command",
                    onDismiss: onDismiss
                )
                return true
            }

            try? await Task.sleep(nanoseconds: 50_000_000)
            guard await editModeSelectionIsStillActive(selection) else {
                await cancelEditModeDelivery(
                    transcription: transcription,
                    reason: "Edit mode selection changed before direct command dispatch",
                    onDismiss: onDismiss
                )
                return true
            }

            logger.notice("🎤 Edit mode command detected: \(editCommand.rawValue, privacy: .private)")
            editCommand.execute()
            transcription.transcriptionStatus = TranscriptionStatus.completed.rawValue
            try? modelContext.save()
            NotificationCenter.default.post(name: .transcriptionCompleted, object: transcription)
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

        guard await editModeSelectionIsStillActive(selection) else {
            await cancelEditModeDelivery(
                transcription: transcription,
                reason: "Edit mode selection changed before enhancement request",
                onDismiss: onDismiss
            )
            return true
        }

        onStateChange(.enhancing)

        do {
            let (editedText, editDuration, substitution) = try await enhancementService.enhanceForEditMode(
                instruction: text, selectedText: selectedText
            )
            logger.notice("📝 Edit mode result: \(editedText, privacy: .private)")

            if shouldCancel() { await onCleanup(); return true }

            guard await editModeSelectionIsStillActive(selection) else {
                await cancelEditModeDelivery(
                    transcription: transcription,
                    reason: "Edit mode selection changed while enhancement was running",
                    onDismiss: onDismiss
                )
                return true
            }

            try? await Task.sleep(nanoseconds: 50_000_000)
            guard await editModeSelectionIsStillActive(selection) else {
                await cancelEditModeDelivery(
                    transcription: transcription,
                    reason: "Edit mode selection changed before replacement paste",
                    onDismiss: onDismiss
                )
                return true
            }

            let pasteResult = await CursorPaster.pasteAtCursorAndWaitUntilPosted(
                editedText,
                targetPID: selection.pid,
                beforePosting: {
                    guard !shouldCancel() else { return false }
                    return await self.editModeSelectionIsStillActive(selection)
                }
            )
            guard pasteResult.didPostPasteCommand else {
                await cancelEditModeDelivery(
                    transcription: transcription,
                    reason: "Edit mode replacement paste was not posted",
                    onDismiss: onDismiss
                )
                return true
            }

            transcription.enhancedText = editedText
            transcription.enhancementDuration = editDuration
            transcription.aiEnhancementModelName = enhancementService.getAIService()?.currentModel
            transcription.transcriptionStatus = TranscriptionStatus.completed.rawValue
            transcription.aiRequestSystemMessage = enhancementService.lastSystemMessageSent
            transcription.aiRequestUserMessage = enhancementService.lastUserMessageSent
            try? modelContext.save()
            NotificationCenter.default.post(name: .transcriptionCompleted, object: transcription)

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

    private func editModeSelectionIsStillActive(
        _ selection: EditModeSelectionSnapshot
    ) async -> Bool {
        guard NSWorkspace.shared.frontmostApplication?.processIdentifier == selection.pid else {
            return false
        }

        let bundleID = NSRunningApplication(processIdentifier: selection.pid)?.bundleIdentifier
        let evidence = await SelectedTextService.currentEditableSelectionEvidence(
            for: selection.pid,
            searchFocusedWindow: EditModeDetectionPolicy.shouldSearchFocusedWindow(bundleID: bundleID)
        )
        guard NSWorkspace.shared.frontmostApplication?.processIdentifier == selection.pid,
              case .selected(let currentText) = evidence else {
            return false
        }
        return currentText == selection.text
    }

    private func cancelEditModeDelivery(
        transcription: Transcription,
        reason: String,
        onDismiss: @escaping () async -> Void
    ) async {
        logger.warning("\(reason, privacy: .public); aborting")
        transcription.transcriptionStatus = TranscriptionStatus.canceled.rawValue
        try? modelContext.save()
        await onDismiss()
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

        // Query surrounding text via AX API — prefer PID captured at recording start
        let context: SurroundingTextContext?
        let pid = capturedAppPID ?? NSWorkspace.shared.frontmostApplication?.processIdentifier
        if let pid {
            context = SurroundingTextService.shared.querySurroundingText(for: pid)
        } else {
            context = nil
        }

        // Boundary overlap removal is always active. It is narrower than the
        // optional context-aware formatting rules and only removes speech that
        // exactly repeats the text immediately before the cursor.
        let boundaryAdjusted = context.map {
            ContextAwareInsertionService.shared.prepareForInsertion(
                text, textBefore: $0.textBefore)
        } ?? ContextAwareInsertionService.shared.removeAdjacentRepeatedPhrases(text)
        guard contextAwareEnabled else {
            return boundaryAdjusted + (appendSpace && !boundaryAdjusted.isEmpty ? " " : "")
        }
        guard !boundaryAdjusted.isEmpty else { return "" }

        // LLM merge takes precedence for mid-text insertion — if it succeeds, skip rule-based entirely
        let llmMergeEnabled = UserDefaults.standard.bool(forKey: "ContextAwareLLMMergeEnabled")
        if llmMergeEnabled,
           let ctx = context, !ctx.textBefore.isEmpty, !ctx.textAfter.isEmpty,
           let enhancementService, enhancementService.isConfigured {
            do {
                let (merged, _) = try await enhancementService.enhanceMerge(
                    insertedText: boundaryAdjusted,
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
            boundaryAdjusted, context: context, appendTrailingSpace: appendSpace)
    }
}
