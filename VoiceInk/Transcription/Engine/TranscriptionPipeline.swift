import Foundation
import AVFoundation
import AppKit
import SwiftData
import os

/// Handles the full post-recording pipeline:
/// transcribe → filter → format → word-replace → Chinese post-process → voice-command → prompt-detect → AI enhance → validate → save → paste → dismiss
@MainActor
class TranscriptionPipeline {
    let modelContext: ModelContext
    private let serviceRegistry: TranscriptionServiceRegistry
    private let enhancementService: AIEnhancementService?
    private let promptDetectionService = PromptDetectionService()
    let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "TranscriptionPipeline")

    var licenseViewModel: LicenseViewModel

    init(
        modelContext: ModelContext,
        serviceRegistry: TranscriptionServiceRegistry,
        enhancementService: AIEnhancementService?
    ) {
        self.modelContext = modelContext
        self.serviceRegistry = serviceRegistry
        self.enhancementService = enhancementService
        self.licenseViewModel = LicenseViewModel()
    }

    /// Run the full pipeline for a given transcription record.
    /// - Parameters:
    ///   - transcription: The pending Transcription SwiftData object to populate and save.
    ///   - audioURL: The recorded audio file.
    ///   - model: The transcription model to use.
    ///   - session: An active streaming session if one was prepared, otherwise nil.
    ///   - isEditMode: Whether Edit Mode is active (fork feature).
    ///   - editModeSelectedText: The selected text for Edit Mode replacement (fork feature).
    ///   - onStateChange: Called when the pipeline moves to a new recording state (e.g. `.enhancing`).
    ///   - shouldCancel: Returns true if the user requested cancellation.
    ///   - onCancel: Called when cancellation is detected to cancel active session state.
    ///   - onDismiss: Called at the end to dismiss the recorder panel.
    ///   - onEditModeComplete: Called when Edit Mode finishes, with optional dictionary suggestion.
    ///   - requestCandidateReview: Called when confidence routing wants the user to choose among candidates.
    func run(
        transcription: Transcription,
        audioURL: URL,
        model: any TranscriptionModel,
        session: TranscriptionSession?,
        isEditMode: Bool = false,
        editModeSelectedText: String? = nil,
        capturedAppPID: pid_t? = nil,
        onStateChange: @escaping (RecordingState) -> Void,
        shouldCancel: () -> Bool,
        onCancel: @escaping () async -> Void,
        onDismiss: @escaping () async -> Void,
        onEditModeComplete: ((WordSubstitution?) -> Void)? = nil,
        requestCandidateReview: ((VocoConfidenceAssessment) async -> String?)? = nil
    ) async {
        var finalPastedText: String?
        var promptDetectionResult: PromptDetectionService.PromptDetectionResult?
        var didInsertSessionMetric = false
        let postProcessor = ChinesePostProcessingService.shared

        logger.notice("🔄 Starting transcription...")

        func restorePromptDetectionSettingsIfNeeded() async {
            if let result = promptDetectionResult,
               let enhancementService,
               result.shouldEnableAI {
                await promptDetectionService.restoreOriginalSettings(result, to: enhancementService)
            }
        }

        func restorePromptDetectionSettingsAndDismiss(afterRestore: () -> Void = {}) async {
            await restorePromptDetectionSettingsIfNeeded()
            afterRestore()
            await onDismiss()
        }

        func finishCanceledTranscription() async {
            await onCancel()
            await restorePromptDetectionSettingsIfNeeded()

            let canceledDuration: TimeInterval?
            if transcription.duration > 0 {
                canceledDuration = nil
            } else {
                let duration = await AudioFileMetadata.duration(for: audioURL)
                canceledDuration = duration > 0 ? duration : nil
            }

            transcription.markAsCanceledTranscription(
                duration: canceledDuration,
                modelName: transcription.transcriptionModelName ?? model.displayName
            )

            do {
                try modelContext.save()
            } catch {
                logger.error("Failed to save canceled transcription: \(error.localizedDescription, privacy: .public)")
            }
        }

        if shouldCancel() {
            await finishCanceledTranscription()
            return
        }

        do {
            let transcriptionStart = Date()
            var text: String
            if let session {
                text = try await session.transcribe(audioURL: audioURL)
            } else {
                text = try await serviceRegistry.transcribe(audioURL: audioURL, model: model)
            }
            let rawASRText = text
            transcription.rawTranscript = rawASRText
            transcription.asrEngineID = VocoCanonicalizationPipeline.asrEngineID(for: model)
            transcription.languageMode = VocoCanonicalizationPipeline.selectedLanguageMode()
            logger.notice("📝 Transcript: \(text, privacy: .private)")
            text = TranscriptionOutputFilter.filter(text)
            logger.notice("📝 Output filter result: \(text, privacy: .private)")
            let transcriptionDuration = Date().timeIntervalSince(transcriptionStart)

            let powerModeManager = PowerModeManager.shared
            let activePowerModeConfig = powerModeManager.currentActiveConfiguration
            let powerModeName = (activePowerModeConfig?.isEnabled == true) ? activePowerModeConfig?.name : nil
            let powerModeEmoji = (activePowerModeConfig?.isEnabled == true) ? activePowerModeConfig?.emoji : nil

            if shouldCancel() { await finishCanceledTranscription(); return }

            text = text.trimmingCharacters(in: .whitespacesAndNewlines)

            if UserDefaults.standard.bool(forKey: "IsTextFormattingEnabled") {
                text = WhisperTextFormatter.format(text)
                logger.notice("📝 Formatted transcript: \(text, privacy: .private)")
            }

            text = WordReplacementService.shared.applyReplacements(to: text, using: modelContext)
            logger.notice("📝 WordReplacement: \(text, privacy: .private)")

            // Set model provider for confidence routing
            postProcessor.lastModelProvider = model.provider

            // Pre-compute audio duration for Qwen3 speech rate check
            let preAudioAsset = AVURLAsset(url: audioURL)
            let preAudioDuration = (try? CMTimeGetSeconds(await preAudioAsset.load(.duration))) ?? 0.0
            postProcessor.lastAudioDuration = preAudioDuration

            // === Language-aware Chinese Post-Processing Pipeline ===
            let detectedLanguage: String? = (model.provider == .qwen3)
                ? serviceRegistry.qwen3TranscriptionService.lastDetectedLanguage
                : nil
            let containsHan = text.unicodeScalars.contains {
                (0x4E00...0x9FFF).contains($0.value) || (0x3400...0x4DBF).contains($0.value)
                    || (0x20000...0x2A6DF).contains($0.value)
            }
            let shouldRunChinesePostProcessing = postProcessor.isEnabled
                && (detectedLanguage == nil || detectedLanguage == "Chinese" || containsHan)

            if let lang = detectedLanguage {
                ChinesePostProcessingService.debugLog("LANGUAGE_TAG: \(lang) | shouldRunChinePP=\(shouldRunChinesePostProcessing) | text(\(text.count)): \(text)")
                logger.notice("🏷️ Detected language: \(lang, privacy: .public)")
            }

            var ppNeedsLLM = true
            if shouldRunChinesePostProcessing {
                let ppResult = postProcessor.process(text)
                text = ppResult.processedText
                ppNeedsLLM = ppResult.needsLLMCorrection
                logger.notice("📝 ChinesePostProcessing: \(text, privacy: .private) (steps: \(ppResult.appliedSteps.joined(separator: ", ")), needsLLM: \(ppNeedsLLM))")

                // Severe repetition → discard output (Whisper hallucination)
                if let repInfo = ppResult.repetitionInfo, repInfo.isSevere {
                    logger.warning("⚠️ Severe repetition detected (\(String(format: "%.0f%%", repInfo.repetitionRatio * 100))), discarding output")
                    transcription.text = "Discarded: severe repetition"
                    transcription.transcriptionStatus = TranscriptionStatus.failed.rawValue
                    try? modelContext.save()
                    await onDismiss()
                    return
                }
            } else if postProcessor.isEnabled {
                ppNeedsLLM = false
                logger.notice("📝 Skipping Chinese post-processing (detected language: \(detectedLanguage ?? "unknown", privacy: .public))")
            }

            let cleanedText = TranscriptionOutputFilter.applyUserCleanupPreferences(text)
            if cleanedText != text {
                logger.notice("📝 User cleanup preferences applied: \(cleanedText, privacy: .private)")
            }
            text = cleanedText

            let normalizationResult = VocoCanonicalizationPipeline.normalize(
                text,
                rawTranscript: rawASRText,
                model: model,
                modelContext: modelContext,
                transcription: transcription
            )
            text = normalizationResult.normalizedText
            if !normalizationResult.replacements.isEmpty || !normalizationResult.suggestions.isEmpty {
                logger.notice(
                    "📝 Canonicalization: replacements=\(normalizationResult.replacements.count, privacy: .public), suggestions=\(normalizationResult.suggestions.count, privacy: .public), result=\(text, privacy: .private)"
                )
            }
            let confidenceAssessment = VocoCanonicalizationPipeline.confidenceAssessment(
                for: normalizationResult,
                rawTranscript: rawASRText
            )

            if !isEditMode,
               confidenceAssessment.route == .reviewSuggested,
               let requestCandidateReview,
               let selectedCandidate = await requestCandidateReview(confidenceAssessment),
               !selectedCandidate.isEmpty {
                text = selectedCandidate
                transcription.selectedCandidate = selectedCandidate
                transcription.recordCandidateReviewFeedback(
                    normalizationResult: normalizationResult,
                    confidenceAssessment: confidenceAssessment,
                    selectedCandidate: selectedCandidate,
                    rawTranscript: rawASRText
                )
                logger.notice("📝 Candidate review selected text: \(text, privacy: .private)")
            }

            let actualDuration = preAudioDuration

            transcription.text = text
            transcription.duration = actualDuration
            transcription.transcriptionModelName = model.displayName
            transcription.transcriptionDuration = transcriptionDuration
            transcription.powerModeName = powerModeName
            transcription.powerModeEmoji = powerModeEmoji
            finalPastedText = text

            // === Edit Mode Branch (fork feature) ===
            if isEditMode, let selectedText = editModeSelectedText {
                let handled = await handleEditMode(
                    text: text,
                    selectedText: selectedText,
                    transcription: transcription,
                    enhancementService: enhancementService,
                    onStateChange: onStateChange,
                    shouldCancel: shouldCancel,
                    onCleanup: { await finishCanceledTranscription() },
                    onDismiss: onDismiss,
                    onEditModeComplete: onEditModeComplete
                )
                if handled { return }
            }

            // === Voice command detection (normal mode) ===
            if let command = VoiceCommandService.shared.detectCommand(in: text) {
                logger.notice("🎤 Voice command detected: \(command.rawValue, privacy: .private)")
                transcription.transcriptionStatus = TranscriptionStatus.completed.rawValue
                try? modelContext.save()
                NotificationCenter.default.post(name: .transcriptionCompleted, object: transcription)
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                    command.execute()
                }
                await onDismiss()
                return
            }

            // === Prompt detection ===
            if let enhancementService, enhancementService.isConfigured {
                let detectionResult = promptDetectionService.analyzeText(text, with: enhancementService)
                promptDetectionResult = detectionResult
                await promptDetectionService.applyDetectionResult(detectionResult, to: enhancementService)
            }

            // Determine if AI Enhancement should be skipped (confidence routing)
            let confidenceSkip = postProcessor.isEnabled && !ppNeedsLLM

            // Skip short enhancement (upstream feature)
            let isSkipShortEnhancementEnabled = UserDefaults.standard.bool(forKey: "SkipShortEnhancement")
            let savedThreshold = UserDefaults.standard.integer(forKey: "ShortEnhancementWordThreshold")
            let shortEnhancementWordThreshold = savedThreshold > 0 ? savedThreshold : 3
            let shortSkip = isSkipShortEnhancementEnabled && WordCounter.count(in: text) <= shortEnhancementWordThreshold && !(promptDetectionResult?.shouldEnableAI == true)

            let shouldSkipEnhancement = confidenceSkip || shortSkip
            ChinesePostProcessingService.debugLog("PIPELINE: shouldSkip=\(shouldSkipEnhancement) (confidence=\(confidenceSkip), short=\(shortSkip)), ppNeedsLLM=\(ppNeedsLLM), postProcessorEnabled=\(postProcessor.isEnabled), enhancementEnabled=\(enhancementService?.isEnhancementEnabled ?? false), isConfigured=\(enhancementService?.isConfigured ?? false) | text(\(text.count)): \(text)")

            if !shouldSkipEnhancement,
               let enhancementService,
               enhancementService.isEnhancementEnabled,
               enhancementService.isConfigured {
                if shouldCancel() { await finishCanceledTranscription(); return }

                onStateChange(.enhancing)
                let textForAI = promptDetectionResult?.processedText ?? text

                do {
                    let (enhancedText, enhancementDuration, promptName) = try await enhancementService.enhance(textForAI)
                    logger.notice("📝 AI enhancement: \(enhancedText, privacy: .private)")

                    // === LLM response validation ===
                    if postProcessor.isEnabled && postProcessor.isLLMValidationEnabled {
                        let customVocabWords = CustomVocabularyService.shared.getCustomVocabularyWords(from: modelContext)
                        let protectedTerms = customVocabWords + CorrectionProtectionList.shared.allWords()

                        // Fetch word replacements for cross-script validation
                        let wrDescriptor = FetchDescriptor<WordReplacement>(predicate: #Predicate { $0.isEnabled })
                        let wordReplacementPairs: [(original: String, replacement: String)] = (try? modelContext.fetch(wrDescriptor))?.map {
                            (original: $0.originalText, replacement: $0.replacementText)
                        } ?? []

                        let validation = postProcessor.llmResponseValidator.validate(
                            response: enhancedText,
                            original: textForAI,
                            protectedTerms: protectedTerms,
                            wordReplacements: wordReplacementPairs,
                            customVocabulary: customVocabWords
                        )
                        ChinesePostProcessingService.debugLog(
                            "LLM_VALIDATION: isValid=\(validation.isValid), reasons=\(validation.reasons.joined(separator: ",")) | original(\(textForAI.count)): \(textForAI) | enhanced(\(enhancedText.count)): \(enhancedText)"
                        )
                        if !validation.isValid {
                            var retrySucceeded = false

                            if validation.isRetryable {
                                logger.notice("🔄 LLM validation failed (\(validation.reasons.joined(separator: ","))), attempting conservative retry")
                                do {
                                    let uncertainWords = postProcessor.lastUncertainWords
                                    let (retryResult, retryDuration) = try await enhancementService.enhanceConservative(
                                        textForAI, uncertainWords: uncertainWords
                                    )
                                    let retryValidation = postProcessor.llmResponseValidator.validate(
                                        response: retryResult,
                                        original: textForAI,
                                        protectedTerms: protectedTerms,
                                        wordReplacements: wordReplacementPairs,
                                        customVocabulary: customVocabWords
                                    )
                                    ChinesePostProcessingService.debugLog(
                                        "CONSERVATIVE_RETRY: isValid=\(retryValidation.isValid), reasons=\(retryValidation.reasons.joined(separator: ",")) | result(\(retryResult.count)): \(retryResult)"
                                    )
                                    if retryValidation.isValid {
                                        transcription.enhancedText = retryResult
                                        finalPastedText = retryResult
                                        transcription.enhancementDuration = (transcription.enhancementDuration ?? 0) + retryDuration
                                        retrySucceeded = true
                                    }
                                } catch {
                                    logger.warning("⚠️ Conservative retry error: \(error.localizedDescription)")
                                }
                            }

                            if !retrySucceeded {
                                if validation.isRetryable {
                                    // CJK overlap check: detect hallucination
                                    let originalCJK = Set(textForAI.unicodeScalars.filter {
                                        (0x4E00...0x9FFF).contains($0.value) || (0x3400...0x4DBF).contains($0.value)
                                    })
                                    let enhancedCJK = Set(enhancedText.unicodeScalars.filter {
                                        (0x4E00...0x9FFF).contains($0.value) || (0x3400...0x4DBF).contains($0.value)
                                    })
                                    let overlap = originalCJK.intersection(enhancedCJK).count
                                    let overlapRatio = originalCJK.isEmpty ? 1.0 : Double(overlap) / Double(originalCJK.count)
                                    ChinesePostProcessingService.debugLog(
                                        "CJK_OVERLAP_CHECK: originalCJK=\(originalCJK.count), enhancedCJK=\(enhancedCJK.count), overlap=\(overlap), ratio=\(String(format: "%.2f", overlapRatio))"
                                    )
                                    if overlapRatio < 0.3 {
                                        logger.warning("⚠️ LLM hallucination detected (CJK overlap \(String(format: "%.0f", overlapRatio * 100))%), falling back to pre-LLM text")
                                    } else {
                                        logger.warning("⚠️ LLM validation failed but retryable — using enhanced text over unpunctuated fallback")
                                        transcription.enhancedText = enhancedText
                                        finalPastedText = enhancedText
                                    }
                                } else {
                                    logger.warning("⚠️ LLM response invalid (non-retryable), falling back to pre-LLM text")
                                }
                            }
                        } else {
                            transcription.enhancedText = enhancedText
                            finalPastedText = enhancedText
                        }
                    } else {
                        transcription.enhancedText = enhancedText
                        finalPastedText = enhancedText
                    }

                    // === Post-LLM punctuation density check ===
                    if let acceptedText = finalPastedText, acceptedText.count >= 10 {
                        let cjkPunct: Set<Character> = ["，", "。", "？", "！", "、", "；", "："]
                        let pCount = acceptedText.filter { cjkPunct.contains($0) }.count
                        let expected = max(acceptedText.count / 20, 1)
                        let maxCJKSpan = 12
                        var hasLongSpan = false
                        var cjkRunCount = 0
                        for char in acceptedText {
                            if cjkPunct.contains(char) {
                                cjkRunCount = 0
                            } else if char.unicodeScalars.first.map({ (0x4E00...0x9FFF).contains($0.value) || (0x3400...0x4DBF).contains($0.value) }) == true {
                                cjkRunCount += 1
                                if cjkRunCount > maxCJKSpan {
                                    hasLongSpan = true
                                    break
                                }
                            }
                        }
                        if pCount < expected || hasLongSpan {
                            ChinesePostProcessingService.debugLog(
                                "POST_LLM_PUNCT_CHECK: len=\(acceptedText.count), punctCount=\(pCount), expected=\(expected), longSpan=\(hasLongSpan), triggering conservative retry for punctuation"
                            )
                            logger.notice("🔄 Post-LLM punctuation insufficient (count=\(pCount)/\(expected), longSpan=\(hasLongSpan)), retrying for punctuation")
                            do {
                                let (retryResult, retryDuration) = try await enhancementService.enhanceConservative(
                                    acceptedText, uncertainWords: []
                                )
                                let retryPunctCount = retryResult.filter { cjkPunct.contains($0) }.count
                                var retryStillHasLongSpan = false
                                if hasLongSpan {
                                    var rrc = 0
                                    for char in retryResult {
                                        if cjkPunct.contains(char) { rrc = 0 }
                                        else if char.unicodeScalars.first.map({ (0x4E00...0x9FFF).contains($0.value) || (0x3400...0x4DBF).contains($0.value) }) == true {
                                            rrc += 1
                                            if rrc > maxCJKSpan { retryStillHasLongSpan = true; break }
                                        }
                                    }
                                }
                                if retryPunctCount > pCount && !retryStillHasLongSpan {
                                    ChinesePostProcessingService.debugLog(
                                        "POST_LLM_PUNCT_RETRY: accepted, punctCount \(pCount)→\(retryPunctCount) | result(\(retryResult.count)): \(retryResult)"
                                    )
                                    transcription.enhancedText = retryResult
                                    finalPastedText = retryResult
                                    transcription.enhancementDuration = (transcription.enhancementDuration ?? 0) + retryDuration
                                } else if hasLongSpan {
                                    // Try comma-only prompt
                                    ChinesePostProcessingService.debugLog(
                                        "POST_LLM_PUNCT_RETRY: still longSpan (punctCount \(pCount)→\(retryPunctCount)), trying comma insertion prompt"
                                    )
                                    let commaInput = retryPunctCount > pCount ? retryResult : (finalPastedText ?? acceptedText)
                                    do {
                                        let (commaResult, commaDuration) = try await enhancementService.enhanceCommaInsertion(commaInput)
                                        let commaInputPunctCount = commaInput.filter { cjkPunct.contains($0) }.count
                                        let commaPunctCount = commaResult.filter { cjkPunct.contains($0) }.count
                                        let commaInputStripped = String(commaInput.filter { !cjkPunct.contains($0) })
                                        let commaResultStripped = String(commaResult.filter { !cjkPunct.contains($0) })
                                        let commaTextUnchanged = commaInputStripped == commaResultStripped
                                        if commaPunctCount > commaInputPunctCount && commaTextUnchanged {
                                            ChinesePostProcessingService.debugLog(
                                                "POST_LLM_COMMA_INSERT: accepted, punctCount \(commaInputPunctCount)→\(commaPunctCount) | result(\(commaResult.count)): \(commaResult)"
                                            )
                                            transcription.enhancedText = commaResult
                                            finalPastedText = commaResult
                                            transcription.enhancementDuration = (transcription.enhancementDuration ?? 0) + commaDuration
                                        } else if !commaTextUnchanged {
                                            ChinesePostProcessingService.debugLog(
                                                "POST_LLM_COMMA_INSERT: rejected (text changed) | input: \(commaInputStripped) | result: \(commaResultStripped)"
                                            )
                                        } else {
                                            ChinesePostProcessingService.debugLog(
                                                "POST_LLM_COMMA_INSERT: rejected (no improvement), punctCount \(commaInputPunctCount)→\(commaPunctCount)"
                                            )
                                        }
                                    } catch {
                                        logger.warning("⚠️ Comma insertion retry error: \(error.localizedDescription)")
                                    }
                                } else {
                                    ChinesePostProcessingService.debugLog(
                                        "POST_LLM_PUNCT_RETRY: rejected (no improvement), punctCount \(pCount)→\(retryPunctCount)"
                                    )
                                }
                            } catch {
                                logger.warning("⚠️ Post-LLM punctuation retry error: \(error.localizedDescription)")
                            }
                        }
                    }

                    // === Final fallback: rule-based punctuation insertion ===
                    if let currentText = finalPastedText, currentText.count >= 10 {
                        let cjkPunctFinal: Set<Character> = ["，", "。", "？", "！", "、", "；", "："]
                        let finalPunctCount = currentText.filter { cjkPunctFinal.contains($0) }.count
                        let finalExpected = max(currentText.count / 20, 1)
                        var finalHasLongSpan = false
                        var finalCjkRun = 0
                        for char in currentText {
                            if cjkPunctFinal.contains(char) { finalCjkRun = 0 }
                            else if char.unicodeScalars.first.map({ (0x4E00...0x9FFF).contains($0.value) || (0x3400...0x4DBF).contains($0.value) }) == true {
                                finalCjkRun += 1
                                if finalCjkRun > 12 { finalHasLongSpan = true; break }
                            }
                        }
                        if finalPunctCount < finalExpected || finalHasLongSpan {
                            let ruleResult = RuleBasedPunctuationInserter.insert(into: currentText)
                            let rulePunctCount = ruleResult.filter { cjkPunctFinal.contains($0) }.count
                            if rulePunctCount > finalPunctCount {
                                ChinesePostProcessingService.debugLog(
                                    "RULE_BASED_PUNCT: applied, punctCount \(finalPunctCount)→\(rulePunctCount) | result(\(ruleResult.count)): \(ruleResult)"
                                )
                                logger.notice("📝 Rule-based punctuation applied as final fallback (punct \(finalPunctCount)→\(rulePunctCount))")
                                transcription.enhancedText = ruleResult
                                finalPastedText = ruleResult
                            }
                        }
                    }

                    transcription.aiEnhancementModelName = enhancementService.getAIService()?.currentModel
                    transcription.promptName = promptName
                    transcription.enhancementDuration = enhancementDuration
                    transcription.aiRequestSystemMessage = enhancementService.lastSystemMessageSent
                    transcription.aiRequestUserMessage = enhancementService.lastUserMessageSent
                } catch {
                    let errorDescription = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
                    transcription.enhancedText = "Enhancement failed: \(errorDescription)"
                    let shortReason = String(errorDescription.prefix(80))
                    await MainActor.run {
                        NotificationManager.shared.showNotification(
                        title: String(localized: "Enhancement failed: \(shortReason)"),
                            type: .warning
                        )
                    }
                    if shouldCancel() { await finishCanceledTranscription(); return }
                }
            } else if shouldSkipEnhancement {
                logger.notice("📝 Skipping AI enhancement (confidence routing)")
                ChinesePostProcessingService.debugLog("SKIPPED_LLM: confidence routing skipped | finalPastedText(\(finalPastedText?.count ?? 0)): \(finalPastedText ?? "nil")")

                // Safety net: if skipped text has insufficient punctuation density or long CJK span, force LLM
                let cjkPunctuation: Set<Character> = ["，", "。", "？", "！", "、", "；", "："]
                let pastedText = finalPastedText ?? ""
                let punctCount = pastedText.filter { cjkPunctuation.contains($0) }.count
                let expectedPunct = pastedText.count / 20
                var safetyNetLongSpan = false
                if pastedText.count >= 10 {
                    var safetyRunCount = 0
                    for char in pastedText {
                        if cjkPunctuation.contains(char) {
                            safetyRunCount = 0
                        } else if char.unicodeScalars.first.map({ (0x4E00...0x9FFF).contains($0.value) || (0x3400...0x4DBF).contains($0.value) }) == true {
                            safetyRunCount += 1
                            if safetyRunCount > 12 {
                                safetyNetLongSpan = true
                                break
                            }
                        }
                    }
                }
                let insufficientPunct = pastedText.count >= 10 && (punctCount < max(expectedPunct, 1) || safetyNetLongSpan)
                ChinesePostProcessingService.debugLog("SAFETY_NET_CHECK: len=\(pastedText.count), punctCount=\(punctCount), expected=\(max(expectedPunct, 1)), longSpan=\(safetyNetLongSpan), willTrigger=\(insufficientPunct)")
                if insufficientPunct,
                   let enhancementService,
                   enhancementService.isEnhancementEnabled,
                   enhancementService.isConfigured {
                    logger.notice("📝 Safety net triggered: long text without punctuation, forcing LLM")
                    onStateChange(.enhancing)
                    let textForAI = promptDetectionResult?.processedText ?? text
                    do {
                        let (enhancedText, enhancementDuration, promptName) = try await enhancementService.enhance(textForAI)
                        logger.notice("📝 Safety net AI enhancement: \(enhancedText, privacy: .private)")
                        transcription.enhancedText = enhancedText
                        finalPastedText = enhancedText
                        transcription.aiEnhancementModelName = enhancementService.getAIService()?.currentModel
                        transcription.promptName = promptName
                        transcription.enhancementDuration = enhancementDuration
                        transcription.aiRequestSystemMessage = enhancementService.lastSystemMessageSent
                        transcription.aiRequestUserMessage = enhancementService.lastUserMessageSent
                    } catch {
                        logger.warning("⚠️ Safety net enhancement failed: \(error.localizedDescription)")
                    }
                }
            }

            // === Personal Style Guard: keep LLM output from becoming template prose ===
            if UserDefaults.standard.bool(forKey: PersonalStyleGuardService.enabledKey),
               let currentText = finalPastedText,
               currentText != transcription.text {
                let styleGuard = PersonalStyleGuardService.shared.validate(
                    response: currentText,
                    original: transcription.text
                )
                if !styleGuard.isValid {
                    transcription.recordStyleGuardRejection(
                        response: currentText,
                        reasons: styleGuard.reasons
                    )
                    transcription.enhancedText = nil
                    finalPastedText = transcription.text
                    logger.warning("⚠️ Personal Style Guard rejected enhancement: \(styleGuard.reasons.joined(separator: ","), privacy: .public)")
                }
            }

            // === Post-LLM comma cleanup: remove wrongly inserted commas after 的/了 ===
            if let currentText = finalPastedText, currentText.count >= 3 {
                let cleaned = PostLLMCommaCleanup.clean(currentText, originalText: transcription.text)
                if cleaned != currentText {
                    ChinesePostProcessingService.debugLog(
                        "COMMA_CLEANUP: \(currentText) → \(cleaned)"
                    )
                    logger.notice("📝 Post-LLM comma cleanup applied")
                    finalPastedText = cleaned
                    if transcription.enhancedText != nil {
                        transcription.enhancedText = cleaned
                    }
                }
            }

            transcription.transcriptionStatus = TranscriptionStatus.completed.rawValue

        } catch {
            let errorDescription = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            let recoverySuggestion = (error as? LocalizedError)?.recoverySuggestion ?? ""
            let fullErrorText = recoverySuggestion.isEmpty ? errorDescription : "\(errorDescription) \(recoverySuggestion)"

            if let nativeAppleError = error as? NativeAppleTranscriptionService.ServiceError,
               case .assetDownloadRequired = nativeAppleError {
                NotificationManager.shared.showNotification(
                    title: errorDescription,
                    type: .error,
                    duration: 5.0
                )
            }

            transcription.text = "Transcription Failed: \(fullErrorText)"
            transcription.transcriptionStatus = TranscriptionStatus.failed.rawValue
        }

        func saveTranscriptionAndPostCompletion() {
            if transcription.transcriptionStatus == TranscriptionStatus.completed.rawValue {
                do {
                    didInsertSessionMetric = try SessionMetricRecorder.recordRecorderSession(
                        transcription: transcription,
                        model: model,
                        in: modelContext
                    )
                } catch {
                    logger.error("Failed to record session metric: \(error.localizedDescription, privacy: .public)")
                }
            }

            do {
                try modelContext.save()
                if didInsertSessionMetric {
                    NotificationCenter.default.post(name: .sessionMetricsDidChange, object: nil)
                }
                NotificationCenter.default.post(name: .transcriptionCompleted, object: transcription)
            } catch {
                logger.error("Failed to save transcription: \(error.localizedDescription, privacy: .public)")
            }
        }

        if shouldCancel() {
            await finishCanceledTranscription()
            return
        }

        if var textToPaste = finalPastedText,
           transcription.transcriptionStatus == TranscriptionStatus.completed.rawValue {
            if case .trialExpired = licenseViewModel.licenseState {
                textToPaste = """
                    Your trial has expired. Upgrade to Voco Pro at tryvoiceink.com/buy
                    \n\(textToPaste)
                    """
            }

            // Enforce vocabulary casing as the final text processing step
            textToPaste = WordReplacementService.shared.enforceVocabularyCasing(
                text: textToPaste, using: modelContext)

            // Add pre-LLM text to context memory to avoid enhancement feedback loops.
            if postProcessor.isEnabled && postProcessor.isContextMemoryEnabled {
                postProcessor.contextMemory.add(transcription.text)
            }

            // === Context-Aware Insertion (fork feature) ===
            let pastedText = await applyContextAwareInsertion(textToPaste, enhancementService: enhancementService, capturedAppPID: capturedAppPID)
            let autoSendKey = PowerModeManager.shared.currentActiveConfiguration?.autoSendKey

            // Capture element while target field still has focus, before Cmd+V fires.
            let autoLearn = AutoLearnVocabularyService.shared
            if let element = autoLearn.captureFocusedElement() {
                autoLearn.prepareMonitoring(pastedText: pastedText, element: element, modelContext: modelContext)
            }

            _ = await CursorPaster.startPasteAtCursor(pastedText).value
            SoundManager.shared.playStopSound()
            await restorePromptDetectionSettingsAndDismiss {
                if let autoSendKey, autoSendKey.isEnabled {
                    Task { @MainActor in
                        try? await Task.sleep(nanoseconds: 500_000_000)
                        CursorPaster.performAutoSend(autoSendKey)
                    }
                }
            }
        } else {
            await restorePromptDetectionSettingsAndDismiss()
        }

        saveTranscriptionAndPostCompletion()
    }
}
