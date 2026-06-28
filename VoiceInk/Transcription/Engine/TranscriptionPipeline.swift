import Foundation
import SwiftData
import os

/// Handles the full post-recording pipeline:
/// transcribe → filter → format → word-replace → AI enhance → deliver → save
@MainActor
class TranscriptionPipeline {
    struct AssistantHooks {
        let isFollowUp: Bool
        let sendFollowUp: (String, Transcription) async -> Void
        let startResponse: (String, EnhancementRuntimeConfiguration) async -> Void
        let showResponse: (String, String?) async -> Void
        let failResponse: (String) async -> Void

        static let inactive = AssistantHooks(
            isFollowUp: false,
            sendFollowUp: { _, _ in },
            startResponse: { _, _ in },
            showResponse: { _, _ in },
            failResponse: { _ in }
        )
    }

    let modelContext: ModelContext
    private let serviceRegistry: TranscriptionServiceRegistry
    let enhancementService: AIEnhancementService?
    private let delivery = TranscriptionDelivery()
    let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "TranscriptionPipeline")
    private static let minContentCharactersForAutoPunctuation = 5
    private static let autoPunctuationMarks: Set<Character> = ["，", "。", "？", "！", "、", "；", "："]
    private static let asciiPunctuationMarks: Set<Character> = [",", ".", "?", "!", ";", ":"]

    init(
        modelContext: ModelContext,
        serviceRegistry: TranscriptionServiceRegistry,
        enhancementService: AIEnhancementService?
    ) {
        self.modelContext = modelContext
        self.serviceRegistry = serviceRegistry
        self.enhancementService = enhancementService
    }

    /// Run the full pipeline for a given transcription record.
    /// - Parameters:
    ///   - transcription: The pending Transcription SwiftData object to populate and save.
    ///   - audioURL: The recorded audio file.
    ///   - transcriptionConfiguration: Mode-resolved transcription engine settings for this phase.
    ///   - session: An active streaming session if one was prepared, otherwise nil.
    ///   - onStateChange: Called when the pipeline moves to a new recording state (e.g. `.enhancing`).
    ///   - shouldCancel: Returns true if the user requested cancellation.
    ///   - onCancel: Called when cancellation is detected to cancel active session state.
    ///   - onDismiss: Called when delivery should close the recorder panel.
    func run(
        transcription: Transcription,
        audioURL: URL,
        transcriptionConfiguration: TranscriptionRuntimeConfiguration,
        formattingConfiguration resolveFormattingConfiguration: @escaping () -> TranscriptionFormattingConfiguration,
        session: TranscriptionSession?,
        enhancementConfiguration: @escaping () -> EnhancementRuntimeConfiguration?,
        recordingContextSnapshot: @escaping () async -> RecordingContextSnapshot? = { nil },
        outputConfiguration: @escaping () -> OutputRuntimeConfiguration,
        isEditMode: Bool = false,
        editModeSelectedText: String? = nil,
        capturedAppPID: pid_t? = nil,
        onStateChange: @escaping (RecordingState) -> Void,
        shouldCancel: () -> Bool,
        onCancel: @escaping () async -> Void,
        onDismiss: @escaping () async -> Void,
        onEditModeComplete: ((WordSubstitution?) -> Void)? = nil,
        assistant: AssistantHooks = .inactive
    ) async {
        let model = transcriptionConfiguration.model
        var finalText: String?
        var didInsertSessionMetric = false
        var responseError: String?
        var outputForDelivery: OutputRuntimeConfiguration?
        var responseConfig: EnhancementRuntimeConfiguration?
        let postProcessor = ChinesePostProcessingService.shared
        var shadowRawASRText: String?
        var shadowPostProcessingTrace = ChinesePostProcessingTrace()
        var shadowConfidenceAssessment: VocoConfidenceAssessment?

        func finishCanceledTranscription() async {
            await onCancel()

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
                text = try await serviceRegistry.transcribe(
                    audioURL: audioURL,
                    model: model,
                    context: transcriptionConfiguration.requestContext
                )
            }
            let rawASRText = text
            shadowRawASRText = rawASRText
            text = TranscriptionOutputFilter.filter(text)
            let transcriptionDuration = Date().timeIntervalSince(transcriptionStart)

            if shouldCancel() { await finishCanceledTranscription(); return }

            text = text.trimmingCharacters(in: .whitespacesAndNewlines)
            let formattingConfiguration = resolveFormattingConfiguration()

            if formattingConfiguration.isTextFormattingEnabled {
                text = ParagraphFormatter.format(text)
            }

            let actualDuration = await AudioFileMetadata.duration(for: audioURL)
            postProcessor.lastModelProvider = model.provider
            postProcessor.lastAudioDuration = actualDuration

            let detectedLanguage = model.provider == .qwen3
                ? serviceRegistry.qwen3TranscriptionService.lastDetectedLanguage
                : nil
            let containsHan = text.unicodeScalars.contains {
                (0x4E00...0x9FFF).contains($0.value)
                    || (0x3400...0x4DBF).contains($0.value)
                    || (0x20000...0x2A6DF).contains($0.value)
            }
            let shouldRunChinesePostProcessing = postProcessor.isEnabled
                && (detectedLanguage == nil || detectedLanguage == "Chinese" || containsHan)
            var postProcessingNeedsLLM = true
            if shouldRunChinesePostProcessing {
                let result = postProcessor.process(text)
                shadowPostProcessingTrace = result.trace
                text = result.processedText
                postProcessingNeedsLLM = result.needsLLMCorrection
                if result.repetitionInfo?.isSevere == true {
                    transcription.text = "Discarded: severe repetition"
                    transcription.transcriptionStatus = TranscriptionStatus.failed.rawValue
                    try? modelContext.save()
                    await onDismiss()
                    return
                }
            } else if postProcessor.isEnabled {
                postProcessingNeedsLLM = false
            }

            let cleanedText = TranscriptionOutputFilter.applyCleanupPreferences(
                text,
                punctuationMode: formattingConfiguration.punctuationCleanupMode,
                shouldLowercase: formattingConfiguration.lowercaseTranscription
            )
            let normalizedOutput = VocoCanonicalizationPipeline.normalizeWithAssessment(
                cleanedText,
                rawTranscript: rawASRText,
                model: model,
                modelContext: modelContext,
                transcription: transcription,
                appName: enhancementService?.cachedAppName,
                windowTitle: enhancementService?.cachedWindowTitle
            )
            let normalizationResult = normalizedOutput.normalizationResult
            let confidenceAssessment = normalizedOutput.confidenceAssessment
            shadowConfidenceAssessment = confidenceAssessment
            text = normalizationResult.normalizedText

            let modeMetadata = transcriptionConfiguration.metadata

            transcription.text = text
            transcription.duration = actualDuration
            transcription.transcriptionModelName = model.displayName
            transcription.transcriptionDuration = transcriptionDuration
            transcription.modeName = modeMetadata.name
            transcription.modeEmoji = modeMetadata.emoji
            finalText = text

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

            if let command = VoiceCommandService.shared.detectCommand(in: text) {
                transcription.transcriptionStatus = TranscriptionStatus.completed.rawValue
                try? modelContext.save()
                NotificationCenter.default.post(name: .transcriptionCompleted, object: transcription)
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                    command.execute()
                }
                await onDismiss()
                return
            }

            if !assistant.isFollowUp {
                let resolvedEnhancementConfiguration = enhancementConfiguration()
                let resolvedOutputConfiguration = outputConfiguration()
                let shouldRespondInRecorder = resolvedOutputConfiguration.outputMode == .respond &&
                    resolvedEnhancementConfiguration?.isEnabled == true &&
                    resolvedEnhancementConfiguration.map { configuration in
                        enhancementService?.isConfigured(for: configuration) == true
                    } == true
                outputForDelivery = resolvedOutputConfiguration
                responseConfig = shouldRespondInRecorder ? resolvedEnhancementConfiguration : nil

                let isSkipShortEnhancementEnabled = UserDefaults.standard.bool(forKey: "SkipShortEnhancement")
                let savedThreshold = UserDefaults.standard.integer(forKey: "ShortEnhancementWordThreshold")
                let shortEnhancementWordThreshold = savedThreshold > 0 ? savedThreshold : 3
                let shouldSkipEnhancement = !shouldRespondInRecorder && (
                    (postProcessor.isEnabled && !postProcessingNeedsLLM)
                        || (isSkipShortEnhancementEnabled
                            && WordCounter.count(in: text) <= shortEnhancementWordThreshold)
                )

                if let enhancementService,
                   let resolvedEnhancementConfiguration,
                   resolvedEnhancementConfiguration.isEnabled,
                   enhancementService.isConfigured(for: resolvedEnhancementConfiguration),
                   !shouldSkipEnhancement {
                    if shouldCancel() { await finishCanceledTranscription(); return }

                    onStateChange(.enhancing)
                    if shouldRespondInRecorder {
                        await assistant.startResponse(text, resolvedEnhancementConfiguration)
                    }

                    do {
                        let contextSnapshot = await recordingContextSnapshot()
                        let (enhancedText, enhancementDuration, promptName) = try await enhancementService.enhance(
                            text,
                            configuration: resolvedEnhancementConfiguration,
                            contextSnapshot: contextSnapshot
                        )
                        transcription.enhancementDuration = enhancementDuration
                        let acceptedText = await validateEnhancedText(
                            enhancedText,
                            originalText: text,
                            enhancementService: enhancementService,
                            postProcessor: postProcessor,
                            transcription: transcription
                        )
                        transcription.enhancedText = acceptedText == text ? nil : acceptedText
                        transcription.aiEnhancementModelName = resolvedEnhancementConfiguration.modelName ?? resolvedEnhancementConfiguration.provider?.defaultModel
                        transcription.promptName = promptName
                        transcription.aiRequestSystemMessage = enhancementService.lastSystemMessageSent
                        transcription.aiRequestUserMessage = enhancementService.lastUserMessageSent
                        finalText = acceptedText
                    } catch {
                        let errorDescription = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
                        transcription.enhancedText = "Enhancement failed: \(errorDescription)"
                        responseError = errorDescription
                        let shortReason = String(errorDescription.prefix(80))
                        await MainActor.run {
                            NotificationManager.shared.showNotification(
                                title: "Enhancement failed: \(shortReason)",
                                type: .warning
                            )
                        }
                        if shouldCancel() { await finishCanceledTranscription(); return }
                    }
                }
            }

            transcription.transcriptionStatus = TranscriptionStatus.completed.rawValue
        } catch {
            let errorDescription = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription

            if let nativeAppleError = error as? NativeAppleTranscriptionService.ServiceError,
               nativeAppleError.shouldShowNotification {
                await MainActor.run {
                    NotificationManager.shared.showNotification(
                        title: errorDescription,
                        type: .error,
                        duration: 5.0
                    )
                }
            }

            transcription.text = "Transcription Failed: \(errorDescription)"
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

        let resolvedOutput = outputForDelivery ?? outputConfiguration()
        if transcription.transcriptionStatus == TranscriptionStatus.completed.rawValue,
           !assistant.isFollowUp,
           resolvedOutput.outputMode == .paste,
           let text = finalText {
            var preparedText = WordReplacementService.shared.enforceVocabularyCasing(
                text: text,
                using: modelContext
            )
            preparedText = await applyContextAwareInsertion(
                preparedText,
                enhancementService: enhancementService,
                capturedAppPID: capturedAppPID
            )
            finalText = preparedText
            transcription.finalPastedText = preparedText

            if postProcessor.isEnabled && postProcessor.isContextMemoryEnabled {
                postProcessor.contextMemory.add(transcription.text)
            }
        }

        logPhoneticShadowSnapshot(
            transcription: transcription,
            audioURL: audioURL,
            model: model,
            rawASRText: shadowRawASRText,
            postProcessingTrace: shadowPostProcessingTrace,
            confidenceAssessment: shadowConfidenceAssessment,
            finalText: finalText
        )

        await delivery.deliver(
            TranscriptionDelivery.Request(
                transcription: transcription,
                text: finalText,
                output: resolvedOutput,
                responseConfig: responseConfig,
                responseError: responseError,
                isAssistantFollowUp: assistant.isFollowUp
            ),
            actions: TranscriptionDelivery.Actions(
                setState: onStateChange,
                dismiss: onDismiss,
                sendFollowUp: assistant.sendFollowUp,
                showResponse: assistant.showResponse,
                failResponse: assistant.failResponse
            )
        )

        saveTranscriptionAndPostCompletion()
    }

    private func logPhoneticShadowSnapshot(
        transcription: Transcription,
        audioURL: URL,
        model: any TranscriptionModel,
        rawASRText: String?,
        postProcessingTrace: ChinesePostProcessingTrace,
        confidenceAssessment: VocoConfidenceAssessment?,
        finalText: String?
    ) {
        guard PhoneticShadowLogger.isShadowLoggingEnabled else { return }

        let durationMs = transcription.duration > 0 ? transcription.duration * 1000 : nil
        let latencyMs = totalLatencyMs(
            transcriptionDuration: transcription.transcriptionDuration,
            enhancementDuration: transcription.enhancementDuration
        )
        let pipeline = PhoneticShadowPipeline(
            asrEngine: shadowASREngineName(for: model.provider),
            rawASR: rawASRText,
            afterOpenCC: postProcessingTrace.afterOpenCC,
            afterPinyinCorrector: postProcessingTrace.afterPinyinCorrector,
            afterHomophoneCorrection: postProcessingTrace.afterHomophoneCorrection,
            afterNasalCorrection: postProcessingTrace.afterNasalCorrection,
            afterPersonalCorrection: postProcessingTrace.afterPersonalCorrection,
            llmEnhanced: transcription.enhancedText,
            finalInserted: finalText,
            route: confidenceAssessment?.route.rawValue,
            confidenceScore: confidenceAssessment?.score,
            avgLogprob: ChinesePostProcessingService.shared.lastAvgLogProb == 0 ? nil : ChinesePostProcessingService.shared.lastAvgLogProb,
            latencyMs: latencyMs
        )
        let audio = PhoneticShadowAudio(
            audioAssetId: audioURL.lastPathComponent,
            durationMs: durationMs
        )
        let event = PhoneticShadowEvent.pipelineSnapshot(
            utteranceId: transcription.id.uuidString,
            transcriptionDbId: transcription.id.uuidString,
            pipeline: pipeline,
            audio: audio
        )
        PhoneticShadowLogger.shared.log(event)
    }

    private func shadowASREngineName(for provider: ModelProvider) -> String {
        switch provider {
        case .qwen3, .qwen3CoreML:
            return "Qwen3-ASR"
        case .whisper, .whisperMLX, .whisperCoreML:
            return "Whisper"
        default:
            return "unknown"
        }
    }

    private func totalLatencyMs(transcriptionDuration: TimeInterval?, enhancementDuration: TimeInterval?) -> Double? {
        let total = (transcriptionDuration ?? 0) + (enhancementDuration ?? 0)
        return total > 0 ? total * 1000 : nil
    }

    private func validateEnhancedText(
        _ enhancedText: String,
        originalText: String,
        enhancementService: AIEnhancementService,
        postProcessor: ChinesePostProcessingService,
        transcription: Transcription
    ) async -> String {
        var acceptedText = enhancedText
        let customVocabulary = CustomVocabularyService.shared.getCustomVocabularyWords(from: modelContext)

        if postProcessor.isEnabled && postProcessor.isLLMValidationEnabled {
            let protectedTerms = customVocabulary + CorrectionProtectionList.shared.allWords()
            let insertedProtectedTerms = VocoAutoApplyModelService.shared.protectedTermGuardTerms()
            let descriptor = FetchDescriptor<WordReplacement>(predicate: #Predicate { $0.isEnabled })
            let replacements = (try? modelContext.fetch(descriptor))?.map {
                (original: $0.originalText, replacement: $0.replacementText)
            } ?? []
            let validation = postProcessor.llmResponseValidator.validate(
                response: acceptedText,
                original: originalText,
                protectedTerms: protectedTerms,
                insertedProtectedTerms: insertedProtectedTerms,
                wordReplacements: replacements,
                customVocabulary: customVocabulary
            )

            if !validation.isValid {
                if validation.isRetryable,
                   let retry = try? await enhancementService.enhanceConservative(
                    originalText,
                    uncertainWords: postProcessor.lastUncertainWords
                   ),
                   postProcessor.llmResponseValidator.validate(
                    response: retry.0,
                    original: originalText,
                    protectedTerms: protectedTerms,
                    insertedProtectedTerms: insertedProtectedTerms,
                    wordReplacements: replacements,
                    customVocabulary: customVocabulary
                   ).isValid {
                    acceptedText = retry.0
                    transcription.enhancementDuration = (transcription.enhancementDuration ?? 0) + retry.1
                } else {
                    acceptedText = originalText
                }
            }
        }

        if PersonalStyleGuardService.isEnabled(), acceptedText != originalText {
            let result = PersonalStyleGuardService.shared.validate(
                response: acceptedText,
                original: originalText
            )
            if !result.isValid {
                transcription.recordStyleGuardRejection(
                    response: acceptedText,
                    reasons: result.reasons
                )
                acceptedText = originalText
            }
        }

        acceptedText = PostLLMCommaCleanup.clean(acceptedText, originalText: originalText)
        let contentLength = Self.autoPunctuationContentLength(acceptedText)
        if contentLength >= Self.minContentCharactersForAutoPunctuation {
            let punctuation = Self.autoPunctuationMarks
            let punctuationCount = acceptedText.filter { punctuation.contains($0) }.count
            if punctuationCount < max(contentLength / 20, 1) {
                let ruleResult = RuleBasedPunctuationInserter.insert(into: acceptedText)
                if ruleResult.filter({ punctuation.contains($0) }).count > punctuationCount {
                    acceptedText = ruleResult
                }
            }
        }
        acceptedText = VocoCanonicalizationService.removeStandaloneVocabularyTerminalPeriod(
            acceptedText,
            vocabularyWords: customVocabulary
        )

        return acceptedText
    }

    private static func autoPunctuationContentLength(_ text: String) -> Int {
        text.filter {
            !$0.isWhitespace &&
                !$0.isNewline &&
                !autoPunctuationMarks.contains($0) &&
                !asciiPunctuationMarks.contains($0)
        }.count
    }
}
