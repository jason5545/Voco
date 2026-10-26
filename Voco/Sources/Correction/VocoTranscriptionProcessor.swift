import Foundation
import SwiftData
import os

@MainActor
final class VocoTranscriptionProcessor {
    struct AfterASRResult {
        let text: String
        let shouldContinue: Bool
        let skipEnhancement: Bool

        static let stop = AfterASRResult(text: "", shouldContinue: false, skipEnhancement: false)
    }

    let modelContext: ModelContext
    let enhancementService: AIEnhancementService?
    private let serviceRegistry: VocoTranscriptionServiceRegistry
    private let dictationPrivacy: VocoDictationPrivacyController?
    private let shadowLogger: PhoneticShadowLogger
    let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "TranscriptionPipeline")
    private let postProcessor = ChinesePostProcessingService.shared

    private var isEphemeral: Bool { dictationPrivacy?.isEphemeral ?? false }

    /// Test seam: production sends the real key events.
    static var executeVoiceCommand: (VoiceCommand) -> Void = { $0.execute() }

    private static let minContentCharactersForAutoPunctuation = 5
    private static let autoPunctuationMarks: Set<Character> = ["，", "。", "？", "！", "、", "；", "："]
    private static let asciiPunctuationMarks: Set<Character> = [",", ".", "?", "!", ";", ":"]

    private var shadowRawASRText: String?
    private var shadowPostProcessingTrace = ChinesePostProcessingTrace()
    private var shadowConfidenceAssessment: VocoConfidenceAssessment?

    init(
        modelContext: ModelContext,
        serviceRegistry: VocoTranscriptionServiceRegistry,
        enhancementService: AIEnhancementService?,
        dictationPrivacy: VocoDictationPrivacyController? = nil,
        shadowLogger: PhoneticShadowLogger = .shared
    ) {
        self.modelContext = modelContext
        self.serviceRegistry = serviceRegistry
        self.enhancementService = enhancementService
        self.dictationPrivacy = dictationPrivacy
        self.shadowLogger = shadowLogger
    }

    func afterASR(
        text rawText: String,
        transcription: Transcription,
        audioURL: URL,
        model: any TranscriptionModel,
        transcriptionConfiguration: TranscriptionRuntimeConfiguration,
        formattingConfiguration: TranscriptionFormattingConfiguration,
        transcriptionStart: Date,
        editModeSelection: EditModeSelectionSnapshot?,
        onStateChange: @escaping (RecordingState) -> Void,
        shouldCancel: @escaping () -> Bool,
        onCancel: @escaping () async -> Void,
        onDismiss: @escaping () async -> Void,
        onEditModeComplete: ((WordSubstitution?) -> Void)?
    ) async -> AfterASRResult {
        var text = rawText
        shadowRawASRText = rawText

        if model.provider == .qwen3 {
            let adapterMetadata = serviceRegistry.qwen3TranscriptionService.lastAdapterMetadata
            transcription.recordQwen3AdapterMetadata(adapterMetadata)
            transcription.recordQwen3SpecialistRoutingMetadata(
                serviceRegistry.qwen3TranscriptionService.lastSpecialistRoutingMetadata
            )
            let adapterPath = adapterMetadata.adapterPath ?? "none"
            let adapterSHA256 = adapterMetadata.adapterSHA256 ?? "none"
            let adapterLoadError = adapterMetadata.adapterLoadError ?? "none"
            logger.info(
                "Qwen3-ASR row adapter state detected=\(adapterMetadata.adapterDetected, privacy: .public) loaded=\(adapterMetadata.adapterLoaded, privacy: .public) applied=\(adapterMetadata.adapterApplied, privacy: .public) path=\(adapterPath, privacy: .public) sha256=\(adapterSHA256, privacy: .public) error=\(adapterLoadError, privacy: .public)"
            )
        }

        text = TranscriptionOutputFilter.filter(text)
        let transcriptionDuration = Date().timeIntervalSince(transcriptionStart)
        if shouldCancel() {
            await onCancel()
            return .stop
        }

        text = text.trimmingCharacters(in: .whitespacesAndNewlines)
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
                if !isEphemeral {
                    try? modelContext.save()
                }
                await onDismiss()
                return .stop
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
            rawTranscript: rawText,
            model: model,
            modelContext: modelContext,
            transcription: transcription,
            appName: enhancementService?.cachedAppName,
            windowTitle: enhancementService?.cachedWindowTitle
        )
        let normalizationResult = normalizedOutput.normalizationResult
        shadowConfidenceAssessment = normalizedOutput.confidenceAssessment
        text = normalizationResult.normalizedText

        let modeMetadata = transcriptionConfiguration.metadata
        transcription.text = text
        transcription.duration = actualDuration
        transcription.transcriptionModelName = model.displayName
        transcription.transcriptionDuration = transcriptionDuration
        transcription.modeName = modeMetadata.name
        transcription.modeEmoji = modeMetadata.emoji

        if let editModeSelection {
            let handled = await handleEditMode(
                text: text,
                selection: editModeSelection,
                transcription: transcription,
                enhancementService: enhancementService,
                onStateChange: onStateChange,
                shouldCancel: shouldCancel,
                onCleanup: onCancel,
                onDismiss: onDismiss,
                onEditModeComplete: onEditModeComplete
            )
            if handled {
                return .stop
            }
        }

        if let command = VoiceCommandService.shared.detectCommand(in: text) {
            // Ephemeral only skips persistence; the command itself is just local key events.
            if !isEphemeral {
                transcription.transcriptionStatus = TranscriptionStatus.completed.rawValue
                try? modelContext.save()
                NotificationCenter.default.post(name: .transcriptionCompleted, object: transcription)
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                Self.executeVoiceCommand(command)
            }
            await onDismiss()
            return .stop
        }

        return AfterASRResult(
            text: text,
            shouldContinue: true,
            skipEnhancement: postProcessor.isEnabled && !postProcessingNeedsLLM
        )
    }

    func postEnhancement(
        _ enhancedText: String,
        originalText: String,
        transcription: Transcription
    ) async -> String {
        guard let enhancementService else { return enhancedText }
        return await validateEnhancedText(
            enhancedText,
            originalText: originalText,
            enhancementService: enhancementService,
            postProcessor: postProcessor,
            transcription: transcription
        )
    }

    func preDelivery(
        text: String?,
        transcription: Transcription,
        audioURL: URL,
        model: any TranscriptionModel,
        output: OutputRuntimeConfiguration,
        isAssistantFollowUp: Bool,
        capturedAppPID: pid_t?
    ) async -> String? {
        var finalText = text
        if transcription.transcriptionStatus == TranscriptionStatus.completed.rawValue,
           !isAssistantFollowUp,
           output.outputMode == .paste,
           let text {
            var preparedText = WordReplacementService.shared.enforceVocabularyCasing(
                text: text,
                using: modelContext
            )
            preparedText = await applyContextAwareInsertion(
                preparedText,
                enhancementService: enhancementService,
                capturedAppPID: capturedAppPID,
                isEphemeral: isEphemeral
            )
            preparedText = ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: preparedText)
            finalText = preparedText
            transcription.finalPastedText = preparedText
        }

        if !isEphemeral {
            logPhoneticShadowSnapshot(
                transcription: transcription,
                audioURL: audioURL,
                model: model,
                rawASRText: shadowRawASRText,
                postProcessingTrace: shadowPostProcessingTrace,
                confidenceAssessment: shadowConfidenceAssessment,
                finalText: finalText
            )
        }
        return finalText
    }

    func record(transcription: Transcription, model: any TranscriptionModel) -> Bool {
        guard !isEphemeral,
              transcription.transcriptionStatus == TranscriptionStatus.completed.rawValue else {
            return false
        }
        if postProcessor.isEnabled && postProcessor.isContextMemoryEnabled {
            postProcessor.contextMemory.add(transcription.text)
        }
        do {
            return try SessionMetricRecorder.recordRecorderSession(
                transcription: transcription,
                model: model,
                in: modelContext
            )
        } catch {
            logger.error("Failed to record session metric: \(error.localizedDescription, privacy: .public)")
            return false
        }
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
        VocoEvidenceScope.record { [shadowLogger] in
            shadowLogger.log(event)
        }
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
        acceptedText = ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: acceptedText)
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
