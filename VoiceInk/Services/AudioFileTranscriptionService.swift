import Foundation
import SwiftUI
import AVFoundation
import SwiftData
import os

@MainActor
class AudioTranscriptionService: ObservableObject {
    @Published var isTranscribing = false
    @Published var currentError: TranscriptionError?

    private let modelContext: ModelContext
    private let enhancementService: AIEnhancementService?
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "AudioTranscriptionService")
    private let serviceRegistry: TranscriptionServiceRegistry

    enum TranscriptionError: Error {
        case noAudioFile
        case transcriptionFailed
        case modelNotLoaded
        case invalidAudioFormat
    }

    init(modelContext: ModelContext, engine: VoiceInkEngine) {
        self.modelContext = modelContext
        self.enhancementService = engine.enhancementService
        self.serviceRegistry = TranscriptionServiceRegistry(modelProvider: engine.whisperModelManager, modelsDirectory: engine.whisperModelManager.modelsDirectory, modelContext: modelContext)
    }

    init(modelContext: ModelContext, serviceRegistry: TranscriptionServiceRegistry, enhancementService: AIEnhancementService?) {
        self.modelContext = modelContext
        self.enhancementService = enhancementService
        self.serviceRegistry = serviceRegistry
    }
    
    func retranscribeAudio(
        from url: URL,
        using model: any TranscriptionModel,
        mode: ModeConfig? = nil,
        sourceTranscription: Transcription? = nil
    ) async throws -> Transcription {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw TranscriptionError.noAudioFile
        }
        
        await MainActor.run {
            isTranscribing = true
        }
        
        do {
            let mode = mode ?? ModeManager.shared.currentEffectiveConfiguration
            let language = TranscriptionLanguageSupport.validLanguageOrFallback(
                mode?.selectedLanguage,
                for: model,
                realtimeEnabled: mode?.isRealtimeTranscriptionEnabled
            )
            let requestContext = TranscriptionRequestContext(
                language: language,
                prompt: UserDefaults.standard.string(forKey: "TranscriptionPrompt"),
                usesQwen3AudioAdapter: false
            )
            let modeName = (mode?.isEnabled == true) ? mode?.name : nil
            let modeEmoji = (mode?.isEnabled == true) ? mode?.icon.value : nil

            let transcriptionStart = Date()
            var text = try await serviceRegistry.transcribe(
                audioURL: url,
                model: model,
                context: requestContext
            )
            let rawASRText = text
            let transcriptionDuration = Date().timeIntervalSince(transcriptionStart)
            text = TranscriptionOutputFilter.filter(text)
            text = text.trimmingCharacters(in: .whitespacesAndNewlines)
            let formattingConfiguration = ModeRuntimeResolver.transcriptionFormattingConfiguration(mode: mode)

            if formattingConfiguration.isTextFormattingEnabled {
                text = ParagraphFormatter.format(text)
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
                correctionPolicy: .skipPostASRCorrectionModels
            )
            let normalizationResult = normalizedOutput.normalizationResult
            let confidenceAssessment = normalizedOutput.confidenceAssessment
            text = normalizationResult.normalizedText

            let audioAsset = AVURLAsset(url: url)
            let duration = CMTimeGetSeconds(try await audioAsset.load(.duration))
            let recordingsDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("com.prakashjoshipax.VoiceInk")
                .appendingPathComponent("Recordings")
            
            let fileName = "retranscribed_\(UUID().uuidString).wav"
            let permanentURL = recordingsDirectory.appendingPathComponent(fileName)
            
            do {
                try FileManager.default.createDirectory(at: recordingsDirectory, withIntermediateDirectories: true)
                try FileManager.default.copyItem(at: url, to: permanentURL)
            } catch {
                logger.error("❌ Failed to create permanent copy of audio: \(error.localizedDescription, privacy: .public)")
                isTranscribing = false
                throw error
            }
            
            let permanentURLString = permanentURL.absoluteString

            let originalText = text

            func styleGuardedEnhancedText(_ enhancedText: String) -> (acceptedText: String?, rejection: PersonalStyleGuardResult?) {
                guard PersonalStyleGuardService.isEnabled() else {
                    return (enhancedText, nil)
                }

                let result = PersonalStyleGuardService.shared.validate(
                    response: enhancedText,
                    original: originalText
                )
                return result.isValid ? (enhancedText, nil) : (nil, result)
            }

            func recordRetranscriptionSource(on transcription: Transcription) {
                guard let sourceTranscription else { return }
                let feedbackSignal = transcription.recordRetranscriptionAnalysis(source: sourceTranscription)
                CorrectionFeedbackLearningService.stageLearningCandidates(
                    from: feedbackSignal,
                    in: modelContext
                )
            }
            let enhancementConfiguration = enhancementService
                .flatMap { service in
                    service.getAIService().map { aiService in
                        ModeRuntimeResolver.currentEnhancementConfiguration(
                            mode: mode,
                            enhancementService: service,
                            aiService: aiService
                        )
                    }
                }

            // Apply AI enhancement if enabled
            if let enhancementService = enhancementService,
               let enhancementConfiguration,
               enhancementConfiguration.isEnabled,
               enhancementService.isConfigured(for: enhancementConfiguration) {
                do {
                    let (enhancedText, enhancementDuration, promptName) = try await enhancementService.enhance(
                        text,
                        configuration: enhancementConfiguration
                    )
                    let styleGuard = styleGuardedEnhancedText(enhancedText)
                    let newTranscription = Transcription(
                        text: originalText,
                        duration: duration,
                        enhancedText: styleGuard.acceptedText,
                        audioFileURL: permanentURLString,
                        transcriptionModelName: model.displayName,
                        aiEnhancementModelName: enhancementConfiguration.modelName ?? enhancementConfiguration.provider?.defaultModel,
                        promptName: promptName,
                        transcriptionDuration: transcriptionDuration,
                        enhancementDuration: enhancementDuration,
                        aiRequestSystemMessage: enhancementService.lastSystemMessageSent,
                        aiRequestUserMessage: enhancementService.lastUserMessageSent,
                        modeName: modeName,
                        modeEmoji: modeEmoji,
                        rawTranscript: rawASRText,
                        normalizedTranscript: normalizationResult.normalizedText,
                        activeContextIDs: normalizationResult.activeContextIDs,
                        canonicalizationReplacements: normalizationResult.replacements,
                        canonicalizationSuggestions: normalizationResult.suggestions,
                        asrEngineID: VocoCanonicalizationPipeline.asrEngineID(for: model),
                        languageMode: VocoCanonicalizationPipeline.selectedLanguageMode(),
                        confidenceAssessment: confidenceAssessment,
                        styleGuardReasons: styleGuard.rejection?.reasons ?? [],
                        styleGuardRejectedText: styleGuard.rejection == nil ? nil : enhancedText
                    )
                    recordRetranscriptionSource(on: newTranscription)
                    modelContext.insert(newTranscription)
                    do {
                        try modelContext.save()
                        NotificationCenter.default.post(name: .transcriptionCreated, object: newTranscription)
                        NotificationCenter.default.post(name: .transcriptionCompleted, object: newTranscription)
                    } catch {
                        logger.error("❌ Failed to save transcription: \(error.localizedDescription, privacy: .public)")
                    }
                    await MainActor.run {
                        isTranscribing = false
                    }

                    return newTranscription
                } catch {
                    let newTranscription = Transcription(
                        text: originalText,
                        duration: duration,
                        audioFileURL: permanentURLString,
                        transcriptionModelName: model.displayName,
                        promptName: nil,
                        transcriptionDuration: transcriptionDuration,
                        modeName: modeName,
                        modeEmoji: modeEmoji,
                        rawTranscript: rawASRText,
                        normalizedTranscript: normalizationResult.normalizedText,
                        activeContextIDs: normalizationResult.activeContextIDs,
                        canonicalizationReplacements: normalizationResult.replacements,
                        canonicalizationSuggestions: normalizationResult.suggestions,
                        asrEngineID: VocoCanonicalizationPipeline.asrEngineID(for: model),
                        languageMode: VocoCanonicalizationPipeline.selectedLanguageMode(),
                        confidenceAssessment: confidenceAssessment
                    )
                    recordRetranscriptionSource(on: newTranscription)
                    modelContext.insert(newTranscription)
                    do {
                        try modelContext.save()
                        NotificationCenter.default.post(name: .transcriptionCreated, object: newTranscription)
                        NotificationCenter.default.post(name: .transcriptionCompleted, object: newTranscription)
                    } catch {
                        logger.error("❌ Failed to save transcription: \(error.localizedDescription, privacy: .public)")
                    }

                    await MainActor.run {
                        isTranscribing = false
                    }

                    return newTranscription
                }
            } else {
                let newTranscription = Transcription(
                    text: originalText,
                    duration: duration,
                    audioFileURL: permanentURLString,
                    transcriptionModelName: model.displayName,
                    promptName: nil,
                    transcriptionDuration: transcriptionDuration,
                    modeName: modeName,
                    modeEmoji: modeEmoji,
                    rawTranscript: rawASRText,
                    normalizedTranscript: normalizationResult.normalizedText,
                    activeContextIDs: normalizationResult.activeContextIDs,
                    canonicalizationReplacements: normalizationResult.replacements,
                    canonicalizationSuggestions: normalizationResult.suggestions,
                    asrEngineID: VocoCanonicalizationPipeline.asrEngineID(for: model),
                    languageMode: VocoCanonicalizationPipeline.selectedLanguageMode(),
                    confidenceAssessment: confidenceAssessment
                )
                recordRetranscriptionSource(on: newTranscription)
                modelContext.insert(newTranscription)
                do {
                    try modelContext.save()
                    NotificationCenter.default.post(name: .transcriptionCreated, object: newTranscription)
                    NotificationCenter.default.post(name: .transcriptionCompleted, object: newTranscription)
                } catch {
                    logger.error("❌ Failed to save transcription: \(error.localizedDescription, privacy: .public)")
                }

                await MainActor.run {
                    isTranscribing = false
                }

                return newTranscription
            }
        } catch {
            logger.error("❌ Transcription failed: \(error.localizedDescription, privacy: .public)")
            currentError = .transcriptionFailed
            isTranscribing = false
            throw error
        }
    }
}
