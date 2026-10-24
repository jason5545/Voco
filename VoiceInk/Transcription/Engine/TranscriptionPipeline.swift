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
    private let serviceRegistry: VocoTranscriptionServiceRegistry
    let enhancementService: AIEnhancementService?
    private let delivery = TranscriptionDelivery()
    let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "TranscriptionPipeline")

    init(
        modelContext: ModelContext,
        serviceRegistry: VocoTranscriptionServiceRegistry,
        enhancementService: AIEnhancementService?
    ) {
        self.modelContext = modelContext
        self.serviceRegistry = serviceRegistry
        self.enhancementService = enhancementService
    }

    func run(
        transcription: Transcription,
        audioURL: URL,
        transcriptionConfiguration: TranscriptionRuntimeConfiguration,
        formattingConfiguration resolveFormattingConfiguration: @escaping () -> TranscriptionFormattingConfiguration,
        session: TranscriptionSession?,
        enhancementConfiguration: @escaping () -> EnhancementRuntimeConfiguration?,
        recordingContextSnapshot: @escaping () async -> RecordingContextSnapshot? = { nil },
        outputConfiguration: @escaping () -> OutputRuntimeConfiguration,
        editModeSelection: EditModeSelectionSnapshot? = nil,
        capturedAppPID: pid_t? = nil,
        onStateChange: @escaping (RecordingState) -> Void,
        shouldCancel: @escaping () -> Bool,
        onCancel: @escaping () async -> Void,
        onDismiss: @escaping () async -> Void,
        onEditModeComplete: ((WordSubstitution?) -> Void)? = nil,
        assistant: AssistantHooks = .inactive
    ) async {
        let model = transcriptionConfiguration.model
        var finalText: String?
        var responseError: String?
        var outputForDelivery: OutputRuntimeConfiguration?
        var responseConfig: EnhancementRuntimeConfiguration?
        let vocoProcessor = VocoTranscriptionProcessor(
            modelContext: modelContext,
            serviceRegistry: serviceRegistry,
            enhancementService: enhancementService
        )

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
            let transcriptionDuration = Date().timeIntervalSince(transcriptionStart)

            let afterASR = await vocoProcessor.afterASR(
                text: text,
                transcription: transcription,
                audioURL: audioURL,
                model: model,
                transcriptionConfiguration: transcriptionConfiguration,
                formattingConfiguration: resolveFormattingConfiguration(),
                transcriptionDuration: transcriptionDuration,
                editModeSelection: editModeSelection,
                onStateChange: onStateChange,
                shouldCancel: shouldCancel,
                onCancel: finishCanceledTranscription,
                onDismiss: onDismiss,
                onEditModeComplete: onEditModeComplete
            )
            guard afterASR.shouldContinue else { return }
            text = afterASR.text
            finalText = text

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
                    afterASR.skipEnhancement
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
                        let acceptedText = await vocoProcessor.postEnhancement(
                            enhancedText,
                            originalText: text,
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
            var didInsertSessionMetric = false
            if transcription.transcriptionStatus == TranscriptionStatus.completed.rawValue {
                didInsertSessionMetric = vocoProcessor.record(transcription: transcription, model: model)
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
        finalText = await vocoProcessor.preDelivery(
            text: finalText,
            transcription: transcription,
            audioURL: audioURL,
            model: model,
            output: resolvedOutput,
            isAssistantFollowUp: assistant.isFollowUp,
            capturedAppPID: capturedAppPID
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
}
