import Foundation
import SwiftUI
import AVFoundation
import SwiftData
import AppKit
import os

@MainActor
class VoiceInkEngine: NSObject, ObservableObject {
    private enum RecordingUseCase {
        case newSession
        case assistantFollowUp

        var isAssistantFollowUp: Bool {
            self == .assistantFollowUp
        }
    }

    @Published var recordingState: RecordingState = .idle
    @Published var shouldCancelRecording = false
    @Published var partialTranscript: String = ""
    var currentSession: TranscriptionSession?
    private var currentSessionTranscriptionConfiguration: TranscriptionRuntimeConfiguration?
    private var activeRecordingStartID: UUID?
    private var activePipelineTranscriptionID: UUID?
    private var canceledPipelineTranscriptionIDs = Set<UUID>()
    private var activeRecordingUseCase: RecordingUseCase = .newSession
    private var activePipelineUseCase: RecordingUseCase = .newSession
    private var activeRecordingContextStore: RecordingContextSnapshotStore?
    private var activeRecordingContextTasks: [Task<Void, Never>] = []
    private var capturedFrontAppPID: pid_t?

    let recorder = Recorder()
    var recordedFile: URL? = nil
    let recordingsDirectory: URL

    // Injected managers
    let whisperModelManager: WhisperModelManager
    let transcriptionModelManager: TranscriptionModelManager
    weak var recorderUIManager: RecorderPanelPresenting?
    weak var prewarmService: ModelPrewarmService?

    let modelContext: ModelContext
    internal let serviceRegistry: TranscriptionServiceRegistry
    let enhancementService: AIEnhancementService?
    let assistantSession = AssistantSession()
    let assistantChat: AssistantChatService?
    internal let pipeline: TranscriptionPipeline

    let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "VoiceInkEngine")

    init(
        modelContext: ModelContext,
        whisperModelManager: WhisperModelManager,
        transcriptionModelManager: TranscriptionModelManager,
        enhancementService: AIEnhancementService? = nil
    ) {
        self.modelContext = modelContext
        self.whisperModelManager = whisperModelManager
        self.transcriptionModelManager = transcriptionModelManager
        self.enhancementService = enhancementService
        if let aiService = enhancementService?.getAIService() {
            self.assistantChat = AssistantChatService(
                modelContext: modelContext,
                aiService: aiService
            )
        } else {
            self.assistantChat = nil
        }

        self.recordingsDirectory = AppIdentifiers.appSupportDirectory.appendingPathComponent("Recordings")

        self.serviceRegistry = TranscriptionServiceRegistry(
            modelProvider: whisperModelManager,
            modelsDirectory: whisperModelManager.modelsDirectory,
            modelContext: modelContext
        )
        self.pipeline = TranscriptionPipeline(
            modelContext: modelContext,
            serviceRegistry: serviceRegistry,
            enhancementService: enhancementService
        )

        super.init()

        setupNotifications()
        createRecordingsDirectoryIfNeeded()
    }

    private func createRecordingsDirectoryIfNeeded() {
        do {
            try FileManager.default.createDirectory(at: recordingsDirectory, withIntermediateDirectories: true, attributes: nil)
        } catch {
            logger.error("❌ Error creating recordings directory: \(error.localizedDescription, privacy: .public)")
        }
    }

    func getEnhancementService() -> AIEnhancementService? {
        return enhancementService
    }

    // MARK: - Toggle Record

    func toggleRecord(modeId: UUID? = nil, isAssistantFollowUp: Bool = false) async {
        if recordingState == .starting {
            await cancelRecording()
            return
        }

        if recordingState == .recording {
            activePipelineUseCase = activeRecordingUseCase
            activeRecordingUseCase = .newSession
            activeRecordingStartID = nil
            partialTranscript = ""
            recordingState = .transcribing
            await recorder.stopRecording()
            await waitForEditModeDetectionIfNeeded()

            if let recordedFile {
                if !shouldCancelRecording {
                    let transcription = makeRecordingTranscription(
                        for: recordedFile,
                        text: "",
                        duration: 0,
                        transcriptionStatus: .pending
                    )
                    modelContext.insert(transcription)
                    try? modelContext.save()
                    NotificationCenter.default.post(name: .transcriptionCreated, object: transcription)

                    await runPipeline(
                        on: transcription,
                        audioURL: recordedFile,
                        contextStore: activeRecordingContextStore
                    )
                } else {
                    await finishActiveRecorderCancellation()
                }
            } else {
                cancelCurrentSession()
                if !shouldCancelRecording {
                    logger.error("❌ No recorded file found after stopping recording")
                }
                recordingState = .idle
                await cleanupResources()
            }
        } else {
            StartupTracer.checkpoint("toggleRecord_start_branch")
            let canContinueAssistantSession = isAssistantFollowUp && assistantSession.canSendFollowUp
            let recordingUseCase: RecordingUseCase = canContinueAssistantSession ? .assistantFollowUp : .newSession

            activePipelineTranscriptionID = nil
            shouldCancelRecording = false
            partialTranscript = ""
            activeRecordingUseCase = recordingUseCase
            clearActiveRecordingContext()

            if !recordingUseCase.isAssistantFollowUp {
                assistantSession.reset()
            }
            StartupTracer.checkpoint("toggleRecord_context_reset_done")

            guard requestRecordPermission() else {
                logger.error("Recording permission denied")
                return
            }

            StartupTracer.checkpoint("toggleRecord_permission_granted")
            let startID = UUID()
            activeRecordingStartID = startID
            let capturedApp = NSWorkspace.shared.frontmostApplication
            capturedFrontAppPID = capturedApp?.processIdentifier
            StartupTracer.checkpoint("toggleRecord_frontmost_app_captured")
            cacheEditModeAppContext(capturedAppName: capturedApp?.localizedName)
            StartupTracer.checkpoint("toggleRecord_edit_context_cached")
            let activeModeTask = ActiveWindowService.shared.beginApplyingConfiguration(modeId: modeId) { [weak self] in
                guard let self else { return false }
                return self.activeRecordingStartID == startID && !self.shouldCancelRecording
            }
            StartupTracer.checkpoint("toggleRecord_active_config_started")

            do {
                let startFlow = try await EngineRecordingStartFlow.run(
                    makeRecordingURL: {
                        let fileName = "\(UUID().uuidString).wav"
                        return self.recordingsDirectory.appendingPathComponent(fileName)
                    },
                    setRecordedFile: {
                        self.recordedFile = $0
                    },
                    setAudioChunkCallback: {
                        self.recorder.onAudioChunk = $0
                    },
                    setRecordingStateStarting: {
                        self.recordingState = .starting
                    },
                    scheduleSystemMute: {
                        self.recorder.scheduleSystemMute()
                    },
                    startRecording: {
                        try await self.recorder.startRecording(toOutputFile: $0)
                    }
                )
                let permanentURL = startFlow.recordingURL
                let pendingChunks = startFlow.pendingChunks

                guard self.activeRecordingStartID == startID,
                      self.recorderUIManager?.isRecorderPanelVisible ?? false,
                      !self.shouldCancelRecording else {
                    activeModeTask.cancel()
                    let shouldKeepRecordingFile = self.shouldCancelRecording
                    if self.activeRecordingStartID == startID {
                        await self.recorder.stopRecording()
                        if !shouldKeepRecordingFile {
                            self.recordedFile = nil
                        }
                        self.recordingState = .idle
                        self.activeRecordingStartID = nil
                    }
                    return
                }

                self.recordingState = .recording

                await activeModeTask.value

                guard self.recordingState == .recording,
                      self.activeRecordingStartID == startID,
                      !self.shouldCancelRecording else {
                    return
                }

                self.startRecordingContextCapture()

                guard let transcriptionConfiguration = ModeRuntimeResolver.transcriptionConfiguration(
                    transcriptionModelManager: self.transcriptionModelManager
                ) else {
                    NotificationManager.shared.showNotification(title: "No AI Model Selected", type: .error)
                    await self.recorder.stopRecording()
                    try? FileManager.default.removeItem(at: permanentURL)
                    self.recordedFile = nil
                    self.recordingState = .idle
                    self.activeRecordingStartID = nil
                    self.clearActiveRecordingContext()
                    await self.cleanupResources()
                    await self.recorderUIManager?.dismissRecorderPanel()
                    return
                }

                if self.serviceRegistry.shouldUseRealtimeTranscription(for: transcriptionConfiguration) {
                    let session = self.serviceRegistry.createSession(
                        for: transcriptionConfiguration,
                        onPartialTranscript: { [weak self] partial in
                            Task { @MainActor in
                                guard let self,
                                      self.activeRecordingStartID == startID,
                                      self.recordingState == .recording else {
                                    return
                                }
                                self.partialTranscript = partial
                            }
                        }
                    )
                    self.currentSession = session
                    self.currentSessionTranscriptionConfiguration = transcriptionConfiguration
                    let realCallback = try await session.prepare(
                        configuration: transcriptionConfiguration
                    )

                    if let realCallback {
                        self.recorder.onAudioChunk = realCallback
                        let buffered = pendingChunks.withLock { chunks -> [Data] in
                            let result = chunks
                            chunks.removeAll()
                            return result
                        }
                        for chunk in buffered { realCallback(chunk) }
                    }
                } else {
                    self.currentSession = nil
                    self.currentSessionTranscriptionConfiguration = nil
                    self.recorder.onAudioChunk = nil
                    pendingChunks.withLock { $0.removeAll() }
                }

                Task { @MainActor [weak self] in
                    guard let self else { return }

                    let currentModel = ModeRuntimeResolver.transcriptionConfiguration(
                        transcriptionModelManager: self.transcriptionModelManager
                    )?.model

                    if let model = currentModel,
                       model.provider == .whisper {
                        if let localWhisperModel = self.whisperModelManager.availableModels.first(where: { $0.name == model.name }),
                           self.whisperModelManager.whisperContext == nil {
                            do {
                                try await self.whisperModelManager.loadModel(localWhisperModel)
                            } catch {
                                self.logger.error("❌ Model loading failed: \(error.localizedDescription, privacy: .public)")
                            }
                        }
                    } else if let fluidAudioModel = currentModel as? FluidAudioModel {
                        try? await self.serviceRegistry.fluidAudioTranscriptionService.loadModel(for: fluidAudioModel)
                    } else if let mlxModel = currentModel as? WhisperMLXModel {
                        try? await self.serviceRegistry.whisperMLXTranscriptionService.preloadModel(for: mlxModel)
                    }

                }

            } catch {
                activeModeTask.cancel()
                logger.error("Recording failed to start: \(error.localizedDescription, privacy: .public)")
                await recorder.stopRecording()
                cancelCurrentSession()
                if let recordedFile = self.recordedFile {
                    try? FileManager.default.removeItem(at: recordedFile)
                }
                recordingState = .idle
                recordedFile = nil
                activeRecordingStartID = nil
                clearActiveRecordingContext()
                await cleanupResources()
                NotificationManager.shared.showNotification(title: "Recording failed to start", type: .error)
                await recorderUIManager?.dismissRecorderPanel()
            }
        }
    }

    private func requestRecordPermission() -> Bool {
        true
    }

    // MARK: - Recording Context

    private func startRecordingContextCapture() {
        clearActiveRecordingContext()

        let store = RecordingContextSnapshotStore()
        activeRecordingContextStore = store
        activeRecordingContextTasks = RecordingContextCaptureService.startCapture(into: store)
    }

    private func clearActiveRecordingContext() {
        activeRecordingContextTasks.forEach { $0.cancel() }
        activeRecordingContextTasks.removeAll()
        activeRecordingContextStore = nil
    }

    // MARK: - Pipeline Dispatch

    private func waitForEditModeDetectionIfNeeded(timeoutNanoseconds: UInt64 = 500_000_000) async {
        guard let task = forkState.editModeDetectionTask else { return }

        let completed = await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                await task.value
                return true
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: timeoutNanoseconds)
                return false
            }

            let completed = await group.next() ?? false
            group.cancelAll()
            return completed
        }

        if !completed {
            task.cancel()
        }
        forkState.editModeDetectionTask = nil
    }

    private func runPipeline(
        on transcription: Transcription,
        audioURL: URL,
        contextStore: RecordingContextSnapshotStore?
    ) async {
        guard let transcriptionConfiguration = currentSessionTranscriptionConfiguration ??
            ModeRuntimeResolver.transcriptionConfiguration(transcriptionModelManager: transcriptionModelManager) else {
            transcription.text = "Transcription Failed: No model selected"
            transcription.transcriptionStatus = TranscriptionStatus.failed.rawValue
            try? modelContext.save()
            recordingState = .idle
            activePipelineUseCase = .newSession
            return
        }

        let session = currentSession
        let transcriptionID = transcription.id
        activePipelineTranscriptionID = transcriptionID

        await pipeline.run(
            transcription: transcription,
            audioURL: audioURL,
            transcriptionConfiguration: transcriptionConfiguration,
            formattingConfiguration: {
                ModeRuntimeResolver.transcriptionFormattingConfiguration()
            },
            session: session,
            enhancementConfiguration: { [weak self] in
                guard let self,
                      let enhancementService = self.enhancementService,
                      let aiService = enhancementService.getAIService() else {
                    return nil
                }
                return ModeRuntimeResolver.currentEnhancementConfiguration(
                    enhancementService: enhancementService,
                    aiService: aiService
                )
            },
            recordingContextSnapshot: {
                await MainActor.run {
                    contextStore?.snapshot
                }
            },
            outputConfiguration: {
                ModeRuntimeResolver.outputConfiguration()
            },
            isEditMode: forkState.isEditMode,
            editModeSelectedText: forkState.editModeSelectedText,
            capturedAppPID: capturedFrontAppPID,
            onStateChange: { [weak self] state in
                guard let self, self.activePipelineTranscriptionID == transcriptionID else { return }
                self.recordingState = state
            },
            shouldCancel: { [weak self] in
                guard let self else { return false }
                return self.canceledPipelineTranscriptionIDs.contains(transcriptionID)
                    || (self.activePipelineTranscriptionID == transcriptionID && self.shouldCancelRecording)
            },
            onCancel: { [weak self, session] in
                guard let self else { return }
                self.cancelPipelineSession(transcriptionID: transcriptionID, session: session)
            },
            onDismiss: { [weak self] in
                guard let self, self.activePipelineTranscriptionID == transcriptionID else { return }
                self.forkState.clearEditMode()
                await self.recorderUIManager?.dismissRecorderPanel()
            },
            onEditModeComplete: { [weak self] substitution in
                guard let self else { return }
                self.forkState.pendingDictionaryEntry = substitution
                self.forkState.clearEditMode()
                self.recordingState = .idle
                self.startDictionaryDismissTimer()
            },
            assistant: TranscriptionPipeline.AssistantHooks(
                isFollowUp: activePipelineUseCase.isAssistantFollowUp,
                sendFollowUp: { [weak self] text, transcription in
                    guard let self, self.activePipelineTranscriptionID == transcriptionID else { return }
                    await self.sendAssistantFollowUp(text, transcription: transcription)
                },
                startResponse: { [weak self] transcript, configuration in
                    guard let self, self.activePipelineTranscriptionID == transcriptionID else { return }
                    self.assistantSession.beginInitialResponse(
                        transcript: transcript,
                        provider: configuration.provider,
                        modelName: configuration.modelName ?? configuration.provider?.defaultModel,
                        modeName: configuration.mode?.name,
                        modeEmoji: configuration.mode?.icon.legacyEmojiValue,
                        promptName: configuration.prompt?.title
                    )
                },
                showResponse: { [weak self] response, systemPrompt in
                    guard let self, self.activePipelineTranscriptionID == transcriptionID else { return }
                    await self.completeAssistantResponse(response, systemPrompt: systemPrompt)
                },
                failResponse: { [weak self] message in
                    guard let self, self.activePipelineTranscriptionID == transcriptionID else { return }
                    self.assistantSession.fail(message)
                }
            )
        )

        let didFinishActivePipeline = activePipelineTranscriptionID == transcriptionID
        if didFinishActivePipeline {
            await finishRecorderSession()
            scheduleModelResourceCleanup()
            activePipelineTranscriptionID = nil
            currentSession = nil
            currentSessionTranscriptionConfiguration = nil
            recordedFile = nil
            shouldCancelRecording = false
            activePipelineUseCase = .newSession
            clearActiveRecordingContext()
            capturedFrontAppPID = nil
        }
        canceledPipelineTranscriptionIDs.remove(transcriptionID)

        if didFinishActivePipeline &&
            (recordingState == .transcribing || recordingState == .enhancing || recordingState == .busy) {
            recordingState = .idle
        }
    }

    // MARK: - Cancellation

    func cancelRecording() async {
        let shouldFinishSessionImmediately: Bool
        switch recordingState {
        case .starting, .recording:
            requestRecordingCancellation()
            await finishActiveRecorderCancellation()
            shouldFinishSessionImmediately = true
        case .transcribing, .enhancing:
            requestRecordingCancellation()
            partialTranscript = ""
            recordingState = .idle
            shouldFinishSessionImmediately = false
        case .idle, .busy:
            partialTranscript = ""
            shouldCancelRecording = false
            recordingState = .idle
            shouldFinishSessionImmediately = true
        }

        if shouldFinishSessionImmediately {
            await finishRecorderSession()
        }
    }

    func resetRecordingSession() async {
        cancelCurrentSession()
        activeRecordingStartID = nil
        activePipelineTranscriptionID = nil
        canceledPipelineTranscriptionIDs.removeAll()
        shouldCancelRecording = false
        partialTranscript = ""
        assistantSession.reset()
        activeRecordingUseCase = .newSession
        activePipelineUseCase = .newSession
        clearActiveRecordingContext()
        forkState.clearEditMode()
        await recorder.stopRecording()
        recordedFile = nil
        recordingState = .idle
        await cleanupResources()
        await finishRecorderSession()
    }

    private func requestRecordingCancellation() {
        shouldCancelRecording = true

        if (recordingState == .transcribing || recordingState == .enhancing),
           let activePipelineTranscriptionID {
            canceledPipelineTranscriptionIDs.insert(activePipelineTranscriptionID)
        }

        cancelCurrentSession()
    }

    private func finishActiveRecorderCancellation() async {
        activeRecordingStartID = nil
        clearActiveRecordingContext()
        forkState.clearEditMode()
        await recorder.stopRecording()
        await saveCanceledRecording()
        recordedFile = nil
        partialTranscript = ""
        recordingState = .idle
        await cleanupResources()
    }

    private func saveCanceledRecording() async {
        guard let recordedFile,
              FileManager.default.fileExists(atPath: recordedFile.path)
        else { return }

        let duration = await AudioFileMetadata.duration(for: recordedFile)
        let transcription = makeRecordingTranscription(
            for: recordedFile,
            text: Transcription.canceledTranscriptionText,
            duration: duration,
            transcriptionStatus: .canceled
        )

        modelContext.insert(transcription)

        do {
            try modelContext.save()
            NotificationCenter.default.post(name: .transcriptionCreated, object: transcription)
        } catch {
            logger.error("Failed to save canceled recording: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func makeRecordingTranscription(
        for audioURL: URL,
        text: String,
        duration: TimeInterval,
        transcriptionStatus: TranscriptionStatus
    ) -> Transcription {
        let modeMetadata = currentModeMetadata()

        return Transcription(
            text: text,
            duration: duration,
            audioFileURL: audioURL.absoluteString,
            transcriptionModelName: ModeRuntimeResolver.transcriptionConfiguration(
                transcriptionModelManager: transcriptionModelManager
            )?.model.displayName,
            modeName: modeMetadata.name,
            modeEmoji: modeMetadata.emoji,
            transcriptionStatus: transcriptionStatus
        )
    }

    private func currentModeMetadata() -> (name: String?, emoji: String?) {
        guard let mode = ModeManager.shared.currentEffectiveConfiguration,
              mode.isEnabled else {
            return (nil, nil)
        }

        return (mode.name, mode.icon.legacyEmojiValue)
    }

    // MARK: - Resource Cleanup

    private func cancelPipelineSession(transcriptionID: UUID, session: TranscriptionSession?) {
        session?.cancel()

        guard activePipelineTranscriptionID == transcriptionID else {
            logger.notice("Skipping stale pipeline cleanup")
            return
        }

        currentSession = nil
        currentSessionTranscriptionConfiguration = nil
    }

    private func cancelCurrentSession() {
        currentSession?.cancel()
        currentSession = nil
        currentSessionTranscriptionConfiguration = nil
    }

    private func finishRecorderSession() async {
        enhancementService?.clearCapturedContexts()
    }

    func cleanupResources() async {
        logger.notice("cleanupResources: releasing model resources")
        activeRecordingStartID = nil
        activeRecordingUseCase = .newSession
        await whisperModelManager.cleanupResources()
        await serviceRegistry.cleanup()
        logger.notice("cleanupResources: completed")
    }

    // MARK: - Notification Handling

    func setupNotifications() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handlePromptChange),
            name: .promptDidChange,
            object: nil
        )
    }

    @objc func handlePromptChange() {
        Task {
            let currentPrompt = UserDefaults.standard.string(forKey: "TranscriptionPrompt")
                ?? whisperModelManager.whisperPrompt.transcriptionPrompt
            if let context = whisperModelManager.whisperContext {
                await context.setPrompt(currentPrompt)
            }
        }
    }
}

@MainActor
enum EngineRecordingStartFlow {
    struct Output {
        let recordingURL: URL
        let pendingChunks: OSAllocatedUnfairLock<[Data]>
    }

    static func run(
        makeRecordingURL: @MainActor () -> URL,
        setRecordedFile: @MainActor (URL) -> Void,
        setAudioChunkCallback: @MainActor (@escaping (Data) -> Void) -> Void,
        setRecordingStateStarting: @MainActor () -> Void,
        scheduleSystemMute: @MainActor () -> Void,
        startRecording: @MainActor (URL) async throws -> Void,
        checkpoint: @MainActor (String) -> Void = { StartupTracer.checkpoint($0) },
        endTrace: @MainActor (String) -> Void = { StartupTracer.end($0) }
    ) async throws -> Output {
        let recordingURL = makeRecordingURL()
        setRecordedFile(recordingURL)
        checkpoint("toggleRecord_recording_file_prepared")

        let pendingChunks = OSAllocatedUnfairLock(initialState: [Data]())
        setAudioChunkCallback { data in
            pendingChunks.withLock { $0.append(data) }
        }
        checkpoint("toggleRecord_audio_callback_set")

        setRecordingStateStarting()
        checkpoint("toggleRecord_state_set_starting")
        scheduleSystemMute()

        checkpoint("toggleRecord_before_startRecording")
        try await startRecording(recordingURL)
        endTrace("recorder_startRecording_done")

        return Output(recordingURL: recordingURL, pendingChunks: pendingChunks)
    }
}

enum AudioFileMetadata {
    static func duration(for url: URL) async -> TimeInterval {
        let asset = AVURLAsset(url: url)
        guard let duration = try? await asset.load(.duration) else { return 0 }
        let seconds = CMTimeGetSeconds(duration)
        return seconds.isFinite ? seconds : 0
    }
}
