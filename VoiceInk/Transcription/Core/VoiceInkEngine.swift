import Foundation
import SwiftUI
import AVFoundation
import SwiftData
import AppKit
import os

@MainActor
class VoiceInkEngine: NSObject, ObservableObject {
    @Published var recordingState: RecordingState = .idle
    @Published var shouldCancelRecording = false
    var partialTranscript: String = ""
    var currentSession: TranscriptionSession?

    let recorder = Recorder()
    var recordedFile: URL? = nil
    let recordingsDirectory: URL
    /// PID of the frontmost app captured at recording start, used for AX queries after transcription.
    private var capturedFrontAppPID: pid_t?

    // Injected managers
    let whisperModelManager: WhisperModelManager
    let transcriptionModelManager: TranscriptionModelManager
    weak var recorderUIManager: RecorderUIManager?
    weak var prewarmService: ModelPrewarmService?

    let modelContext: ModelContext
    internal let serviceRegistry: TranscriptionServiceRegistry
    let enhancementService: AIEnhancementService?
    internal let pipeline: TranscriptionPipeline

    let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "VoiceInkEngine")

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

        let appSupportDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("com.prakashjoshipax.VoiceInk")
        self.recordingsDirectory = appSupportDirectory.appendingPathComponent("Recordings")

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

        if let enhancementService {
            PowerModeSessionManager.shared.configure(engine: self, enhancementService: enhancementService)
        }

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

    func toggleRecord(powerModeId: UUID? = nil) async {
        logger.notice("toggleRecord called – state=\(String(describing: self.recordingState), privacy: .public)")

        if recordingState == .recording {
            partialTranscript = ""
            recordingState = .transcribing
            await recorder.stopRecording()

            if let recordedFile {
                if !shouldCancelRecording {
                    let audioAsset = AVURLAsset(url: recordedFile)
                    let duration = (try? CMTimeGetSeconds(await audioAsset.load(.duration))) ?? 0.0

                    let transcription = Transcription(
                        text: "",
                        duration: duration,
                        audioFileURL: recordedFile.absoluteString,
                        transcriptionStatus: .pending
                    )
                    modelContext.insert(transcription)
                    try? modelContext.save()
                    NotificationCenter.default.post(name: .transcriptionCreated, object: transcription)

                    await runPipeline(on: transcription, audioURL: recordedFile)
                } else {
                    currentSession?.cancel()
                    currentSession = nil
                    capturedFrontAppPID = nil
                    try? FileManager.default.removeItem(at: recordedFile)
                    recordingState = .idle
                    await cleanupResources()
                }
            } else {
                logger.error("❌ No recorded file found after stopping recording")
                currentSession?.cancel()
                currentSession = nil
                recordingState = .idle
                await cleanupResources()
            }
        } else {
            StartupTracer.checkpoint("toggleRecord_start_branch")
            logger.notice("toggleRecord: entering start-recording branch")
            guard transcriptionModelManager.currentTranscriptionModel != nil else {
                NotificationManager.shared.showNotification(title: String(localized: "No AI Model Selected"), type: .error)
                return
            }
            shouldCancelRecording = false
            partialTranscript = ""

            // Capture frontmost app BEFORE entering Task (Voco becomes frontmost inside Task)
            let capturedFrontApp = NSWorkspace.shared.frontmostApplication
            capturedFrontAppPID = capturedFrontApp?.processIdentifier
            let capturedAppName = capturedFrontApp?.localizedName

            // Set recording state immediately for responsive UI
            recordingState = .recording
            StartupTracer.checkpoint("toggleRecord_state_set_recording")

            do {
                let fileName = "\(UUID().uuidString).wav"
                let permanentURL = recordingsDirectory.appendingPathComponent(fileName)
                recordedFile = permanentURL

                let pendingChunks = OSAllocatedUnfairLock(initialState: [Data]())
                recorder.onAudioChunk = { data in
                    pendingChunks.withLock { $0.append(data) }
                }

                // Inline await: recorder.startRecording suspends into audioSetupQueue,
                // freeing MainActor so SwiftUI render runs concurrently with Core Audio setup.
                StartupTracer.checkpoint("toggleRecord_before_startRecording")
                try await recorder.startRecording(toOutputFile: permanentURL)

                StartupTracer.end("recorder_startRecording_done")

                guard recorderUIManager?.isMiniRecorderVisible ?? false, !shouldCancelRecording else {
                    recorder.stopRecording()
                    recordedFile = nil
                    recordingState = .idle
                    return
                }

                logger.notice("toggleRecord: recording started successfully, state=recording")

                // Apply PowerMode configuration in background — don't block session/model prep
                Task { await ActiveWindowService.shared.applyConfiguration(powerModeId: powerModeId) }

                if recordingState == .recording,
                   let model = transcriptionModelManager.currentTranscriptionModel {
                    let session = serviceRegistry.createSession(
                        for: model,
                        onPartialTranscript: { [weak self] partial in
                            Task { @MainActor in
                                self?.partialTranscript = partial
                            }
                        }
                    )
                    currentSession = session
                    let realCallback = try await session.prepare(model: model)

                    if let realCallback {
                        recorder.onAudioChunk = realCallback
                        let buffered = pendingChunks.withLock { chunks -> [Data] in
                            let result = chunks
                            chunks.removeAll()
                            return result
                        }
                        for chunk in buffered { realCallback(chunk) }
                    } else {
                        recorder.onAudioChunk = nil
                        pendingChunks.withLock { $0.removeAll() }
                    }
                }

                Task.detached { [weak self] in
                    guard let self else { return }

                    if let model = await self.transcriptionModelManager.currentTranscriptionModel {
                        switch model.provider {
                        case .local:
                            if let localWhisperModel = await self.whisperModelManager.availableModels.first(where: { $0.name == model.name }),
                               await self.whisperModelManager.whisperContext == nil {
                                do {
                                    try await self.whisperModelManager.loadModel(localWhisperModel)
                                } catch {
                                    await self.logger.error("❌ Model loading failed: \(error.localizedDescription, privacy: .public)")
                                }
                            }
                        case .parakeet:
                            if let parakeetModel = model as? ParakeetModel {
                                try? await self.serviceRegistry.parakeetTranscriptionService.loadModel(for: parakeetModel)
                            }
                        case .whisperMLX:
                            if let mlxModel = model as? WhisperMLXModel {
                                try? await self.serviceRegistry.whisperMLXTranscriptionService.preloadModel(for: mlxModel)
                            }
                        case .qwen3, .qwen3CoreML:
                            break // Qwen3 loads on demand during transcription
                        default:
                            break
                        }
                    }

                    if let enhancementService = await self.enhancementService {
                        // Cache app context from EditModeCacheService (fork feature)
                        await MainActor.run {
                            self.cacheEditModeAppContext(capturedAppName: capturedAppName)
                        }

                        guard !Task.isCancelled else { return }

                        let shouldCaptureClipboard = await MainActor.run {
                            enhancementService.useClipboardContext
                        }
                        if shouldCaptureClipboard {
                            await MainActor.run {
                                enhancementService.captureClipboardContext()
                            }
                        }

                        guard !Task.isCancelled else { return }

                        let shouldCaptureScreen = await MainActor.run {
                            enhancementService.useScreenCaptureContext
                        }
                        if shouldCaptureScreen {
                            await enhancementService.captureScreenContext()
                        }
                    }
                }

            } catch {
                logger.error("❌ Failed to start recording: \(error.localizedDescription, privacy: .public)")
                NotificationManager.shared.showNotification(title: String(localized: "Recording failed to start"), type: .error)
                logger.notice("toggleRecord: calling dismissMiniRecorder from error handler")
                await recorderUIManager?.dismissMiniRecorder()
                recordedFile = nil
            }
        }
    }

    // MARK: - Pipeline Dispatch

    private func runPipeline(on transcription: Transcription, audioURL: URL) async {
        guard let model = transcriptionModelManager.currentTranscriptionModel else {
            transcription.text = "Transcription Failed: No model selected"
            transcription.transcriptionStatus = TranscriptionStatus.failed.rawValue
            try? modelContext.save()
            recordingState = .idle
            return
        }

        let session = currentSession
        currentSession = nil

        let pid = capturedFrontAppPID
        capturedFrontAppPID = nil

        await runPipelineWithForkFeatures(
            transcription: transcription,
            audioURL: audioURL,
            model: model,
            session: session,
            capturedAppPID: pid
        )

        shouldCancelRecording = false
        if recordingState != .idle {
            recordingState = .idle
        }
    }

    // MARK: - Resource Cleanup

    func cleanupResources() async {
        cancelScheduledModelCleanup()
        logger.notice("cleanupResources: releasing model resources")
        await whisperModelManager.cleanupResources()
        await serviceRegistry.cleanup()
        logger.notice("cleanupResources: completed")
    }

    // MARK: - Notification Handling

    func setupNotifications() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleLicenseStatusChanged),
            name: .licenseStatusChanged,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handlePromptChange),
            name: .promptDidChange,
            object: nil
        )
    }

    @objc func handleLicenseStatusChanged() {
        pipeline.licenseViewModel = LicenseViewModel()
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
