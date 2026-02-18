import Foundation
import SwiftUI
import AVFoundation
import SwiftData
import AppKit
import KeyboardShortcuts
import os

// MARK: - Recording State Machine
enum RecordingState: Equatable {
    case idle
    case starting
    case recording
    case transcribing
    case enhancing
    case busy
}

@MainActor
class WhisperState: NSObject, ObservableObject {
    @Published var recordingState: RecordingState = .idle
    @Published var isModelLoaded = false
    @Published var loadedLocalModel: WhisperModel?
    @Published var currentTranscriptionModel: (any TranscriptionModel)?
    @Published var isModelLoading = false
    @Published var availableModels: [WhisperModel] = []
    @Published var allAvailableModels: [any TranscriptionModel] = PredefinedModels.models
    @Published var clipboardMessage = ""
    @Published var miniRecorderError: String?
    @Published var shouldCancelRecording = false
    var partialTranscript: String = ""
    var currentSession: TranscriptionSession?


    @Published var recorderType: String = UserDefaults.standard.string(forKey: "RecorderType") ?? "mini" {
        didSet {
            if isMiniRecorderVisible {
                if oldValue == "notch" {
                    notchWindowManager?.hide()
                    notchWindowManager = nil
                } else {
                    miniWindowManager?.hide()
                    miniWindowManager = nil
                }
                Task { @MainActor in
                    try? await Task.sleep(nanoseconds: 50_000_000)
                    showRecorderPanel()
                }
            }
            UserDefaults.standard.set(recorderType, forKey: "RecorderType")
        }
    }
    
    @Published var isMiniRecorderVisible = false {
        didSet {
            // Dispatch asynchronously to avoid "Publishing changes from within view updates" warning
            DispatchQueue.main.async { [self] in
                if isMiniRecorderVisible {
                    showRecorderPanel()
                } else {
                    hideRecorderPanel()
                }
            }
        }
    }
    
    var whisperContext: WhisperContext?
    let recorder = Recorder()
    var recordedFile: URL? = nil
    let whisperPrompt = WhisperPrompt()
    
    // Prompt detection service for trigger word handling
    private let promptDetectionService = PromptDetectionService()
    
    let modelContext: ModelContext
    
    internal var serviceRegistry: TranscriptionServiceRegistry!
    
    private var modelUrl: URL? {
        let possibleURLs = [
            Bundle.main.url(forResource: "ggml-base.en", withExtension: "bin", subdirectory: "Models"),
            Bundle.main.url(forResource: "ggml-base.en", withExtension: "bin"),
            Bundle.main.bundleURL.appendingPathComponent("Models/ggml-base.en.bin")
        ]
        
        for url in possibleURLs {
            if let url = url, FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }
        return nil
    }
    
    private enum LoadError: Error {
        case couldNotLocateModel
    }
    
    let modelsDirectory: URL
    let recordingsDirectory: URL
    let enhancementService: AIEnhancementService?
    var licenseViewModel: LicenseViewModel
    let logger = Logger(subsystem: "com.jasonchien.voco", category: "WhisperState")
    var notchWindowManager: NotchWindowManager?
    var miniWindowManager: MiniWindowManager?
    
    // For model progress tracking
    @Published var downloadProgress: [String: Double] = [:]
    @Published var parakeetDownloadStates: [String: Bool] = [:]
    
    init(modelContext: ModelContext, enhancementService: AIEnhancementService? = nil) {
        self.modelContext = modelContext
        let appSupportDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("com.jasonchien.Voco")
        
        self.modelsDirectory = appSupportDirectory.appendingPathComponent("WhisperModels")
        self.recordingsDirectory = appSupportDirectory.appendingPathComponent("Recordings")
        
        self.enhancementService = enhancementService
        self.licenseViewModel = LicenseViewModel()
        
        super.init()
        
        // Configure the session manager
        if let enhancementService = enhancementService {
            PowerModeSessionManager.shared.configure(whisperState: self, enhancementService: enhancementService)
        }

        // Initialize the transcription service registry
        self.serviceRegistry = TranscriptionServiceRegistry(whisperState: self, modelsDirectory: self.modelsDirectory)
        
        setupNotifications()
        createModelsDirectoryIfNeeded()
        createRecordingsDirectoryIfNeeded()
        loadAvailableModels()
        loadCurrentTranscriptionModel()
        refreshAllAvailableModels()
    }
    
    private func createRecordingsDirectoryIfNeeded() {
        do {
            try FileManager.default.createDirectory(at: recordingsDirectory, withIntermediateDirectories: true, attributes: nil)
        } catch {
            logger.error("Error creating recordings directory: \(error.localizedDescription)")
        }
    }
    
    func toggleRecord(powerModeId: UUID? = nil) async {
        logger.notice("toggleRecord called – state=\(String(describing: self.recordingState))")
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

                    await transcribeAudio(on: transcription)
                } else {
                    currentSession?.cancel()
                    currentSession = nil
                    try? FileManager.default.removeItem(at: recordedFile)
                    await MainActor.run {
                        recordingState = .idle
                    }
                    await cleanupModelResources()
                }
            } else {
                logger.error("❌ No recorded file found after stopping recording")
                currentSession?.cancel()
                currentSession = nil
                await MainActor.run {
                    recordingState = .idle
                }
            }
        } else {
            logger.notice("toggleRecord: entering start-recording branch")
            guard currentTranscriptionModel != nil else {
                await MainActor.run {
                    NotificationManager.shared.showNotification(
                        title: "No AI Model Selected",
                        type: .error
                    )
                }
                return
            }
            shouldCancelRecording = false
            partialTranscript = ""
            requestRecordPermission { [self] granted in
                if granted {
                    Task {
                        do {
                            // --- Prepare permanent file URL ---
                            let fileName = "\(UUID().uuidString).wav"
                            let permanentURL = self.recordingsDirectory.appendingPathComponent(fileName)
                            self.recordedFile = permanentURL

                            // Buffer chunks from the start; session created after Power Mode resolves
                            let pendingChunks = OSAllocatedUnfairLock(initialState: [Data]())
                            self.recorder.onAudioChunk = { data in
                                pendingChunks.withLock { $0.append(data) }
                            }

                            // Check for user corrections from previous paste
                            AutoDictionaryService.shared.checkForUserCorrections(modelContext: self.modelContext)

                            // Start recording immediately — no waiting for network
                            try await self.recorder.startRecording(toOutputFile: permanentURL)

                            await MainActor.run {
                                self.recordingState = .recording
                            }
                            self.logger.notice("toggleRecord: recording started successfully, state=recording")

                            // Power Mode resolves while recording runs (~50-200ms)
                            await ActiveWindowService.shared.applyConfiguration(powerModeId: powerModeId)

                            // Create session with the resolved model (skip if user already stopped)
                            if self.recordingState == .recording, let model = self.currentTranscriptionModel {
                                let session = self.serviceRegistry.createSession(for: model, onPartialTranscript: { [weak self] partial in
                                    Task { @MainActor in
                                        self?.partialTranscript = partial
                                    }
                                })
                                self.currentSession = session
                                let realCallback = try await session.prepare(model: model)

                                if let realCallback = realCallback {
                                    // Swap callback first so new chunks go straight to the session
                                    self.recorder.onAudioChunk = realCallback
                                    // Then flush anything that was buffered before the swap
                                    let buffered = pendingChunks.withLock { chunks -> [Data] in
                                        let result = chunks
                                        chunks.removeAll()
                                        return result
                                    }
                                    for chunk in buffered { realCallback(chunk) }
                                } else {
                                    self.recorder.onAudioChunk = nil
                                    pendingChunks.withLock { $0.removeAll() }
                                }
                            }

                            // Load model and capture context in background without blocking
                            Task.detached { [weak self] in
                                guard let self = self else { return }

                                // Only load model if it's a local model and not already loaded
                                if let model = await self.currentTranscriptionModel, model.provider == .local {
                                    if let localWhisperModel = await self.availableModels.first(where: { $0.name == model.name }),
                                       await self.whisperContext == nil {
                                        do {
                                            try await self.loadModel(localWhisperModel)
                                        } catch {
                                            await self.logger.error("❌ Model loading failed: \(error.localizedDescription)")
                                        }
                                    }
                                } else if let parakeetModel = await self.currentTranscriptionModel as? ParakeetModel {
                                    try? await self.serviceRegistry.parakeetTranscriptionService.loadModel(for: parakeetModel)
                                }

                                if let enhancementService = await self.enhancementService {
                                    await MainActor.run {
                                        enhancementService.captureClipboardContext()
                                    }
                                    await enhancementService.captureScreenContext()
                                }
                            }

                        } catch {
                            self.logger.error("❌ Failed to start recording: \(error.localizedDescription)")
                            await NotificationManager.shared.showNotification(title: "Recording failed to start", type: .error)
                            self.logger.notice("toggleRecord: calling dismissMiniRecorder from error handler")
                            await self.dismissMiniRecorder()
                            // Do not remove the file on a failed start, to preserve all recordings.
                            self.recordedFile = nil
                        }
                    }
                } else {
                    logger.error("❌ Recording permission denied.")
                }
            }
        }
    }
    
    private func requestRecordPermission(response: @escaping (Bool) -> Void) {
        response(true)
    }
    
    private func transcribeAudio(on transcription: Transcription) async {
        guard let urlString = transcription.audioFileURL, let url = URL(string: urlString) else {
            logger.error("❌ Invalid audio file URL in transcription object.")
            await MainActor.run {
                recordingState = .idle
            }
            transcription.text = "Transcription Failed: Invalid audio file URL"
            transcription.transcriptionStatus = TranscriptionStatus.failed.rawValue
            try? modelContext.save()
            return
        }

        if shouldCancelRecording {
            await MainActor.run {
                recordingState = .idle
            }
            await cleanupModelResources()
            return
        }

        await MainActor.run {
            recordingState = .transcribing
        }

        // Play stop sound when transcription starts with a small delay
        Task {
            let isSystemMuteEnabled = UserDefaults.standard.bool(forKey: "isSystemMuteEnabled")
            if isSystemMuteEnabled {
                try? await Task.sleep(nanoseconds: 200_000_000) // 200 milliseconds delay
            }
            await MainActor.run {
                SoundManager.shared.playStopSound()
            }
        }

        defer {
            if shouldCancelRecording {
                Task {
                    await cleanupModelResources()
                }
            }
        }

        logger.notice("🔄 Starting transcription...")
        
        var finalPastedText: String?
        var promptDetectionResult: PromptDetectionService.PromptDetectionResult?
        let postProcessor = ChinesePostProcessingService.shared

        do {
            guard let model = currentTranscriptionModel else {
                throw WhisperStateError.transcriptionFailed
            }

            let transcriptionStart = Date()
            var text: String
            if let session = currentSession {
                text = try await session.transcribe(audioURL: url)
                currentSession = nil
            } else {
                text = try await serviceRegistry.transcribe(audioURL: url, model: model)
            }
            logger.notice("📝 Transcript: \(text, privacy: .public)")
            text = TranscriptionOutputFilter.filter(text)
            logger.notice("📝 Output filter result: \(text, privacy: .public)")
            let transcriptionDuration = Date().timeIntervalSince(transcriptionStart)

            let powerModeManager = PowerModeManager.shared
            let activePowerModeConfig = powerModeManager.currentActiveConfiguration
            let powerModeName = (activePowerModeConfig?.isEnabled == true) ? activePowerModeConfig?.name : nil
            let powerModeEmoji = (activePowerModeConfig?.isEnabled == true) ? activePowerModeConfig?.emoji : nil

            if await checkCancellationAndCleanup() { return }

            text = text.trimmingCharacters(in: .whitespacesAndNewlines)

            if UserDefaults.standard.bool(forKey: "IsTextFormattingEnabled") {
                text = WhisperTextFormatter.format(text)
                logger.notice("📝 Formatted transcript: \(text, privacy: .public)")
            }

            text = WordReplacementService.shared.applyReplacements(to: text, using: modelContext)
            logger.notice("📝 WordReplacement: \(text, privacy: .public)")

            // === Chinese Post-Processing Pipeline ===
            var ppNeedsLLM = true
            if postProcessor.isEnabled {
                let ppResult = postProcessor.process(text)
                text = ppResult.processedText
                ppNeedsLLM = ppResult.needsLLMCorrection
                logger.notice("📝 ChinesePostProcessing: \(text, privacy: .public) (steps: \(ppResult.appliedSteps.joined(separator: ", ")), needsLLM: \(ppNeedsLLM))")

                // Severe repetition → discard output (Whisper hallucination)
                if let repInfo = ppResult.repetitionInfo, repInfo.isSevere {
                    logger.warning("⚠️ Severe repetition detected (\(String(format: "%.0f%%", repInfo.repetitionRatio * 100))), discarding output")
                    transcription.text = "Discarded: severe repetition"
                    transcription.transcriptionStatus = TranscriptionStatus.failed.rawValue
                    try? modelContext.save()
                    await self.dismissMiniRecorder()
                    shouldCancelRecording = false
                    return
                }
            }

            let audioAsset = AVURLAsset(url: url)
            let actualDuration = (try? CMTimeGetSeconds(await audioAsset.load(.duration))) ?? 0.0
            
            transcription.text = text
            transcription.duration = actualDuration
            transcription.transcriptionModelName = model.displayName
            transcription.transcriptionDuration = transcriptionDuration
            transcription.powerModeName = powerModeName
            transcription.powerModeEmoji = powerModeEmoji
            finalPastedText = text

            // Voice command detection — intercept before AI enhancement
            if let command = VoiceCommandService.shared.detectCommand(in: text) {
                logger.notice("🎤 Voice command detected: \(command.rawValue, privacy: .public)")
                transcription.transcriptionStatus = TranscriptionStatus.completed.rawValue
                try? modelContext.save()
                NotificationCenter.default.post(name: .transcriptionCompleted, object: transcription)
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                    command.execute()
                }
                await self.dismissMiniRecorder()
                shouldCancelRecording = false
                return
            }

            if let enhancementService = enhancementService, enhancementService.isConfigured {
                let detectionResult = await promptDetectionService.analyzeText(text, with: enhancementService)
                promptDetectionResult = detectionResult
                await promptDetectionService.applyDetectionResult(detectionResult, to: enhancementService)
            }

            // Determine if AI Enhancement should be skipped (confidence routing)
            let shouldSkipEnhancement = postProcessor.isEnabled && !ppNeedsLLM
            ChinesePostProcessingService.debugLog("WHISPER_STATE: shouldSkip=\(shouldSkipEnhancement), ppNeedsLLM=\(ppNeedsLLM), postProcessorEnabled=\(postProcessor.isEnabled), enhancementEnabled=\(enhancementService?.isEnhancementEnabled ?? false), isConfigured=\(enhancementService?.isConfigured ?? false) | text(\(text.count)): \(text)")

            if !shouldSkipEnhancement,
               let enhancementService = enhancementService,
               enhancementService.isEnhancementEnabled,
               enhancementService.isConfigured {
                if await checkCancellationAndCleanup() { return }

                await MainActor.run { self.recordingState = .enhancing }
                let textForAI = promptDetectionResult?.processedText ?? text

                do {
                    let (enhancedText, enhancementDuration, promptName) = try await enhancementService.enhance(textForAI)
                    logger.notice("📝 AI enhancement: \(enhancedText, privacy: .public)")

                    // LLM response validation
                    if postProcessor.isEnabled && postProcessor.isLLMValidationEnabled {
                        let isValid = postProcessor.llmResponseValidator.isValid(response: enhancedText, original: textForAI)
                        ChinesePostProcessingService.debugLog("LLM_VALIDATION: isValid=\(isValid) | original(\(textForAI.count)): \(textForAI) | enhanced(\(enhancedText.count)): \(enhancedText)")
                        if !isValid {
                            logger.warning("⚠️ LLM response invalid, falling back to pre-LLM text")
                            // Keep text as-is (pre-LLM), don't use enhancedText
                        } else {
                            transcription.enhancedText = enhancedText
                            finalPastedText = enhancedText
                        }
                    } else {
                        transcription.enhancedText = enhancedText
                        finalPastedText = enhancedText
                    }

                    transcription.aiEnhancementModelName = enhancementService.getAIService()?.currentModel
                    transcription.promptName = promptName
                    transcription.enhancementDuration = enhancementDuration
                    transcription.aiRequestSystemMessage = enhancementService.lastSystemMessageSent
                    transcription.aiRequestUserMessage = enhancementService.lastUserMessageSent
                } catch {
                    transcription.enhancedText = "Enhancement failed: \(error)"

                    if await checkCancellationAndCleanup() { return }
                }
            } else if shouldSkipEnhancement {
                logger.notice("📝 Skipping AI enhancement (confidence routing)")
                ChinesePostProcessingService.debugLog("SKIPPED_LLM: confidence routing skipped | finalPastedText(\(finalPastedText?.count ?? 0)): \(finalPastedText ?? "nil")")

                // Safety net: if skipped text has insufficient punctuation density, force LLM
                let cjkPunctuation: Set<Character> = ["，", "。", "？", "！", "、", "；", "："]
                let pastedText = finalPastedText ?? ""
                let punctCount = pastedText.filter { cjkPunctuation.contains($0) }.count
                let expectedPunct = pastedText.count / 20
                let insufficientPunct = pastedText.count >= 10 && punctCount < max(expectedPunct, 1)
                ChinesePostProcessingService.debugLog("SAFETY_NET_CHECK: len=\(pastedText.count), punctCount=\(punctCount), expected=\(max(expectedPunct, 1)), willTrigger=\(insufficientPunct)")
                if insufficientPunct,
                   let enhancementService = enhancementService,
                   enhancementService.isEnhancementEnabled,
                   enhancementService.isConfigured {
                    logger.notice("📝 Safety net triggered: long text without punctuation, forcing LLM")
                    await MainActor.run { self.recordingState = .enhancing }
                    let textForAI = promptDetectionResult?.processedText ?? text
                    do {
                        let (enhancedText, enhancementDuration, promptName) = try await enhancementService.enhance(textForAI)
                        logger.notice("📝 Safety net AI enhancement: \(enhancedText, privacy: .public)")
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

            transcription.transcriptionStatus = TranscriptionStatus.completed.rawValue

        } catch {
            let errorDescription = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            let recoverySuggestion = (error as? LocalizedError)?.recoverySuggestion ?? ""
            let fullErrorText = recoverySuggestion.isEmpty ? errorDescription : "\(errorDescription) \(recoverySuggestion)"

            transcription.text = "Transcription Failed: \(fullErrorText)"
            transcription.transcriptionStatus = TranscriptionStatus.failed.rawValue
        }

        try? modelContext.save()

        if transcription.transcriptionStatus == TranscriptionStatus.completed.rawValue {
            NotificationCenter.default.post(name: .transcriptionCompleted, object: transcription)
        }

        if await checkCancellationAndCleanup() { return }

        if var textToPaste = finalPastedText, transcription.transcriptionStatus == TranscriptionStatus.completed.rawValue {
            // Enforce vocabulary casing as the final text processing step (after AI enhancement)
            textToPaste = WordReplacementService.shared.enforceVocabularyCasing(
                text: textToPaste, using: modelContext)

            if case .trialExpired = licenseViewModel.licenseState {
                textToPaste = """
                    Your trial has expired. Upgrade to Voco Pro at tryvoiceink.com/buy
                    \n\(textToPaste)
                    """
            }

            // Add to context memory for future LLM disambiguation
            if postProcessor.isEnabled && postProcessor.isContextMemoryEnabled {
                postProcessor.contextMemory.add(textToPaste)
            }

            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                // Snapshot text field state before pasting for auto-dictionary learning
                AutoDictionaryService.shared.capturePrePasteSnapshot(insertedText: textToPaste)

                CursorPaster.pasteAtCursor(textToPaste + " ")

                let powerMode = PowerModeManager.shared
                if let activeConfig = powerMode.currentActiveConfiguration, activeConfig.isAutoSendEnabled {
                    // Slight delay to ensure the paste operation completes
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                        CursorPaster.pressEnter()
                    }
                }
            }
        }

        if let result = promptDetectionResult,
           let enhancementService = enhancementService,
           result.shouldEnableAI {
            await promptDetectionService.restoreOriginalSettings(result, to: enhancementService)
        }

        await self.dismissMiniRecorder()

        shouldCancelRecording = false
    }

    func getEnhancementService() -> AIEnhancementService? {
        return enhancementService
    }
    
    private func checkCancellationAndCleanup() async -> Bool {
        if shouldCancelRecording {
            await cleanupModelResources()
            return true
        }
        return false
    }

    private func cleanupAndDismiss() async {
        await dismissMiniRecorder()
    }
}
