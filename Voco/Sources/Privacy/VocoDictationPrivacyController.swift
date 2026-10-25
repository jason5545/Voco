import Combine
import Foundation
import SwiftData

@MainActor
final class VocoDictationPrivacyController: ObservableObject {
    @Published private(set) var isEphemeral = false
    @Published private(set) var privacyNotice: String?

    private let detector: any VocoEphemeralSignalDetecting
    private let onPrivacyNotice: (String) -> Void
    var onStateChange: (() -> Void)?
    private var sessionIsActive = false
    private var sessionID: UUID?
    private var pendingTranscription: Transcription?
    private var pendingTranscriptionWasInserted = false
    private var usedCloudASR = false
    private var didPresentPrivacyNotice = false
    private var postProcessorSnapshot: PostProcessorSnapshot?

    private struct PostProcessorSnapshot {
        let avgLogProb: Double
        let modelProvider: ModelProvider?
        let audioDuration: Double
        let uncertainWords: [UncertainWord]
        let wordConfidences: [WordConfidence]
    }

    init(
        detector: (any VocoEphemeralSignalDetecting)? = nil,
        cleanStaleRecordings: Bool = false,
        onPrivacyNotice: ((String) -> Void)? = nil
    ) {
        self.detector = detector ?? VocoEphemeralSignalDetector()
        self.onPrivacyNotice = onPrivacyNotice ?? { title in
            NotificationManager.shared.showNotification(title: title, type: .warning, duration: 7.0)
        }
        if cleanStaleRecordings {
            try? FileManager.default.removeItem(at: Self.ephemeralRecordingsRoot)
        }
    }

    var hasActiveSession: Bool { sessionIsActive }
    var didUseCloudASR: Bool { usedCloudASR }
    var shouldSkipRefineCleanup = false

    static var cloudASRRejectedMessage: String {
        String(localized: "Ephemeral dictation requires a local transcription model. Cloud ASR is disabled for this recording.")
    }

    static var cloudASRAlreadySentMessage: String {
        String(localized: "Cloud ASR was already used for this dictation.")
    }

    static var ephemeralRecordingsRoot: URL {
        FileManager.default.temporaryDirectory.appendingPathComponent(
            "VocoEphemeralRecordings",
            isDirectory: true
        )
    }

    func beginSessionIfNeeded() {
        if sessionIsActive {
            refreshSignal()
            return
        }

        sessionIsActive = true
        sessionID = UUID()
        shouldSkipRefineCleanup = false
        isEphemeral = false
        privacyNotice = nil
        pendingTranscription = nil
        pendingTranscriptionWasInserted = false
        usedCloudASR = false
        didPresentPrivacyNotice = false

        let postProcessor = ChinesePostProcessingService.shared
        postProcessorSnapshot = PostProcessorSnapshot(
            avgLogProb: postProcessor.lastAvgLogProb,
            modelProvider: postProcessor.lastModelProvider,
            audioDuration: postProcessor.lastAudioDuration,
            uncertainWords: postProcessor.lastUncertainWords,
            wordConfidences: postProcessor.lastWordConfidences
        )
        refreshSignal()
    }

    @discardableResult
    func refreshSignal() -> Bool {
        if detector.isEphemeral() {
            isEphemeral = true
        }
        presentPrivacyNoticeIfNeeded()
        onStateChange?()
        return isEphemeral
    }

    func markCloudASR() {
        usedCloudASR = true
        presentPrivacyNoticeIfNeeded()
        onStateChange?()
    }

    func makeRecordingURL(default defaultURL: URL) -> URL {
        guard isEphemeral, let sessionID else { return defaultURL }
        let directory = Self.ephemeralRecordingsRoot.appendingPathComponent(
            sessionID.uuidString,
            isDirectory: true
        )
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent("recording.wav")
    }

    func trackPendingTranscription(_ transcription: Transcription, inserted: Bool) {
        pendingTranscription = transcription
        pendingTranscriptionWasInserted = inserted
    }

    func discardPendingTranscriptionIfNeeded(in modelContext: ModelContext) {
        defer {
            pendingTranscription = nil
            pendingTranscriptionWasInserted = false
        }
        guard isEphemeral, pendingTranscriptionWasInserted, let pendingTranscription else { return }

        modelContext.delete(pendingTranscription)
        do {
            try modelContext.save()
            NotificationCenter.default.post(name: .transcriptionDeleted, object: nil)
        } catch {
            return
        }
    }

    func removeRecordingIfNeeded(at audioURL: URL) {
        guard isEphemeral else { return }
        try? FileManager.default.removeItem(at: audioURL)

        let parent = audioURL.deletingLastPathComponent()
        if parent.deletingLastPathComponent().path == Self.ephemeralRecordingsRoot.path {
            try? FileManager.default.removeItem(at: parent)
        }
    }

    func endSession() {
        if isEphemeral, let postProcessorSnapshot {
            let postProcessor = ChinesePostProcessingService.shared
            postProcessor.lastAvgLogProb = postProcessorSnapshot.avgLogProb
            postProcessor.lastModelProvider = postProcessorSnapshot.modelProvider
            postProcessor.lastAudioDuration = postProcessorSnapshot.audioDuration
            postProcessor.lastUncertainWords = postProcessorSnapshot.uncertainWords
            postProcessor.lastWordConfidences = postProcessorSnapshot.wordConfidences
        }

        shouldSkipRefineCleanup = isEphemeral
        sessionIsActive = false
        sessionID = nil
        isEphemeral = false
        privacyNotice = nil
        pendingTranscription = nil
        pendingTranscriptionWasInserted = false
        usedCloudASR = false
        didPresentPrivacyNotice = false
        postProcessorSnapshot = nil
        onStateChange?()
    }

    private func presentPrivacyNoticeIfNeeded() {
        guard isEphemeral, usedCloudASR, !didPresentPrivacyNotice else { return }
        didPresentPrivacyNotice = true
        let message = Self.cloudASRAlreadySentMessage
        privacyNotice = message
        onPrivacyNotice(message)
    }
}

enum VocoDictationASRPolicy {
    static func usesCloudASR(_ model: any TranscriptionModel) -> Bool {
        switch model.provider {
        case .whisper, .fluidAudio, .transcribeCpp, .nativeApple,
             .qwen3, .qwen3CoreML, .whisperMLX, .whisperCoreML:
            return false
        default:
            return true
        }
    }
}
