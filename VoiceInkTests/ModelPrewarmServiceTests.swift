import Foundation
import Testing
@testable import Voco

@MainActor
struct ModelPrewarmServiceTests {
    @Test func reschedulingWakePrewarmCoalescesToOneRun() async throws {
        let fixture = try PrewarmTestFixture()
        let service = fixture.makeService(prewarmDelay: .milliseconds(80))

        service.schedulePrewarm(trigger: "first")
        service.schedulePrewarm(trigger: "second")
        try await fixture.transcriber.waitForTranscribeCallCount(1)
        try await Task.sleep(for: .milliseconds(120))

        #expect(fixture.transcriber.transcribeCallCount == 1)
    }

    @Test func wakePrewarmStaysSingleFlightWhileActive() async throws {
        let fixture = try PrewarmTestFixture(transcribeDuration: .milliseconds(150))
        let service = fixture.makeService(prewarmDelay: .zero)

        service.schedulePrewarm(trigger: "first")
        try await Task.sleep(for: .milliseconds(20))
        service.schedulePrewarm(trigger: "second")
        try await Task.sleep(for: .milliseconds(80))

        #expect(fixture.transcriber.transcribeCallCount == 1)
    }
}

@MainActor
private final class PrewarmTestFixture {
    let transcriber: FakePrewarmTranscriber
    let transcriptionModelManager: TranscriptionModelManager
    let audioURL: URL
    private let userDefaults: UserDefaults
    private let userDefaultsSuiteName: String

    init(transcribeDuration: Duration? = nil) throws {
        self.transcriber = FakePrewarmTranscriber(transcribeDuration: transcribeDuration)
        self.userDefaultsSuiteName = "ModelPrewarmServiceTests.\(UUID().uuidString)"
        self.userDefaults = try #require(UserDefaults(suiteName: userDefaultsSuiteName))

        let modelsDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)

        let whisperModelManager = WhisperModelManager(modelsDirectory: modelsDirectory)
        let fluidAudioModelManager = FluidAudioModelManager()
        let transcriptionModelManager = TranscriptionModelManager(
            whisperModelManager: whisperModelManager,
            fluidAudioModelManager: fluidAudioModelManager
        )
        transcriptionModelManager.currentTranscriptionModel = try #require(
            TranscriptionModelRegistry.models.first(where: { $0.provider == .whisperMLX })
        )
        self.transcriptionModelManager = transcriptionModelManager

        let audioURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        try Data().write(to: audioURL)
        self.audioURL = audioURL

        userDefaults.set(true, forKey: "PrewarmModelOnWake")
    }

    deinit {
        userDefaults.removePersistentDomain(forName: userDefaultsSuiteName)
    }

    func makeService(prewarmDelay: Duration) -> ModelPrewarmService {
        ModelPrewarmService(
            transcriptionModelManager: transcriptionModelManager,
            serviceRegistry: transcriber,
            prewarmAudioURL: audioURL,
            prewarmDelay: prewarmDelay,
            userDefaults: userDefaults,
            observeWorkspaceNotifications: false,
            scheduleInitialPrewarm: false
        )
    }
}

@MainActor
private final class FakePrewarmTranscriber: ModelPrewarmTranscribing {
    private(set) var transcribeCallCount = 0
    private let transcribeDuration: Duration?

    init(transcribeDuration: Duration?) {
        self.transcribeDuration = transcribeDuration
    }

    func transcribe(
        audioURL: URL,
        model: any TranscriptionModel,
        context: TranscriptionRequestContext
    ) async throws -> String {
        transcribeCallCount += 1

        if let transcribeDuration {
            try await Task.sleep(for: transcribeDuration)
        }

        return ""
    }

    func waitForTranscribeCallCount(_ expectedCount: Int, timeout: Duration = .seconds(2)) async throws {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)

        while transcribeCallCount < expectedCount && clock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
    }
}
