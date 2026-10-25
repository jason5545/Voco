import AppKit
import OSLog
import SwiftData
import Testing
@testable import Voco

@MainActor
private final class FakeEphemeralDetector: VocoEphemeralSignalDetecting {
    var value = false
    var values: [Bool] = []

    func isEphemeral() -> Bool {
        if !values.isEmpty {
            return values.removeFirst()
        }
        return value
    }
}

@MainActor
private final class FakeRecorder: Recorder {
    var startCount = 0
    var stopCount = 0

    override func startRecording(toOutputFile url: URL) async throws {
        startCount += 1
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data().write(to: url)
    }

    override func stopRecording() async {
        stopCount += 1
    }
}

@MainActor
private final class FakeTranscriptionService: TranscriptionService {
    let text: String
    var callCount = 0
    var onTranscribe: (() -> Void)?

    init(text: String) {
        self.text = text
    }

    func transcribe(
        audioURL: URL,
        model: any TranscriptionModel,
        context: TranscriptionRequestContext
    ) async throws -> String {
        callCount += 1
        onTranscribe?()
        return text
    }
}

@MainActor
private final class FakeTranscriptionRegistry: VocoTranscriptionServiceRegistry {
    let fakeService: FakeTranscriptionService

    init(
        modelProvider: any WhisperModelProvider,
        modelsDirectory: URL,
        modelContext: ModelContext,
        text: String
    ) {
        self.fakeService = FakeTranscriptionService(text: text)
        super.init(
            modelProvider: modelProvider,
            modelsDirectory: modelsDirectory,
            modelContext: modelContext
        )
    }

    override func shouldUseRealtimeTranscription(for configuration: TranscriptionRuntimeConfiguration) -> Bool {
        false
    }

    override func createSession(
        for configuration: TranscriptionRuntimeConfiguration,
        onPartialTranscript: ((String) -> Void)? = nil
    ) -> TranscriptionSession {
        FileTranscriptionSession(service: fakeService)
    }

    override func transcribe(
        audioURL: URL,
        model: any TranscriptionModel,
        context: TranscriptionRequestContext = .currentDefaults
    ) async throws -> String {
        try await fakeService.transcribe(audioURL: audioURL, model: model, context: context)
    }

    override func cleanup() async {}
}

private final class SpyAIService: AIService {
    override var connectedProviders: [AIProvider] { [.custom] }
    override func availableModels(for provider: AIProvider) -> [String] { ["test-model"] }
}

@MainActor
private final class SpyEnhancementService: AIEnhancementService {
    var enhanceCount = 0

    override func isConfigured(for configuration: EnhancementRuntimeConfiguration) -> Bool {
        true
    }

    override func enhance(
        _ text: String,
        configuration: EnhancementRuntimeConfiguration,
        contextSnapshot: RecordingContextSnapshot? = nil
    ) async throws -> AIEnhancementResult {
        enhanceCount += 1
        return AIEnhancementResult(
            text: "\(text)X",
            duration: 0.01,
            promptName: "test-prompt",
            systemMessage: "system",
            userMessage: "user"
        )
    }
}

@MainActor
private final class FakeRecorderPanel: RecorderPanelPresenting {
    var isRecorderPanelVisible = true
    var dismissCount = 0

    func dismissRecorderPanel() async {
        dismissCount += 1
        isRecorderPanelVisible = false
    }
}

private struct DeliveryRecord {
    let text: String
    let isEphemeral: Bool
    let sendKey: FinishAndSendKey
}

private final class DeliveryBox {
    var records: [DeliveryRecord] = []
}

private final class NoticeBox {
    var titles: [String] = []
}

private final class ScreenCallBox {
    var count = 0
}

@MainActor
private final class ScopedDefaults {
    private let defaults = UserDefaults.standard
    private var original: [String: Any?] = [:]

    func set(_ value: Any?, forKey key: String) {
        if original[key] == nil {
            original[key] = defaults.object(forKey: key)
        }
        defaults.set(value, forKey: key)
    }

    func restore() {
        for (key, value) in original {
            if let value {
                defaults.set(value, forKey: key)
            } else {
                defaults.removeObject(forKey: key)
            }
        }
    }
}

@MainActor
private final class EphemeralTestFixture {
    let container: ModelContainer
    let modelContext: ModelContext
    let model: any TranscriptionModel
    let detector = FakeEphemeralDetector()
    let recorder = FakeRecorder()
    let registry: FakeTranscriptionRegistry
    let enhancement: SpyEnhancementService
    let privacy: VocoDictationPrivacyController
    let engine: VoiceInkEngine
    let panel = FakeRecorderPanel()
    let deliveryBox = DeliveryBox()
    let noticeBox = NoticeBox()
    let screenCallBox = ScreenCallBox()
    let shadowLogger: PhoneticShadowLogger
    let recordingsDirectory: URL
    let evidenceDirectory: URL

    private let root: URL
    private let defaults = ScopedDefaults()
    private let modeManager = ModeManager.shared
    private var oldConfigurations: [ModeConfig] = []
    private var oldActiveConfiguration: ModeConfig?
    private var oldPostProcessingEnabled = true

    init(enhancementEnabled: Bool = false, useCloudModel: Bool = false) throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("EphemeralDictationTests-\(UUID().uuidString)", isDirectory: true)
        recordingsDirectory = root.appendingPathComponent("Recordings", isDirectory: true)
        evidenceDirectory = root.appendingPathComponent("ShadowLogs", isDirectory: true)
        try FileManager.default.createDirectory(at: recordingsDirectory, withIntermediateDirectories: true)
        oldPostProcessingEnabled = ChinesePostProcessingService.shared.isEnabled
        ChinesePostProcessingService.shared.isEnabled = false

        for (key, value) in [
            "ChinesePostProcessingEnabled": false,
            "VocoPhoneticShadowLoggingEnabled": true,
            "VocoRuntimeCorrectionModelEnabled": false,
            "VocoAutoApplyModelEnabled": false,
            "ContextAwareInsertionEnabled": false,
            "ContextAwareLLMMergeEnabled": false,
            "SkipShortEnhancement": false,
            "AppendTrailingSpace": false,
            "IsTextFormattingEnabled": false,
            "IsAutoLearnDictionaryEnabled": false,
        ] {
            defaults.set(value, forKey: key)
        }

        let schema = Schema([
            Transcription.self,
            SessionMetric.self,
            VocabularyWord.self,
            WordReplacement.self,
        ])
        let configuration = ModelConfiguration(
            "ephemeral-tests-\(UUID().uuidString)",
            schema: schema,
            isStoredInMemoryOnly: true
        )
        container = try ModelContainer(for: schema, configurations: [configuration])
        modelContext = ModelContext(container)

        if useCloudModel {
            model = CustomCloudModel(
                name: "cloud-test",
                displayName: "Cloud Test",
                description: "Cloud test model",
                apiEndpoint: "https://example.invalid/transcribe",
                modelName: "cloud-test"
            )
        } else {
            model = NativeAppleModel(
                name: "apple-speech",
                displayName: "Apple Speech",
                description: "Local test model",
                isMultilingualModel: true,
                supportedLanguages: ["auto": "Auto"]
            )
        }

        let modelsDirectory = root.appendingPathComponent("Models", isDirectory: true)
        let whisperManager = WhisperModelManager(modelsDirectory: modelsDirectory)
        let fluidManager = FluidAudioModelManager()
        let modelManager = TranscriptionModelManager(
            whisperModelManager: whisperManager,
            fluidAudioModelManager: fluidManager
        )
        modelManager.allAvailableModels = [model]
        modelManager.currentTranscriptionModel = model

        registry = FakeTranscriptionRegistry(
            modelProvider: whisperManager,
            modelsDirectory: modelsDirectory,
            modelContext: modelContext,
            text: "raw"
        )
        enhancement = SpyEnhancementService(
            aiService: SpyAIService(),
            modelContext: modelContext
        )
        enhancement.customPrompts = [
            CustomPrompt(title: "test-prompt", promptText: "test")
        ]

        let detector = self.detector
        privacy = VocoDictationPrivacyController(
            detector: detector,
            onPrivacyNotice: { [noticeBox] title in
                noticeBox.titles.append(title)
            }
        )
        shadowLogger = PhoneticShadowLogger(logDirectory: evidenceDirectory)
        let deliveryBox = self.deliveryBox
        let delivery = TranscriptionDelivery(pasteOverride: {
            text,
            transcription,
            _,
            sendKey,
            isEphemeral,
            dismiss in
            deliveryBox.records.append(
                DeliveryRecord(text: text, isEphemeral: isEphemeral, sendKey: sendKey)
            )
            if !isEphemeral {
                transcription.recordPasteAttempt(text: text, didPostCommand: true)
            }
            await dismiss()
        })
        let screenCallBox = self.screenCallBox
        engine = VoiceInkEngine(
            modelContext: modelContext,
            whisperModelManager: whisperManager,
            transcriptionModelManager: modelManager,
            enhancementService: enhancement,
            recorder: recorder,
            recordingsDirectory: recordingsDirectory,
            serviceRegistry: registry,
            delivery: delivery,
            dictationPrivacy: privacy,
            screenTextProvider: {
                screenCallBox.count += 1
                return "screen-context"
            },
            shadowLogger: shadowLogger
        )
        engine.recorderUIManager = panel

        oldConfigurations = modeManager.configurations
        oldActiveConfiguration = modeManager.activeConfiguration
        let promptID = enhancement.customPrompts.first?.id.uuidString
        let mode = ModeConfig(
            name: "EphemeralTests",
            isAIEnhancementEnabled: enhancementEnabled,
            selectedPrompt: enhancementEnabled ? promptID : nil,
            selectedTranscriptionModelName: model.name,
            isRealtimeTranscriptionEnabled: false,
            selectedLanguage: "auto",
            useScreenCapture: true,
            selectedAIProvider: AIProvider.custom.rawValue,
            selectedAIModel: "test-model",
            outputMode: .paste,
            isDefault: true
        )
        modeManager.replaceConfigurations([mode])
        modeManager.setActiveConfiguration(mode)
    }

    func completeDictation() async {
        panel.isRecorderPanelVisible = true
        await engine.toggleRecord()
        await engine.toggleRecord()
    }

    @discardableResult
    func insertLegacySession() throws -> (transcription: Transcription, audioURL: URL) {
        let audioURL = recordingsDirectory.appendingPathComponent("legacy.wav")
        try Data([1, 2, 3]).write(to: audioURL)
        let transcription = Transcription(
            text: "legacy",
            duration: 1,
            audioFileURL: audioURL.absoluteString,
            transcriptionStatus: .completed
        )
        modelContext.insert(transcription)
        try modelContext.save()
        return (transcription, audioURL)
    }

    func insertReplacement() {
        let replacement = WordReplacement(
            originalText: "raw",
            replacementText: "corrected",
            isEnabled: true,
            source: WordReplacement.sourceUser
        )
        modelContext.insert(replacement)
        try? modelContext.save()
    }

    func count<T: PersistentModel>(of type: T.Type) -> Int {
        (try? modelContext.fetchCount(FetchDescriptor<T>())) ?? -1
    }

    func audioFiles() -> [URL] {
        (try? FileManager.default.contentsOfDirectory(
            at: recordingsDirectory,
            includingPropertiesForKeys: nil
        )) ?? []
    }

    func evidenceFiles() -> [URL] {
        shadowLogger.flush()
        return (try? FileManager.default.contentsOfDirectory(
            at: evidenceDirectory,
            includingPropertiesForKeys: nil
        )) ?? []
    }

    func tearDown() {
        shadowLogger.flush()
        modeManager.replaceConfigurations(oldConfigurations)
        modeManager.setActiveConfiguration(oldActiveConfiguration)
        ChinesePostProcessingService.shared.isEnabled = oldPostProcessingEnabled
        defaults.restore()
        try? FileManager.default.removeItem(at: root)
    }
}

@MainActor
struct EphemeralDictationTests {
    @Test func signalMatcherRequiresT3BundleAndMarker() {
        let bundleIDs: Set<String> = ["com.jason5545.t3code.nightly"]
        let marked = VocoEphemeralAXContent(windowText: ["無痕對話 — test"])
        #expect(VocoEphemeralSignalMatcher.isEphemeral(
            bundleID: "com.jason5545.t3code.nightly",
            installedT3BundleIDs: bundleIDs,
            content: marked
        ))
        #expect(!VocoEphemeralSignalMatcher.isEphemeral(
            bundleID: "com.example.other",
            installedT3BundleIDs: bundleIDs,
            content: marked
        ))
        #expect(!VocoEphemeralSignalMatcher.isEphemeral(
            bundleID: "com.jason5545.t3code.nightly",
            installedT3BundleIDs: bundleIDs,
            content: VocoEphemeralAXContent(windowText: ["T3 Code (Nightly)"])
        ))
        #expect(!VocoEphemeralSignalMatcher.isEphemeral(
            bundleID: nil,
            installedT3BundleIDs: bundleIDs,
            content: marked
        ))
    }

    @Test func detectorAcceptsWindowAndWebAreaTitleOrDescription() {
        let bundleID = "com.jason5545.t3code.nightly"
        let cases = [
            VocoEphemeralAXContent(windowText: ["prefix 無痕對話 suffix"]),
            VocoEphemeralAXContent(windowText: [], webAreaText: ["無痕對話 — thread"]),
            VocoEphemeralAXContent(windowText: [], webAreaText: ["title", "desc 無痕對話"]),
        ]
        for content in cases {
            let detector = VocoEphemeralSignalDetector(
                frontmostApplication: { (1234, bundleID) },
                installedT3BundleIDs: { [bundleID] },
                readAX: { _ in content }
            )
            #expect(detector.isEphemeral())
        }
    }

    @Test func sessionLatchesEphemeralUntilEndAndPresentsCloudNoticeOnce() {
        let detector = FakeEphemeralDetector()
        let notices = NoticeBox()
        let privacy = VocoDictationPrivacyController(
            detector: detector,
            onPrivacyNotice: { notices.titles.append($0) }
        )
        privacy.beginSessionIfNeeded()
        #expect(!privacy.isEphemeral)

        detector.value = true
        privacy.refreshSignal()
        #expect(privacy.isEphemeral)
        detector.value = false
        privacy.refreshSignal()
        #expect(privacy.isEphemeral)

        privacy.markCloudASR()
        privacy.markCloudASR()
        #expect(privacy.didUseCloudASR)
        #expect(privacy.privacyNotice != nil)
        #expect(notices.titles.count == 1)

        privacy.endSession()
        #expect(!privacy.isEphemeral)
        privacy.beginSessionIfNeeded()
        #expect(!privacy.isEphemeral)
        privacy.endSession()
    }

    @Test func ephemeralDictationLeavesNoPersistentSideEffects() async throws {
        let fixture = try EphemeralTestFixture(enhancementEnabled: true)
        defer { fixture.tearDown() }
        let legacy = try fixture.insertLegacySession()
        fixture.insertReplacement()
        fixture.detector.value = true

        await fixture.completeDictation()

        #expect(fixture.deliveryBox.records.count == 1)
        #expect(fixture.deliveryBox.records.first?.text == "corrected")
        #expect(fixture.deliveryBox.records.first?.isEphemeral == true)
        #expect(fixture.deliveryBox.records.first?.sendKey == FinishAndSendKey.none)
        #expect(fixture.count(of: Transcription.self) == 1)
        #expect(fixture.count(of: SessionMetric.self) == 0)
        #expect(fixture.count(of: VocabularyWord.self) == 0)
        #expect(fixture.count(of: WordReplacement.self) == 1)
        #expect(fixture.audioFiles().map(\.lastPathComponent) == ["legacy.wav"])
        #expect(fixture.evidenceFiles().isEmpty)
        #expect(fixture.enhancement.enhanceCount == 0)
        #expect(fixture.screenCallBox.count == 0)
        #expect(fixture.registry.fakeService.callCount == 1)
        #expect(legacy.transcription.text == "legacy")
        #expect(FileManager.default.fileExists(atPath: legacy.audioURL.path))
        #expect(!fixture.privacy.hasActiveSession)
        #expect(!fixture.privacy.isEphemeral)
    }

    @Test func normalDictationKeepsPersistentBehavior() async throws {
        let fixture = try EphemeralTestFixture(enhancementEnabled: true)
        defer { fixture.tearDown() }
        fixture.insertReplacement()

        await fixture.completeDictation()

        #expect(fixture.count(of: Transcription.self) == 1)
        #expect(fixture.count(of: SessionMetric.self) == 1)
        #expect(fixture.audioFiles().count == 1)
        #expect(fixture.evidenceFiles().count == 1)
        #expect(fixture.enhancement.enhanceCount == 1)
        #expect(fixture.screenCallBox.count == 1)
        #expect(fixture.deliveryBox.records.first?.isEphemeral == false)
        #expect(fixture.deliveryBox.records.first?.text == "correctedX")
        let row = try #require(fixture.modelContext.fetch(FetchDescriptor<Transcription>()).first)
        #expect(row.transcriptionStatus == TranscriptionStatus.completed.rawValue)
        #expect(row.enhancedText == "correctedX")
        #expect(row.finalPastedText == "correctedX")
    }

    @Test func previousNormalSessionIsUntouchedWhenNextTurnsEphemeral() async throws {
        let fixture = try EphemeralTestFixture()
        defer { fixture.tearDown() }
        fixture.detector.value = false
        await fixture.completeDictation()

        let firstRow = try #require(fixture.modelContext.fetch(FetchDescriptor<Transcription>()).first)
        let firstAudio = try #require(firstRow.audioFileURL.flatMap(URL.init(string:)))
        #expect(fixture.count(of: Transcription.self) == 1)
        #expect(fixture.audioFiles().count == 1)
        #expect(fixture.evidenceFiles().count == 1)

        fixture.detector.value = true
        await fixture.completeDictation()

        #expect(fixture.count(of: Transcription.self) == 1)
        #expect(firstRow.text == "raw")
        #expect(FileManager.default.fileExists(atPath: firstAudio.path))
        #expect(fixture.audioFiles().count == 1)
        #expect(fixture.evidenceFiles().count == 1)
        #expect(fixture.recorder.startCount == 2)
        #expect(fixture.deliveryBox.records.map(\.isEphemeral) == [false, true])
    }

    @Test func signalAtPasteDeletesOnlyCurrentPendingRowAndAudio() async throws {
        let fixture = try EphemeralTestFixture(enhancementEnabled: true)
        defer { fixture.tearDown() }
        _ = try fixture.insertLegacySession()
        fixture.detector.values = [false, false, false, false, false, false, true]

        await fixture.completeDictation()

        #expect(fixture.count(of: Transcription.self) == 1)
        #expect(fixture.audioFiles().map(\.lastPathComponent) == ["legacy.wav"])
        #expect(fixture.evidenceFiles().isEmpty)
        #expect(fixture.deliveryBox.records.first?.isEphemeral == true)
        #expect(fixture.enhancement.enhanceCount == 1)
        #expect(fixture.screenCallBox.count == 1)
    }

    @Test func cloudASRIsRejectedBeforeRecordingStarts() async throws {
        let fixture = try EphemeralTestFixture(useCloudModel: true)
        defer { fixture.tearDown() }
        fixture.detector.value = true

        await fixture.engine.toggleRecord()

        #expect(fixture.recorder.startCount == 0)
        #expect(fixture.count(of: Transcription.self) == 0)
        #expect(fixture.deliveryBox.records.isEmpty)
        #expect(!fixture.privacy.hasActiveSession)
        #expect(fixture.registry.fakeService.callCount == 0)
    }

    @Test func lateCloudASRStillPastesAndPresentsNotice() async throws {
        let fixture = try EphemeralTestFixture(useCloudModel: true)
        defer { fixture.tearDown() }
        fixture.detector.values = [false, false, false, false, false, false, true]

        await fixture.completeDictation()

        #expect(fixture.recorder.startCount == 1)
        #expect(fixture.deliveryBox.records.first?.isEphemeral == true)
        #expect(fixture.count(of: Transcription.self) == 0)
        #expect(fixture.noticeBox.titles.count == 1)
    }

    @Test func cancelEphemeralRecordingLeavesNoCanceledRowOrAudio() async throws {
        let fixture = try EphemeralTestFixture()
        defer { fixture.tearDown() }
        fixture.detector.value = true

        await fixture.engine.toggleRecord()
        await fixture.engine.cancelRecording()

        #expect(fixture.count(of: Transcription.self) == 0)
        #expect(fixture.audioFiles().isEmpty)
        #expect(fixture.evidenceFiles().isEmpty)
        #expect(!fixture.privacy.hasActiveSession)
    }

    @Test func clipboardMarkersAreOnlyAddedForEphemeralWrites() {
        let pasteboard = NSPasteboard.general
        let oldItems = (pasteboard.pasteboardItems ?? []).map { item in
            item.types.compactMap { type in
                item.data(forType: type).map { (type, $0) }
            }
        }
        defer {
            pasteboard.clearContents()
            let restored = oldItems.map { item -> NSPasteboardItem in
                let restoredItem = NSPasteboardItem()
                for (type, data) in item {
                    restoredItem.setData(data, forType: type)
                }
                return restoredItem
            }
            if !restored.isEmpty {
                pasteboard.writeObjects(restored)
            }
        }

        #expect(ClipboardManager.setClipboard("secret", transient: false, concealed: true))
        #expect(pasteboard.data(forType: NSPasteboard.PasteboardType("org.nspasteboard.ConcealedType")) != nil)
        #expect(pasteboard.data(forType: NSPasteboard.PasteboardType("org.nspasteboard.TransientType")) != nil)

        #expect(ClipboardManager.setClipboard("ordinary", transient: false, concealed: false))
        #expect(pasteboard.data(forType: NSPasteboard.PasteboardType("org.nspasteboard.ConcealedType")) == nil)
        #expect(pasteboard.data(forType: NSPasteboard.PasteboardType("org.nspasteboard.TransientType")) == nil)
    }

    @Test func evidenceBufferWritesOnlyAfterCommit() {
        let buffer = VocoEvidenceBuffer()
        var writes = 0
        buffer.append { writes += 1 }
        #expect(writes == 0)
        buffer.commit()
        #expect(writes == 1)
        buffer.append { writes += 1 }
        buffer.discard()
        #expect(writes == 1)
    }

    @Test func outputFilterUnifiedLogContainsMetadataNotTranscript() async throws {
        let marker = "sensitive-marker-\(UUID().uuidString)"
        let since = Date().addingTimeInterval(-2)
        _ = TranscriptionOutputFilter.filter(marker)
        try await Task.sleep(nanoseconds: 100_000_000)

        let store = try OSLogStore(scope: .system)
        let entries = try store.getEntries(
            at: store.position(date: since),
            matching: NSPredicate(format: "subsystem == %@", AppIdentifiers.subsystem)
        )
        let messages = entries.compactMap { entry -> String? in
            guard let log = entry as? OSLogEntryLog,
                  log.category == "TranscriptionOutputFilter" else {
                return nil
            }
            return log.composedMessage
        }
        let latest = try #require(messages.last { $0.contains("outputChars") })
        #expect(latest.contains("outputChars"))
        #expect(!latest.contains(marker))
    }
}
