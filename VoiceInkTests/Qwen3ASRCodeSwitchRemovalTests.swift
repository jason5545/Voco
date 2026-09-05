import Foundation
import MLX
import Testing
@testable import Voco

/// Regression tests for the 2026-09-05 removal of the Qwen3-ASR auto-mode
/// code-switch remap second decode pass. ReplayLab eval493 showed the remap
/// made things strictly worse on the promoted adapter (accepted 4: 0 improved,
/// 4 regressed, CER 0.0289 → 0.0309), so auto mode must decode exactly once.
struct Qwen3ASRCodeSwitchRemovalTests {
    @Test func autoModeRunsExactlyOneDecodePass() throws {
        let spy = try GenerateTextSpyModel(stubText: "我們今天來測試中文輸入法")

        let result = try spy.transcribe(audio: Self.silence, language: nil)

        #expect(spy.generateTextCalls.count == 1)
        #expect(spy.generateTextCalls.first == .some(nil))
        #expect(result.text == "我們今天來測試中文輸入法")
        // NLLanguageRecognizer detection is still reported for downstream routing.
        #expect(result.detectedLanguage == "Chinese")
    }

    @Test func autoModeCodeSwitchOutputDoesNotTriggerSecondPass() throws {
        // Previously this Chinese-dominant output with Latin terms triggered the
        // remap re-decode with language "English". It must now stay single-pass.
        let spy = try GenerateTextSpyModel(stubText: "我今天想用 VSCode 寫一個 plugin")

        let result = try spy.transcribe(audio: Self.silence, language: nil)

        #expect(spy.generateTextCalls.count == 1)
        #expect(spy.generateTextCalls.first == .some(nil))
        #expect(result.text == "我今天想用 VSCode 寫一個 plugin")
        // detectedLanguage 由 NLLanguageRecognizer 決定（混合語可能不是 "Chinese"），
        // 重點是無論偵測結果為何都不得觸發第二趟解碼。
    }

    @Test func manualLanguagePassesThroughUnchanged() throws {
        // Manual language still flows into generateText, which builds the
        // `language <Lang>` + <asr_text> prompt prefix (that builder is unchanged).
        let spy = try GenerateTextSpyModel(stubText: "你好在嗎")

        let result = try spy.transcribe(audio: Self.silence, language: "Chinese")

        #expect(spy.generateTextCalls == ["Chinese"])
        #expect(result.detectedLanguage == nil)
    }

    /// End-to-end smoke with real model weights: both auto and manual modes must
    /// decode exactly once. Gated like the other model smoke tests.
    @Test func realModelDecodesOnceInAutoAndManualModes() async throws {
        guard ProcessInfo.processInfo.environment["VOCO_QWEN3_SINGLE_PASS_SMOKE"] == "1" else {
            return
        }
        let registryModel = try #require(
            TranscriptionModelRegistry.models.compactMap { $0 as? Qwen3Model }
                .first(where: { $0.name == "qwen3-asr-1.7b-8bit" })
        )
        let modelDirectory = Qwen3ModelManager.modelDirectory(for: registryModel.modelId)
        guard FileManager.default.fileExists(atPath: modelDirectory.path) else { return }

        let audioURL = Self.projectRootURL()
            .appendingPathComponent("LocalModels/EdgeTTSSmoke/century-wind-2072-stock-183.wav")
        guard FileManager.default.fileExists(atPath: audioURL.path) else { return }
        let audio = try readWAVSamples(from: audioURL)

        let spy = try GenerateTextSpyModel(
            audioConfig: registryModel.modelSize.audioConfig,
            textConfig: registryModel.modelSize.textConfig,
            callThrough: true
        )
        try spy.load(from: modelDirectory, modelSize: registryModel.modelSize)

        let autoResult = try spy.transcribe(audio: audio, language: nil)
        #expect(spy.generateTextCalls.count == 1)
        #expect(spy.generateTextCalls.first == .some(nil))
        #expect(!autoResult.text.isEmpty)

        spy.reset()
        let manualResult = try spy.transcribe(audio: audio, language: "Chinese")
        #expect(spy.generateTextCalls == ["Chinese"])
        #expect(manualResult.detectedLanguage == nil)
        #expect(!manualResult.text.isEmpty)
    }

    private static let silence = [Float](repeating: 0, count: 1600)

    private static func projectRootURL() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}

private final class GenerateTextSpyModel: Qwen3ASRModel {
    private(set) var generateTextCalls: [String?] = []
    var stubText: String
    let callThrough: Bool

    init(
        stubText: String,
        callThrough: Bool = false
    ) throws {
        self.stubText = stubText
        self.callThrough = callThrough
        try super.init()
        // transcribe() only nil-checks the decoder before generateText; the spy
        // never runs it, and MLX arrays stay unevaluated, so this is cheap.
        self.textDecoder = Qwen3QuantizedTextModel(config: .small)
    }

    init(
        audioConfig: Qwen3AudioEncoderConfig,
        textConfig: Qwen3TextDecoderConfig,
        callThrough: Bool
    ) throws {
        self.stubText = ""
        self.callThrough = callThrough
        try super.init(audioConfig: audioConfig, textConfig: textConfig)
    }

    func reset() {
        generateTextCalls = []
    }

    override func generateText(
        audioEmbeds: MLXArray,
        textDecoder: Qwen3QuantizedTextModel,
        language: String?,
        prompt: String?,
        maxTokens: Int,
        decodingOptions: Qwen3DecodingOptions
    ) throws -> TranscriptionResult {
        generateTextCalls.append(language)
        if callThrough {
            return try super.generateText(
                audioEmbeds: audioEmbeds,
                textDecoder: textDecoder,
                language: language,
                prompt: prompt,
                maxTokens: maxTokens,
                decodingOptions: decodingOptions
            )
        }
        return TranscriptionResult(
            text: stubText,
            avgLogProb: -0.1,
            tokenCount: 8,
            detectedLanguage: nil,
            uncertainWords: [],
            wordConfidences: []
        )
    }
}
