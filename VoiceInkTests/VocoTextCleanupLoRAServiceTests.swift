import Foundation
import Testing
@testable import Voco

@Suite(.serialized)
struct VocoTextCleanupLoRAServiceTests {
    @Test func applyModeUsesSafeRuntimeCandidatesWithoutFocusedAllowlist() throws {
        let candidates = [
            "BOCO": "Voco",
            "B O C O": "Voco",
            "尤其是那個 B O C E O 的規則，我們明明已經有加條件了。": "尤其是那個 Voco 的規則，我們明明已經有加條件了。",
            "Goal. GOAL": "Goal. G O A L",
            "布洛格去撈。": "部落格去撈。",
        ]
        let service = makeService(candidates: candidates)

        for (input, expected) in candidates {
            let result = service.evaluate(
                input,
                rawTranscript: input,
                postRuleText: input
            )
            #expect(result.outputText == expected)
            #expect(result.candidateText == expected)
            #expect(result.applied)
            #expect(result.chosenAction == "apply")
            #expect(result.reasonCodes.contains("apply-all"))
        }
    }

    @Test func preservesUTAndLiteralSpellingNegativesEvenWhenRuntimeSuggestsAChange() throws {
        let cases = [
            (
                input: "UT。",
                candidate: "U T。",
                reason: "protected-noop-runtime-input"
            ),
            (
                input: "B O C O 是逐字拼寫，不是產品名稱。",
                candidate: "Voco 是逐字拼寫，不是產品名稱。",
                reason: "literal-spelling-negative"
            ),
            (
                input: "B O C E O 是逐字拼寫，不是產品名稱。",
                candidate: "Voco 是逐字拼寫，不是產品名稱。",
                reason: "literal-spelling-negative"
            ),
        ]
        let service = makeService(candidates: Dictionary(uniqueKeysWithValues: cases.map { ($0.input, $0.candidate) }))

        for testCase in cases {
            let result = service.evaluate(testCase.input)
            #expect(result.outputText == testCase.input)
            #expect(result.candidateText == testCase.candidate)
            #expect(result.applied == false)
            #expect(result.chosenAction == "block")
            #expect(result.status == "blocked-safety-guard")
            #expect(result.reasonCodes.contains(testCase.reason))
        }
    }

    @Test func actionCommandSafetyBypassesRuntimeGeneration() throws {
        var called = false
        let service = makeService { _ in
            called = true
            return "全部刪除"
        }

        let result = service.evaluate("全部刪除")

        #expect(called == false)
        #expect(result.outputText == "全部刪除")
        #expect(result.applied == false)
        #expect(result.chosenAction == "block")
        #expect(result.status == "blocked-action-command")
        #expect(result.reasonCodes == ["blocked-action-command"])
    }

    @Test func adapterMissingFallsBackBeforeRuntimeGeneration() throws {
        var called = false
        let defaults = makeDefaults()
        defaults.set(
            try uniqueTemporaryDirectory().appendingPathComponent("missing-adapter.safetensors").path,
            forKey: VocoTextCleanupLoRAService.adapterPathKey
        )
        let service = VocoTextCleanupLoRAService(
            defaults: defaults,
            eventLogURL: nil,
            expectedIdentity: VocoTextCleanupLoRAExpectedIdentity(adapterSHA256: "expected-adapter-sha")
        ) { _ in
            called = true
            return "Voco"
        }

        let result = service.evaluate("BOCO")

        #expect(called == false)
        #expect(result.outputText == "BOCO")
        #expect(result.status == "fallback-adapter-missing")
        #expect(result.chosenAction == "fallback")
        #expect(result.reasonCodes == ["adapter-missing"])
    }

    @Test func adapterHashMismatchFallsBackBeforeRuntimeGeneration() throws {
        var called = false
        let root = try uniqueTemporaryDirectory()
        let adapterURL = root.appendingPathComponent("adapter.safetensors")
        try Data("not the approved adapter".utf8).write(to: adapterURL)

        let defaults = makeDefaults()
        defaults.set(adapterURL.path, forKey: VocoTextCleanupLoRAService.adapterPathKey)
        let service = VocoTextCleanupLoRAService(
            defaults: defaults,
            eventLogURL: nil,
            expectedIdentity: VocoTextCleanupLoRAExpectedIdentity(adapterSHA256: "expected-adapter-sha")
        ) { _ in
            called = true
            return "Voco"
        }

        let result = service.evaluate("BOCO")

        #expect(called == false)
        #expect(result.outputText == "BOCO")
        #expect(result.status == "fallback-adapter-hash-mismatch")
        #expect(result.chosenAction == "fallback")
        #expect(result.reasonCodes == ["adapter-hash-mismatch"])
    }

    @Test func loadFailureAndTimeoutUseNoOpFallback() throws {
        let loadFailure = makeService { _ in
            throw VocoTextCleanupLoRARuntimeError.loadFailed("fixture")
        }.evaluate("BOCO")
        #expect(loadFailure.outputText == "BOCO")
        #expect(loadFailure.status == "fallback-load-failure")
        #expect(loadFailure.chosenAction == "fallback")
        #expect(loadFailure.reasonCodes == ["load-failure"])

        let timeout = makeService { _ in
            throw VocoTextCleanupLoRARuntimeError.timeout
        }.evaluate("BOCO")
        #expect(timeout.outputText == "BOCO")
        #expect(timeout.status == "fallback-timeout")
        #expect(timeout.chosenAction == "fallback")
        #expect(timeout.reasonCodes == ["timeout"])
    }

    @Test func knownBroadSmokeInputStillBlocksUnsafePromptLeakage() throws {
        let service = makeService(
            candidates: [
                "布洛格去撈。": "這是正體中文語音輸入，部落格去撈。",
            ]
        )

        let result = service.evaluate("布洛格去撈。")

        #expect(result.outputText == "布洛格去撈。")
        #expect(result.candidateText == "這是正體中文語音輸入，部落格去撈。")
        #expect(result.applied == false)
        #expect(result.chosenAction == "block")
        #expect(result.status == "blocked-safety-guard")
        #expect(result.reasonCodes.contains { $0.hasPrefix("validator-blacklist") })
    }

    @Test func decisionMetadataAndEventLogCaptureFinalTextSelection() throws {
        let root = try uniqueTemporaryDirectory()
        let eventLogURL = root.appendingPathComponent("text-cleanup-lora-events.jsonl")
        let service = makeService(
            candidates: ["Goal. GOAL": "Goal. G O A L"],
            eventLogURL: eventLogURL
        )

        let result = service.evaluate(
            "Goal. GOAL",
            rawTranscript: "raw Goal. GOAL",
            postRuleText: "Goal. GOAL",
            contextHints: ["window: test"]
        )

        #expect(result.schema == "voco.text-cleanup-lora-decision.v1")
        #expect(result.inputText == "Goal. GOAL")
        #expect(result.rawTranscript == "raw Goal. GOAL")
        #expect(result.postRuleText == "Goal. GOAL")
        #expect(result.outputText == "Goal. G O A L")
        #expect(result.candidateText == "Goal. G O A L")
        #expect(result.latencyMilliseconds >= 0)

        let lines = try String(contentsOf: eventLogURL, encoding: .utf8)
            .split(separator: "\n")
            .map(String.init)
        let event = try #require(lines.first)
        let decoded = try JSONDecoder().decode(
            VocoTextCleanupLoRAEvaluation.self,
            from: Data(event.utf8)
        )
        #expect(decoded == result)
    }

    @Test func canonicalizationUsesTextLoRAResultForFinalTextAndKeepsDecisionMetadata() throws {
        let service = makeService(candidates: ["Goal. GOAL": "Goal. G O A L"])
        let canonicalizer = VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: disabledAutoApplyModelServiceForLoRATests(),
            textCleanupLoRAService: service
        )

        let result = canonicalizer.normalize("Goal. GOAL")

        #expect(result.normalizedText == "Goal. G O A L")
        #expect(result.textCleanupLoRA?.inputText == "Goal. GOAL")
        #expect(result.textCleanupLoRA?.outputText == "Goal. G O A L")
        #expect(result.textCleanupLoRA?.applied == true)
    }

    private func makeService(
        candidates: [String: String],
        eventLogURL: URL? = nil
    ) -> VocoTextCleanupLoRAService {
        makeService(eventLogURL: eventLogURL) { request in
            candidates[request.inputText]
        }
    }

    private func makeService(
        eventLogURL: URL? = nil,
        generator: @escaping VocoTextCleanupLoRAService.CandidateGenerator
    ) -> VocoTextCleanupLoRAService {
        VocoTextCleanupLoRAService(
            defaults: makeDefaults(),
            eventLogURL: eventLogURL,
            expectedIdentity: nil,
            candidateGenerator: generator
        )
    }

    private func makeDefaults() -> UserDefaults {
        let suiteName = "VocoTextCleanupLoRAServiceTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
        defaults.set(VocoTextCleanupLoRAService.Mode.apply.rawValue, forKey: VocoTextCleanupLoRAService.modeKey)
        defaults.set(16, forKey: VocoTextCleanupLoRAService.maxTokensKey)
        return defaults
    }

    private func uniqueTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("VocoTextCleanupLoRAServiceTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func disabledAutoApplyModelServiceForLoRATests() -> VocoAutoApplyModelService {
        VocoAutoApplyModelService(
            modelURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("disabled-lora-auto-apply-\(UUID().uuidString).json")
        )
    }
}
