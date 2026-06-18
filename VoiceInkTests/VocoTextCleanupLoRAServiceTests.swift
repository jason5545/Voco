import Foundation
import Testing
@testable import Voco

@Suite(.serialized)
struct VocoTextCleanupLoRAServiceTests {
    @Test func appDefaultsKeepApplyModeButPointAtV5FocusedAdapter() throws {
        #expect(AppDefaults.defaultValues[VocoTextCleanupLoRAService.modeKey] as? String == VocoTextCleanupLoRAService.Mode.apply.rawValue)
        #expect(AppDefaults.defaultValues[VocoTextCleanupLoRAService.adapterPathKey] as? String == VocoTextCleanupLoRAService.defaultAdapterPath)
        #expect(AppDefaults.defaultValues[VocoTextCleanupLoRAService.auditReportPathKey] as? String == VocoTextCleanupLoRAService.defaultAuditReportPath)
        #expect(VocoTextCleanupLoRAService.defaultAdapterPath.contains("v5-focused-promotion"))
        #expect(VocoTextCleanupLoRAService.expectedAdapterSHA256 == "f253bacd64ab58cbbed73e5b439a49d9e757bc1698be84b402438116cc735932")
    }

    @Test func v5FocusedAdapterWeightsAndAuditIdentityLoadWhenPresent() throws {
        let adapterURL = URL(fileURLWithPath: VocoTextCleanupLoRAService.defaultAdapterPath)
        let auditURL = URL(fileURLWithPath: VocoTextCleanupLoRAService.defaultAuditReportPath)
        guard FileManager.default.fileExists(atPath: adapterURL.path),
              FileManager.default.fileExists(atPath: auditURL.path)
        else { return }

        let store = try Qwen3TextDecoderLoRAStore.load(from: adapterURL, config: .large)
        #expect(store.adapterURL.path == adapterURL.path)
        #expect(store.appliedProjectionCount > 0)

        let defaults = makeDefaults()
        defaults.set(adapterURL.path, forKey: VocoTextCleanupLoRAService.adapterPathKey)
        defaults.set(auditURL.path, forKey: VocoTextCleanupLoRAService.auditReportPathKey)
        let service = VocoTextCleanupLoRAService(
            defaults: defaults,
            eventLogURL: nil,
            expectedIdentity: VocoTextCleanupLoRAExpectedIdentity(
                adapterSHA256: VocoTextCleanupLoRAService.expectedAdapterSHA256,
                auditReportSHA256: VocoTextCleanupLoRAService.expectedAuditReportSHA256
            )
        ) { request in
            #expect(request.identity?.adapterSHA256 == VocoTextCleanupLoRAService.expectedAdapterSHA256)
            #expect(request.identity?.adapterVersion == "v5-focused-promotion")
            #expect(request.identity?.promotedApplyClass == "focused-runtime-lora")
            #expect(request.identity?.applyAllAllowed == false)
            #expect(request.identity?.releaseOrDeployAllowed == false)
            return "Goal. G O A L"
        }

        let result = service.evaluate("Goal. GOAL")
        #expect(result.outputText == "Goal. G O A L")
        #expect(result.reasonCodes.contains("focused-runtime-lora"))
        #expect(result.reasonCodes.contains("apply-all-not-approved"))
    }

    @Test func applyModeAppliesOnlyFocusedRuntimeFamilies() throws {
        let cases = [
            (
                input: "BOCO 的規則要接回 Voco。",
                candidate: "Voco 的規則要接回 Voco。",
                reason: "focused-voco-brand-context"
            ),
            (
                input: "尤其是那個 B O C O 的規則，我們明明已經有加條件了。",
                candidate: "尤其是那個 Voco 的規則，我們明明已經有加條件了。",
                reason: "focused-voco-brand-context"
            ),
            (
                input: "尤其是那個 B O C E O 的規則，我們明明已經有加條件了。",
                candidate: "尤其是那個 Voco 的規則，我們明明已經有加條件了。",
                reason: "focused-voco-brand-context"
            ),
            (
                input: "Goal. GOAL",
                candidate: "Goal. G O A L",
                reason: "focused-goal-acronym"
            ),
            (
                input: "Lisa 的座右銘是今日もいい日だ。",
                candidate: "LiSA 的座右銘是今日もいい日だ。",
                reason: "focused-lisa-japanese-quote"
            ),
        ]
        let service = makeService(candidates: Dictionary(uniqueKeysWithValues: cases.map { ($0.input, $0.candidate) }))

        for testCase in cases {
            let result = service.evaluate(
                testCase.input,
                rawTranscript: testCase.input,
                postRuleText: testCase.input
            )
            #expect(result.outputText == testCase.candidate)
            #expect(result.candidateText == testCase.candidate)
            #expect(result.applied)
            #expect(result.chosenAction == "apply")
            #expect(result.reasonCodes.contains("focused-runtime-lora"))
            #expect(result.reasonCodes.contains("apply-all-not-approved"))
            #expect(result.reasonCodes.contains(testCase.reason))
            #expect(!result.reasonCodes.contains("apply-all"))
        }
    }

    @Test func broadApplyAllCandidateIsBlockedEvenWhenCandidateLooksSafe() throws {
        let service = makeService(candidates: ["布洛格去撈。": "部落格去撈。"])

        let result = service.evaluate("布洛格去撈。")

        #expect(result.outputText == "布洛格去撈。")
        #expect(result.candidateText == "部落格去撈。")
        #expect(result.applied == false)
        #expect(result.chosenAction == "block")
        #expect(result.status == "blocked-broad-apply-all")
        #expect(result.reasonCodes.contains("broad-apply-all-blocked"))
        #expect(result.reasonCodes.contains("focused-scope-required"))
    }

    @Test func preservesUTAndLiteralSpellingNegatives() throws {
        let service = makeService(
            candidates: [
                "UT。": "UT。",
                "B O C O 是逐字拼寫，不是產品名稱。": "Voco 是逐字拼寫，不是產品名稱。",
                "B O C E O 是逐字拼寫，不是產品名稱。": "Voco 是逐字拼寫，不是產品名稱。",
            ]
        )

        let ut = service.evaluate("UT。")
        #expect(ut.outputText == "UT。")
        #expect(ut.applied == false)
        #expect(ut.reasonCodes.contains("focused-ut-preserve"))

        for input in [
            "B O C O 是逐字拼寫，不是產品名稱。",
            "B O C E O 是逐字拼寫，不是產品名稱。",
        ] {
            let result = service.evaluate(input)
            #expect(result.outputText == input)
            #expect(result.applied == false)
            #expect(result.chosenAction == "block")
            #expect(result.status == "blocked-safety-guard")
            #expect(result.reasonCodes.contains("literal-spelling-negative"))
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
        #expect(result.chosenAction == "action-command-bypass")
        #expect(result.status == "action-command-bypass")
        #expect(result.reasonCodes == ["action-command-bypass"])
    }

    @Test func exportsReplayLabRuntimeShadowSmokeWhenRequested() throws {
        let outputPath = ProcessInfo.processInfo.environment["VOCO_TEXT_LORA_SHADOW_SMOKE_JSONL"]
            .flatMap { $0.isEmpty ? nil : $0 }
            ?? "/private/tmp/voco-text-lora-runtime-shadow-latest.jsonl"
        let outputURL = URL(fileURLWithPath: outputPath)
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        if fileManager.fileExists(atPath: outputURL.path) {
            try fileManager.removeItem(at: outputURL)
        }

        let focusedService = VocoTextCleanupLoRAService(
            defaults: makeDefaults(),
            eventLogURL: outputURL
        ) { request in
            switch request.inputText {
            case "Goal. GOAL":
                return "Goal. G O A L"
            case "布洛格去撈。":
                return "部落格去撈。"
            default:
                return nil
            }
        }
        _ = focusedService.evaluate("Goal. GOAL")
        _ = focusedService.evaluate("布洛格去撈。")
        _ = focusedService.evaluate("全部刪除")

        let v6AdapterURL = URL(
            fileURLWithPath: "/Users/jianruicheng/GitHub/VocoReplayLab/local-adapters/qwen3-asr-cleanup-lora-20260615-v6-apply-all-microstep/adapters.safetensors"
        )
        let v6AuditURL = URL(
            fileURLWithPath: "/Users/jianruicheng/GitHub/VocoReplayLab/artifacts/lora-qwen3-asr-cleanup-20260615-v6-apply-all-microstep/apply-all-audit/report.json"
        )
        if fileManager.fileExists(atPath: v6AdapterURL.path),
           fileManager.fileExists(atPath: v6AuditURL.path) {
            let v6Defaults = makeDefaults()
            v6Defaults.set(v6AdapterURL.path, forKey: VocoTextCleanupLoRAService.adapterPathKey)
            v6Defaults.set(v6AuditURL.path, forKey: VocoTextCleanupLoRAService.auditReportPathKey)
            let v6Service = VocoTextCleanupLoRAService(
                defaults: v6Defaults,
                eventLogURL: outputURL
            ) { request in
                #expect(request.identity?.adapterSHA256 == VocoTextCleanupLoRAService.v6ComparisonAdapterSHA256)
                return "LiSA 的座右銘是今日もいい日だ。"
            }
            _ = v6Service.evaluate("Lisa 的座右銘是今日もいい日だ。")
        }

        let lines = try String(contentsOf: outputURL, encoding: .utf8)
            .split(separator: "\n")
            .map(String.init)
        #expect(lines.count >= 4)
        let events = try lines.map { line in
            try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any]
        }
        #expect(events.allSatisfy { $0?["eventId"] as? String != nil })
        #expect(events.allSatisfy { $0?["adapterHash"] as? String != nil })
        #expect(events.allSatisfy { $0?["applyAllAllowed"] as? Bool == false })
        #expect(events.allSatisfy { $0?["releaseOrDeployAllowed"] as? Bool == false })
    }

    @Test func shadowModeRecordsCandidateWithoutChangingOutput() throws {
        let defaults = makeDefaults(mode: .shadow)
        let service = makeService(
            defaults: defaults,
            candidates: ["Goal. GOAL": "Goal. G O A L"]
        )

        let result = service.evaluate("Goal. GOAL")

        #expect(result.outputText == "Goal. GOAL")
        #expect(result.candidateText == "Goal. G O A L")
        #expect(result.applied == false)
        #expect(result.chosenAction == "shadow")
        #expect(result.status == "shadow-candidate")
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

        let result = service.evaluate("BOCO 的規則要接回 Voco。")

        #expect(called == false)
        #expect(result.outputText == "BOCO 的規則要接回 Voco。")
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

        let result = service.evaluate("BOCO 的規則要接回 Voco。")

        #expect(called == false)
        #expect(result.outputText == "BOCO 的規則要接回 Voco。")
        #expect(result.status == "fallback-adapter-hash-mismatch")
        #expect(result.chosenAction == "fallback")
        #expect(result.reasonCodes == ["adapter-hash-mismatch"])
    }

    @Test func decisionMetadataAndEventLogCaptureFocusedBoundary() throws {
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
        #expect(result.rawTranscript == "raw Goal. GOAL")
        #expect(result.postRuleText == "Goal. GOAL")
        #expect(result.outputText == "Goal. G O A L")
        #expect(result.reasonCodes.contains("apply-all-not-approved"))

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

    @Test func canonicalizationUsesFocusedTextLoRAResultAndKeepsDecisionMetadata() throws {
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
        #expect(result.textCleanupLoRA?.reasonCodes.contains("focused-runtime-lora") == true)
    }

    @Test func canonicalizationKeepsBroadApplyAllBlocked() throws {
        let service = makeService(candidates: ["布洛格去撈。": "部落格去撈。"])
        let canonicalizer = VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: disabledAutoApplyModelServiceForLoRATests(),
            textCleanupLoRAService: service
        )

        let result = canonicalizer.normalize("布洛格去撈。")

        #expect(result.normalizedText == "布洛格去撈。")
        #expect(result.textCleanupLoRA?.status == "blocked-broad-apply-all")
        #expect(result.textCleanupLoRA?.reasonCodes.contains("broad-apply-all-blocked") == true)
    }

    private func makeService(
        defaults: UserDefaults? = nil,
        candidates: [String: String],
        eventLogURL: URL? = nil
    ) -> VocoTextCleanupLoRAService {
        makeService(defaults: defaults, eventLogURL: eventLogURL) { request in
            candidates[request.inputText]
        }
    }

    private func makeService(
        defaults: UserDefaults? = nil,
        eventLogURL: URL? = nil,
        generator: @escaping VocoTextCleanupLoRAService.CandidateGenerator
    ) -> VocoTextCleanupLoRAService {
        VocoTextCleanupLoRAService(
            defaults: defaults ?? makeDefaults(),
            eventLogURL: eventLogURL,
            expectedIdentity: nil,
            candidateGenerator: generator
        )
    }

    private func makeDefaults(mode: VocoTextCleanupLoRAService.Mode = .apply) -> UserDefaults {
        let suiteName = "VocoTextCleanupLoRAServiceTests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
        defaults.set(mode.rawValue, forKey: VocoTextCleanupLoRAService.modeKey)
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
