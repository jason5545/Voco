import Foundation
import CryptoKit
import Testing
@testable import Voco

struct VocoRuntimeCorrectionModelServiceTests {
    @Test func shadowContractLogsDecisionWithoutChangingFinalText() throws {
        let root = try runtimeTemporaryDirectory()
        let artifact = try writeRuntimeShadowArtifact(in: root)
        let eventLog = root.appendingPathComponent("shadow-events.jsonl")
        let defaults = try runtimeTemporaryDefaults()
        defaults.set(true, forKey: VocoRuntimeCorrectionModelService.enabledKey)
        let service = VocoRuntimeCorrectionModelService(
            artifactURL: artifact,
            eventLogURL: eventLog,
            defaults: defaults
        )

        let evaluation = service.evaluate(
            VocoRuntimeCorrectionFeatures(
                rawTranscript: "首先用 C O I 先補一下吧。",
                canonicalizedText: "首先用 CLI 先補一下吧。",
                postRuleText: "首先用 CLI 先補一下吧。",
                contextHints: ["Voco", "CLI"],
                candidateSpans: [
                    VocoRuntimeCorrectionCandidate(source: "C O I", target: "CLI", score: 0.91)
                ]
            )
        )

        #expect(service.status.isAvailable)
        #expect(evaluation.outputText == "首先用 CLI 先補一下吧。")
        #expect(evaluation.changed == false)
        #expect(evaluation.decision?.runtimeMode == "shadow")
        #expect(evaluation.decision?.chosenAction == "noop")
        #expect(evaluation.decision?.fallbackReason == "shadow-contract-fixture-no-runtime-model")

        let events = try readRuntimeDecisionEvents(eventLog)
        #expect(events.count == 1)
        #expect(events[0].rawTranscript == "首先用 C O I 先補一下吧。")
        #expect(events[0].finalText == "首先用 CLI 先補一下吧。")
        #expect(events[0].candidates.map(\.target) == ["CLI"])
    }

    @Test func disabledRuntimeModelKeepsDeterministicJsonOutput() throws {
        let root = try runtimeTemporaryDirectory()
        let artifact = try writeRuntimeShadowArtifact(in: root)
        let eventLog = root.appendingPathComponent("shadow-events.jsonl")
        let runtimeDefaults = try runtimeTemporaryDefaults()
        runtimeDefaults.set(false, forKey: VocoRuntimeCorrectionModelService.enabledKey)
        let runtimeService = VocoRuntimeCorrectionModelService(
            artifactURL: artifact,
            eventLogURL: eventLog,
            defaults: runtimeDefaults
        )
        let autoApplyService = VocoAutoApplyModelService(
            modelURL: try writeRuntimeAutoApplyFixture(in: root),
            defaults: try runtimeTemporaryDefaults()
        )
        let canonicalization = VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: autoApplyService,
            runtimeCorrectionModelService: runtimeService
        )

        let result = canonicalization.normalize("那我們又做了把這個模型給載入了，所以他在我們這個城市裡面的角色是什麼？")

        #expect(result.normalizedText == "那我們又做了把這個模型給載入了，所以他在我們這個程式裡面的角色是什麼？")
        #expect(FileManager.default.fileExists(atPath: eventLog.path) == false)
    }

    @Test func shadowRuntimeModelCannotOverrideDeterministicJsonOutput() throws {
        let root = try runtimeTemporaryDirectory()
        let artifact = try writeRuntimeShadowArtifact(in: root)
        let eventLog = root.appendingPathComponent("shadow-events.jsonl")
        let runtimeDefaults = try runtimeTemporaryDefaults()
        runtimeDefaults.set(true, forKey: VocoRuntimeCorrectionModelService.enabledKey)
        let runtimeService = VocoRuntimeCorrectionModelService(
            artifactURL: artifact,
            eventLogURL: eventLog,
            defaults: runtimeDefaults
        )
        let autoApplyService = VocoAutoApplyModelService(
            modelURL: try writeRuntimeAutoApplyFixture(in: root),
            defaults: try runtimeTemporaryDefaults()
        )
        let canonicalization = VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: autoApplyService,
            runtimeCorrectionModelService: runtimeService
        )

        let result = canonicalization.normalize("那我們又做了把這個模型給載入了，所以他在我們這個城市裡面的角色是什麼？")

        #expect(result.normalizedText == "那我們又做了把這個模型給載入了，所以他在我們這個程式裡面的角色是什麼？")
        let events = try readRuntimeDecisionEvents(eventLog)
        #expect(events.count == 1)
        #expect(events[0].postRuleText == result.normalizedText)
        #expect(events[0].chosenAction == "noop")
    }

    @Test func skipPostASRCorrectionModelsPolicyBypassesAutoApplyAndRuntimeModels() throws {
        let root = try runtimeTemporaryDirectory()
        let artifact = try writeRuntimeShadowArtifact(in: root)
        let eventLog = root.appendingPathComponent("shadow-events.jsonl")
        let runtimeDefaults = try runtimeTemporaryDefaults()
        runtimeDefaults.set(true, forKey: VocoRuntimeCorrectionModelService.enabledKey)
        let runtimeService = VocoRuntimeCorrectionModelService(
            artifactURL: artifact,
            eventLogURL: eventLog,
            defaults: runtimeDefaults
        )
        let autoApplyService = VocoAutoApplyModelService(
            modelURL: try writeRuntimeAutoApplyFixture(in: root),
            defaults: try runtimeTemporaryDefaults()
        )
        let canonicalization = VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: autoApplyService,
            runtimeCorrectionModelService: runtimeService
        )

        let input = "那我們又做了把這個模型給載入了，所以他在我們這個城市裡面的角色是什麼？"
        let result = canonicalization.normalize(
            input,
            correctionPolicy: .skipPostASRCorrectionModels
        )

        #expect(result.normalizedText == input)
        #expect(result.replacements.isEmpty)
        #expect(result.suggestions.isEmpty)
        #expect(FileManager.default.fileExists(atPath: eventLog.path) == false)
    }

    @Test func gatedApplyContractCanDirectlyChangeFinalTextWhenReplaySafeAndApproved() throws {
        let root = try runtimeTemporaryDirectory()
        let artifact = try writeRuntimeGatedApplyArtifact(in: root)
        let eventLog = root.appendingPathComponent("gated-apply-events.jsonl")
        let defaults = try runtimeTemporaryDefaults()
        defaults.set(true, forKey: VocoRuntimeCorrectionModelService.enabledKey)
        let service = VocoRuntimeCorrectionModelService(
            artifactURL: artifact,
            eventLogURL: eventLog,
            defaults: defaults
        )

        let evaluation = service.evaluate(
            VocoRuntimeCorrectionFeatures(
                rawTranscript: "runtime 小模型直接改輸出",
                canonicalizedText: "runtime 小模型直接改輸出",
                postRuleText: "runtime 小模型直接改輸出",
                candidateSpans: [
                    VocoRuntimeCorrectionCandidate(source: "直接改輸出", target: "直接改 final output", score: 0.99)
                ]
            )
        )

        #expect(service.status.isAvailable)
        #expect(service.isGatedApplyEnabled)
        #expect(evaluation.changed)
        #expect(evaluation.outputText == "runtime 小模型直接改 final output")
        #expect(evaluation.decision?.chosenAction == "apply")
        #expect(evaluation.decision?.score == 0.99)
        #expect(evaluation.decision?.reasonCodes.contains("not-worse-than-compiled-json") == true)

        let events = try readRuntimeDecisionEvents(eventLog)
        #expect(events.count == 1)
        #expect(events[0].runtimeMode == "gatedApply")
        #expect(events[0].chosenAction == "apply")
        #expect(events[0].finalText == "runtime 小模型直接改 final output")
    }

    @Test func gatedApplyCanonicalizationUsesPortableCandidateSpanModelForNewGap() throws {
        let root = try runtimeTemporaryDirectory()
        let artifact = try writeRuntimeGatedApplyArtifact(in: root)
        let eventLog = root.appendingPathComponent("gated-apply-events.jsonl")
        let runtimeDefaults = try runtimeTemporaryDefaults()
        runtimeDefaults.set(true, forKey: VocoRuntimeCorrectionModelService.enabledKey)
        let runtimeService = VocoRuntimeCorrectionModelService(
            artifactURL: artifact,
            eventLogURL: eventLog,
            defaults: runtimeDefaults
        )
        let autoApplyService = VocoAutoApplyModelService(
            modelURL: try writeRuntimeAutoApplyFixture(in: root),
            defaults: try runtimeTemporaryDefaults()
        )
        let canonicalization = VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: autoApplyService,
            runtimeCorrectionModelService: runtimeService
        )

        let result = canonicalization.normalize("runtime 小模型直接改輸出")

        #expect(result.normalizedText == "runtime 小模型直接改 final output")
        let events = try readRuntimeDecisionEvents(eventLog)
        #expect(events.count == 1)
        #expect(events[0].chosenAction == "apply")
        #expect(events[0].candidates.map(\.target) == ["直接改 final output"])
    }

    @Test func gatedApplyCanonicalizationCannotRegressCompiledJsonRuleBaseline() throws {
        let root = try runtimeTemporaryDirectory()
        let artifact = try writeRuntimeGatedApplyArtifact(
            in: root,
            candidateEntries: """
                {
                  "id": "unsafe-inverse-city-to-program",
                  "source": "程式",
                  "target": "城市",
                  "score": 0.999
                }
            """
        )
        let eventLog = root.appendingPathComponent("gated-apply-events.jsonl")
        let runtimeDefaults = try runtimeTemporaryDefaults()
        runtimeDefaults.set(true, forKey: VocoRuntimeCorrectionModelService.enabledKey)
        let runtimeService = VocoRuntimeCorrectionModelService(
            artifactURL: artifact,
            eventLogURL: eventLog,
            defaults: runtimeDefaults
        )
        let autoApplyService = VocoAutoApplyModelService(
            modelURL: try writeRuntimeAutoApplyFixture(in: root),
            defaults: try runtimeTemporaryDefaults()
        )
        let canonicalization = VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: autoApplyService,
            runtimeCorrectionModelService: runtimeService
        )

        let result = canonicalization.normalize("那我們又做了把這個模型給載入了，所以他在我們這個城市裡面的角色是什麼？")

        #expect(result.normalizedText == "那我們又做了把這個模型給載入了，所以他在我們這個程式裡面的角色是什麼？")
        let events = try readRuntimeDecisionEvents(eventLog)
        #expect(events.count == 1)
        #expect(events[0].chosenAction == "block")
        #expect(events[0].fallbackReason == "deterministic-rule-priority")
        #expect(events[0].candidates.map(\.target) == ["城市"])
    }

    @Test func gatedApplyCannotOverrideDeterministicJsonRuleOutput() throws {
        let root = try runtimeTemporaryDirectory()
        let artifact = try writeRuntimeGatedApplyArtifact(in: root)
        let eventLog = root.appendingPathComponent("gated-apply-events.jsonl")
        let defaults = try runtimeTemporaryDefaults()
        defaults.set(true, forKey: VocoRuntimeCorrectionModelService.enabledKey)
        let service = VocoRuntimeCorrectionModelService(
            artifactURL: artifact,
            eventLogURL: eventLog,
            defaults: defaults
        )

        let evaluation = service.evaluate(
            VocoRuntimeCorrectionFeatures(
                rawTranscript: "那我們又做了把這個模型給載入了，所以他在我們這個城市裡面的角色是什麼？",
                canonicalizedText: "那我們又做了把這個模型給載入了，所以他在我們這個城市裡面的角色是什麼？",
                postRuleText: "那我們又做了把這個模型給載入了，所以他在我們這個程式裡面的角色是什麼？",
                deterministicRuleFires: [
                    VocoAutoApplyPolicyFire(
                        policyId: "runtime-test-city-to-program",
                        policyType: "scopedReplacement",
                        autoApplyMode: "apply",
                        sourcePattern: "城市",
                        targetText: "程式",
                        sourceSlices: ["城市"]
                    )
                ],
                candidateSpans: [
                    VocoRuntimeCorrectionCandidate(source: "程式", target: "城市", score: 0.999)
                ]
            )
        )

        #expect(evaluation.outputText == "那我們又做了把這個模型給載入了，所以他在我們這個程式裡面的角色是什麼？")
        #expect(evaluation.changed == false)
        #expect(evaluation.decision?.chosenAction == "block")
        #expect(evaluation.decision?.fallbackReason == "deterministic-rule-priority")
        #expect(evaluation.decision?.reasonCodes.contains("not-worse-than-compiled-json") == true)
    }

    @Test func gatedApplyArtifactWithoutNotWorseReadinessCannotLoad() throws {
        let root = try runtimeTemporaryDirectory()
        let artifact = try writeRuntimeGatedApplyArtifact(in: root, notWorse: false)
        let eventLog = root.appendingPathComponent("gated-apply-events.jsonl")
        let defaults = try runtimeTemporaryDefaults()
        defaults.set(true, forKey: VocoRuntimeCorrectionModelService.enabledKey)
        let service = VocoRuntimeCorrectionModelService(
            artifactURL: artifact,
            eventLogURL: eventLog,
            defaults: defaults
        )

        let evaluation = service.evaluate(
            VocoRuntimeCorrectionFeatures(
                rawTranscript: "runtime 小模型直接改輸出",
                canonicalizedText: "runtime 小模型直接改輸出",
                postRuleText: "runtime 小模型直接改輸出",
                candidateSpans: [
                    VocoRuntimeCorrectionCandidate(source: "直接改輸出", target: "直接改 final output", score: 0.99)
                ]
            )
        )

        #expect(service.status.isAvailable == false)
        #expect(evaluation.outputText == "runtime 小模型直接改輸出")
        #expect(FileManager.default.fileExists(atPath: eventLog.path) == false)
    }

    @Test func gatedApplyArtifactWithInvalidCandidateSpanModelCannotLoad() throws {
        let root = try runtimeTemporaryDirectory()
        let artifact = try writeRuntimeGatedApplyArtifact(
            in: root,
            candidateEntries: """
                {
                  "id": "invalid-empty-source",
                  "source": "",
                  "target": "直接改 final output",
                  "score": 0.99
                }
            """
        )
        let eventLog = root.appendingPathComponent("gated-apply-events.jsonl")
        let defaults = try runtimeTemporaryDefaults()
        defaults.set(true, forKey: VocoRuntimeCorrectionModelService.enabledKey)
        let service = VocoRuntimeCorrectionModelService(
            artifactURL: artifact,
            eventLogURL: eventLog,
            defaults: defaults
        )

        let evaluation = service.evaluate(
            VocoRuntimeCorrectionFeatures(
                rawTranscript: "runtime 小模型直接改輸出",
                canonicalizedText: "runtime 小模型直接改輸出",
                postRuleText: "runtime 小模型直接改輸出"
            )
        )

        #expect(service.status.isAvailable == false)
        #expect(evaluation.outputText == "runtime 小模型直接改輸出")
        #expect(FileManager.default.fileExists(atPath: eventLog.path) == false)
    }

    @Test func gatedApplyArtifactWithNestedRelativeModelPathCanLoad() throws {
        let root = try runtimeTemporaryDirectory()
        let artifact = try writeRuntimeGatedApplyArtifact(
            in: root,
            modelRelativePath: "models/runtime-candidate-spans.fixture"
        )
        let defaults = try runtimeTemporaryDefaults()
        defaults.set(true, forKey: VocoRuntimeCorrectionModelService.enabledKey)
        let service = VocoRuntimeCorrectionModelService(
            artifactURL: artifact,
            eventLogURL: root.appendingPathComponent("gated-apply-events.jsonl"),
            defaults: defaults
        )

        let evaluation = service.evaluate(
            VocoRuntimeCorrectionFeatures(
                rawTranscript: "runtime 小模型直接改輸出",
                canonicalizedText: "runtime 小模型直接改輸出",
                postRuleText: "runtime 小模型直接改輸出"
            )
        )

        #expect(service.status.isAvailable)
        #expect(evaluation.outputText == "runtime 小模型直接改 final output")
    }

    @Test func gatedApplyArtifactWithUnsafeModelPathCannotLoad() throws {
        let root = try runtimeTemporaryDirectory()
        let unsafePaths = [
            "../runtime-candidate-spans.fixture",
            "/private/tmp/runtime-candidate-spans.fixture",
            "models//runtime-candidate-spans.fixture",
            "models/./runtime-candidate-spans.fixture",
            "proposal-ranker-model.joblib"
        ]

        for unsafePath in unsafePaths {
            let artifactRoot = root.appendingPathComponent(UUID().uuidString, isDirectory: true)
            let artifact = try writeRuntimeGatedApplyArtifact(
                in: artifactRoot,
                artifactModelPath: unsafePath
            )
            let defaults = try runtimeTemporaryDefaults()
            defaults.set(true, forKey: VocoRuntimeCorrectionModelService.enabledKey)
            let service = VocoRuntimeCorrectionModelService(
                artifactURL: artifact,
                eventLogURL: artifactRoot.appendingPathComponent("gated-apply-events.jsonl"),
                defaults: defaults
            )

            let evaluation = service.evaluate(
                VocoRuntimeCorrectionFeatures(
                    rawTranscript: "runtime 小模型直接改輸出",
                    canonicalizedText: "runtime 小模型直接改輸出",
                    postRuleText: "runtime 小模型直接改輸出"
                )
            )

            #expect(service.status.isAvailable == false)
            #expect(evaluation.outputText == "runtime 小模型直接改輸出")
        }
    }

    @Test func runtimeCorrectionArtifactChangesAreReloadedAutomatically() async throws {
        let root = try runtimeTemporaryDirectory()
        let artifact = root.appendingPathComponent("runtime-correction-artifact.json")
        let eventLog = root.appendingPathComponent("gated-apply-events.jsonl")
        let defaults = try runtimeTemporaryDefaults()
        defaults.set(true, forKey: VocoRuntimeCorrectionModelService.enabledKey)
        let service = VocoRuntimeCorrectionModelService(
            artifactURL: artifact,
            eventLogURL: eventLog,
            defaults: defaults
        )

        #expect(service.status.isAvailable == false)
        try await Task.sleep(nanoseconds: 200_000_000)
        _ = try writeRuntimeGatedApplyArtifact(in: root)
        try await waitForRuntimeCorrectionModelAvailability(service)

        let evaluation = service.evaluate(
            VocoRuntimeCorrectionFeatures(
                rawTranscript: "runtime 小模型直接改輸出",
                canonicalizedText: "runtime 小模型直接改輸出",
                postRuleText: "runtime 小模型直接改輸出"
            )
        )

        #expect(service.status.isAvailable)
        #expect(evaluation.outputText == "runtime 小模型直接改 final output")
    }

    @Test func gatedApplyFixtureReplayDoesNotRegressRuleBaseline() throws {
        let root = try runtimeTemporaryDirectory()
        let artifact = try writeRuntimeGatedApplyArtifact(in: root)
        let defaults = try runtimeTemporaryDefaults()
        defaults.set(true, forKey: VocoRuntimeCorrectionModelService.enabledKey)
        let service = VocoRuntimeCorrectionModelService(
            artifactURL: artifact,
            eventLogURL: root.appendingPathComponent("gated-apply-events.jsonl"),
            defaults: defaults
        )
        let ruleFire = VocoAutoApplyPolicyFire(
            policyId: "runtime-test-city-to-program",
            policyType: "scopedReplacement",
            autoApplyMode: "apply",
            sourcePattern: "城市",
            targetText: "程式",
            sourceSlices: ["城市"]
        )
        let cases: [(String, VocoRuntimeCorrectionFeatures, String, String)] = [
            (
                "rule baseline is preserved",
                VocoRuntimeCorrectionFeatures(
                    rawTranscript: "城市角色",
                    canonicalizedText: "城市角色",
                    postRuleText: "程式角色",
                    deterministicRuleFires: [ruleFire],
                    candidateSpans: [
                        VocoRuntimeCorrectionCandidate(source: "程式", target: "城市", score: 0.999)
                    ]
                ),
                "程式角色",
                "block"
            ),
            (
                "action command is never rewritten",
                VocoRuntimeCorrectionFeatures(
                    rawTranscript: "全部刪除",
                    canonicalizedText: "全部刪除",
                    postRuleText: "全部刪除",
                    actionCommand: true,
                    candidateSpans: [
                        VocoRuntimeCorrectionCandidate(source: "全部刪除", target: "全部都刪掉", score: 0.999)
                    ]
                ),
                "全部刪除",
                "block"
            ),
            (
                "new non-rule gap can improve",
                VocoRuntimeCorrectionFeatures(
                    rawTranscript: "runtime 小模型直接改輸出",
                    canonicalizedText: "runtime 小模型直接改輸出",
                    postRuleText: "runtime 小模型直接改輸出",
                    candidateSpans: [
                        VocoRuntimeCorrectionCandidate(source: "直接改輸出", target: "直接改 final output", score: 0.99)
                    ]
                ),
                "runtime 小模型直接改 final output",
                "apply"
            ),
            (
                "low confidence candidate is ignored",
                VocoRuntimeCorrectionFeatures(
                    rawTranscript: "runtime 小模型直接改輸出",
                    canonicalizedText: "runtime 小模型直接改輸出",
                    postRuleText: "runtime 小模型直接改輸出",
                    candidateSpans: [
                        VocoRuntimeCorrectionCandidate(source: "直接改輸出", target: "直接改 final output", score: 0.9)
                    ]
                ),
                "runtime 小模型直接改輸出",
                "noop"
            )
        ]

        for (name, features, expectedOutput, expectedAction) in cases {
            let evaluation = service.evaluate(features)
            #expect(evaluation.outputText == expectedOutput, "\(name) output")
            #expect(evaluation.decision?.chosenAction == expectedAction, "\(name) action")
        }
    }

    @Test func actionCommandBypassesRuntimeModelTextRewrite() throws {
        let root = try runtimeTemporaryDirectory()
        let artifact = try writeRuntimeShadowArtifact(in: root)
        let eventLog = root.appendingPathComponent("shadow-events.jsonl")
        let defaults = try runtimeTemporaryDefaults()
        defaults.set(true, forKey: VocoRuntimeCorrectionModelService.enabledKey)
        let service = VocoRuntimeCorrectionModelService(
            artifactURL: artifact,
            eventLogURL: eventLog,
            defaults: defaults
        )

        let evaluation = service.evaluate(
            VocoRuntimeCorrectionFeatures(
                rawTranscript: "全部刪除",
                canonicalizedText: "全部刪除",
                postRuleText: "全部刪除",
                actionCommand: true,
                candidateSpans: [
                    VocoRuntimeCorrectionCandidate(source: "全部刪除", target: "全部都刪掉", score: 0.99)
                ]
            )
        )

        #expect(evaluation.outputText == "全部刪除")
        #expect(evaluation.decision?.chosenAction == "block")
        #expect(evaluation.decision?.fallbackReason == "action-command-bypass")

        let events = try readRuntimeDecisionEvents(eventLog)
        #expect(events.count == 1)
        #expect(events[0].finalText == "全部刪除")
    }

    @Test func missingArtifactFallsBackToPostRuleText() throws {
        let root = try runtimeTemporaryDirectory()
        let missingArtifact = root.appendingPathComponent("missing-runtime-correction-artifact.json")
        let defaults = try runtimeTemporaryDefaults()
        defaults.set(true, forKey: VocoRuntimeCorrectionModelService.enabledKey)
        let service = VocoRuntimeCorrectionModelService(
            artifactURL: missingArtifact,
            eventLogURL: root.appendingPathComponent("shadow-events.jsonl"),
            defaults: defaults
        )

        let evaluation = service.evaluate(
            VocoRuntimeCorrectionFeatures(
                rawTranscript: "原始",
                canonicalizedText: "規則後",
                postRuleText: "規則後"
            )
        )

        #expect(service.status.isAvailable == false)
        #expect(evaluation.outputText == "規則後")
        #expect(evaluation.decision == nil)
    }

    @Test func missingPostASRCorrectionModelsKeepCanonicalizationUsable() throws {
        let root = try runtimeTemporaryDirectory()
        let eventLog = root.appendingPathComponent("shadow-events.jsonl")
        let runtimeDefaults = try runtimeTemporaryDefaults()
        runtimeDefaults.set(true, forKey: VocoRuntimeCorrectionModelService.enabledKey)
        let runtimeService = VocoRuntimeCorrectionModelService(
            artifactURL: root.appendingPathComponent("missing-runtime-correction-artifact.json"),
            eventLogURL: eventLog,
            defaults: runtimeDefaults
        )
        let autoApplyService = VocoAutoApplyModelService(
            modelURL: root.appendingPathComponent("missing-auto-apply-model.json"),
            defaults: try runtimeTemporaryDefaults()
        )
        let canonicalization = VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: autoApplyService,
            runtimeCorrectionModelService: runtimeService
        )

        let input = "沒有安裝後處理模型時，基本轉錄結果仍然可以保存。"
        let result = canonicalization.normalize(input)

        #expect(autoApplyService.status.isAvailable == false)
        #expect(runtimeService.status.isAvailable == false)
        #expect(result.normalizedText == input)
        #expect(result.replacements.isEmpty)
        #expect(result.suggestions.isEmpty)
        #expect(FileManager.default.fileExists(atPath: eventLog.path) == false)
    }

    @Test func joblibCannotBeLoadedByRuntimeOrCompiledJsonBoundaries() throws {
        let root = try runtimeTemporaryDirectory()
        let joblib = root.appendingPathComponent("proposal-ranker-model.joblib")
        try Data([0x80, 0x04]).write(to: joblib)
        let defaults = try runtimeTemporaryDefaults()
        defaults.set(true, forKey: VocoRuntimeCorrectionModelService.enabledKey)

        let runtimeService = VocoRuntimeCorrectionModelService(
            artifactURL: joblib,
            eventLogURL: root.appendingPathComponent("shadow-events.jsonl"),
            defaults: defaults
        )
        let autoApplyService = VocoAutoApplyModelService(
            modelURL: joblib,
            defaults: try runtimeTemporaryDefaults()
        )

        #expect(runtimeService.status.isAvailable == false)
        #expect(runtimeService.evaluate(.init(rawTranscript: "a", canonicalizedText: "b", postRuleText: "b")).outputText == "b")
        #expect(autoApplyService.status.isAvailable == false)
        #expect(autoApplyService.evaluate("全部刪除").outputText == "全部刪除")
    }
}

private func runtimeTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("VocoRuntimeCorrectionModelServiceTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func runtimeTemporaryDefaults() throws -> UserDefaults {
    let suiteName = "VocoRuntimeCorrectionModelServiceTests-\(UUID().uuidString)"
    let defaults = try #require(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
    return defaults
}

private func waitForRuntimeCorrectionModelAvailability(
    _ service: VocoRuntimeCorrectionModelService,
    timeout: TimeInterval = 3.0
) async throws {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        if service.status.isAvailable {
            return
        }
        try await Task.sleep(nanoseconds: 50_000_000)
    }
}

private func writeRuntimeShadowArtifact(in root: URL) throws -> URL {
    let url = root.appendingPathComponent("runtime-correction-artifact.json")
    let artifact = """
    {
      "schema": "voco.runtime-correction-model.v1",
      "artifactId": "runtime-correction-shadow-test-contract",
      "runtimeMode": "shadow",
      "intendedUse": "runtime shadow correction contract; does not replace compiled JSON rules",
      "model": {
        "format": "none",
        "modelType": "shadow-contract-only",
        "path": "",
        "portableRuntime": false,
        "sha256": ""
      },
      "approval": {
        "allowedModes": ["shadow"],
        "runtimeAllowed": false
      },
      "sourceRanker": {
        "runtimeUsableDirectly": false
      },
      "safety": {
        "actionCommandBypass": true,
        "artifactMissingFallback": "return-post-rule-text",
        "compiledJsonLoaderMayLoadJoblib": false,
        "jsonExactRulePriority": true,
        "timeoutFallback": "return-post-rule-text"
      },
      "decisionSchema": {
        "schema": "voco.runtime-correction-decision.v1",
        "actions": ["noop", "suggest", "apply", "block"],
        "requiresEvidenceEvent": true,
        "requiresReasonCodes": true,
        "requiresScore": true
      },
      "candidateGenerator": {
        "required": true,
        "schema": "voco.runtime-candidate-generator.v1",
        "sha256": "candidate-generator-test-sha"
      }
    }
    """
    try Data(artifact.utf8).write(to: url)
    return url
}

private func writeRuntimeGatedApplyArtifact(
    in root: URL,
    notWorse: Bool = true,
    candidateEntries: String? = nil,
    modelRelativePath: String = "runtime-candidate-spans.fixture",
    artifactModelPath: String? = nil
) throws -> URL {
    let modelURL = root.appendingPathComponent(modelRelativePath)
    try FileManager.default.createDirectory(
        at: modelURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    let entries = candidateEntries ?? """
        {
          "id": "runtime-direct-output",
          "source": "直接改輸出",
          "target": "直接改 final output",
          "score": 0.99
        }
    """
    let model = """
    {
      "schema": "voco.runtime-candidate-spans.v1",
      "candidates": [
    \(entries)
      ]
    }
    """
    try Data(model.utf8).write(to: modelURL)
    let modelSha = try sha256Hex(of: modelURL)

    let url = root.appendingPathComponent("runtime-correction-artifact.json")
    let artifact = """
    {
      "schema": "voco.runtime-correction-model.v1",
      "artifactId": "runtime-correction-gated-apply-test-contract",
      "runtimeMode": "gatedApply",
      "intendedUse": "runtime gated apply correction contract; preserves compiled JSON rule baseline",
      "model": {
        "format": "candidate-spans-v1",
        "modelType": "candidate-ranker",
        "path": "\(artifactModelPath ?? modelRelativePath)",
        "portableRuntime": true,
        "sha256": "\(modelSha)"
      },
      "approval": {
        "allowedModes": ["gatedApply"],
        "approvedAt": "2026-06-13T10:00:00Z",
        "approvedBy": "Jason",
        "approvalToken": "jason-approved-runtime-gated-apply-test",
        "requiresJasonApprovalForApply": true,
        "runtimeAllowed": true
      },
      "sourceRanker": {
        "runtimeUsableDirectly": false
      },
      "safety": {
        "actionCommandBypass": true,
        "artifactMissingFallback": "return-post-rule-text",
        "compiledJsonLoaderMayLoadJoblib": false,
        "jsonExactRulePriority": true,
        "notWorseThanCompiledJson": \(notWorse),
        "timeoutFallback": "return-post-rule-text"
      },
      "decisionSchema": {
        "schema": "voco.runtime-correction-decision.v1",
        "actions": ["noop", "suggest", "apply", "block"],
        "requiresEvidenceEvent": true,
        "requiresReasonCodes": true,
        "requiresScore": true
      },
      "candidateGenerator": {
        "required": true,
        "schema": "voco.runtime-candidate-generator.v1",
        "sha256": "candidate-generator-test-sha"
      },
      "thresholdConfig": {
        "shadow": 0.0,
        "suggest": 0.85,
        "gatedApply": 0.97
      },
      "runtimeReadiness": {
        "actionCommandBypassVerified": true,
        "baselineReplayPass": true,
        "finalTextRegressionCount": \(notWorse ? 0 : 1),
        "gatedApplyReplayPass": \(notWorse),
        "notWorseThanCompiledJson": \(notWorse),
        "unsafeApplyFalsePositiveCount": 0
      }
    }
    """
    try Data(artifact.utf8).write(to: url)
    return url
}

private func writeRuntimeAutoApplyFixture(in root: URL) throws -> URL {
    let url = root.appendingPathComponent("full-db.auto-apply-model.json")
    let fixture = """
    {
      "policyCounts": { "apply": 1 },
      "policyTypeCounts": { "scopedReplacement": 1 },
      "safetyContract": [],
      "protectedTermAllowlistGuards": [],
      "mergedReplayReadiness": {
        "mergedAutoApplyModelReady": true
      },
      "policies": [
        {
          "policyId": "runtime-test-city-to-program",
          "autoApplyMode": "apply",
          "policyType": "scopedReplacement",
          "sourcePattern": "城市",
          "targetText": "程式",
          "sourceSlices": ["城市"],
          "reviewGateConflictRows": []
        }
      ]
    }
    """
    try Data(fixture.utf8).write(to: url)
    return url
}

private func sha256Hex(of url: URL) throws -> String {
    let data = try Data(contentsOf: url)
    let digest = SHA256.hash(data: data)
    return digest.map { String(format: "%02x", $0) }.joined()
}

private func readRuntimeDecisionEvents(_ url: URL) throws -> [VocoRuntimeCorrectionDecision] {
    let data = try Data(contentsOf: url)
    let lines = String(decoding: data, as: UTF8.self)
        .split(separator: "\n")
        .map(String.init)
    return try lines.map { line in
        try JSONDecoder().decode(VocoRuntimeCorrectionDecision.self, from: Data(line.utf8))
    }
}
