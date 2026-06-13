import Foundation
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

private func readRuntimeDecisionEvents(_ url: URL) throws -> [VocoRuntimeCorrectionDecision] {
    let data = try Data(contentsOf: url)
    let lines = String(decoding: data, as: UTF8.self)
        .split(separator: "\n")
        .map(String.init)
    return try lines.map { line in
        try JSONDecoder().decode(VocoRuntimeCorrectionDecision.self, from: Data(line.utf8))
    }
}
