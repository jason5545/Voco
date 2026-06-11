import Foundation
import Testing
@testable import Voco

struct VocoAutoApplyModelServiceTests {
    @Test func missingModelIsUnavailableAndSettingsToggleIsOffDisabled() throws {
        let service = VocoAutoApplyModelService(
            modelURL: try temporaryDirectory().appendingPathComponent("fixture.auto-apply.json"),
            defaults: try temporaryDefaults()
        )

        #expect(service.status.isAvailable == false)
        #expect(service.settingsToggleIsOn == false)
        #expect(service.settingsToggleIsEnabled == false)
        #expect(service.evaluate("Cloud 的 OPUS 模型").outputText == "Cloud 的 OPUS 模型")
    }

    @Test func validReadyModelIsAvailableAndDefaultsOn() throws {
        let url = try writeFixture(ready: true)
        let service = VocoAutoApplyModelService(modelURL: url, defaults: try temporaryDefaults())

        #expect(service.status.isAvailable == true)
        #expect(service.settingsToggleIsOn == true)
        #expect(service.settingsToggleIsEnabled == true)
    }

    @Test func invalidJSONAndReadinessFalseAreUnavailable() throws {
        let invalidURL = try temporaryDirectory().appendingPathComponent("invalid-fixture.json")
        try Data("{".utf8).write(to: invalidURL)
        let invalid = VocoAutoApplyModelService(modelURL: invalidURL, defaults: try temporaryDefaults())
        #expect(invalid.status.isAvailable == false)
        #expect(invalid.evaluate("Cloud 的 OPUS 模型").outputText == "Cloud 的 OPUS 模型")

        let notReady = VocoAutoApplyModelService(
            modelURL: try writeFixture(ready: false),
            defaults: try temporaryDefaults()
        )
        #expect(notReady.status.isAvailable == false)
        #expect(notReady.evaluate("Cloud 的 OPUS 模型").outputText == "Cloud 的 OPUS 模型")
    }

    @Test func userSettingOffBlocksAutoApplyEvenWhenReady() throws {
        let defaults = try temporaryDefaults()
        defaults.set(false, forKey: VocoAutoApplyModelService.enabledKey)
        let service = VocoAutoApplyModelService(
            modelURL: try writeFixture(ready: true),
            defaults: defaults
        )

        #expect(service.status.isAvailable == true)
        #expect(service.settingsToggleIsOn == false)
        #expect(service.evaluate("Cloud 的 OPUS 模型").outputText == "Cloud 的 OPUS 模型")
    }

    @Test func exactWholeUtteranceAndScopedReplacementPoliciesApply() throws {
        let service = VocoAutoApplyModelService(
            modelURL: try writeFixture(ready: true),
            defaults: try temporaryDefaults()
        )

        let exact = service.evaluate("W 零三的 VT 時間是什麼呢？")
        #expect(exact.outputText == "W零三的VT時間是什麼呢？")
        #expect(exact.applied.map(\.policyId) == ["exact-fixture-question"])

        let scoped = service.evaluate("我在測 Cloud 的 OPUS 模型")
        #expect(scoped.outputText == "我在測 Claude 的 OPUS 模型")
        #expect(scoped.applied.map(\.policyId) == ["scoped-fixture-claude"])
    }

    @Test func suggestPoliciesDoNotChangeOutputButEmitEvidence() throws {
        let service = VocoAutoApplyModelService(
            modelURL: try writeFixture(ready: true),
            defaults: try temporaryDefaults()
        )

        let result = service.evaluate("答案給我匯入")
        #expect(result.outputText == "答案給我匯入")
        #expect(result.applied.isEmpty)
        #expect(result.suggestions.map(\.policyId) == ["suggest-fixture-file-import"])
    }

    @Test func actionCommandsAreBlockedFromTextAutoApply() throws {
        let service = VocoAutoApplyModelService(
            modelURL: try writeFixture(ready: true),
            defaults: try temporaryDefaults()
        )

        let result = service.evaluate("全部刪除")
        #expect(result.outputText == "全部刪除")
        #expect(result.applied.isEmpty)
        #expect(result.suggestions.isEmpty)
    }

    @Test func questionParticlePunctuationIsNotStripped() throws {
        let service = VocoAutoApplyModelService(
            modelURL: try writeFixture(ready: true),
            defaults: try temporaryDefaults()
        )

        let result = service.evaluate("你要喝嗎？")
        #expect(result.outputText == "你要喝嗎？")
        #expect(!result.outputText.hasSuffix("嗎"))
    }

    @Test func gitignoreProtectsProductionAndLocalModelPaths() throws {
        let gitignore = try String(
            contentsOf: URL(fileURLWithPath: "/Users/jianruicheng/GitHub/Voco/.gitignore"),
            encoding: .utf8
        )

        #expect(gitignore.contains("LocalModels/"))
        #expect(gitignore.contains("Support/AutoApplyModels/"))
        #expect(gitignore.contains("full-db.auto-apply-model.json"))
    }

    private func writeFixture(ready: Bool) throws -> URL {
        let url = try temporaryDirectory().appendingPathComponent("small-auto-apply-fixture.json")
        let data = try #require(fixtureJSON(ready: ready).data(using: .utf8))
        try data.write(to: url)
        return url
    }

    private func fixtureJSON(ready: Bool) -> String {
        """
        {
          "policyCounts": { "apply": 3, "suggest": 1 },
          "policyTypeCounts": { "exactTrainablePair": 2, "scopedReplacement": 2 },
          "safetyContract": [
            "exact trainable-pair policies may auto-apply only on normalized whole-utterance match",
            "Voco action commands such as 全部刪除 are blocked from text auto-apply training",
            "unresolved exact-input conflicts across source slices are blocked from the merged model"
          ],
          "mergedReplayReadiness": {
            "mergedAutoApplyModelReady": \(ready ? "true" : "false"),
            "failures": []
          },
          "policies": [
            {
              "policyId": "exact-fixture-question",
              "autoApplyMode": "apply",
              "policyType": "exactTrainablePair",
              "exactInputRequired": true,
              "inputStrictKey": "w 零三的 vt 時間是什麼呢?",
              "sourcePattern": "W 零三的 VT 時間是什麼呢？",
              "targetText": "W零三的VT時間是什麼呢？",
              "sourceSlices": ["rerawPre12022"]
            },
            {
              "policyId": "exact-fixture-ma",
              "autoApplyMode": "apply",
              "policyType": "exactTrainablePair",
              "exactInputRequired": true,
              "inputStrictKey": "你要喝嗎?",
              "sourcePattern": "你要喝嗎？",
              "targetText": "你要喝嗎？",
              "sourceSlices": ["currentRaw"]
            },
            {
              "policyId": "scoped-fixture-claude",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "Cloud 的 OPUS 模型",
              "targetText": "Claude 的 OPUS 模型",
              "scopedSourcePhrase": "Cloud 的 OPUS 模型",
              "contextAliasesAny": [],
              "contextTokensAny": [],
              "sourceSlices": ["currentRaw"]
            },
            {
              "policyId": "suggest-fixture-file-import",
              "autoApplyMode": "suggest",
              "policyType": "scopedReplacement",
              "sourcePattern": "答案給我匯入",
              "targetText": "檔案給我匯入",
              "scopedSourcePhrase": "答案給我匯入",
              "contextAliasesAny": [],
              "contextTokensAny": [],
              "sourceSlices": ["currentRaw"]
            }
          ]
        }
        """
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("VocoAutoApplyModelServiceTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func temporaryDefaults() throws -> UserDefaults {
        let suiteName = "VocoAutoApplyModelServiceTests-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
        return defaults
    }
}
