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

    @Test func modelFileChangesAreReloadedAutomatically() async throws {
        let url = try temporaryDirectory().appendingPathComponent("watched-auto-apply-fixture.json")
        try writeFixture(to: url, ready: true)
        let service = VocoAutoApplyModelService(modelURL: url, defaults: try temporaryDefaults())

        #expect(service.evaluate("我在測 Cloud 的 OPUS 模型").outputText == "我在測 Claude 的 OPUS 模型")

        try await Task.sleep(nanoseconds: 100_000_000)
        try writeFixture(to: url, ready: true, scopedClaudeTarget: "Claude Reloaded 的 OPUS 模型")

        try await waitUntil {
            service.evaluate("我在測 Cloud 的 OPUS 模型").outputText == "我在測 Claude Reloaded 的 OPUS 模型"
        }
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

    @Test func migratedSwiftContextRulesApplyOnlyWithLockContext() throws {
        let service = VocoAutoApplyModelService(
            modelURL: try writeFixture(ready: true),
            defaults: try temporaryDefaults()
        )

        let transcriptionContext = "Voco retranscribe ASR 語音辨識 轉錄 技能"
        #expect(
            service.evaluate(
                "再跑一次轉怒的技能吧。",
                context: transcriptionContext
            ).outputText == "再跑一次轉錄的技能吧。"
        )
        #expect(
            service.evaluate(
                "再跑一次轉路的技能吧。",
                context: transcriptionContext
            ).outputText == "再跑一次轉錄的技能吧。"
        )
        #expect(
            service.evaluate(
                "重新轉入的技能",
                context: transcriptionContext
            ).outputText == "重新轉錄的技能"
        )
        #expect(service.evaluate("把資料轉入系統", context: "資料匯入 表格").outputText == "把資料轉入系統")

        let repositoryContext = "repo GitHub commit push 遠端"
        #expect(
            service.evaluate(
                "但是不要推 Ripper，並非是人名。",
                context: repositoryContext
            ).outputText == "但是不要推 repo，並非是人名。"
        )
        #expect(
            service.evaluate(
                "然後我那個reaper只能是repo，r e p o。",
                context: repositoryContext
            ).outputText == "然後我那個repo只能是repo，r e p o。"
        )
        #expect(service.evaluate("這是一個 reaper 音訊工具。").outputText == "這是一個 reaper 音訊工具。")

        let cloudflareContext = "Cloudflare Workers D1 GitHub repo 專案部署"
        #expect(
            service.evaluate(
                "目前我的初步構想是跑在 Load Fail的Workers，然後由D One來去做處理。",
                context: cloudflareContext
            ).outputText == "目前我的初步構想是跑在 Cloudflare的Workers，然後由D1來去做處理。"
        )
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

    private func writeFixture(ready: Bool, scopedClaudeTarget: String = "Claude 的 OPUS 模型") throws -> URL {
        let url = try temporaryDirectory().appendingPathComponent("small-auto-apply-fixture.json")
        try writeFixture(to: url, ready: ready, scopedClaudeTarget: scopedClaudeTarget)
        return url
    }

    private func writeFixture(
        to url: URL,
        ready: Bool,
        scopedClaudeTarget: String = "Claude 的 OPUS 模型"
    ) throws {
        let data = try #require(
            fixtureJSON(ready: ready, scopedClaudeTarget: scopedClaudeTarget).data(using: .utf8)
        )
        try data.write(to: url)
    }

    private func fixtureJSON(ready: Bool, scopedClaudeTarget: String = "Claude 的 OPUS 模型") -> String {
        """
        {
          "policyCounts": { "apply": 14, "suggest": 1 },
          "policyTypeCounts": { "exactTrainablePair": 2, "scopedReplacement": 13 },
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
              "targetText": "\(scopedClaudeTarget)",
              "scopedSourcePhrase": "Cloud 的 OPUS 模型",
              "contextAliasesAny": [],
              "contextTokensAny": [],
              "sourceSlices": ["currentRaw"]
            },
            {
              "policyId": "migrated-swift-transcription-zhuan-ru",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "轉入",
              "targetText": "轉錄",
              "scopedSourcePhrase": "轉入",
              "contextAliasesAny": [],
              "contextTokensAny": ["轉錄", "retranscribe", "技能", "語音", "辨識", "ASR", "Voco"],
              "contextRequired": true,
              "sourceSlices": ["controlEvidence"]
            },
            {
              "policyId": "migrated-swift-transcription-zhuan-nu",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "轉怒",
              "targetText": "轉錄",
              "scopedSourcePhrase": "轉怒",
              "contextAliasesAny": [],
              "contextTokensAny": ["轉錄", "retranscribe", "技能", "語音", "辨識", "ASR", "Voco"],
              "contextRequired": true,
              "sourceSlices": ["controlEvidence"]
            },
            {
              "policyId": "migrated-swift-transcription-zhuan-lu",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "轉路",
              "targetText": "轉錄",
              "scopedSourcePhrase": "轉路",
              "contextAliasesAny": [],
              "contextTokensAny": ["轉錄", "retranscribe", "技能", "語音", "辨識", "ASR", "Voco"],
              "contextRequired": true,
              "sourceSlices": ["controlEvidence"]
            },
            {
              "policyId": "migrated-swift-repo-push-ripper-spaced",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "推 Ripper",
              "targetText": "推 repo",
              "scopedSourcePhrase": "推 Ripper",
              "contextAliasesAny": [],
              "contextTokensAny": ["repo", "r e p o", "GitHub", "commit", "push", "推到", "推送", "遠端", "main", "origin"],
              "contextRequired": true,
              "sourceSlices": ["controlEvidence"]
            },
            {
              "policyId": "migrated-swift-repo-push-ripper-tight",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "推Ripper",
              "targetText": "推 repo",
              "scopedSourcePhrase": "推Ripper",
              "contextAliasesAny": [],
              "contextTokensAny": ["repo", "r e p o", "GitHub", "commit", "push", "推到", "推送", "遠端", "main", "origin"],
              "contextRequired": true,
              "sourceSlices": ["controlEvidence"]
            },
            {
              "policyId": "migrated-swift-repo-push-reaper-spaced",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "推 reaper",
              "targetText": "推 repo",
              "scopedSourcePhrase": "推 reaper",
              "contextAliasesAny": [],
              "contextTokensAny": ["repo", "r e p o", "GitHub", "commit", "push", "推到", "推送", "遠端", "main", "origin"],
              "contextRequired": true,
              "sourceSlices": ["controlEvidence"]
            },
            {
              "policyId": "migrated-swift-repo-push-reaper-tight",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "推reaper",
              "targetText": "推 repo",
              "scopedSourcePhrase": "推reaper",
              "contextAliasesAny": [],
              "contextTokensAny": ["repo", "r e p o", "GitHub", "commit", "push", "推到", "推送", "遠端", "main", "origin"],
              "contextRequired": true,
              "sourceSlices": ["controlEvidence"]
            },
            {
              "policyId": "migrated-swift-repo-reaper",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "reaper",
              "targetText": "repo",
              "scopedSourcePhrase": "reaper",
              "contextAliasesAny": [],
              "contextTokensAny": ["repo", "r e p o", "GitHub", "commit", "push", "推到", "推送", "遠端", "main", "origin"],
              "contextRequired": true,
              "sourceSlices": ["controlEvidence"]
            },
            {
              "policyId": "migrated-swift-repo-ripper",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "Ripper",
              "targetText": "repo",
              "scopedSourcePhrase": "Ripper",
              "contextAliasesAny": [],
              "contextTokensAny": ["repo", "r e p o", "GitHub", "commit", "push", "推到", "推送", "遠端", "main", "origin"],
              "contextRequired": true,
              "sourceSlices": ["controlEvidence"]
            },
            {
              "policyId": "migrated-swift-cloudflare-load-fail",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "Load Fail",
              "targetText": "Cloudflare",
              "scopedSourcePhrase": "Load Fail",
              "contextAliasesAny": [],
              "contextTokensAny": ["Cloudflare", "Workers", "D1", "D 1", "Durable Object", "repo", "GitHub", "專案", "部署"],
              "contextRequired": true,
              "sourceSlices": ["controlEvidence"]
            },
            {
              "policyId": "migrated-swift-cloudflare-d-one",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "D One",
              "targetText": "D1",
              "scopedSourcePhrase": "D One",
              "contextAliasesAny": [],
              "contextTokensAny": ["Cloudflare", "Workers", "D1", "D 1", "Durable Object", "repo", "GitHub", "專案", "部署"],
              "contextRequired": true,
              "sourceSlices": ["controlEvidence"]
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

    private func waitUntil(_ condition: @escaping () -> Bool) async throws {
        for _ in 0..<60 {
            if condition() { return }
            try await Task.sleep(nanoseconds: 50_000_000)
        }

        try #require(condition())
    }

    private func temporaryDefaults() throws -> UserDefaults {
        let suiteName = "VocoAutoApplyModelServiceTests-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
        return defaults
    }
}
