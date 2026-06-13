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

    @Test func replacedPoliciesLoadButDoNotApplyOrSuggest() throws {
        let service = VocoAutoApplyModelService(
            modelURL: try writeFixture(ready: true),
            defaults: try temporaryDefaults()
        )

        let result = service.evaluate("推送到吉他。", context: "GitHub repo push")
        #expect(service.status.isAvailable == true)
        #expect(result.outputText == "推送到吉他。")
        #expect(result.applied.isEmpty)
        #expect(result.suggestions.isEmpty)
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

    @Test func mingdeAllowlistPhrasesRemainAllowed() throws {
        let service = VocoAutoApplyModelService(
            modelURL: try writeFixture(ready: true),
            defaults: try temporaryDefaults()
        )

        for text in ["明德捷運站。", "明德水庫。", "明德路附近。", "施明德。"] {
            let result = service.evaluate(text)
            #expect(result.outputText == text)
            #expect(result.applied.isEmpty)
            #expect(result.guardBlocks.isEmpty)
            #expect(result.requiresReview == false)
        }
    }

    @Test func mingdeScopedPolicyAppliesOnlyWhenContextLockMatches() throws {
        let service = VocoAutoApplyModelService(
            modelURL: try writeFixture(ready: true),
            defaults: try temporaryDefaults()
        )

        let result = service.evaluate("我們最明德變更應該有加了自動學習。")
        #expect(result.outputText == "我們最近的變更應該有加了自動學習。")
        #expect(result.applied.map(\.policyId) == ["scoped-fixture-mingde-recent-change"])
        #expect(result.guardBlocks.isEmpty)
    }

    @Test func mingdeOutsideAllowlistWithoutPolicySupportIsReviewOnlyGuarded() throws {
        let service = VocoAutoApplyModelService(
            modelURL: try writeFixture(ready: true),
            defaults: try temporaryDefaults()
        )

        let text = "這個明德變更怪怪的。"
        let result = service.evaluate(text)
        #expect(result.outputText == text)
        #expect(result.applied.isEmpty)
        #expect(result.guardBlocks.map(\.guardId) == ["protected-term-allowlist.mingde"])
        #expect(result.guardBlocks.first?.reason == VocoAutoApplyModelService.protectedTermGuardReason)
        #expect(result.requiresReview == true)

        let normalization = VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: service
        ).normalize(text, activeContextIDs: [])
        #expect(normalization.normalizedText == text)
        #expect(normalization.suggestions.contains {
            $0.reason == VocoAutoApplyModelService.protectedTermGuardReason &&
                $0.termID == "auto-apply-model.guard.protected-term-allowlist.mingde"
        })

        let assessment = VocoConfidenceGateService().assess(normalizationResult: normalization)
        #expect(assessment.route == .reviewSuggested)
        #expect(assessment.reasons.contains(VocoAutoApplyModelService.protectedTermGuardReason))
        #expect(assessment.reviewTriggers.contains {
            $0.id == VocoAutoApplyModelService.protectedTermGuardReason
        })
    }

    @Test func protectedTermGuardFallsBackFromPrimaryCandidateSurface() throws {
        let service = VocoAutoApplyModelService(
            modelURL: try writeFixture(ready: true),
            defaults: try temporaryDefaults()
        )

        let guardedText = "我明德是這個是 repo。"
        let rawText = "我講的是這個是 repo。"
        let normalization = VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: service
        ).normalize(guardedText, activeContextIDs: [])
        let assessment = VocoConfidenceGateService().assess(
            normalizationResult: normalization,
            rawTranscript: rawText
        )

        #expect(normalization.suggestions.contains {
            $0.reason == VocoAutoApplyModelService.protectedTermGuardReason
        })
        #expect(assessment.route == .reviewSuggested)
        #expect(assessment.selectedCandidate == rawText)
        #expect(assessment.candidates.first == rawText)
        #expect(assessment.candidateLabels.first == "Recommended")
        #expect(assessment.candidates.contains(guardedText))
        #expect(assessment.labelForCandidate(at: assessment.candidates.firstIndex(of: guardedText) ?? -1) == "Guarded output")
    }

    @Test func bocoScopedReplacementAppliesToVoco() throws {
        let service = VocoAutoApplyModelService(
            modelURL: try writeFixture(ready: true),
            defaults: try temporaryDefaults()
        )

        let result = service.evaluate("你傳一個 prompt 去給那個 Boco 那邊。")
        #expect(result.outputText == "你傳一個 prompt 去給那個 Voco 那邊。")
        #expect(result.applied.map(\.policyId) == ["scoped-fixture-boco-voco"])
        #expect(result.guardBlocks.isEmpty)
    }

    @Test func protectedTermAllowlistGuardComesFromModelMetadata() throws {
        let service = VocoAutoApplyModelService(
            modelURL: try writeFixture(ready: true, includeProtectedTermGuards: false),
            defaults: try temporaryDefaults()
        )

        let result = service.evaluate("這個明德變更怪怪的。")
        #expect(result.outputText == "這個明德變更怪怪的。")
        #expect(result.guardBlocks.isEmpty)
        #expect(result.requiresReview == false)
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

    private func writeFixture(
        ready: Bool,
        scopedClaudeTarget: String = "Claude 的 OPUS 模型",
        includeProtectedTermGuards: Bool = true
    ) throws -> URL {
        let url = try temporaryDirectory().appendingPathComponent("small-auto-apply-fixture.json")
        try writeFixture(
            to: url,
            ready: ready,
            scopedClaudeTarget: scopedClaudeTarget,
            includeProtectedTermGuards: includeProtectedTermGuards
        )
        return url
    }

    private func writeFixture(
        to url: URL,
        ready: Bool,
        scopedClaudeTarget: String = "Claude 的 OPUS 模型",
        includeProtectedTermGuards: Bool = true
    ) throws {
        let data = try #require(
            fixtureJSON(
                ready: ready,
                scopedClaudeTarget: scopedClaudeTarget,
                includeProtectedTermGuards: includeProtectedTermGuards
            ).data(using: .utf8)
        )
        try data.write(to: url)
    }

    private func fixtureJSON(
        ready: Bool,
        scopedClaudeTarget: String = "Claude 的 OPUS 模型",
        includeProtectedTermGuards: Bool = true
    ) -> String {
        let protectedTermGuardsJSON = includeProtectedTermGuards ? """
          "protectedTermAllowlistGuards": [
            {
              "guardId": "protected-term-allowlist.mingde",
              "reason": "\(VocoAutoApplyModelService.protectedTermGuardReason)",
              "term": "明德",
              "allowedPhrases": ["明德捷運站", "明德水庫", "明德路", "施明德"]
            }
          ],
        """ : ""

        return """
        {
          "policyCounts": { "apply": 15, "suggest": 1, "replaced": 1 },
          "policyTypeCounts": { "exactTrainablePair": 2, "scopedReplacement": 15 },
          "safetyContract": [
            "exact trainable-pair policies may auto-apply only on normalized whole-utterance match",
            "Voco action commands such as 全部刪除 are blocked from text auto-apply training",
            "unresolved exact-input conflicts across source slices are blocked from the merged model"
          ],
          "mergedReplayReadiness": {
            "mergedAutoApplyModelReady": \(ready ? "true" : "false"),
            "failures": []
          },
        \(protectedTermGuardsJSON)
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
              "policyId": "scoped-fixture-mingde-recent-change",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "最明德變更",
              "targetText": "最近的變更",
              "scopedSourcePhrase": "最明德變更",
              "contextAliasesAny": [],
              "contextTokensAny": ["變更", "自動學習", "昨天晚上", "最近"],
              "contextRequired": true,
              "sourceSlices": ["controlEvidence"]
            },
            {
              "policyId": "scoped-fixture-boco-voco",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "Boco",
              "targetText": "Voco",
              "scopedSourcePhrase": "Boco",
              "contextAliasesAny": [],
              "contextTokensAny": [],
              "sourceSlices": ["controlEvidence"]
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
            },
            {
              "policyId": "replaced-fixture-github",
              "autoApplyMode": "replaced",
              "policyType": "scopedReplacement",
              "sourcePattern": "吉他",
              "targetText": "GitHub",
              "scopedSourcePhrase": "吉他",
              "contextAliasesAny": [],
              "contextTokensAny": ["GitHub", "repo", "push"],
              "contextRequired": true,
              "sourceSlices": ["controlEvidence"]
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
