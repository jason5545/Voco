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

    @Test func checkedInDemoAutoApplyFixtureLoadsAndAppliesExampleData() throws {
        let url = projectRootURL()
            .appendingPathComponent("examples/correction-model-demo/AutoApplyModels/full-db.auto-apply-model.json")
        let service = VocoAutoApplyModelService(modelURL: url, defaults: try temporaryDefaults())

        #expect(service.status.isAvailable == true)
        #expect(service.settingsToggleIsOn == true)
        #expect(service.settingsToggleIsEnabled == true)

        let exact = service.evaluate("open AI API key 要放在哪裡？")
        #expect(exact.outputText == "OpenAI API key 要放在哪裡？")
        #expect(exact.applied.map(\.policyId) == ["demo-exact-openai-question"])

        let scoped = service.evaluate("我用 VS code 開這個 repo。", context: "editor 開發 程式")
        #expect(scoped.outputText == "我用 VS Code 開這個 repo。")
        #expect(scoped.applied.map(\.policyId) == ["demo-scoped-vs-code"])

        let noContext = service.evaluate("我用 VS code 開這個 repo。", context: "一般聊天")
        #expect(noContext.outputText == "我用 VS code 開這個 repo。")
        #expect(noContext.applied.isEmpty)
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

    @Test func cjkBoundaryGuardedScopedReplacementDoesNotOverreachIntoContinuationWords() throws {
        let service = VocoAutoApplyModelService(
            modelURL: try writeFixture(ready: true),
            defaults: try temporaryDefaults()
        )

        let name = service.evaluate("剛剛有提到尖銳成。")
        #expect(name.outputText == "剛剛有提到簡瑞成。")
        #expect(name.applied.map(\.policyId) == ["scoped-fixture-jian-rui-cheng"])
        #expect(name.applied.first?.familyId == "name.jian-rui-cheng")
        #expect(name.applied.first?.sourceBoundaryMode == VocoAutoApplyModelService.cjkUnsafeContinuationBoundaryMode)

        let commonPhrase = service.evaluate("這個意見很尖銳成分很高")
        #expect(commonPhrase.outputText == "這個意見很尖銳成分很高")
        #expect(commonPhrase.applied.isEmpty)
    }

    @Test func indexedRuntimeV2ExactWholeUtteranceMatchesV1AndKeepsRuntimeGuards() throws {
        let v1 = VocoAutoApplyModelService(
            modelURL: try writeFixture(ready: true),
            defaults: try temporaryDefaults()
        )
        let v2 = VocoAutoApplyModelService(
            modelURL: try writeIndexedRuntimeV2Fixture(ready: true),
            defaults: try temporaryDefaults()
        )

        let v1Exact = v1.evaluate("W 零三的 VT 時間是什麼呢？")
        let v2Exact = v2.evaluate("W 零三的 VT 時間是什麼呢？")
        #expect(v2.status.isAvailable == true)
        #expect(v2Exact.outputText == v1Exact.outputText)
        #expect(v2Exact.applied.map(\.policyId) == v1Exact.applied.map(\.policyId))

        let manualOverrideInput = "不要動到原來的城市碼。"
        let v1ManualOverrideExact = v1.evaluate(manualOverrideInput)
        let v2ManualOverrideExact = v2.evaluate(manualOverrideInput)
        #expect(v1ManualOverrideExact.outputText == "不要動到原來的程式碼。")
        #expect(v1ManualOverrideExact.applied.map(\.policyId) == ["exact-fixture-manual-override-code"])
        #expect(v2ManualOverrideExact.outputText == v1ManualOverrideExact.outputText)
        #expect(v2ManualOverrideExact.applied.map(\.policyId) == v1ManualOverrideExact.applied.map(\.policyId))

        let scoped = v2.evaluate("我在測 Cloud 的 OPUS 模型")
        #expect(scoped.outputText == "我在測 Claude 的 OPUS 模型")
        #expect(scoped.applied.map(\.policyId) == ["scoped-fixture-claude"])

        let suggestion = v2.evaluate("答案給我匯入")
        #expect(suggestion.outputText == "答案給我匯入")
        #expect(suggestion.suggestions.map(\.policyId) == ["suggest-fixture-file-import"])

        let command = v2.evaluate("全部刪除")
        #expect(command.outputText == "全部刪除")
        #expect(command.applied.isEmpty)
        #expect(command.suggestions.isEmpty)

        let guarded = v2.evaluate("這個明德變更怪怪的。")
        #expect(guarded.outputText == "這個明德變更怪怪的。")
        #expect(guarded.guardBlocks.map(\.guardId) == ["protected-term-allowlist.mingde"])
    }

    @Test func malformedIndexedRuntimeV2FallsBackToReadableV1Payload() throws {
        let url = try temporaryDirectory().appendingPathComponent("v2-marker-with-v1-fallback.json")
        try Data(indexedRuntimeV2DecodeFailureWithV1FallbackJSON().utf8).write(to: url)
        let service = VocoAutoApplyModelService(modelURL: url, defaults: try temporaryDefaults())

        let exact = service.evaluate("W 零三的 VT 時間是什麼呢？")
        #expect(service.status.isAvailable == true)
        #expect(exact.outputText == "W零三的VT時間是什麼呢？")
        #expect(exact.applied.map(\.policyId) == ["exact-fixture-question"])
    }

    @Test func brokenIndexedRuntimeV2ReloadKeepsPreviousModel() throws {
        let url = try writeIndexedRuntimeV2Fixture(ready: true)
        let service = VocoAutoApplyModelService(modelURL: url, defaults: try temporaryDefaults())
        #expect(service.evaluate("W 零三的 VT 時間是什麼呢？").outputText == "W零三的VT時間是什麼呢？")

        try Data(brokenIndexedRuntimeV2FixtureJSON().utf8).write(to: url)
        service.reload()

        #expect(service.status.isAvailable == true)
        #expect(service.status.isDegraded == true)
        #expect(service.status.message == String(localized: "Model reload failed, using previous version"))
        #expect(service.evaluate("W 零三的 VT 時間是什麼呢？").outputText == "W零三的VT時間是什麼呢？")
    }

    @Test func largeSyntheticModelKeepsIndexedExactScopedSuggestAndGuardBehavior() throws {
        let url = try temporaryDirectory().appendingPathComponent("large-synthetic-auto-apply-fixture.json")
        try Data(largeSyntheticFixtureJSON(fillerExactCount: 6_000).utf8).write(to: url)
        let service = VocoAutoApplyModelService(modelURL: url, defaults: try temporaryDefaults())

        let exact = service.evaluate("Cloud 的 OPUS 模型")
        #expect(exact.outputText == "Exact whole utterance wins")
        #expect(exact.applied.map(\.policyId) == ["large-exact-cloud"])
        #expect(exact.suggestions.map(\.policyId) == ["large-suggest-cloud"])

        let scoped = service.evaluate("alpha beta")
        #expect(scoped.outputText == "gamma gamma")
        #expect(scoped.applied.map(\.policyId) == ["large-scoped-alpha-beta", "large-scoped-beta-gamma"])

        let contextLocked = service.evaluate("請幫我轉入文字", context: "Voco ASR 轉錄")
        #expect(contextLocked.outputText == "請幫我轉錄文字")
        #expect(contextLocked.applied.map(\.policyId) == ["large-scoped-context"])
        #expect(service.evaluate("請幫我轉入文字", context: "表格匯入").outputText == "請幫我轉入文字")

        let guarded = service.evaluate("這個明德變更怪怪的。")
        #expect(guarded.outputText == "這個明德變更怪怪的。")
        #expect(guarded.applied.isEmpty)
        #expect(guarded.guardBlocks.map(\.guardId) == ["large-protected-mingde"])

        let command = service.evaluate("全部刪除")
        #expect(command.outputText == "全部刪除")
        #expect(command.applied.isEmpty)
        #expect(command.suggestions.isEmpty)
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

    @Test func canonicalizationDoesNotCreateProtectedVocabularyPhoneticCandidate() async throws {
        try await requireLoadedPinyinDatabase()

        let service = VocoAutoApplyModelService(
            modelURL: try writeFixture(ready: true),
            defaults: try temporaryDefaults()
        )
        let canonicalizer = VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: service
        )
        let vocabulary = VocoCanonicalizationService.vocabularyTerms(from: ["明德"])
        let original = "所以你整體看我的過癮障礙到底到了什麼程度？我越來越懷疑自己比我自己想的嚴重了。"

        let result = canonicalizer.normalize(
            original,
            activeContextIDs: [],
            additionalTerms: vocabulary
        )

        #expect(result.normalizedText == original)
        #expect(result.replacements.isEmpty)
        #expect(result.suggestions.isEmpty)
    }

    @Test func canonicalizationDoesNotCommitAllowedProtectedPhraseWhenRawLacksProtectedTerm() async throws {
        try await requireLoadedPinyinDatabase()

        let service = VocoAutoApplyModelService(
            modelURL: try writeFixture(ready: true),
            defaults: try temporaryDefaults()
        )
        let canonicalizer = VocoCanonicalizationService(
            contextPacks: [],
            autoApplyModelService: service
        )
        let vocabulary = VocoCanonicalizationService.vocabularyTerms(from: ["明德水庫"])
        let original = "我們在民德水庫旁邊。"

        let result = canonicalizer.normalize(
            original,
            activeContextIDs: [],
            additionalTerms: vocabulary
        )

        #expect(result.normalizedText == original)
        #expect(result.replacements.isEmpty)
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

    @Test func policyProposalRankerArtifactIsShadowFixtureAndIgnoredByRuntime() throws {
        let root = try temporaryDirectory()
        let modelDirectory = root.appendingPathComponent("AutoApplyModels", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
        let compiledRuntimeModel = modelDirectory.appendingPathComponent(VocoAutoApplyModelService.modelFileName)
        try writeFixture(
            to: compiledRuntimeModel,
            ready: true,
            scopedClaudeTarget: "Compiled Runtime Claude 的 OPUS 模型"
        )
        let artifactDirectory = try writePolicyProposalRankerFixture(under: root)

        let manifest = try jsonObject(at: artifactDirectory.appendingPathComponent("dataset-manifest.json"))
        let report = try jsonObject(at: artifactDirectory.appendingPathComponent("proposal-ranker-report.json"))
        let safetyGate = try jsonObject(
            at: artifactDirectory
                .appendingPathComponent("proposal-release-gate-dry-run", isDirectory: true)
                .appendingPathComponent("proposal-safety-gate.report.json")
        )
        let manifestBoundary = try #require(manifest["safetyBoundary"] as? [String])
        let reportBoundary = try #require(report["safetyBoundary"] as? [String])

        #expect((manifest["intendedUse"] as? String)?.contains("not a Voco runtime model") == true)
        #expect((report["intendedUse"] as? String)?.contains("not a Voco runtime auto-apply model") == true)
        #expect(manifestBoundary.contains {
            $0.contains("full-db.auto-apply-model.json") && $0.localizedCaseInsensitiveContains("runtime")
        })
        #expect(reportBoundary.contains {
            $0.localizedCaseInsensitiveContains("Predicted apply") && $0.localizedCaseInsensitiveContains("proposal")
        })
        #expect(reportBoundary.contains {
            $0.localizedCaseInsensitiveContains("ReplayLab") && $0.localizedCaseInsensitiveContains("compiled")
        })
        let valid = try #require(report["valid"] as? [String: Any])
        let test = try #require(report["test"] as? [String: Any])
        #expect(valid["unsafeApplyFalsePositiveCount"] as? Int == 0)
        #expect(test["unsafeApplyFalsePositiveCount"] as? Int == 0)
        let rankerGate = try #require(safetyGate["rankerGate"] as? [String: Any])
        let candidateReplay = try #require(safetyGate["candidateReplay"] as? [String: Any])
        let rawInputReplay = try #require(safetyGate["rawInputReplay"] as? [String: Any])
        let activeModelDiff = try #require(safetyGate["activeModelDiff"] as? [String: Any])
        let readiness = try #require(safetyGate["readiness"] as? [String: Any])
        let runtimeBoundaryAudit = try #require(safetyGate["runtimeBoundaryAudit"] as? [String: Any])
        #expect(safetyGate["schema"] as? String == "voco.policy-proposal-safety-gate.v2")
        #expect(rankerGate["proposalCount"] as? Int == 4898)
        #expect(rankerGate["predictedApplyCount"] as? Int == 4524)
        #expect(rankerGate["acceptedForCompileCount"] as? Int == 4524)
        #expect(rankerGate["unsafeApplyFalsePositiveCount"] as? Int == 0)
        #expect(rankerGate["applyMissCount"] as? Int == 26)
        #expect(readiness["dryRunSafetyGatePass"] as? Bool == true)
        #expect(readiness["productionRuntimeAllowed"] as? Bool == false)
        #expect(readiness["releaseReady"] as? Bool == true)
        #expect((candidateReplay["unexpectedChanges"] as? [Any])?.isEmpty == true)
        #expect((rawInputReplay["unexpectedChanges"] as? [Any])?.isEmpty == true)
        #expect((candidateReplay["inheritedBaselineUnexpectedChanges"] as? [Any])?.count == 1)
        #expect((rawInputReplay["inheritedBaselineUnexpectedChanges"] as? [Any])?.count == 1)
        #expect((candidateReplay["acceptedManualCorpusChanges"] as? [Any])?.isEmpty == true)
        #expect((rawInputReplay["acceptedManualCorpusChanges"] as? [Any])?.isEmpty == true)
        #expect((readiness["blockers"] as? [String])?.isEmpty == true)
        #expect((readiness["warnings"] as? [String])?.contains {
            $0.contains("not an install approval")
        } == true)
        #expect(activeModelDiff["candidateIsSubsetOfActive"] as? Bool == true)
        #expect(activeModelDiff["candidateCoversActiveApplyPolicies"] as? Bool == true)
        #expect(activeModelDiff["droppedActiveApplyPolicyCount"] as? Int == 0)
        #expect(activeModelDiff["addedPolicyCount"] as? Int == 0)
        #expect(activeModelDiff["changedPolicyCount"] as? Int == 0)
        #expect(runtimeBoundaryAudit["installOrActivateCommandEmitted"] as? Bool == false)
        #expect(runtimeBoundaryAudit["joblibActivationAllowed"] as? Bool == false)
        #expect(runtimeBoundaryAudit["rankerModelIsRuntimeModel"] as? Bool == false)
        #expect(FileManager.default.fileExists(
            atPath: artifactDirectory.appendingPathComponent("proposal-ranker-model.joblib").path
        ))

        let service = VocoAutoApplyModelService(modelURL: compiledRuntimeModel, defaults: try temporaryDefaults())
        let result = service.evaluate("我在測 Cloud 的 OPUS 模型")

        #expect(service.status.isAvailable == true)
        #expect(service.status.modelURL == compiledRuntimeModel)
        #expect(result.outputText == "我在測 Compiled Runtime Claude 的 OPUS 模型")
        #expect(result.applied.map(\.policyId) == ["scoped-fixture-claude"])
    }

    @Test func gitignoreProtectsProductionAndLocalModelPaths() throws {
        let gitignore = try String(
            contentsOf: projectRootURL().appendingPathComponent(".gitignore"),
            encoding: .utf8
        )

        #expect(gitignore.contains("LocalModels/"))
        #expect(gitignore.contains("Support/AutoApplyModels/"))
        #expect(gitignore.contains("full-db.auto-apply-model.json"))
    }

    @Test func workerSyncInstallsValidatedRemoteModelAndUpdatesStatus() async throws {
        let root = try temporaryDirectory()
        let modelDirectory = root.appendingPathComponent("AutoApplyModels", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
        let activeModel = modelDirectory.appendingPathComponent(VocoAutoApplyModelService.modelFileName)
        try writeFixture(to: activeModel, ready: true)

        let remoteData = try #require(fixtureJSON(ready: true, scopedClaudeTarget: "Remote Claude 的 OPUS 模型").data(using: .utf8))
        let remoteSha = VocoAutoApplyModelService.sha256Hex(for: remoteData)
        let manifestData = try #require(workerManifestJSON(modelSha: remoteSha).data(using: .utf8))
        let recorder = WorkerSyncRequestRecorder()
        let client = VocoAutoApplyWorkerSyncClient(
            workerURL: URL(string: "https://worker.example")!,
            transport: { request in
                recorder.paths.append(request.url?.path ?? "")
                let response = try Self.httpResponse(for: request, statusCode: 200)
                switch request.url?.path {
                case "/v1/auto-apply/manifest":
                    return (manifestData, response)
                case "/v1/auto-apply/models/\(remoteSha)":
                    return (remoteData, response)
                default:
                    Issue.record("Unexpected Worker request: \(request.url?.absoluteString ?? "")")
                    return (Data(), response)
                }
            }
        )
        let service = VocoAutoApplyModelService(
            modelURL: activeModel,
            defaults: try temporaryDefaults(),
            workerSyncClient: client,
            workerSyncKeyProvider: { "secret" },
            modelBackupRetention: 2
        )

        let outcome = await service.syncFromWorker()
        let result = service.evaluate("我在測 Cloud 的 OPUS 模型")
        let backups = try FileManager.default.contentsOfDirectory(
            at: modelDirectory.appendingPathComponent("Backups", isDirectory: true),
            includingPropertiesForKeys: nil
        )

        #expect(outcome.state == .installed)
        #expect(result.outputText == "我在測 Remote Claude 的 OPUS 模型")
        #expect(service.status.remoteLatestSha256 == remoteSha)
        #expect(service.status.remoteIsInSync == true)
        #expect(backups.count == 1)
        #expect(recorder.paths == ["/v1/auto-apply/manifest", "/v1/auto-apply/models/\(remoteSha)"])
    }

    @Test func workerSync404KeepsLocalLastKnownGoodModel() async throws {
        let activeModel = try writeFixture(ready: true)
        let localSha = VocoAutoApplyModelService.sha256HexForFileIfExists(activeModel)
        let recorder = WorkerSyncRequestRecorder()
        let client = VocoAutoApplyWorkerSyncClient(
            workerURL: URL(string: "https://worker.example")!,
            transport: { request in
                recorder.paths.append(request.url?.path ?? "")
                let response = try Self.httpResponse(for: request, statusCode: 404)
                return (Data("not found".utf8), response)
            }
        )
        let service = VocoAutoApplyModelService(
            modelURL: activeModel,
            defaults: try temporaryDefaults(),
            workerSyncClient: client,
            workerSyncKeyProvider: { "wrong-secret" }
        )

        let outcome = await service.syncFromWorker()
        let result = service.evaluate("我在測 Cloud 的 OPUS 模型")

        #expect(outcome.state == .keptLocal)
        #expect(service.status.isAvailable == true)
        #expect(service.status.localModelSha256 == localSha)
        #expect(service.status.remoteLatestSha256 == nil)
        #expect(result.outputText == "我在測 Claude 的 OPUS 模型")
        #expect(recorder.paths == ["/v1/auto-apply/manifest"])
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

    private func writeIndexedRuntimeV2Fixture(ready: Bool) throws -> URL {
        let url = try temporaryDirectory().appendingPathComponent("small-auto-apply-runtime-v2-fixture.json")
        try Data(indexedRuntimeV2FixtureJSON(ready: ready).utf8).write(to: url)
        return url
    }

    private func indexedRuntimeV2FixtureJSON(ready: Bool) -> String {
        """
        {
          "schemaVersion": 1,
          "runtimeSchemaVersion": \(VocoAutoApplyModelService.supportedRuntimeSchemaVersion),
          "modelFormat": "voco-auto-apply-runtime-indexed-v2",
          "autoApplyModelVersion": "indexed-v2-fixture",
          "generatedAt": "2026-06-20T00:00:00Z",
          "policyCounts": { "apply": 5, "suggest": 1, "replaced": 1, "blocked": 0 },
          "policyTypeCounts": { "exactTrainablePair": 3, "scopedReplacement": 3 },
          "safetyContract": [
            "exact trainable-pair policies may auto-apply only on normalized whole-utterance match",
            "Voco action commands such as 全部刪除 are blocked from text auto-apply training"
          ],
          "mergedReplayReadiness": {
            "mergedAutoApplyModelReady": \(ready ? "true" : "false"),
            "failures": []
          },
          "actionCommandGuards": [
            { "surface": "全部刪除" },
            { "surface": "全部删除" }
          ],
          "protectedTermAllowlistGuards": [
            {
              "guardId": "protected-term-allowlist.mingde",
              "reason": "\(VocoAutoApplyModelService.protectedTermGuardReason)",
              "term": "明德",
              "allowedPhrases": ["明德捷運站", "明德水庫", "明德路", "施明德"]
            }
          ],
          "exactApplyPolicyByStrictKey": {
            "w 零三的 vt 時間是什麼呢?": {
              "policyId": "exact-fixture-question",
              "sourcePattern": "W 零三的 VT 時間是什麼呢？",
              "targetText": "W零三的VT時間是什麼呢？",
              "sourceSlices": ["rerawPre12022"]
            },
            "不要動到原來的城市碼。": {
              "policyId": "exact-fixture-manual-override-code",
              "sourcePattern": "不要動到原來的城市碼。",
              "targetText": "不要動到原來的程式碼。",
              "sourceSlices": ["currentRaw"]
            },
            "全部刪除": {
              "policyId": "exact-action-command-delete-all",
              "sourcePattern": "全部刪除",
              "targetText": "全部刪掉",
              "sourceSlices": ["synthetic"]
            }
          },
          "scopedApplyPolicies": [
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
            }
          ],
          "suggestPolicies": [
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

    private func indexedRuntimeV2DecodeFailureWithV1FallbackJSON() -> String {
        var json = fixtureJSON(ready: true)
        json = json.replacingOccurrences(
            of: "{",
            with: """
            {
              "runtimeSchemaVersion": 2,
              "modelFormat": "voco-auto-apply-runtime-indexed-v2",
              "exactApplyPolicyByStrictKey": "not-a-map",
            """,
            options: [],
            range: json.startIndex..<json.index(after: json.startIndex)
        )
        return json
    }

    private func workerManifestJSON(modelSha: String) -> String {
        """
        {
          "schema": "voco.auto-apply-worker-sync-manifest.v1",
          "phase": "\(VocoAutoApplyWorkerSyncClient.phase)",
          "version": "worker-test",
          "createdAt": "2026-06-24T00:00:00Z",
          "source": "replaylab",
          "modelFileName": "\(VocoAutoApplyModelService.modelFileName)",
          "modelSha256": "\(modelSha)",
          "schemaVersion": null,
          "runtimeSchemaVersion": null,
          "autoApplyModelVersion": "worker-test-model",
          "generatedAt": "2026-06-24T00:00:00Z",
          "policyCounts": { "apply": 17, "suggest": 1, "replaced": 1 },
          "policyTypeCounts": { "exactTrainablePair": 3, "scopedReplacement": 16 },
          "readiness": {
            "mergedAutoApplyModelReady": true,
            "failures": []
          },
          "privacy": {
            "transcriptUploadAllowed": false,
            "evidenceUploadAllowed": false,
            "workerDecisionAllowed": false
          }
        }
        """
    }

    private func brokenIndexedRuntimeV2FixtureJSON() -> String {
        """
        {
          "schemaVersion": 1,
          "runtimeSchemaVersion": 2,
          "modelFormat": "voco-auto-apply-runtime-indexed-v2",
          "mergedReplayReadiness": {
            "mergedAutoApplyModelReady": true
          },
          "exactApplyPolicyByStrictKey": "not-a-map"
        }
        """
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
          "policyCounts": { "apply": 16, "suggest": 1, "replaced": 1 },
          "policyTypeCounts": { "exactTrainablePair": 3, "scopedReplacement": 15 },
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
              "policyId": "exact-fixture-manual-override-code",
              "autoApplyMode": "apply",
              "policyType": "exactTrainablePair",
              "exactInputRequired": true,
              "inputStrictKey": "不要動到原來的城市碼。",
              "sourcePattern": "不要動到原來的城市碼。",
              "targetText": "不要動到原來的程式碼。",
              "reviewGateConflictRows": [12606],
              "manualOverrideRows": [12606],
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
              "policyId": "scoped-fixture-jian-rui-cheng",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "尖銳成",
              "targetText": "簡瑞成",
              "scopedSourcePhrase": "尖銳成",
              "contextAliasesAny": [],
              "contextTokensAny": [],
              "sourceSlices": ["manualControlPlane", "migratedPCTSeed"],
              "sourceBoundaryMode": "\(VocoAutoApplyModelService.cjkUnsafeContinuationBoundaryMode)",
              "familyId": "name.jian-rui-cheng",
              "familyRole": "alias",
              "migrationSource": "migrated-pct-seed"
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

    private func largeSyntheticFixtureJSON(fillerExactCount: Int) -> String {
        let exactInput = "Cloud 的 OPUS 模型"
        var policies = (0..<fillerExactCount).map { index in
            """
            {
              "policyId": "large-filler-exact-\(index)",
              "autoApplyMode": "apply",
              "policyType": "exactTrainablePair",
              "exactInputRequired": true,
              "inputStrictKey": "synthetic exact \(index)",
              "sourcePattern": "synthetic exact \(index)",
              "targetText": "synthetic target \(index)",
              "sourceSlices": ["synthetic"]
            }
            """
        }

        policies.append(
            """
            {
              "policyId": "large-exact-cloud",
              "autoApplyMode": "apply",
              "policyType": "exactTrainablePair",
              "exactInputRequired": true,
              "inputStrictKey": "\(VocoAutoApplyModelService.strictTextKey(exactInput))",
              "sourcePattern": "\(exactInput)",
              "targetText": "Exact whole utterance wins",
              "sourceSlices": ["synthetic"]
            }
            """
        )
        policies.append(
            """
            {
              "policyId": "large-scoped-cloud",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "Cloud",
              "targetText": "Claude",
              "scopedSourcePhrase": "Cloud",
              "contextAliasesAny": [],
              "contextTokensAny": [],
              "sourceSlices": ["synthetic"]
            }
            """
        )
        policies.append(
            """
            {
              "policyId": "large-scoped-alpha-beta",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "alpha",
              "targetText": "beta",
              "scopedSourcePhrase": "alpha",
              "contextAliasesAny": [],
              "contextTokensAny": [],
              "sourceSlices": ["synthetic"]
            }
            """
        )
        policies.append(
            """
            {
              "policyId": "large-scoped-beta-gamma",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "beta",
              "targetText": "gamma",
              "scopedSourcePhrase": "beta",
              "contextAliasesAny": [],
              "contextTokensAny": [],
              "sourceSlices": ["synthetic"]
            }
            """
        )
        policies.append(
            """
            {
              "policyId": "large-scoped-context",
              "autoApplyMode": "apply",
              "policyType": "scopedReplacement",
              "sourcePattern": "轉入",
              "targetText": "轉錄",
              "scopedSourcePhrase": "轉入",
              "contextAliasesAny": [],
              "contextTokensAny": ["ASR", "轉錄", "Voco"],
              "contextRequired": true,
              "sourceSlices": ["synthetic"]
            }
            """
        )
        policies.append(
            """
            {
              "policyId": "large-suggest-cloud",
              "autoApplyMode": "suggest",
              "policyType": "scopedReplacement",
              "sourcePattern": "\(exactInput)",
              "targetText": "Claude 的 OPUS 模型",
              "scopedSourcePhrase": "\(exactInput)",
              "contextAliasesAny": [],
              "contextTokensAny": [],
              "sourceSlices": ["synthetic"]
            }
            """
        )

        return """
        {
          "policyCounts": { "apply": \(fillerExactCount + 5), "suggest": 1, "replaced": 0 },
          "policyTypeCounts": { "exactTrainablePair": \(fillerExactCount + 1), "scopedReplacement": 5 },
          "safetyContract": [
            "exact trainable-pair policies may auto-apply only on normalized whole-utterance match",
            "Voco action commands such as 全部刪除 are blocked from text auto-apply training"
          ],
          "mergedReplayReadiness": {
            "mergedAutoApplyModelReady": true,
            "failures": []
          },
          "protectedTermAllowlistGuards": [
            {
              "guardId": "large-protected-mingde",
              "reason": "\(VocoAutoApplyModelService.protectedTermGuardReason)",
              "term": "明德",
              "allowedPhrases": ["明德水庫"]
            }
          ],
          "policies": [
            \(policies.joined(separator: ",\n"))
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

    private func requireLoadedPinyinDatabase() async throws {
        for _ in 0..<100 {
            if PinyinDatabase.shared.isLoaded { return }
            try await Task.sleep(nanoseconds: 50_000_000)
        }

        try #require(PinyinDatabase.shared.isLoaded)
    }

    private func temporaryDefaults() throws -> UserDefaults {
        let suiteName = "VocoAutoApplyModelServiceTests-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
        return defaults
    }

    private func writePolicyProposalRankerFixture(under root: URL) throws -> URL {
        let artifactDirectory = root
            .appendingPathComponent("artifacts", isDirectory: true)
            .appendingPathComponent("policy-proposal-model-20260613-active-122458", isDirectory: true)
        try FileManager.default.createDirectory(at: artifactDirectory, withIntermediateDirectories: true)

        let manifest = """
        {
          "datasetType": "post-asr-policy-proposal-decision",
          "intendedUse": "train/evaluate a proposal classifier or ranker; not a Voco runtime model",
          "counts": {
            "proposals": 4898,
            "decisions": { "apply": 4550, "block": 93, "abstain": 255 },
            "splits": { "train": 3938, "valid": 483, "test": 477 }
          },
          "mergedModel": "artifacts/active-auto-apply-model-snapshots/20260613-122458-current-active-after-13168-cloi-cli/full-db.auto-apply-model.json",
          "safetyBoundary": [
            "Rows are training/evaluation examples for proposal decisions only.",
            "Voco runtime must continue to load compiled full-db.auto-apply-model.json, not model outputs.",
            "A generated proposal must pass replay gates before it can be compiled into runtime JSON."
          ]
        }
        """
        let report = """
        {
          "applyThreshold": 0.6,
          "datasetDir": "artifacts/policy-proposal-model-20260613-active-122458",
          "intendedUse": "rank/classify post-ASR policy proposals before replay; not a Voco runtime auto-apply model",
          "labels": ["apply", "suggest", "block", "abstain"],
          "modelType": "tfidf-charword-logistic-regression-policy-proposal-ranker",
          "safetyBoundary": [
            "Predicted apply is only a proposal decision.",
            "A generated proposal must pass ReplayLab gates before it is compiled into runtime JSON.",
            "The current dataset has only three suggest examples, all in train; suggest metrics are not meaningful yet."
          ],
          "valid": { "rows": 483, "unsafeApplyFalsePositiveCount": 0, "applyMissCount": 6 },
          "test": { "rows": 477, "unsafeApplyFalsePositiveCount": 0, "applyMissCount": 2 }
        }
        """
        try Data(manifest.utf8).write(to: artifactDirectory.appendingPathComponent("dataset-manifest.json"))
        try Data(report.utf8).write(to: artifactDirectory.appendingPathComponent("proposal-ranker-report.json"))
        try Data([0x80, 0x04, 0x70, 0x72, 0x6f, 0x70, 0x6f, 0x73, 0x61, 0x6c]).write(
            to: artifactDirectory.appendingPathComponent("proposal-ranker-model.joblib")
        )
        let safetyGateDirectory = artifactDirectory.appendingPathComponent("proposal-release-gate-dry-run", isDirectory: true)
        try FileManager.default.createDirectory(at: safetyGateDirectory, withIntermediateDirectories: true)
        let safetyGate = """
        {
          "schema": "voco.policy-proposal-safety-gate.v2",
          "rankerGate": {
            "proposalCount": 4898,
            "predictedApplyCount": 4524,
            "acceptedForCompileCount": 4524,
            "unsafeApplyFalsePositiveCount": 0,
            "applyMissCount": 26
          },
          "candidateReplay": {
            "readiness": { "autoApplyModelReady": true },
            "sentinelFailures": [],
            "unexpectedChanges": [],
            "inheritedBaselineUnexpectedChanges": [{ "rowPk": 12291 }],
            "acceptedManualCorpusChanges": []
          },
          "rawInputReplay": {
            "readiness": { "rawInputReplayPass": true },
            "sentinelFailures": [],
            "unexpectedChanges": [],
            "inheritedBaselineUnexpectedChanges": [{ "rowPk": 12291 }],
            "acceptedManualCorpusChanges": []
          },
          "activeModelDiff": {
            "activePolicyCounts": { "apply": 4550, "blocked": 1, "replaced": 17 },
            "candidatePolicyCounts": { "apply": 4550, "blocked": 1, "replaced": 17 },
            "policyCountDelta": { "apply": 0, "blocked": 0, "replaced": 0 },
            "addedPolicyCount": 0,
            "removedPolicyCount": 0,
            "changedPolicyCount": 0,
            "candidateCoversActiveApplyPolicies": true,
            "droppedActiveApplyPolicyCount": 0,
            "droppedActiveApplyPolicyIds": [],
            "candidateIsSubsetOfActive": true
          },
          "readiness": {
            "dryRunSafetyGatePass": true,
            "productionRuntimeAllowed": false,
            "releaseReady": true,
            "blockers": [],
            "warnings": [
              "dry-run candidate is not an install approval",
              "ranker artifact is evaluated only as proposal/shadow fixture",
              "suggest has no valid/test support; do not treat suggest as a release signal"
            ]
          },
          "runtimeBoundaryAudit": {
            "candidateModelFilename": "full-db.auto-apply-model.json",
            "candidateModelFilenameAllowed": true,
            "installOrActivateCommandEmitted": false,
            "joblibActivationAllowed": false,
            "rankerModelIsRuntimeModel": false,
            "productionRuntimeAllowed": false
          }
        }
        """
        try Data(safetyGate.utf8).write(to: safetyGateDirectory.appendingPathComponent("proposal-safety-gate.report.json"))
        return artifactDirectory
    }

    private func jsonObject(at url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func projectRootURL() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static func httpResponse(for request: URLRequest, statusCode: Int) throws -> HTTPURLResponse {
        guard let url = request.url,
              let response = HTTPURLResponse(
                url: url,
                statusCode: statusCode,
                httpVersion: nil,
                headerFields: nil
              )
        else {
            throw WorkerSyncTestError.invalidHTTPResponse
        }
        return response
    }
}

private final class WorkerSyncRequestRecorder {
    var paths: [String] = []
}

private enum WorkerSyncTestError: Error {
    case invalidHTTPResponse
}
