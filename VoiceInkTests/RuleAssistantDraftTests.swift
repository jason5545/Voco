import Foundation
import Testing
@testable import Voco

@Suite(.serialized)
struct RuleAssistantDraftParsingTests {
    @Test func locateMultipleObjects() {
        let answer = """
        第一則：{"eventType": "correction", "sourceText": "小振", "targetText": "小鎮"}
        再來：{"eventType": "correction", "sourceText": "甲", "targetText": "乙"}
        """
        let located = RuleAssistantDraft.locateAllJSON(in: answer)
        #expect(located.count == 2)
        let drafts = RuleAssistantDraft.parseAll(located)
        #expect(drafts.count == 2)
        #expect(drafts[0].sourceText == "小振")
        #expect(drafts[1].targetText == "乙")
    }

    @Test func draftsArrayWrapper() {
        let answer = """
        {"drafts": [
            {"eventType": "correction", "sourceText": "a", "targetText": "b"},
            {"eventType": "correction", "sourceText": "c", "targetText": "d"}
        ]}
        """
        let drafts = RuleAssistantDraft.parseAll(RuleAssistantDraft.locateAllJSON(in: answer))
        #expect(drafts.count == 2)
    }

    @Test func fencedDraftIsFound() {
        let answer = """
        我建議這條規則：
        ```json
        {"eventType": "correction", "sourceText": "失重", "targetText": "釋出"}
        ```
        """
        let drafts = RuleAssistantDraft.parseAll(RuleAssistantDraft.locateAllJSON(in: answer))
        #expect(drafts.count == 1)
        #expect(drafts[0].targetText == "釋出")
    }

    @Test func parseAllCapsAtEight() {
        let entries = (0..<12).map { #"{"eventType": "correction", "sourceText": "s\#($0)", "targetText": "t\#($0)"}"# }
        let answer = entries.joined(separator: " ")
        let drafts = RuleAssistantDraft.parseAll(RuleAssistantDraft.locateAllJSON(in: answer))
        #expect(drafts.count == RuleAssistantConstants.maxDraftsPerAnswer)
    }

    @Test func invalidJsonObjectsAreSkipped() {
        let answer = #"{"not": "a draft"} {"eventType": "correction", "sourceText": "a", "targetText": "b"}"#
        let drafts = RuleAssistantDraft.parseAll(RuleAssistantDraft.locateAllJSON(in: answer))
        #expect(drafts.count == 1)
    }

    @Test func draftWrappedUnderDraftKey() {
        let parsed = RuleAssistantDraft.parse([
            "draft": ["eventType": "correction", "sourceText": "a", "targetText": "b"],
        ])
        #expect(parsed?.sourceText == "a")
    }
}

@Suite(.serialized)
struct RuleAssistantQuestionParseTests {
    @Test func nestedQuestionWithCandidatesAndIds() {
        let parsed = RuleAssistantQuestion.parse([
            "question": [
                "id": "q7",
                "prompt": "哪個錯？",
                "multiSelect": true,
                "options": [
                    ["id": "a", "label": "西賴 → CLI", "detail": "程式語境", "surface": "西賴", "target": "CLI"],
                    ["label": "這筆沒錯"],
                ],
            ] as [String: Any],
        ])
        #expect(parsed?.id == "q7")
        #expect(parsed?.multiSelect == true)
        #expect(parsed?.options.count == 2)
        #expect(parsed?.options[0].isCandidate == true)
        #expect(parsed?.options[0].detail == "程式語境")
        // Missing ids get letters by position; existing ids are kept.
        #expect(parsed?.options[1].id == "b")
        #expect(parsed?.options[1].isCandidate == false)
    }

    @Test func bareShapeWithStringOptionsAndQuestionKeyAsPrompt() {
        let parsed = RuleAssistantQuestion.parse([
            "question": "A 還是 B？",
            "options": ["A", "B"],
        ])
        #expect(parsed?.prompt == "A 還是 B？")
        #expect(parsed?.multiSelect == false)
        #expect(parsed?.options.map(\.id) == ["a", "b"])
        #expect(parsed?.options.map(\.label) == ["A", "B"])
    }

    @Test func duplicateIdsAreMadeUnique() {
        let parsed = RuleAssistantQuestion.parse([
            "prompt": "?",
            "options": [["id": "a", "label": "1"], ["id": "a", "label": "2"]],
        ])
        #expect(parsed?.options.map(\.id) == ["a", "b1"])
    }

    @Test func malformedQuestionsFailClosed() {
        #expect(RuleAssistantQuestion.parse(["question": ["prompt": "x", "options": [] as [Any]] as [String: Any]]) == nil)
        #expect(RuleAssistantQuestion.parse(["question": ["options": [["label": "x"]]] as [String: Any]]) == nil)
        #expect(RuleAssistantQuestion.parse(["question": ["prompt": "x", "options": [["detail": "no label"]]] as [String: Any]]) == nil)
        #expect(RuleAssistantQuestion.parse(["eventType": "correction", "sourceText": "a", "targetText": "b"]) == nil)
        let long = String(repeating: "很", count: 401)
        #expect(RuleAssistantQuestion.parse(["prompt": long, "options": ["x"]]) == nil)
    }

    @Test func optionsWithTheSameSurfaceAndTargetCollapseToOne() {
        let parsed = RuleAssistantQuestion.parse([
            "prompt": "?",
            "options": [
                ["id": "a", "label": "資料架 → 資料夾（只改這句）", "surface": "資料架", "target": "資料夾"],
                ["id": "b", "label": "資料架 → 資料夾（語境限定）", "surface": "資料架", "target": "資料夾"],
                ["id": "c", "label": "資料架 → 資料夾（任何語境）", "surface": "資料架", "target": "資料夾"],
                ["id": "d", "label": "這筆沒錯"],
                ["id": "e", "label": "資料架 → 資料庫", "surface": "資料架", "target": "資料庫"],
            ],
        ])
        #expect(parsed?.options.map(\.id) == ["a", "d", "e"])
    }

    @Test func optionsAreCappedAtEightAndDraftsIgnoreQuestions() {
        let options = (1...12).map { ["label": "選項 \($0)"] }
        let parsed = RuleAssistantQuestion.parse(["prompt": "?", "options": options])
        #expect(parsed?.options.count == 8)
        let answer = "說明\n" + RuleAssistantTestJSON.string(["question": ["prompt": "?", "options": ["a"]] as [String: Any]])
            + "\n" + RuleAssistantTestJSON.string(["eventType": "correction", "sourceText": "a", "targetText": "b"])
        let located = RuleAssistantDraft.locateAllJSON(in: answer)
        #expect(RuleAssistantDraft.parseAll(located).count == 1)
        #expect(RuleAssistantQuestion.parseFirst(located)?.prompt == "?")
    }
}

@Suite(.serialized)
struct RuleAssistantDraftSafetyTests {
    private func draft(
        _ type: String,
        source: String? = nil,
        target: String? = nil,
        pattern: String? = nil,
        familyId: String? = nil,
        aliases: [String] = [],
        contextTokens: [String] = [],
        policyId: String? = nil,
        from: String? = nil,
        to: String? = nil,
        reason: String? = nil,
        disposition: String? = nil
    ) -> RuleAssistantDraft {
        RuleAssistantDraft(
            eventType: type,
            sourceText: source,
            targetText: target,
            sourcePattern: pattern,
            familyId: familyId,
            aliases: aliases,
            contextTokensAny: contextTokens,
            contextAliasesAny: [],
            policyId: policyId,
            fromFamilyId: from,
            toFamilyId: to,
            reason: reason,
            disposition: disposition
        )
    }

    @Test func correctionRequiresDistinctSourceAndTarget() {
        #expect(draft("correction", source: "小振", target: "小鎮").isSafeForWrite())
        #expect(!draft("correction", source: "小振", target: "小振").isSafeForWrite())
        #expect(!draft("correction", source: "  ", target: "小鎮").isSafeForWrite())
        #expect(!draft("correction", source: nil, target: "小鎮").isSafeForWrite())
    }

    @Test func contextLockedNeedsContext() {
        #expect(draft("contextLockedRule", target: "拔草測", pattern: "拔草", contextTokens: ["墓地"]).isSafeForWrite())
        #expect(!draft("contextLockedRule", target: "拔草測", pattern: "拔草").isSafeForWrite())
    }

    @Test func replacementRuleRejectsIdenticalPatternAndTarget() {
        #expect(draft("replacementRule", target: "釋出", pattern: "失重").isSafeForWrite())
        #expect(!draft("replacementRule", target: "失重", pattern: "失重").isSafeForWrite())
    }

    @Test func replacementFamilyNeedsAliasesDistinctFromTarget() {
        #expect(draft("replacementFamily", target: "小鎮", familyId: "fam1", aliases: ["小振"]).isSafeForWrite())
        #expect(!draft("replacementFamily", target: "小鎮", familyId: "fam1", aliases: ["小鎮"]).isSafeForWrite())
        #expect(!draft("replacementFamily", target: "小鎮", familyId: "fam1").isSafeForWrite())
    }

    @Test func tombstoneNeedsReasonAndDisposition() {
        #expect(draft("tombstone", target: "釋出", pattern: "失重", reason: "錯規則", disposition: "replaced").isSafeForWrite())
        #expect(!draft("tombstone", target: "釋出", pattern: "失重", reason: "錯規則").isSafeForWrite())
        #expect(!draft("tombstone", target: "釋出", pattern: "失重", reason: "錯規則", disposition: "other").isSafeForWrite())
    }

    @Test func moveAliasNeedsDestinationFamily() {
        #expect(draft("moveAliasToFamily", pattern: "小振", from: "fam1", to: "fam2").isSafeForWrite())
        #expect(!draft("moveAliasToFamily", pattern: "小振", from: "fam1", to: "fam1").isSafeForWrite())
        #expect(!draft("moveAliasToFamily", pattern: "小振").isSafeForWrite())
    }

    @Test func mergeNeedsTwoDistinctFamilies() {
        #expect(draft("mergeReplacementFamilies", from: "fam1", to: "fam2").isSafeForWrite())
        #expect(!draft("mergeReplacementFamilies", from: "fam1", to: "fam1").isSafeForWrite())
    }

    @Test func replaceContextLockNeedsPolicyIdAndTokens() {
        #expect(draft("replaceContextLockedRule", target: "Mac", pattern: "麥克", contextTokens: ["硬體"], policyId: "manual-context-1", reason: "加語境").isSafeForWrite())
        #expect(!draft("replaceContextLockedRule", target: "Mac", pattern: "麥克", contextTokens: ["硬體"], reason: "加語境").isSafeForWrite())
        #expect(!draft("replaceContextLockedRule", target: "Mac", pattern: "麥克", policyId: "manual-context-1", reason: "加語境").isSafeForWrite())
        #expect(!draft("replaceContextLockedRule", target: "Mac", pattern: "麥克", contextTokens: ["硬體"], policyId: "manual-context-1").isSafeForWrite())
    }

    @MainActor
    @Test func replaceContextLockIsATransactionButNotAFamilyOne() {
        let entry = draft("replaceContextLockedRule", target: "Mac", pattern: "麥克", contextTokens: ["硬體"], policyId: "manual-context-1", reason: "加語境")
        #expect(entry.isTransaction)
        #expect(!entry.isFamilyTransaction)
        #expect(entry.writeToolName() == "replace_auto_apply_context_locked_rule")
        // Allowed from a 語境限定 choice; auto-guess (scan) still refuses it.
        #expect(RuleAssistantSession.gateReason(for: entry, kind: .choice(candidateChosen: true)) == nil)
        #expect(RuleAssistantSession.gateReason(for: entry, kind: .scan) != nil)
        #expect(RuleAssistantSession.gateReason(for: entry, kind: .manual) == nil)
    }

    @MainActor
    @Test func replacePlanLineCountsAsAPlan() {
        #expect(RuleAssistantSession.missingJSONNudge(answer: "[replace] manual-context-1 → 麥克 → Mac", kind: .manual) == .draft)
    }

    @Test func unknownTypeIsNeverSafe() {
        #expect(!draft("deleteEverything", source: "a", target: "b").isSafeForWrite())
    }

    @Test func oversizedFieldsAreRefused() {
        let huge = String(repeating: "好", count: 2_001)
        #expect(!draft("correction", source: huge, target: "b").isSafeForWrite())
    }

    @MainActor
    @Test func rescopeGateRequiresExactSubstitutionAndScopeType() {
        let source = "但是你的思維另有說。"
        let target = "但是你的思維鏈有說。"
        let broad = draft("replacementRule", target: "思維鏈有說", pattern: "思維另有說")
        #expect(RuleAssistantSession.gateReason(for: broad, kind: .rescope(scope: .broad, sourceText: source, targetText: target)) == nil)
        let shorter = draft("replacementRule", target: "思維鏈有", pattern: "思維另有")
        #expect(RuleAssistantSession.gateReason(for: shorter, kind: .rescope(scope: .broad, sourceText: source, targetText: target)) == nil)
        let wrong = draft("replacementRule", target: "思維鏈", pattern: "思維")
        #expect(RuleAssistantSession.gateReason(for: wrong, kind: .rescope(scope: .broad, sourceText: source, targetText: target))?.contains("套回原句") == true)
        let family = draft("replacementFamily", target: "思維鏈有說", familyId: "f1", aliases: ["思維另有說"])
        #expect(RuleAssistantSession.gateReason(for: family, kind: .rescope(scope: .broad, sourceText: source, targetText: target))?.contains("只接受 replacementRule") == true)
        let correction = draft("correction", source: source, target: target)
        #expect(RuleAssistantSession.gateReason(for: correction, kind: .rescope(scope: .broad, sourceText: source, targetText: target)) == nil)

        let latinSource = "整個麥克鍵盤就會泛油光"
        let latinTarget = "整個 Mac 鍵盤就會泛油光"
        let latin = draft("replacementRule", target: "Mac 鍵盤", pattern: "麥克鍵盤")
        #expect(RuleAssistantSession.gateReason(for: latin, kind: .rescope(scope: .broad, sourceText: latinSource, targetText: latinTarget)) == nil)
        let missingBoundarySpace = draft("replacementRule", target: "Mac 鍵盤", pattern: "麥克鍵盤")
        #expect(
            RuleAssistantSession.gateReason(
                for: missingBoundarySpace,
                kind: .rescope(scope: .broad, sourceText: latinSource, targetText: "整個Mac 鍵盤就會泛油光")
            )?.contains("套回原句後只差中英之間的空格") == true
        )
        // The message has to tell the model what to change, not just that it is wrong.
        let spacingReason = RuleAssistantSession.gateReason(
            for: missingBoundarySpace,
            kind: .rescope(scope: .broad, sourceText: latinSource, targetText: "整個Mac 鍵盤就會泛油光")
        )
        #expect(spacingReason?.contains("targetText 只放正確的英文片段本身") == true)
        #expect(spacingReason?.contains("不要把周圍的中文一起放進來") == true)
    }

    /// Record voco:row:24701: an earlier card already published 託登斯寫的 → Codex寫的, so the rescoped rule only
    /// owns the remaining span. Against 原句 alone the substitution never reproduces 修正後; against 原句 after
    /// the live runtime it does, and the gate must let it through.
    @MainActor
    @Test func rescopeGateAcceptsARuleThatOnlyCoversWhatTheLiveRuntimeLeftBehind() {
        let source = "你還是要給託登斯特朗，因為這個是託登斯寫的。"
        let target = "你還是要給 Codex 一個 prompt，因為這個是 Codex 寫的。"
        let afterRuntime = "你還是要給託登斯特朗，因為這個是 Codex 寫的。"
        let rule = draft("replacementRule", target: "Codex 一個 prompt", pattern: "託登斯特朗")
        let withoutRuntime = RuleAssistantSession.gateReason(
            for: rule,
            kind: .rescope(scope: .broad, sourceText: source, targetText: target)
        )
        #expect(withoutRuntime?.contains("套回原句不等於修正後") == true)
        #expect(
            RuleAssistantSession.gateReason(
                for: rule,
                kind: .rescope(scope: .broad, sourceText: source, targetText: target),
                runtimeBaseline: afterRuntime
            ) == nil
        )
        // A runtime baseline never waves through a rule that reproduces neither sentence.
        let wrong = draft("replacementRule", target: "Codex", pattern: "託登斯特朗")
        #expect(
            RuleAssistantSession.gateReason(
                for: wrong,
                kind: .rescope(scope: .broad, sourceText: source, targetText: target),
                runtimeBaseline: afterRuntime
            ) != nil
        )
    }

    @MainActor
    @Test func draftGateKeyIgnoresWordingButSeparatesRules() {
        let first = draft("replacementRule", target: "Codex", pattern: "託登斯")
        let sameRule = draft("replacementRule", source: "另一句原句", target: "Codex", pattern: "託登斯")
        let other = draft("replacementRule", target: "Codex 一個 prompt", pattern: "託登斯特朗")
        #expect(RuleAssistantSession.draftGateKey(first) == RuleAssistantSession.draftGateKey(sameRule))
        #expect(RuleAssistantSession.draftGateKey(first) != RuleAssistantSession.draftGateKey(other))
    }

    @MainActor
    @Test func contextRescopeGateRequiresTokensInSource() {
        let source = "但是你的思維另有說。"
        let target = "但是你的思維鏈有說。"
        let locked = draft("contextLockedRule", target: "思維鏈有說", pattern: "思維另有說", contextTokens: ["思維", "有說"])
        #expect(RuleAssistantSession.gateReason(for: locked, kind: .rescope(scope: .context, sourceText: source, targetText: target)) == nil)
        let missing = draft("contextLockedRule", target: "思維鏈有說", pattern: "思維另有說", contextTokens: ["不存在"])
        #expect(RuleAssistantSession.gateReason(for: missing, kind: .rescope(scope: .context, sourceText: source, targetText: target))?.contains("不存在") == true)
        let broad = draft("replacementRule", target: "思維鏈有說", pattern: "思維另有說")
        #expect(RuleAssistantSession.gateReason(for: broad, kind: .rescope(scope: .context, sourceText: source, targetText: target))?.contains("只接受 contextLockedRule") == true)
    }

    @MainActor
    @Test func choiceNeverAuthorisesBroadButKeepsContextLockAndManualScanRules() {
        let broad = draft("replacementRule", target: "小鎮", pattern: "小振")
        let choiceReason = RuleAssistantSession.gateReason(for: broad, kind: .choice(candidateChosen: true))
        #expect(choiceReason?.contains("草稿卡") == true)
        let locked = draft("contextLockedRule", target: "小鎮", pattern: "小振", contextTokens: ["家"])
        #expect(RuleAssistantSession.gateReason(for: locked, kind: .choice(candidateChosen: true)) == nil)
        #expect(RuleAssistantSession.gateReason(for: broad, kind: .manual) == nil)
        #expect(RuleAssistantSession.gateReason(for: broad, kind: .scan) != nil)
        #expect(RuleAssistantSession.gateReason(for: locked, kind: .scan) == nil)
    }
}

@Suite(.serialized)
struct RuleAssistantDraftArgumentsTests {
    private let context = RuleAssistantContext(
        rowPk: 42,
        timestampMs: 1_700_000_000_000,
        text: "text",
        recordId: "11111111-2222-3333-4444-555555555555"
    )

    @Test func correctionWriteArgumentsAreTrimmedToSchema() {
        let draft = RuleAssistantDraft(
            eventType: "correction",
            sourceText: "小振",
            targetText: "小鎮",
            familyId: "should-not-ship",
            note: "筆記"
        )
        let args = draft.toMcpArguments(context: context)
        #expect(args["actor"] as? String == "voco-rule-assistant")
        #expect(args["correctionSource"] as? String == "voco")
        #expect(args["sourceText"] as? String == "小振")
        #expect(args["targetText"] as? String == "小鎮")
        #expect(args["makeAvailableNow"] as? Bool == true)
        // The note carries the voco row provenance.
        let note = args["note"] as? String ?? ""
        #expect(note.contains("筆記"))
        #expect(note.contains("source=voco:row:42"))
        // Correction schema has no family fields.
        #expect(args["familyId"] == nil)
        // No legacy top-level rowPk.
        #expect(args["rowPk"] == nil)
        let row = args["correctionRow"] as? [String: Any]
        #expect(row?["platform"] as? String == "voco")
        #expect((row?["rowPk"] as? Int64) == 42)
        #expect(row?["recordId"] as? String == "11111111-2222-3333-4444-555555555555")
    }

    @Test func previewArgumentsCarryEventTypeButNotPublishFlag() {
        let draft = RuleAssistantDraft(eventType: "correction", sourceText: "a", targetText: "b")
        let preview = draft.toPreviewArguments(context: context)
        #expect(preview["eventType"] as? String == "correction")
        #expect(preview["makeAvailableNow"] == nil)
        #expect(preview["correctionSource"] as? String == "voco")
    }

    @Test func tombstoneHasNoNote() {
        let draft = RuleAssistantDraft(
            eventType: "tombstone",
            targetText: "釋出",
            sourcePattern: "失重",
            reason: "錯規則",
            disposition: "replaced",
            note: "不該出現"
        )
        let args = draft.toMcpArguments(context: context)
        #expect(args["note"] == nil)
        #expect(args["reason"] as? String == "錯規則")
        #expect(args["disposition"] as? String == "replaced")
    }

    @Test func examplesUseWorkerShape() {
        let draft = RuleAssistantDraft(
            eventType: "replacementRule",
            targetText: "釋出",
            sourcePattern: "失重",
            positiveExamples: [RuleAssistantExample(text: "失重訓練", context: "太空", expectedText: "釋出訓練")]
        )
        let args = draft.toMcpArguments(context: context)
        let positives = args["positiveExamples"] as? [[String: Any]]
        #expect(positives?.count == 1)
        #expect(positives?[0]["text"] as? String == "失重訓練")
        #expect(positives?[0]["context"] as? String == "太空")
        #expect(positives?[0]["expectedText"] as? String == "釋出訓練")
    }
}


@MainActor
@Suite(.serialized)
struct RuleAssistantGuardSuggesterTests {
    private func lexicon(_ prefix: String) -> [(word: String, frequency: Int)] {
        let table: [(String, Int)] = [("資料架構", 900), ("資料架設", 860), ("資料架子", 848), ("資料架構圖", 120), ("架", 500)]
        return table.filter { $0.0 != prefix && $0.0.contains(prefix) }.map { (word: $0.0, frequency: $0.1) }
    }

    @Test func longerWordsMustExistInLexicon() {
        let guards = RuleAssistantGuardSuggester.guards(for: "資料架", lexicon: lexicon)
        #expect(guards == ["資料架構", "資料架設", "資料架子", "資料架構圖"])
        #expect(RuleAssistantGuardSuggester.guards(for: "摩多阿瑞納", lexicon: lexicon).isEmpty)
    }

    @Test func capsAtSixAndSkipsNonCjkOrSingleCharacterSources() {
        let many = RuleAssistantGuardSuggester.guards(for: "資料架") { prefix in
            prefix == "資料架" ? (1...10).map { ("資料架\($0)", 1000 - $0) } : []
        }
        #expect(many.count == RuleAssistantGuardSuggester.maxGuards)
        #expect(RuleAssistantGuardSuggester.guards(for: "modelarena", lexicon: lexicon).isEmpty)
        #expect(RuleAssistantGuardSuggester.guards(for: "架", lexicon: lexicon).isEmpty)
        #expect(RuleAssistantGuardSuggester.guards(for: "西賴 CLI", lexicon: lexicon).isEmpty)
    }

    @Test func withAutoGuardsAddsAndRemovesOnlyTheAutomaticOnes() {
        let draft = RuleAssistantDraft(
            eventType: "replacementRule", targetText: "資料夾", sourcePattern: "資料架",
            negativeExamples: [RuleAssistantExample(text: "資料架構", context: "模型說的", expectedText: "資料架構")]
        )
        let on = RuleAssistantSession.withAutoGuards(draft, guards: ["資料架構", "資料架設"], enabled: true)
        let onTexts: [String] = on.negativeExamples.map { $0.text }
        #expect(onTexts == ["資料架構", "資料架設"])
        let off = RuleAssistantSession.withAutoGuards(on, guards: ["資料架構", "資料架設"], enabled: false)
        // The model-authored example (with context) survives; the automatic ones go.
        let offTexts: [String] = off.negativeExamples.map { $0.text }
        #expect(offTexts == ["資料架構"])
        #expect(off.negativeExamples[0].context == "模型說的")
    }

    @Test func choiceGateRequiresTheSelectedCandidateScopeAndPair() {
        let draft = RuleAssistantDraft(eventType: "replacementRule", targetText: "資料夾", sourcePattern: "資料架")
        let broad = RuleAssistantChoiceCandidate(surface: "資料架", target: "資料夾", scope: .broad)
        #expect(RuleAssistantSession.gateReason(for: draft, kind: .choice(candidateChosen: true, candidates: [broad])) == nil)
        let sentence = RuleAssistantChoiceCandidate(surface: "資料架", target: "資料夾", scope: .sentence)
        #expect(RuleAssistantSession.gateReason(for: draft, kind: .choice(candidateChosen: true, candidates: [sentence]))?.contains("勾選時已選") == true)
        let wrong = RuleAssistantChoiceCandidate(surface: "資料夾", target: "資料架", scope: .broad)
        #expect(RuleAssistantSession.gateReason(for: draft, kind: .choice(candidateChosen: true, candidates: [wrong])) != nil)
    }
}

/// `runtimeReplay`: the record re-run through the runtime installed now, so the model can tell
/// "the current runtime already fixes this" from "no rule covers this".
@Suite(.serialized)
struct RuleAssistantRuntimeReplayTests {
    private static let replay = RuleAssistantRuntimeReplay(
        inputText: "麥克積塊也是一個產產品名",
        outputText: "麥克雞塊也是一個產品名",
        fires: [
            RuleAssistantRuntimeFire(
                policyId: "manual-replacement-1ad7a8372aeedcbc",
                policyType: "scopedReplacement",
                sourcePattern: "麥克積塊",
                targetText: "麥克雞塊"
            ),
            RuleAssistantRuntimeFire(
                policyId: VocoAutoApplyModelService.singlePrefixRestartCollapsePolicyId,
                policyType: VocoAutoApplyModelService.singlePrefixRestartCollapsePolicyType,
                sourcePattern: "產產品",
                targetText: "產品"
            ),
        ],
        modelVersion: "2026-09-11-overlay"
    )

    private func context(replay: RuleAssistantRuntimeReplay?) -> RuleAssistantContext {
        RuleAssistantContext(
            rowPk: 24208,
            timestampMs: 1_700_000_000_000,
            text: "麥克積塊也是一個產產品名",
            autoApplyModelVersion: "2026-09-09-overlay",
            recordId: "11111111-2222-3333-4444-555555555555",
            runtimeReplay: replay
        )
    }

    @Test func recordJSONCarriesReplayOutputAndFires() {
        let json = context(replay: Self.replay).toSafeJSON()
        guard let replay = json["runtimeReplay"] as? [String: Any] else {
            Issue.record("runtimeReplay missing from the record JSON")
            return
        }
        #expect(replay["inputText"] as? String == "麥克積塊也是一個產產品名")
        #expect(replay["outputText"] as? String == "麥克雞塊也是一個產品名")
        #expect(replay["changed"] as? Bool == true)
        #expect(replay["modelVersion"] as? String == "2026-09-11-overlay")
        let fires = replay["fires"] as? [[String: Any]] ?? []
        #expect(fires.count == 2)
        #expect(fires.map { $0["sourcePattern"] as? String } == ["麥克積塊", "產產品"])
        #expect(fires.map { $0["targetText"] as? String } == ["麥克雞塊", "產品"])
        #expect(fires.map { $0["policyId"] as? String }.allSatisfy { $0?.isEmpty == false })
        // The overlay version at transcription time stays separate from the replay's version.
        #expect(json["autoApplyModelVersion"] as? String == "2026-09-09-overlay")
        #expect(json["audioFileURL"] == nil)
    }

    @Test func unchangedReplayReportsNotChanged() {
        let replay = RuleAssistantRuntimeReplay(inputText: "沒有規則命中", outputText: "沒有規則命中")
        #expect(replay.changed == false)
        let json = replay.toJSONObject()
        #expect(json["changed"] as? Bool == false)
        #expect((json["fires"] as? [[String: Any]])?.isEmpty == true)
        #expect(json["modelVersion"] is NSNull)
    }

    @Test func recordJSONIsNullWhenTheRuntimeIsUnavailable() {
        let json = context(replay: nil).toSafeJSON()
        #expect(json["runtimeReplay"] is NSNull)
        #expect(json["text"] as? String == "麥克積塊也是一個產產品名")
    }

    @Test func currentReplayIsNilWithoutALoadedModelAndDoesNotThrow() throws {
        let service = VocoAutoApplyModelService(
            modelURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("missing-auto-apply-\(UUID().uuidString).json")
        )
        #expect(service.status.isAvailable == false)
        #expect(RuleAssistantRuntimeReplay.current(inputText: "麥克積塊", service: service) == nil)
        #expect(RuleAssistantRuntimeReplay.current(inputText: nil, service: service) == nil)
        #expect(RuleAssistantRuntimeReplay.current(inputText: "", service: service) == nil)
    }

    @Test func transcriptionInitReplaysNormalizedTranscriptAndIsInjectable() {
        let transcription = Transcription(
            text: "純文字",
            duration: 0,
            rawTranscript: "原始",
            normalizedTranscript: "標準化後"
        )
        var seen: [String?] = []
        let context = RuleAssistantContext(transcription: transcription, rowPk: 7) { input in
            seen.append(input)
            return RuleAssistantRuntimeReplay(inputText: input ?? "", outputText: "改過了")
        }
        // normalizedTranscript is what reaches the correction layer, so that is what gets replayed.
        #expect(seen == ["標準化後"])
        #expect(context.runtimeReplay?.outputText == "改過了")
        #expect(context.runtimeReplay?.changed == true)

        let plain = Transcription(text: "只有 text", duration: 0)
        var plainSeen: [String?] = []
        _ = RuleAssistantContext(transcription: plain, rowPk: 8) { input in
            plainSeen.append(input)
            return nil
        }
        #expect(plainSeen == ["只有 text"])
    }
}
