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

    @Test func unknownTypeIsNeverSafe() {
        #expect(!draft("deleteEverything", source: "a", target: "b").isSafeForWrite())
    }

    @Test func oversizedFieldsAreRefused() {
        let huge = String(repeating: "好", count: 2_001)
        #expect(!draft("correction", source: huge, target: "b").isSafeForWrite())
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
