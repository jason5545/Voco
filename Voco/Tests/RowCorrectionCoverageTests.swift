import Foundation
import Testing
@testable import Voco

@Suite(.serialized)
struct RowCorrectionCoverageTests {
    private func fire(_ id: String) -> VocoAutoApplyPolicyFire {
        VocoAutoApplyPolicyFire(
            policyId: id,
            policyType: "exact",
            autoApplyMode: "apply",
            sourcePattern: "錯",
            targetText: "對",
            sourceSlices: ["test"]
        )
    }

    private func guardBlock(_ id: String) -> VocoAutoApplyGuardBlock {
        VocoAutoApplyGuardBlock(guardId: id, reason: "protected", term: "詞", blockedText: "錯", allowedPhrases: [])
    }

    @Test func rewriteByPolicyNotSeenAtTranscriptionIsFixedByCurrentRules() {
        let evaluation = VocoAutoApplyEvaluation(
            inputText: "錯句", outputText: "對句", applied: [fire("p1")], suggestions: [], modelVersion: "v9"
        )
        let coverage = RowCorrectionCoverageEvaluator.coverage(priorHitIds: [], evaluation: evaluation)
        #expect(coverage?.state == .fixedByCurrentRules)
        #expect(coverage?.policyIds == ["p1"])
        #expect(coverage?.modelVersion == "v9")
        #expect(coverage?.label.tone == .green)
        #expect(coverage?.label.text == String(localized: "Fixed by current rules"))
    }

    @Test func rewriteByPoliciesAlreadyHitAtTranscriptionIsAppliedAtTranscription() {
        let evaluation = VocoAutoApplyEvaluation(
            inputText: "錯句", outputText: "對句", applied: [fire("p1"), fire("p2")], suggestions: []
        )
        let coverage = RowCorrectionCoverageEvaluator.coverage(priorHitIds: ["p2", "p1", "other"], evaluation: evaluation)
        #expect(coverage?.state == .appliedAtTranscription)
        #expect(coverage?.label.tone == .secondary)
    }

    @Test func anyNewPolicyPromotesToFixedByCurrentRules() {
        let evaluation = VocoAutoApplyEvaluation(
            inputText: "錯句", outputText: "對句", applied: [fire("p1"), fire("p3")], suggestions: []
        )
        let coverage = RowCorrectionCoverageEvaluator.coverage(priorHitIds: ["p1"], evaluation: evaluation)
        #expect(coverage?.state == .fixedByCurrentRules)
        #expect(coverage?.policyIds == ["p1", "p3"])
    }

    @Test func rewriteMatchingStoredTextIsAppliedAtTranscriptionEvenWithoutHitIds() {
        let evaluation = VocoAutoApplyEvaluation(
            inputText: "零八零九", outputText: "0809", applied: [fire("phone")], suggestions: []
        )
        let stored = RowCorrectionCoverageEvaluator.coverage(
            priorHitIds: [], storedTexts: [nil, " 0809 "], evaluation: evaluation
        )
        #expect(stored?.state == .appliedAtTranscription)

        let differs = RowCorrectionCoverageEvaluator.coverage(
            priorHitIds: [], storedTexts: ["零八零九"], evaluation: evaluation
        )
        #expect(differs?.state == .fixedByCurrentRules)

        let empty = RowCorrectionCoverageEvaluator.coverage(
            priorHitIds: [], storedTexts: [""], evaluation: evaluation
        )
        #expect(empty?.state == .fixedByCurrentRules)
    }

    @Test func currencyNormalizationAloneIsNotCoverage() {
        let currency = fire(VocoAutoApplyModelService.currencyNumberNormalizationPolicyId)
        let evaluation = VocoAutoApplyEvaluation(
            inputText: "一千元", outputText: "1000元", applied: [currency], suggestions: []
        )
        #expect(RowCorrectionCoverageEvaluator.coverage(priorHitIds: [], evaluation: evaluation) == nil)
    }

    @Test func guardBlockWithoutRewriteIsBlockedByGuard() {
        let evaluation = VocoAutoApplyEvaluation(
            inputText: "錯句", outputText: "錯句", applied: [], suggestions: [], guardBlocks: [guardBlock("g1"), guardBlock("g1")]
        )
        let coverage = RowCorrectionCoverageEvaluator.coverage(priorHitIds: [], evaluation: evaluation)
        #expect(coverage?.state == .blockedByGuard)
        #expect(coverage?.policyIds == ["g1"])
        #expect(coverage?.label.tone == .orange)
    }

    @Test func suggestionsOnlyAreSuggestOnly() {
        let evaluation = VocoAutoApplyEvaluation(
            inputText: "錯句", outputText: "錯句", applied: [], suggestions: [fire("s1")]
        )
        let coverage = RowCorrectionCoverageEvaluator.coverage(priorHitIds: [], evaluation: evaluation)
        #expect(coverage?.state == .suggestOnly)
        #expect(coverage?.policyIds == ["s1"])
    }

    @Test func unchangedTextWithNothingMatchingHasNoCoverage() {
        let evaluation = VocoAutoApplyEvaluation(inputText: "對句", outputText: "對句", applied: [], suggestions: [])
        #expect(RowCorrectionCoverageEvaluator.coverage(priorHitIds: ["p1"], evaluation: evaluation) == nil)
    }

    @Test func sourceTextPrefersRawTranscriptAndSkipsEmptyOrCanceledRows() {
        let raw = Transcription(text: "對句", duration: 1)
        raw.rawTranscript = "錯句"
        #expect(RowCorrectionCoverageEvaluator.sourceText(for: raw) == "錯句")

        let noRaw = Transcription(text: "對句", duration: 1)
        noRaw.rawTranscript = "   "
        #expect(RowCorrectionCoverageEvaluator.sourceText(for: noRaw) == "對句")

        let canceled = Transcription(text: Transcription.canceledTranscriptionText, duration: 1)
        #expect(RowCorrectionCoverageEvaluator.sourceText(for: canceled) == nil)

        let empty = Transcription(text: "", duration: 1)
        #expect(RowCorrectionCoverageEvaluator.sourceText(for: empty) == nil)
    }
}
