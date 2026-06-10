import Foundation
import Testing
@testable import Voco

struct CorrectionEvidenceClassifierTests {
    @Test func llmAndEnhancedTextDifferencesAreAlwaysUntrusted() async throws {
        let llm = CorrectionEvidenceClassifier.classify(
            CorrectionEvidenceInput(
                source: .llmEnhancement,
                rawText: "修正",
                targetText: "小振"
            )
        )
        #expect(llm.evidenceTier == .t0Untrusted)
        #expect(llm.noiseFlags.contains(.llmOnly))
        #expect(llm.isPurePhoneticCandidate == false)

        let enhancedDiff = CorrectionEvidenceClassifier.classify(
            CorrectionEvidenceInput(
                source: .ztextEnhancedDifference,
                rawText: "失重",
                targetText: "實作"
            )
        )
        #expect(enhancedDiff.evidenceTier == .t0Untrusted)
        #expect(enhancedDiff.noiseFlags.contains(.llmOnly))
    }

    @Test func automaticAndUnselectedCandidatesStayT0() async throws {
        let automatic = CorrectionEvidenceClassifier.classify(
            CorrectionEvidenceInput(
                source: .automaticCorrection,
                rawText: "智商",
                targetText: "諮商"
            )
        )
        #expect(automatic.evidenceTier == .t0Untrusted)

        let unselected = CorrectionEvidenceClassifier.classify(
            CorrectionEvidenceInput(
                source: .candidateNotSelected,
                rawText: "專欄",
                targetText: "專案"
            )
        )
        #expect(unselected.evidenceTier == .t0Untrusted)
    }

    @Test func weakInteractionRequiresSpanAndTimingBeforeT2() async throws {
        let weak = CorrectionEvidenceClassifier.classify(
            CorrectionEvidenceInput(
                source: .userSubstitution,
                rawText: "拍板",
                targetText: "排版"
            )
        )
        #expect(weak.evidenceTier == .t1WeakInteraction)
        #expect(weak.noiseFlags.contains(.selectedSpanMissing))
        #expect(weak.noiseFlags.contains(.correctionTooLate))
        #expect(weak.isPurePhoneticCandidate == false)

        let confirmed = CorrectionEvidenceClassifier.classify(
            CorrectionEvidenceInput(
                source: .userSubstitution,
                rawText: "拍板",
                targetText: "排版",
                selectedRangeLength: 2,
                timeSinceUtteranceMs: 1500
            )
        )
        #expect(confirmed.evidenceTier == .t2ConfirmedSpan)
        #expect(confirmed.noiseFlags.isEmpty)
    }

    @Test func repeatedConfirmedSpanBecomesT3AndGoldBecomesT4() async throws {
        let repeated = CorrectionEvidenceClassifier.classify(
            CorrectionEvidenceInput(
                source: .correctionFeedback,
                rawText: "變吃",
                targetText: "辨識",
                selectedRangeLength: 2,
                timeSinceUtteranceMs: 2000,
                repeatedObservationCount: 3
            )
        )
        #expect(repeated.evidenceTier == .t3ConfirmedRepeated)

        let gold = CorrectionEvidenceClassifier.classify(
            CorrectionEvidenceInput(
                source: .reviewCandidate,
                rawText: "often",
                targetText: "Orphan",
                selectedRangeLength: 5,
                timeSinceUtteranceMs: 1000,
                isGoldConfirmation: true
            )
        )
        #expect(gold.evidenceTier == .t4Gold)
    }

    @Test func negativeEvidenceCoversRejectionRollbackAndAllowlistedOriginal() async throws {
        let rejected = CorrectionEvidenceClassifier.classify(
            CorrectionEvidenceInput(
                source: .rejectedCandidate,
                rawText: "69 輪",
                targetText: "六十九輪"
            )
        )
        #expect(rejected.evidenceTier == .negativeEvidence)

        let rollback = CorrectionEvidenceClassifier.classify(
            CorrectionEvidenceInput(
                source: .rollback,
                rawText: "修正",
                targetText: "小振"
            )
        )
        #expect(rollback.evidenceTier == .negativeEvidence)

        let allowlisted = CorrectionEvidenceClassifier.classify(
            CorrectionEvidenceInput(
                source: .correctionFeedback,
                rawText: "69 輪",
                targetText: "69 輪",
                isExplicitAllowlistedCorrectOriginal: true
            )
        )
        #expect(allowlisted.evidenceTier == .negativeEvidence)
    }

    @Test func noisyFullSentenceRewriteIsNotPromoted() async throws {
        let result = CorrectionEvidenceClassifier.classify(
            CorrectionEvidenceInput(
                source: .userSubstitution,
                rawText: "文他預測",
                targetText: "我想要問他預測這一段到底是不是舊文字被帶進來。",
                selectedRangeLength: nil,
                timeSinceUtteranceMs: 1200,
                activeAppChanged: true
            )
        )

        #expect(result.evidenceTier == .t1WeakInteraction)
        #expect(result.noiseFlags.contains(.selectedSpanMissing))
        #expect(result.noiseFlags.contains(.activeAppChanged))
        #expect(result.noiseFlags.contains(.targetLengthExpansionRatioHigh))
        #expect(result.noiseFlags.contains(.fullSentenceRewriteSuspected))
        #expect(result.noiseFlags.contains(.stalePendingTranscriptSuspected))
    }

    @Test func crossScriptTechnicalTermsAreMarkedAsCrossScriptNotChineseHomophones() async throws {
        let result = CorrectionEvidenceClassifier.classify(
            CorrectionEvidenceInput(
                source: .correctionFeedback,
                rawText: "凹頭",
                targetText: "auto",
                selectedRangeLength: 2,
                timeSinceUtteranceMs: 1000
            )
        )

        #expect(result.evidenceTier == .t2ConfirmedSpan)
        #expect(result.phoneticComparison?.languageMode == .crossScript)
        #expect(result.phoneticComparison?.target.isTechnicalTermCandidate == true)
        #expect(!result.noiseFlags.contains(.crossLanguageReconstruction))
    }

    @Test func crossLanguageReconstructionIsFlaggedWhenNotTechnical() async throws {
        let result = CorrectionEvidenceClassifier.classify(
            CorrectionEvidenceInput(
                source: .correctionFeedback,
                rawText: "你好",
                targetText: "hello there",
                selectedRangeLength: 2,
                timeSinceUtteranceMs: 1000
            )
        )

        #expect(result.evidenceTier == .t1WeakInteraction)
        #expect(result.phoneticComparison?.languageMode == .crossScript)
        #expect(result.noiseFlags.contains(.crossLanguageReconstruction))
    }
}
