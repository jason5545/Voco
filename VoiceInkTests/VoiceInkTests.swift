//
//  VoiceInkTests.swift
//  VoiceInkTests
//
//  Created by Prakash Joshi on 15/10/2024.
//

import Testing
@testable import Voco

struct VoiceInkTests {

    @Test func validatorRejectsPromptLeakage() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "這是正體中文語音輸入，請修正。",
            original: "請修正這段文字"
        )

        #expect(result.isValid == false)
        #expect(result.reasons.contains(where: { $0.contains("blacklist") }))
    }

    @Test func validatorPreservesProtectedTechnicalTerms() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "我剛剛用差值被統一整理這段程式。",
            original: "我剛剛用 Chat GPT 整理這段程式。",
            protectedTerms: ["ChatGPT"]
        )

        #expect(result.isValid == false)
        #expect(result.reasons.contains(where: { $0.contains("dropped-term") }))
    }

    @Test func validatorAllowsScriptNormalizationAndPunctuation() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "網頁版。",
            original: "网页版"
        )

        #expect(result.isValid == true)
    }

    @Test func validatorRejectsAggressiveShortRewrite() async throws {
        let result = LLMResponseValidator.shared.validate(
            response: "網葉斑",
            original: "网页版"
        )

        #expect(result.isValid == false)
        #expect(result.reasons.contains(where: { $0 == "short-edit-budget" }))
    }

    @Test func editModePollingStateCoalescesAcrossRestartWhilePollIsInFlight() async throws {
        var state = EditModePollingState()

        let firstGenerationResult = state.startPolling()
        let firstGeneration = try #require(firstGenerationResult)
        #expect(state.beginNextRefresh() == firstGeneration)
        #expect(state.enqueueRefresh() == .queuedBehindInFlight)

        state.stopPolling()
        #expect(state.shouldApplyResult(for: firstGeneration) == false)

        let secondGenerationResult = state.startPolling()
        let secondGeneration = try #require(secondGenerationResult)
        #expect(secondGeneration > firstGeneration)
        #expect(state.enqueueRefresh() == .coalesced)

        state.finishRefresh()
        #expect(state.beginNextRefresh() == secondGeneration)
    }

    @Test func editModePollingStateIgnoresRefreshRequestsWhenStopped() async throws {
        var state = EditModePollingState()

        #expect(state.enqueueRefresh() == .ignoredStopped)

        let generationResult = state.startPolling()
        let generation = try #require(generationResult)
        #expect(state.shouldContinuePolling(expectedGeneration: generation) == true)

        state.stopPolling()
        #expect(state.shouldContinuePolling(expectedGeneration: generation) == false)
        #expect(state.shouldApplyResult(for: generation) == false)
        #expect(state.enqueueRefresh() == .ignoredStopped)
    }

}
