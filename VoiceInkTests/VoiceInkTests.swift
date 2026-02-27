//
//  VoiceInkTests.swift
//  VoiceInkTests
//
//  Created by Prakash Joshi on 15/10/2024.
//

import Testing
@testable import VoiceInk

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

}
