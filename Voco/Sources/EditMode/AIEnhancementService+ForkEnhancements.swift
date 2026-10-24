import Foundation

extension AIEnhancementService {
    func enhanceConservative(_ text: String, uncertainWords: [UncertainWord]) async throws -> (String, TimeInterval) {
        try await VocoEnhancementRequester.enhanceConservative(
            self,
            text: text,
            uncertainWords: uncertainWords
        )
    }

    func enhanceCommaInsertion(_ text: String) async throws -> (String, TimeInterval) {
        try await VocoEnhancementRequester.enhanceCommaInsertion(self, text: text)
    }

    func enhanceForEditMode(instruction: String, selectedText: String) async throws -> (String, TimeInterval, WordSubstitution?) {
        try await VocoEnhancementRequester.enhanceForEditMode(
            self,
            instruction: instruction,
            selectedText: selectedText
        )
    }

    func enhanceMerge(
        insertedText: String,
        textBefore: String,
        textAfter: String
    ) async throws -> (String, TimeInterval) {
        try await VocoEnhancementRequester.enhanceMerge(
            self,
            insertedText: insertedText,
            textBefore: textBefore,
            textAfter: textAfter
        )
    }
}
