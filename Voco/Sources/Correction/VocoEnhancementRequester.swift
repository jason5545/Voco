import Foundation

@MainActor
enum VocoEnhancementRequester {
    static func enhance(
        _ service: AIEnhancementService,
        text: String,
        configuration: EnhancementRuntimeConfiguration,
        contextSnapshot: RecordingContextSnapshot?
    ) async throws -> AIEnhancementResult {
        guard service.isEnhancementEnabled else {
            throw EnhancementError.enhancementDisabled
        }
        let startTime = Date()
        let result = try await service.makeRequestWithRetry(
            text: text,
            configuration: configuration,
            contextSnapshot: contextSnapshot
        )
        return AIEnhancementResult(
            text: result.text,
            duration: Date().timeIntervalSince(startTime),
            promptName: configuration.prompt?.title,
            systemMessage: result.systemMessage,
            userMessage: result.userMessage
        )
    }

    static func enhanceConservative(
        _ service: AIEnhancementService,
        text: String,
        uncertainWords: [UncertainWord]
    ) async throws -> (String, TimeInterval) {
        let startTime = Date()
        let systemMessage = AIPrompts.conservativeRetryPrompt(
            uncertainWords: uncertainWords.map { $0.text }
        )
        let result = try await service.makeRequest(
            text: text,
            configuration: service.currentRuntimeConfiguration,
            systemMessageOverride: systemMessage
        )
        return (result.text, Date().timeIntervalSince(startTime))
    }

    static func enhanceCommaInsertion(
        _ service: AIEnhancementService,
        text: String
    ) async throws -> (String, TimeInterval) {
        let startTime = Date()
        let result = try await service.makeRequest(
            text: text,
            configuration: service.currentRuntimeConfiguration,
            systemMessageOverride: AIPrompts.commaInsertionPrompt
        )
        return (result.text, Date().timeIntervalSince(startTime))
    }

    static func enhanceForEditMode(
        _ service: AIEnhancementService,
        instruction: String,
        selectedText: String
    ) async throws -> (String, TimeInterval, WordSubstitution?) {
        let startTime = Date()
        guard service.isConfigured else { throw EnhancementError.notConfigured }
        guard !instruction.isEmpty else { return (selectedText, 0, nil) }

        let systemMessage = """
        You are a precise text editor. The user has selected text and given you a spoken instruction to modify it.

        Rules:
        - Apply the instruction to the selected text
        - Return a JSON object (no markdown fences): {"result": "modified text", "substitution": {"from": "original word", "to": "new word"}}
        - "substitution" should contain the single word/phrase pair that was replaced. Set it to null if the edit is not a simple word substitution (e.g. rewriting, reformatting, multi-word changes).
        - Each side of the substitution must be ≤ 20 characters
        - Preserve the original formatting style and language
        - Do NOT add explanations or commentary
        """

        let userMessage = """
        <SELECTED_TEXT>
        \(selectedText)
        </SELECTED_TEXT>

        <INSTRUCTION>
        \(instruction)
        </INSTRUCTION>
        """

        service.lastSystemMessageSent = systemMessage
        service.lastUserMessageSent = userMessage

        let raw = try await service.makeRequestWithRetry(
            text: "",
            configuration: service.currentRuntimeConfiguration,
            systemMessageOverride: systemMessage,
            userMessageOverride: userMessage
        )
        let duration = Date().timeIntervalSince(startTime)

        var jsonString = raw.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if jsonString.hasPrefix("```") {
            let lines = jsonString.components(separatedBy: "\n")
            let stripped = lines.dropFirst().prefix(while: { !$0.hasPrefix("```") })
            jsonString = stripped.joined(separator: "\n")
        }

        if let data = jsonString.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let result = json["result"] as? String {
            var substitution: WordSubstitution?
            if let sub = json["substitution"] as? [String: String],
               let from = sub["from"], let to = sub["to"],
               !from.isEmpty, !to.isEmpty,
               from.count <= 20, to.count <= 20 {
                substitution = WordSubstitution(original: from, replacement: to)
            }
            return (result, duration, substitution)
        }
        return (raw.text, duration, nil)
    }

    static func enhanceMerge(
        _ service: AIEnhancementService,
        insertedText: String,
        textBefore: String,
        textAfter: String
    ) async throws -> (String, TimeInterval) {
        let startTime = Date()
        guard service.isConfigured else { throw EnhancementError.notConfigured }
        guard !insertedText.isEmpty else { return (insertedText, 0) }

        let userMessage = """
        <TEXT_BEFORE_CURSOR>
        \(textBefore)
        </TEXT_BEFORE_CURSOR>

        <INSERTED_TEXT>
        \(insertedText)
        </INSERTED_TEXT>

        <TEXT_AFTER_CURSOR>
        \(textAfter)
        </TEXT_AFTER_CURSOR>
        """

        let result = try await service.makeRequestWithRetry(
            text: "",
            configuration: service.currentRuntimeConfiguration,
            systemMessageOverride: AIPrompts.contextMergePrompt,
            userMessageOverride: userMessage
        )
        return (
            result.text.trimmingCharacters(in: .whitespacesAndNewlines),
            Date().timeIntervalSince(startTime)
        )
    }
}
