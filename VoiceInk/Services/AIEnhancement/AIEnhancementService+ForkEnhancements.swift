// AIEnhancementService+ForkEnhancements.swift
// Fork-specific enhancement methods: conservative retry, comma insertion,
// edit mode, and context-aware merge.
// Isolated from AIEnhancementService.swift to minimize upstream merge conflicts.

import Foundation

extension AIEnhancementService {

    /// Conservative retry: minimal corrections (homophone fixes + punctuation only).
    func enhanceConservative(_ text: String, uncertainWords: [UncertainWord]) async throws -> (String, TimeInterval) {
        let startTime = Date()
        let systemMessage = AIPrompts.conservativeRetryPrompt(
            uncertainWords: uncertainWords.map { $0.text }
        )
        let result = try await makeRequest(
            text: text,
            configuration: currentRuntimeConfiguration,
            systemMessageOverride: systemMessage
        )
        return (result, Date().timeIntervalSince(startTime))
    }

    /// Comma-only insertion retry: adds punctuation without changing text.
    func enhanceCommaInsertion(_ text: String) async throws -> (String, TimeInterval) {
        let startTime = Date()
        let result = try await makeRequest(
            text: text,
            configuration: currentRuntimeConfiguration,
            systemMessageOverride: AIPrompts.commaInsertionPrompt
        )
        return (result, Date().timeIntervalSince(startTime))
    }

    /// Edit Mode: applies a spoken instruction to selected text, returns JSON with result + optional substitution.
    func enhanceForEditMode(instruction: String, selectedText: String) async throws -> (String, TimeInterval, WordSubstitution?) {
        let startTime = Date()
        guard isConfigured else { throw EnhancementError.notConfigured }
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

        await MainActor.run {
            self.lastSystemMessageSent = systemMessage
            self.lastUserMessageSent = userMessage
        }

        let raw = try await makeRequestWithRetry(
            text: "",
            configuration: currentRuntimeConfiguration,
            systemMessageOverride: systemMessage,
            userMessageOverride: userMessage
        )
        let duration = Date().timeIntervalSince(startTime)

        // Try to parse JSON response (handle optional markdown code fences)
        var jsonString = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if jsonString.hasPrefix("```") {
            let lines = jsonString.components(separatedBy: "\n")
            let stripped = lines.dropFirst()
                .prefix(while: { !$0.hasPrefix("```") })
            jsonString = stripped.joined(separator: "\n")
        }

        if let data = jsonString.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let result = json["result"] as? String {
            var substitution: WordSubstitution? = nil
            if let sub = json["substitution"] as? [String: String],
               let from = sub["from"], let to = sub["to"],
               !from.isEmpty, !to.isEmpty,
               from.count <= 20, to.count <= 20 {
                substitution = WordSubstitution(original: from, replacement: to)
            }
            return (result, duration, substitution)
        }

        // Fallback: treat entire response as plain text result, no substitution
        return (raw, duration, nil)
    }

    /// Merge inserted text into surrounding context for seamless reading (fork feature).
    /// Only adjusts the inserted portion — surrounding text is context-only.
    func enhanceMerge(
        insertedText: String,
        textBefore: String,
        textAfter: String
    ) async throws -> (String, TimeInterval) {
        let startTime = Date()
        guard isConfigured else { throw EnhancementError.notConfigured }
        guard !insertedText.isEmpty else { return (insertedText, 0) }

        let systemMessage = AIPrompts.contextMergePrompt
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

        let result = try await makeRequestWithRetry(
            text: "",
            configuration: currentRuntimeConfiguration,
            systemMessageOverride: systemMessage,
            userMessageOverride: userMessage
        )
        let duration = Date().timeIntervalSince(startTime)
        return (result.trimmingCharacters(in: .whitespacesAndNewlines), duration)
    }
}
