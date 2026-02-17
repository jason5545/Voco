import Foundation

/// Validates LLM responses to filter out invalid corrections
/// Reference: xvoice/src/llm/openrouter.py lines 246-273
class LLMResponseValidator {
    static let shared = LLMResponseValidator()

    /// Phrases that should never appear in LLM output (they come from the system prompt)
    private let blacklistPhrases: [String] = [
        "使用臺灣語音輸入",
        "正體中文語音輸入",
        "這是正體中文",
        "臺灣正體中文",
        "語音辨識結果",
    ]

    /// Maximum ratio of response length to original length
    private let maxLengthRatio: Double = 3.0

    private init() {}

    /// Check if LLM response is valid
    /// - Parameters:
    ///   - response: The LLM's response text
    ///   - original: The original text before LLM processing
    /// - Returns: true if the response is valid, false if it should be rejected
    func isValid(response: String, original: String) -> Bool {
        // Check blacklist phrases
        for phrase in blacklistPhrases {
            if response.contains(phrase) {
                return false
            }
        }

        // Check length ratio - response shouldn't be much longer than original
        if Double(response.count) > Double(original.count) * maxLengthRatio {
            return false
        }

        return true
    }
}
