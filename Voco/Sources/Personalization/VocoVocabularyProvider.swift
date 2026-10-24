import Foundation

enum VocoVocabularyProvider {
    static let defaultWords = [
        "M5 Max",
        "M5 Max 128GB",
        "Voco",
        "Claude",
        "Claude Code",
        "Codex",
        "ChatGPT",
        "Cloudflare",
        "Workers",
        "D1",
        "RIME",
        "鼠鬚管",
        "repo",
        "SESSION",
        "UI",
        "UX",
        "Qwen3-ASR 1.7B",
        "ASR",
        "frontier",
        "hook",
    ]

    static func merging(_ persistedWords: [String]) -> [String] {
        uniqueWords(persistedWords + defaultWords)
    }

    private static func uniqueWords(_ words: [String]) -> [String] {
        var seen: Set<String> = []
        var result: [String] = []
        for word in words {
            let normalized = word.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            guard !normalized.isEmpty else { continue }
            if seen.insert(normalized).inserted {
                result.append(word)
            }
        }
        return result
    }
}
