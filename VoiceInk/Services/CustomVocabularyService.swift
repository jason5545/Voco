import Foundation
import SwiftUI
import SwiftData

class CustomVocabularyService {
    static let shared = CustomVocabularyService()

    private let defaultVocabulary = [
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

    private init() {}

    func getCustomVocabulary(from context: ModelContext) -> String {
        let customWords = getCustomVocabularyWords(from: context)
        guard !customWords.isEmpty else {
            return ""
        }

        let wordsText = customWords.joined(separator: ", ")
        return "Important Vocabulary: \(wordsText)"
    }

    func getCustomVocabularyWords(from context: ModelContext) -> [String] {
        let descriptor = FetchDescriptor<VocabularyWord>(sortBy: [SortDescriptor(\VocabularyWord.word)])

        do {
            let items = try context.fetch(descriptor)
            let words = items.map { $0.word } + defaultVocabulary
            return uniqueWords(words)
        } catch {
            return defaultVocabulary
        }
    }

    private func uniqueWords(_ words: [String]) -> [String] {
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
