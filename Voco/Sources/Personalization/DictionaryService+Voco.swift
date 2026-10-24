import OSLog
import SwiftData

extension DictionaryService {
    @discardableResult
    static func addVocabularyWords(
        _ words: [String],
        context: ModelContext
    ) throws -> Int {
        let descriptor = FetchDescriptor<VocabularyWord>()
        let existing = (try? context.fetch(descriptor)) ?? []
        var seen = Set(existing.map { $0.word.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() })
        var insertedCount = 0

        for word in words {
            let trimmed = word.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let key = trimmed.lowercased()
            guard seen.insert(key).inserted else { continue }
            context.insert(VocabularyWord(word: trimmed))
            insertedCount += 1
        }

        guard insertedCount > 0 else { return 0 }
        do {
            try context.save()
            return insertedCount
        } catch {
            context.rollback()
            throw error
        }
    }

    @discardableResult
    static func removeExactDuplicateContent(context: ModelContext, source: String) -> Bool {
        var deletedVocabularyCount = 0
        var deletedReplacementCount = 0

        if let vocabularyWords = try? context.fetch(FetchDescriptor<VocabularyWord>()) {
            var seenWords = Set<String>()
            for vocabularyWord in vocabularyWords.sorted(by: { $0.dateAdded < $1.dateAdded }) {
                let word = vocabularyWord.word
                guard !word.isEmpty else { continue }
                if seenWords.insert(word).inserted {
                    continue
                }
                context.delete(vocabularyWord)
                deletedVocabularyCount += 1
            }
        }

        if let wordReplacements = try? context.fetch(FetchDescriptor<WordReplacement>()) {
            var seenReplacements = Set<[String]>()
            for wordReplacement in wordReplacements.sorted(by: { $0.dateAdded < $1.dateAdded }) {
                let key = [wordReplacement.originalText, wordReplacement.replacementText]
                guard !wordReplacement.originalText.isEmpty || !wordReplacement.replacementText.isEmpty else {
                    continue
                }
                if seenReplacements.insert(key).inserted {
                    continue
                }
                context.delete(wordReplacement)
                deletedReplacementCount += 1
            }
        }

        guard deletedVocabularyCount > 0 || deletedReplacementCount > 0 else {
            return false
        }

        do {
            try context.save()
            logger.notice("Removed exact dictionary duplicates from \(source, privacy: .public): \(deletedVocabularyCount, privacy: .public) vocabulary, \(deletedReplacementCount, privacy: .public) word replacement")
            return true
        } catch {
            context.rollback()
            logger.error("Failed to remove exact dictionary duplicates from \(source, privacy: .public): \(error.localizedDescription, privacy: .public)")
            return false
        }
    }

    private static var logger: Logger {
        Logger(subsystem: "com.prakashjoshipax.voiceink", category: "DictionaryService")
    }
}
