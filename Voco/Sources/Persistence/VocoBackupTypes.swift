import Foundation

struct WordReplacementBackup: Codable, Equatable {
    let originalText: String
    let replacementText: String
    let isEnabled: Bool
    let source: String
    let hitCount: Int
    let dateAdded: Date
    let lastSeenDate: Date

    init(replacement: WordReplacement) {
        self.originalText = replacement.originalText
        self.replacementText = replacement.replacementText
        self.isEnabled = replacement.isEnabled
        self.source = replacement.source
        self.hitCount = replacement.hitCount
        self.dateAdded = replacement.dateAdded
        self.lastSeenDate = replacement.lastSeenDate
    }

    func makeReplacement(
        originalText: String? = nil,
        replacementText: String? = nil
    ) -> WordReplacement {
        let wordReplacement = WordReplacement(
            originalText: originalText ?? self.originalText,
            replacementText: replacementText ?? self.replacementText,
            dateAdded: dateAdded,
            isEnabled: isEnabled,
            source: source
        )
        wordReplacement.hitCount = max(1, hitCount)
        wordReplacement.lastSeenDate = lastSeenDate
        return wordReplacement
    }
}
