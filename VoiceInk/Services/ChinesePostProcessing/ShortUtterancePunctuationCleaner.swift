import Foundation

enum ShortUtterancePunctuationCleaner {
    private static let maxContentCharactersForTerminalPunctuationCleanup = 4
    private static let cjkPunctuation: Set<Character> = ["，", "。", "？", "！", "、", "；", "：", "「", "」", "『", "』", "（", "）", "…"]
    private static let asciiPunctuation: Set<Character> = [",", ".", "?", "!", ";", ":"]
    private static let terminalPunctuation: Set<Character> = ["。", "？", "！", ".", "?", "!", "…"]

    static func removeTerminalSentencePunctuation(from text: String) -> String {
        guard contentLength(in: text) <= maxContentCharactersForTerminalPunctuationCleanup else {
            return text
        }
        guard let lastContentIndex = text.indices.last(where: { !text[$0].isWhitespace && !text[$0].isNewline }) else {
            return text
        }

        var punctuationStart = lastContentIndex
        var didRemovePunctuation = false
        while terminalPunctuation.contains(text[punctuationStart]) {
            didRemovePunctuation = true
            if punctuationStart == text.startIndex { break }
            let previous = text.index(before: punctuationStart)
            if !terminalPunctuation.contains(text[previous]) {
                punctuationStart = previous
                break
            }
            punctuationStart = previous
        }
        guard didRemovePunctuation else { return text }

        let removalStart = terminalPunctuation.contains(text[punctuationStart])
            ? punctuationStart
            : text.index(after: punctuationStart)
        return String(text[..<removalStart]) + String(text[text.index(after: lastContentIndex)...])
    }

    private static func contentLength(in text: String) -> Int {
        text.filter {
            !$0.isWhitespace &&
                !$0.isNewline &&
                !cjkPunctuation.contains($0) &&
                !asciiPunctuation.contains($0)
        }.count
    }
}
