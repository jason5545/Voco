import Foundation

enum ShortUtterancePunctuationCleaner {
    private static let maxContentCharactersForTerminalPunctuationCleanup = 4
    private static let cjkPunctuation: Set<Character> = ["，", "。", "？", "！", "、", "；", "：", "「", "」", "『", "』", "（", "）", "…"]
    private static let asciiPunctuation: Set<Character> = [",", ".", "?", "!", ";", ":"]
    private static let terminalPunctuation: Set<Character> = ["。", "？", "！", ".", "?", "!", "…"]

    /// Separators ASR tends to insert between digit groups; dropped when the utterance is a phone number.
    private static let digitGroupNoise: Set<Character> = ["，", "、"]
    private static let minPhoneDigits = 7
    private static let maxPhoneDigits = 15

    static func removeTerminalSentencePunctuation(from text: String) -> String {
        if let phoneNumber = standalonePhoneNumber(in: text) {
            return phoneNumber
        }
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

    /// When the whole utterance is one phone number (7–15 digits, optional leading "+", digit
    /// groups separated only by spaces, hyphens, or ASR comma noise), returns it with the trailing
    /// sentence punctuation and the comma noise removed; otherwise nil. ASCII "." and "," between
    /// digits are left alone so decimals and thousands separators never get flattened.
    static func standalonePhoneNumber(in text: String) -> String? {
        let leadingWhitespace = text.prefix { $0.isWhitespace || $0.isNewline }
        let trailingWhitespace = text.reversed().prefix { $0.isWhitespace || $0.isNewline }.reversed()
        var core = Substring(text.dropFirst(leadingWhitespace.count).dropLast(trailingWhitespace.count))
        while let last = core.last, terminalPunctuation.contains(last) || digitGroupNoise.contains(last) || cjkPunctuation.contains(last) {
            core = core.dropLast()
        }
        guard let first = core.first, first == "+" || first.isNumber else { return nil }
        guard let last = core.last, last.isNumber else { return nil }

        var digitCount = 0
        var kept = ""
        for (offset, character) in core.enumerated() {
            if character.isNumber {
                digitCount += 1
                kept.append(character)
            } else if character == "+", offset == 0 {
                kept.append(character)
            } else if character == "-" || character == " " {
                kept.append(character)
            } else if digitGroupNoise.contains(character) {
                continue
            } else {
                return nil
            }
        }
        guard (minPhoneDigits...maxPhoneDigits).contains(digitCount) else { return nil }
        let result = String(leadingWhitespace) + kept + String(trailingWhitespace)
        return result == text ? nil : result
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
