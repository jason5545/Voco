import Foundation
import os

final class ContextAwareInsertionService {
    static let shared = ContextAwareInsertionService()
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "ContextAwareInsertion")

    private init() {}

    /// Apply context-aware adjustments to text before pasting.
    /// Returns adjusted text ready for CursorPaster.
    func adjust(_ text: String, context: SurroundingTextContext?, appendTrailingSpace: Bool) -> String {
        guard let ctx = context, ctx.isAvailable else {
            // No context available — use current behavior
            return text + (appendTrailingSpace ? " " : "")
        }

        var result = prepareForInsertion(text, textBefore: ctx.textBefore)

        // Rule 1: Smart leading space (English word boundary)
        result = adjustLeadingSpace(result, textBefore: ctx.textBefore)

        // Rule 2: Prevent duplicate punctuation at junction
        result = preventDuplicatePunctuation(result, textBefore: ctx.textBefore, textAfter: ctx.textAfter)

        // Rule 3: Capitalize first letter if at sentence start
        result = adjustCapitalization(result, textBefore: ctx.textBefore)

        // Rule 4: CJK-Latin spacing
        result = adjustCJKLatinSpacing(result, textBefore: ctx.textBefore, textAfter: ctx.textAfter)

        // Rule 5: Smart trailing space (must be last — depends on final state of result)
        result = adjustTrailingSpace(result, textAfter: ctx.textAfter, appendSpaceSetting: appendTrailingSpace)

        return result
    }

    /// Removes the longest exact overlap between the existing text suffix and
    /// the new dictation prefix. A minimum of two letters or digits avoids
    /// treating a coincidental single character as repeated speech.
    func removeOverlappingPrefix(_ text: String, textBefore: String) -> String {
        guard !text.isEmpty, !textBefore.isEmpty else { return text }

        let incoming = text.drop(while: { $0.isWhitespace })
        let beforeEnd = textBefore.lastIndex(where: { !$0.isWhitespace })
            .map { textBefore.index(after: $0) } ?? textBefore.startIndex
        let before = textBefore[..<beforeEnd]
        guard !incoming.isEmpty, !before.isEmpty else { return text }

        let maximumOverlap = min(before.count, incoming.count)
        for length in stride(from: maximumOverlap, through: 1, by: -1) {
            let overlap = incoming.prefix(length)
            guard before.suffix(length).elementsEqual(overlap),
                  isMeaningfulBoundaryOverlap(overlap),
                  hasLatinWordBoundaries(
                    before: before,
                    incoming: incoming,
                    overlapLength: length,
                    overlap: overlap
                  ) else {
                continue
            }
            return String(incoming.dropFirst(length))
        }
        return text
    }

    private func isMeaningfulBoundaryOverlap(_ overlap: Substring) -> Bool {
        if Self.singleCharacterRestartOverlaps.contains(String(overlap)) { return true }
        return overlap.filter({ $0.isLetter || $0.isNumber }).count >= 2
    }

    func prepareForInsertion(_ text: String, textBefore: String) -> String {
        removeOverlappingPrefix(
            removeAdjacentRepeatedPhrases(text),
            textBefore: textBefore
        )
    }

    /// Collapses an immediately repeated speech phrase inside one dictation.
    /// Three content characters keeps ordinary forms such as「看看」and
    ///「非常非常」outside this automatic correction.
    func removeAdjacentRepeatedPhrases(_ text: String) -> String {
        var characters = Array(text)

        while let repeatedRange = adjacentRepeatedPhraseRange(in: characters) {
            characters.removeSubrange(repeatedRange)
        }
        return String(characters)
    }

    private func adjacentRepeatedPhraseRange(in characters: [Character]) -> Range<Int>? {
        guard characters.count >= 6 else { return nil }

        for start in characters.indices {
            let maximumLength = (characters.count - start) / 2
            guard maximumLength > 0 else { continue }

            for length in stride(from: maximumLength, through: 1, by: -1) {
                let secondStart = start + length
                let secondEnd = secondStart + length
                let phrase = characters[start..<secondStart]
                guard isMeaningfulRepeatedPhrase(phrase),
                      phrase.elementsEqual(characters[secondStart..<secondEnd]) else {
                    continue
                }
                return secondStart..<secondEnd
            }
        }
        return nil
    }

    private func isMeaningfulRepeatedPhrase(_ phrase: ArraySlice<Character>) -> Bool {
        let content = phrase.filter { $0.isLetter || $0.isNumber }
        return content.count >= 3 && Set(content).count >= 2
    }

    private static let singleCharacterRestartOverlaps: Set<String> = ["又", "就", "也", "還", "再", "都", "才", "只"]

    private func hasLatinWordBoundaries(
        before: Substring,
        incoming: Substring,
        overlapLength: Int,
        overlap: Substring
    ) -> Bool {
        if overlap.first?.isASCIILetter == true, before.count > overlapLength {
            let precedingIndex = before.index(before.endIndex, offsetBy: -overlapLength - 1)
            if before[precedingIndex].isASCIILetter { return false }
        }
        if overlap.last?.isASCIILetter == true, incoming.count > overlapLength {
            let followingIndex = incoming.index(incoming.startIndex, offsetBy: overlapLength)
            if incoming[followingIndex].isASCIILetter { return false }
        }
        return true
    }

    // MARK: - Rule Implementations

    /// Rule 1: If both sides are Latin letters, ensure exactly one space between them.
    private func adjustLeadingSpace(_ text: String, textBefore: String) -> String {
        guard !text.isEmpty, !textBefore.isEmpty else { return text }
        let lastBefore = textBefore.last!
        let firstInserted = text.first!

        // Both are Latin letters — need a space between
        let needsSpace = lastBefore.isLetter && !lastBefore.isCJK
            && firstInserted.isLetter && !firstInserted.isCJK

        if needsSpace && lastBefore != " " && firstInserted != " " {
            return " " + text
        }
        // Prevent double space
        if lastBefore == " " && text.hasPrefix(" ") {
            return String(text.dropFirst())
        }
        return text
    }

    /// Rule 2: Remove duplicate punctuation at the insertion boundaries.
    private func preventDuplicatePunctuation(_ text: String, textBefore: String, textAfter: String) -> String {
        guard !text.isEmpty else { return text }
        var result = text

        // Leading duplicate: inserted text starts with same punctuation as textBefore ends with
        if let lastBefore = textBefore.last, let firstInserted = result.first,
           lastBefore == firstInserted && (lastBefore.isPunctuation || lastBefore.isCJKPunctuation) {
            result = String(result.dropFirst())
        }
        // Trailing duplicate: inserted text ends with same punctuation as textAfter starts with
        if !result.isEmpty,
           let lastInserted = result.last, let firstAfter = textAfter.first,
           lastInserted == firstAfter && (lastInserted.isPunctuation || lastInserted.isCJKPunctuation) {
            result = String(result.dropLast())
        }
        return result
    }

    /// Rule 3: Capitalize at sentence start, lowercase mid-sentence (but preserve acronyms).
    private func adjustCapitalization(_ text: String, textBefore: String) -> String {
        guard !text.isEmpty, let firstChar = text.first, firstChar.isLetter else { return text }
        // Only adjust Latin characters
        guard !firstChar.isCJK else { return text }

        let trimmedBefore = textBefore.trimmingCharacters(in: .whitespaces)

        let atSentenceStart = trimmedBefore.isEmpty
            || trimmedBefore.hasSuffix(".")
            || trimmedBefore.hasSuffix("!")
            || trimmedBefore.hasSuffix("?")
            || trimmedBefore.hasSuffix("。")
            || trimmedBefore.hasSuffix("！")
            || trimmedBefore.hasSuffix("？")

        if atSentenceStart {
            // Uppercase first letter
            if firstChar.isLowercase {
                return text.prefix(1).uppercased() + text.dropFirst()
            }
        } else {
            // Mid-sentence: lowercase the first letter, unless it's an acronym (all uppercase word)
            let firstWord = text.prefix(while: { $0.isLetter })
            if firstWord.count > 1 && firstChar.isUppercase
                && firstWord.dropFirst().allSatisfy({ $0.isLowercase }) {
                // Single capitalized word like "Hello" → "hello", but keep "API", "HTTP" etc.
                return text.prefix(1).lowercased() + text.dropFirst()
            }
        }
        return text
    }

    /// Rule 4: Add space at CJK-Latin boundaries.
    private func adjustCJKLatinSpacing(_ text: String, textBefore: String, textAfter: String) -> String {
        guard !text.isEmpty else { return text }
        var result = text

        // Leading boundary: CJK before + Latin inserted (or vice versa)
        if let lastBefore = textBefore.last, lastBefore != " ",
           let firstInserted = result.first, firstInserted != " " {
            if (lastBefore.isCJK && firstInserted.isASCII && firstInserted.isLetter)
                || (lastBefore.isASCII && lastBefore.isLetter && firstInserted.isCJK) {
                result = " " + result
            }
        }
        // Trailing boundary: Latin inserted + CJK after (or vice versa)
        if let lastInserted = result.last, lastInserted != " ",
           let firstAfter = textAfter.first, firstAfter != " " {
            if (lastInserted.isASCII && lastInserted.isLetter && firstAfter.isCJK)
                || (lastInserted.isCJK && firstAfter.isASCII && firstAfter.isLetter) {
                result = result + " "
            }
        }
        return result
    }

    /// Rule 5: Smart trailing space — only add when the character after cursor isn't already a space or punctuation.
    private func adjustTrailingSpace(_ text: String, textAfter: String, appendSpaceSetting: Bool) -> String {
        guard appendSpaceSetting, !text.isEmpty else { return text }

        // If nothing after cursor (end of field), follow the setting
        if textAfter.isEmpty { return text + " " }

        let firstAfter = textAfter.first!
        // Don't append if next char is already space, punctuation, or CJK punctuation
        if firstAfter == " " || firstAfter.isPunctuation || firstAfter.isCJKPunctuation {
            return text
        }
        return text + " "
    }
}

// MARK: - Character Extensions

// Character.isCJK is defined in CorrectionEngine.swift

extension Character {
    fileprivate var isASCIILetter: Bool {
        unicodeScalars.count == 1 && unicodeScalars.first.map {
            (65...90).contains(Int($0.value)) || (97...122).contains(Int($0.value))
        } == true
    }

    var isCJKPunctuation: Bool {
        let cjkPunct: Set<Character> = ["，", "。", "？", "！", "、", "；", "：", "「", "」", "（", "）", "《", "》", "【", "】", "〈", "〉"]
        return cjkPunct.contains(self)
    }
}
