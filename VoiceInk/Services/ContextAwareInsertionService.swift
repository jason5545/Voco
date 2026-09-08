import Foundation
import os

final class ContextAwareInsertionService {
    static let shared = ContextAwareInsertionService()
    static let adjacentRepeatedPhraseRuleID = "context-aware-insertion.adjacent-phrase-dedup.v2"

    struct DeduplicationResult: Equatable {
        let text: String
        let events: [DeduplicationEvent]
    }

    struct DeduplicationEvent: Equatable {
        enum Decision: String, Equatable {
            case removed
            case preservedForReview
        }

        let ruleID: String
        let decision: Decision
        let reason: String
        let beforeText: String
        let afterText: String
        let repeatedPhrase: String
        /// Character offsets in `beforeText` covering both adjacent copies.
        let matchedRange: Range<Int>
        /// Character offsets in `beforeText`; nil when the candidate is preserved.
        let removedRange: Range<Int>?
    }

    private struct AdjacentRepeatedPhraseCandidate {
        let firstRange: Range<Int>
        let duplicateRange: Range<Int>

        var matchedRange: Range<Int> {
            firstRange.lowerBound..<duplicateRange.upperBound
        }
    }

    private struct ReviewCueFamily {
        let reason: String
        let phrases: [String]
        let patterns: [String]
    }

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
    ///「非常非常」outside this automatic correction. Candidates containing
    /// semantic control cues, literal/quoted text, or ASCII and numeric tokens
    /// are preserved for review because an exact character repeat may still
    /// carry meaning.
    func removeAdjacentRepeatedPhrases(_ text: String) -> String {
        let result = deduplicateAdjacentRepeatedPhrases(text)
        result.events.forEach(logDeduplicationEvent)
        return result.text
    }

    func deduplicateAdjacentRepeatedPhrases(_ text: String) -> DeduplicationResult {
        var characters = Array(text)
        var events: [DeduplicationEvent] = []
        var searchStart = 0

        while let candidate = adjacentRepeatedPhraseCandidate(
            in: characters,
            startingAt: searchStart
        ) {
            let beforeText = String(characters)
            let repeatedPhrase = String(characters[candidate.firstRange])
            let clause = surroundingClause(in: characters, matchedRange: candidate.matchedRange)

            if let protectedReason = protectedReviewReason(
                repeatedPhrase: repeatedPhrase,
                surroundingClause: clause,
                isInsideLiteralQuote: isInsideLiteralQuote(
                    in: characters,
                    matchedRange: candidate.matchedRange
                )
            ) {
                events.append(
                    DeduplicationEvent(
                        ruleID: Self.adjacentRepeatedPhraseRuleID,
                        decision: .preservedForReview,
                        reason: protectedReason,
                        beforeText: beforeText,
                        afterText: beforeText,
                        repeatedPhrase: repeatedPhrase,
                        matchedRange: candidate.matchedRange,
                        removedRange: nil
                    )
                )
                searchStart = candidate.duplicateRange.upperBound
                continue
            }

            characters.removeSubrange(candidate.duplicateRange)
            let afterText = String(characters)
            events.append(
                DeduplicationEvent(
                    ruleID: Self.adjacentRepeatedPhraseRuleID,
                    decision: .removed,
                    reason: "exact-adjacent-phrase-repeat",
                    beforeText: beforeText,
                    afterText: afterText,
                    repeatedPhrase: repeatedPhrase,
                    matchedRange: candidate.matchedRange,
                    removedRange: candidate.duplicateRange
                )
            )
            // Re-scan after a mutation so three or more adjacent copies collapse.
            searchStart = 0
        }

        return DeduplicationResult(text: String(characters), events: events)
    }

    private func adjacentRepeatedPhraseCandidate(
        in characters: [Character],
        startingAt searchStart: Int
    ) -> AdjacentRepeatedPhraseCandidate? {
        guard characters.count >= 6, searchStart < characters.count else { return nil }

        for start in searchStart..<characters.count {
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
                return AdjacentRepeatedPhraseCandidate(
                    firstRange: start..<secondStart,
                    duplicateRange: secondStart..<secondEnd
                )
            }
        }
        return nil
    }

    private func isMeaningfulRepeatedPhrase(_ phrase: ArraySlice<Character>) -> Bool {
        let content = phrase.filter { $0.isLetter || $0.isNumber }
        return content.count >= 3 && Set(content).count >= 2
    }

    private func surroundingClause(
        in characters: [Character],
        matchedRange: Range<Int>
    ) -> String {
        var lowerBound = matchedRange.lowerBound
        while lowerBound > 0, !Self.clauseBoundaries.contains(characters[lowerBound - 1]) {
            lowerBound -= 1
        }

        var upperBound = matchedRange.upperBound
        while upperBound < characters.count, !Self.clauseBoundaries.contains(characters[upperBound]) {
            upperBound += 1
        }
        return String(characters[lowerBound..<upperBound])
    }

    private func protectedReviewReason(
        repeatedPhrase: String,
        surroundingClause: String,
        isInsideLiteralQuote: Bool
    ) -> String? {
        if isInsideLiteralQuote
            || repeatedPhrase.contains(where: { Self.quoteMarkers.contains($0) })
            || matches(Self.literalOrMetaCueFamily, in: surroundingClause) {
            return Self.literalOrMetaCueFamily.reason
        }
        if matches(Self.selfRepairCueFamily, in: surroundingClause) {
            return Self.selfRepairCueFamily.reason
        }
        if let reason = firstMatchingReason(
            in: repeatedPhrase,
            families: Self.semanticCueFamilies
        ) {
            return reason
        }
        if let reason = firstMatchingReason(
            in: surroundingClause,
            families: Self.semanticCueFamilies
        ) {
            return reason
        }
        if repeatedPhrase.unicodeScalars.contains(where: {
            $0.isASCII && CharacterSet.alphanumerics.contains($0)
        }) {
            return "review-protected-ascii-or-number"
        }
        return nil
    }

    private func firstMatchingReason(
        in text: String,
        families: [ReviewCueFamily]
    ) -> String? {
        families.first(where: { matches($0, in: text) })?.reason
    }

    private func matches(_ family: ReviewCueFamily, in text: String) -> Bool {
        containsAny(family.phrases, in: text)
            || family.patterns.contains(where: { matchesCuePattern($0, in: text) })
    }

    private func containsAny(_ cues: [String], in text: String) -> Bool {
        cues.contains(where: text.contains)
    }

    private func matchesCuePattern(_ pattern: String, in text: String) -> Bool {
        text.range(of: pattern, options: [.regularExpression, .caseInsensitive]) != nil
    }

    private func isInsideLiteralQuote(
        in characters: [Character],
        matchedRange: Range<Int>
    ) -> Bool {
        for pair in Self.directionalQuotePairs {
            var depth = 0
            for character in characters[..<matchedRange.lowerBound] {
                if character == pair.open {
                    depth += 1
                } else if character == pair.close, depth > 0 {
                    depth -= 1
                }
            }
            if depth > 0,
               characters[matchedRange.upperBound...].contains(pair.close) {
                return true
            }
        }

        for quote in Self.symmetricQuoteCharacters {
            let precedingCount = characters[..<matchedRange.lowerBound]
                .filter { $0 == quote }
                .count
            if precedingCount.isMultiple(of: 2) == false,
               characters[matchedRange.upperBound...].contains(quote) {
                return true
            }
        }
        return false
    }

    private func logDeduplicationEvent(_ event: DeduplicationEvent) {
        let matchedRange = "\(event.matchedRange.lowerBound)..<\(event.matchedRange.upperBound)"
        let removedRange = event.removedRange.map {
            "\($0.lowerBound)..<\($0.upperBound)"
        } ?? "none"
        logger.notice(
            "Adjacent repetition rule=\(event.ruleID, privacy: .public) decision=\(event.decision.rawValue, privacy: .public) reason=\(event.reason, privacy: .public) matchedRange=\(matchedRange, privacy: .public) removedRange=\(removedRange, privacy: .public) repeatedPhrase=\(event.repeatedPhrase, privacy: .private(mask: .hash)) before=\(event.beforeText, privacy: .private) after=\(event.afterText, privacy: .private)"
        )
    }

    private static let clauseBoundaries: Set<Character> = ["。", "！", "？", ".", "!", "?", "；", ";", "\n", "\r"]
    private static let literalOrMetaCueFamily = ReviewCueFamily(
        reason: "review-protected-literal-or-quoted-text",
        phrases: [
            "字面", "原文", "逐字", "照抄", "請寫", "请写", "寫出", "写出",
            "請保留", "请保留", "保留", "請輸出", "请输出", "輸出", "输出",
            "重複", "重复", "不要改", "別改", "别改", "這幾個字", "这几个字",
            "這段文字", "这段文字", "字串", "字符串",
        ],
        patterns: ["\\b(?:literal|literally|verbatim|quote|quoted|string)\\b"]
    )
    private static let selfRepairCueFamily = ReviewCueFamily(
        reason: "review-protected-self-repair-cue",
        phrases: [
            "應該是", "应该是", "我的意思是", "我是說", "我是说", "其實是", "其实是",
            "更正", "修正一下", "應該說", "应该说", "改成", "改口", "重來", "重来",
            "重新說", "重新说", "不對", "不对",
        ],
        patterns: [
            "(?:^|[，,、；;：:\\s])等一下(?:[，,、；;：:]|$)",
            "\\b(?:i\\s+mean|correction|sorry|wait)\\b",
        ]
    )
    private static let semanticCueFamilies = [
        ReviewCueFamily(
            reason: "review-protected-contrast-cue",
            phrases: ["但是", "但", "不過", "不过", "可是", "然而", "卻", "却", "反而"],
            patterns: ["\\b(?:but|however|instead)\\b|\\brather\\s+than\\b"]
        ),
        ReviewCueFamily(
            reason: "review-protected-comparison-cue",
            phrases: [
                "或者", "還是", "还是", "或是", "比較", "比较", "相比", "相較", "相较",
                "比起", "不如", "不同", "一樣", "一样", "前者", "後者", "后者",
            ],
            patterns: ["\\b(?:or|versus|vs)\\b"]
        ),
        ReviewCueFamily(
            reason: "review-protected-negation-cue",
            phrases: [
                "不要", "不能", "不是", "沒有", "没有", "別", "别", "非", "勿", "莫",
                "甭", "不", "沒", "没", "未", "無", "无",
            ],
            patterns: ["\\b(?:not|no|cannot|can't|don't|doesn't)\\b"]
        ),
    ]
    private static let directionalQuotePairs: [(open: Character, close: Character)] = [
        ("「", "」"),
        ("『", "』"),
        ("“", "”"),
        ("‘", "’"),
        ("〈", "〉"),
        ("《", "》"),
    ]
    private static let symmetricQuoteCharacters: Set<Character> = ["\""]
    private static let quoteMarkers: Set<Character> = [
        "「", "」", "『", "』", "“", "”", "‘", "’", "〈", "〉", "《", "》", "\"",
    ]
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
