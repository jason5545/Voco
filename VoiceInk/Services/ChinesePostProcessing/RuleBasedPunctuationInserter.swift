import Foundation

/// Pure Swift rule-based punctuation inserter for CJK text.
/// Used as the final fallback when all LLM-based punctuation attempts fail.
enum RuleBasedPunctuationInserter {

    // MARK: - Break-after particles
    // Insert comma after these words when followed by more CJK text
    private static let breakAfterParticles: [String] = [
        "但是", "因為", "所以", "然後", "如果", "不過", "而且", "或者",
        "可是", "雖然", "因此", "於是", "另外", "而是", "總之", "結果",
        "但", "的", "了", "嘛",
    ]

    // MARK: - Break-before pronouns (only when preceded by 6+ CJK chars)
    private static let breakBeforePronouns: Set<Character> = ["我", "你", "他", "她", "它"]

    // MARK: - Question tail patterns
    private static let questionTails: [String] = [
        "嗎", "呢", "吧",
    ]
    private static let questionPhrases: [String] = [
        "是不是", "會不會", "能不能", "可不可以", "好不好", "對不對",
        "有沒有", "要不要",
    ]

    // MARK: - CJK helpers

    private static func isCJK(_ scalar: Unicode.Scalar) -> Bool {
        (0x4E00...0x9FFF).contains(scalar.value) || (0x3400...0x4DBF).contains(scalar.value)
    }

    private static func isCJKChar(_ c: Character) -> Bool {
        c.unicodeScalars.first.map { isCJK($0) } ?? false
    }

    private static let sentenceEndPunctuation: Set<Character> = ["。", "？", "！", "…"]
    private static let allCJKPunctuation: Set<Character> = ["，", "。", "？", "！", "、", "；", "：", "…"]

    // MARK: - Public API

    /// Insert rule-based punctuation into CJK text that has no/insufficient punctuation.
    /// Returns the text with commas, periods, and question marks inserted at natural break points.
    static func insert(into text: String) -> String {
        guard !text.isEmpty else { return text }

        var chars = Array(text)
        var result: [Character] = []
        var cjkRun = 0  // consecutive CJK chars since last punctuation

        var i = 0
        while i < chars.count {
            let c = chars[i]

            if allCJKPunctuation.contains(c) || c == "," || c == "." || c == "?" || c == "!" {
                result.append(c)
                cjkRun = 0
                i += 1
                continue
            }

            if isCJKChar(c) {
                // Check break-after: does a particle end here?
                var didBreakAfter = false
                if cjkRun >= 2 {
                    for particle in breakAfterParticles {
                        let pLen = particle.count
                        if result.count >= pLen {
                            let tail = String(result.suffix(pLen))
                            if tail == particle {
                                // Check the char at (result.count - pLen) is CJK (not punctuation)
                                // to avoid inserting comma right after another punctuation
                                let beforeParticle = result.count > pLen ? result[result.count - pLen - 1] : nil
                                let afterIsCJK = isCJKChar(c)
                                if afterIsCJK, beforeParticle == nil || (beforeParticle != nil && !allCJKPunctuation.contains(beforeParticle!)) {
                                    // Don't insert if 「的」 is just part of normal flow < 6 chars
                                    if particle == "的" || particle == "了" || particle == "嘛" {
                                        if cjkRun >= 8 {
                                            result.append("，")
                                            cjkRun = 0
                                            didBreakAfter = true
                                        }
                                    } else {
                                        result.append("，")
                                        cjkRun = 0
                                        didBreakAfter = true
                                    }
                                }
                                if didBreakAfter { break }
                            }
                        }
                    }
                }

                // Check break-before: pronoun preceded by 6+ CJK chars
                if !didBreakAfter && cjkRun >= 6 && breakBeforePronouns.contains(c) {
                    // Only break if previous char is CJK (not punctuation)
                    if let last = result.last, isCJKChar(last) {
                        result.append("，")
                        cjkRun = 0
                    }
                }

                // Force comma for very long runs (>15 CJK without punctuation)
                if cjkRun >= 15 {
                    // Try to find a reasonable break point in the last few chars
                    // Simple heuristic: just insert here
                    if let last = result.last, isCJKChar(last) {
                        result.append("，")
                        cjkRun = 0
                    }
                }

                result.append(c)
                cjkRun += 1
            } else {
                // Non-CJK char (space, English, etc.)
                result.append(c)
                // Reset CJK run on spaces/English
                if c == " " || c == "\n" {
                    cjkRun = 0
                }
            }

            i += 1
        }

        // Sentence-end punctuation
        var output = String(result)
        output = addSentenceEndPunctuation(output)

        return output
    }

    // MARK: - Sentence-end punctuation

    private static func addSentenceEndPunctuation(_ text: String) -> String {
        guard !text.isEmpty else { return text }
        var chars = Array(text)

        // Trim trailing whitespace for check
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return text }
        let lastChar = trimmed.last!

        // If already ends with sentence-end punctuation, return as-is
        if sentenceEndPunctuation.contains(lastChar) || lastChar == "，" {
            return text
        }

        // Check for question patterns
        if isQuestion(trimmed) {
            // Find last non-whitespace position and append question mark
            if let idx = chars.lastIndex(where: { !$0.isWhitespace && !$0.isNewline }) {
                chars.insert("？", at: chars.index(after: idx))
                return String(chars)
            }
        }

        // Default: add period if last char is CJK
        if isCJKChar(lastChar) {
            if let idx = chars.lastIndex(where: { !$0.isWhitespace && !$0.isNewline }) {
                chars.insert("。", at: chars.index(after: idx))
                return String(chars)
            }
        }

        return String(chars)
    }

    private static func isQuestion(_ text: String) -> Bool {
        for tail in questionTails {
            if text.hasSuffix(tail) { return true }
        }
        for phrase in questionPhrases {
            if text.contains(phrase) { return true }
        }
        return false
    }
}
