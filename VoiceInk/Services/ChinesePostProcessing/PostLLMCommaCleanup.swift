import Foundation

/// Removes commas that the LLM incorrectly inserted after particles like 的/了.
///
/// The LLM frequently inserts commas that break up valid phrases:
///   "奇怪的，地方" → "奇怪的地方"
///   "修了，一下"   → "修了一下"
///
/// This cleanup runs after all LLM processing as a deterministic safety net.
enum PostLLMCommaCleanup {

    // MARK: - Exception sets

    /// After 的+comma, if the text following the comma starts with any of these,
    /// the comma is likely correct (new clause with subject change or conjunction).
    private static let deCommaKeepPrefixes: [String] = [
        // Multi-char (check first to avoid partial matches)
        "我們", "你們", "他們", "她們",
        "大家", "自己",
        "但是", "不過", "可是", "所以", "因為",
        "而且", "然後", "如果", "雖然", "於是", "因此",
        "只要", "只有", "只是", "即使", "儘管",
        "除非", "否則", "既然", "無論", "不管",
        "不是", "不要", "不能", "不會", "不可",
        // Single-char
        "我", "你", "您", "他", "她", "它",
        "但", "卻", "再", "只",
    ]

    /// Response words that form interjections with 的 (好的/對的/是的/行的).
    /// When these appear right before 的 at sentence start or after punctuation,
    /// 的 is a sentence-final particle (= "OK"), not a modifier.
    private static let interjectionChars: Set<Character> = ["好", "對", "是", "行"]

    /// Punctuation that indicates a sentence/clause boundary.
    private static let boundaryPunctuation: Set<Character> = [
        "，", "。", "？", "！", "、", "；", "：", "…",
        ",", ".", "?", "!", ":", ";",
        "\n",
    ]

    /// Complement patterns after 了 that should never have a comma.
    /// e.g. "修了一下" not "修了，一下"
    private static let leComplementPrefixes: [String] = [
        "一下", "一些", "一點", "一番", "一會", "一陣",
    ]

    /// Characters that form a word with 了 when 了 is the first char.
    /// e.g. "了解" not "了，解"
    /// Note: 然/卻 excluded — too easily confused with 然後/卻是 (conjunctions)
    private static let leWordSecondChars: Set<Character> = [
        "解", "結", "斷", "事", "得", "無",
    ]

    // MARK: - Verb/Preposition + pronoun sets

    /// Prepositions/complements that always take pronoun objects.
    /// Comma between these and a following pronoun is always wrong.
    /// e.g. "把，他帶走" → "把他帶走", "拖累到，他的" → "拖累到他的"
    private static let safePrepositions: Set<Character> = ["把", "被", "替", "到"]

    /// Pronouns that can be verb/preposition objects (multi-char first for prefix matching).
    private static let objectPronouns: [String] = [
        "我們", "你們", "他們", "她們",
        "我", "你", "您", "他", "她", "它",
    ]

    /// Fixed phrases where a comma should never be inserted.
    /// Each entry is (before_comma_suffix, after_comma_prefix).
    /// e.g. "另外，一個" → "另外一個", "其中，一個" → "其中一個"
    private static let fixedPhrases: [(String, String)] = [
        ("另外", "一"), ("其中", "一"), ("其他", "一"),
        ("另外", "也"), ("另外", "還"),
    ]

    /// Sentence-final particles. Comma after these + pronoun is a real clause break.
    /// e.g. "他說了，他的意思是..." — 了 is particle, comma is correct.
    private static let sentenceFinalParticles: Set<Character> = [
        "了", "的", "嗎", "呢", "吧", "啊", "呀", "哦", "嘛", "囉",
    ]

    // MARK: - Public API

    /// Apply comma cleanup rules to LLM-enhanced text.
    /// - Parameters:
    ///   - text: The LLM-enhanced text to clean.
    ///   - originalText: The ASR original text (ZTEXT). When provided, a final
    ///     comparison pass removes commas that the LLM inserted between characters
    ///     that were adjacent in the original.
    static func clean(_ text: String, originalText: String? = nil) -> String {
        guard text.count >= 3 else { return text }
        var result = text
        result = cleanDeComma(result)
        result = cleanLeComma(result)
        result = cleanVerbPronounComma(result)
        result = cleanFixedPhrases(result)
        if let originalText {
            result = cleanByOriginalComparison(result, originalText: originalText)
        }
        return result
    }

    // MARK: - 的+，cleanup

    /// Remove comma after 的 when it incorrectly splits a modifier-noun phrase.
    ///
    /// Rule: 的+，+CJK → remove comma
    /// Exceptions (keep comma):
    /// 1. Text after comma starts with a subject pronoun or conjunction
    /// 2. 的 is part of an interjection (好的/對的/是的 at sentence boundary)
    /// 3. 的 follows a verb phrase (sentence-final nominalizer, e.g. "可以部署的，標記為...")
    private static func cleanDeComma(_ text: String) -> String {
        let chars = Array(text)
        var result: [Character] = []
        var i = 0

        while i < chars.count {
            if chars[i] == "的"
                && i + 1 < chars.count && chars[i + 1] == "，"
                && i + 2 < chars.count && isCJK(chars[i + 2])
            {
                // Exception 1: text after comma starts with a clause-starting word
                let afterComma = String(chars[(i + 2)...])
                let startsNewClause = deCommaKeepPrefixes.contains { afterComma.hasPrefix($0) }

                // Exception 2: interjection pattern (好的，/對的，/是的，)
                // 的 preceded by a response word at sentence start or after punctuation
                let isInterjection: Bool = {
                    guard i >= 1, interjectionChars.contains(chars[i - 1]) else { return false }
                    // The response word must be at start or after punctuation/space
                    if i == 1 { return true }
                    let beforeResponse = chars[i - 2]
                    return boundaryPunctuation.contains(beforeResponse) || beforeResponse == " "
                }()

                // Exception 3: sentence-final 的 after a verb phrase
                // Heuristic: if the segment before 的 ends with a verb-like pattern
                // and the segment after comma also starts a new action (contains 為/成),
                // then 的 is nominalizing, not modifying.
                let isSentenceFinal: Bool = {
                    // Look at the segment after comma until next punctuation
                    let afterSegment = String(chars[(i + 2)...])
                    let nextPunctIdx = afterSegment.firstIndex { boundaryPunctuation.contains($0) }
                    let segment = nextPunctIdx != nil
                        ? String(afterSegment[afterSegment.startIndex..<nextPunctIdx!])
                        : afterSegment
                    // If the segment after comma looks like a verb phrase (contains 為/成/給),
                    // and does NOT look like a noun being modified by 的,
                    // then 的 is likely sentence-final
                    let verbMarkers: [String] = ["為", "成"]
                    return segment.count >= 2 && verbMarkers.contains(where: { segment.contains($0) })
                }()

                let shouldKeep = startsNewClause || isInterjection || isSentenceFinal

                if !shouldKeep {
                    result.append("的")
                    i += 2  // skip 的 and ，
                    continue
                }
            }

            result.append(chars[i])
            i += 1
        }

        return String(result)
    }

    // MARK: - 了+，cleanup

    /// Remove comma after 了 only for known complement/word-split patterns.
    ///
    /// Conservative: only handles clear-cut cases.
    /// - Complement: 了+一下/一些/一點/... (V了一下 construction)
    /// - Word split: 了+解/結/... (了解/了結 are single words)
    private static func cleanLeComma(_ text: String) -> String {
        let chars = Array(text)
        var result: [Character] = []
        var i = 0

        while i < chars.count {
            if chars[i] == "了"
                && i + 1 < chars.count && chars[i + 1] == "，"
                && i + 2 < chars.count
            {
                let afterComma = String(chars[(i + 2)...])

                // Check complement patterns: 了，一下 → 了一下
                let isComplement = leComplementPrefixes.contains { afterComma.hasPrefix($0) }

                // Check word-split: 了，解 → 了解
                let isWordSplit = leWordSecondChars.contains(chars[i + 2])

                if isComplement || isWordSplit {
                    result.append("了")
                    i += 2  // skip 了 and ，
                    continue
                }
            }

            result.append(chars[i])
            i += 1
        }

        return String(result)
    }

    // MARK: - Verb/Prep + ，+ pronoun cleanup

    /// Remove comma between a verb/preposition and its pronoun object.
    ///
    /// Rule 1: Safe preposition (把/被/替/到) + ，+ pronoun → always remove
    /// Rule 2: non-particle CJK + ，+ pronoun + 的 → remove (pronoun is possessive object)
    private static func cleanVerbPronounComma(_ text: String) -> String {
        let chars = Array(text)
        var result: [Character] = []
        var i = 0

        while i < chars.count {
            if chars[i] == "，"
                && i >= 1
                && i + 1 < chars.count
            {
                let beforeComma = chars[i - 1]
                let afterComma = String(chars[(i + 1)...])

                // Find matching pronoun after comma
                let matchedPronoun = objectPronouns.first { afterComma.hasPrefix($0) }

                if let pronoun = matchedPronoun {
                    let afterPronoun = String(afterComma.dropFirst(pronoun.count))

                    // Rule 1: safe preposition before comma
                    if safePrepositions.contains(beforeComma) {
                        i += 1  // skip comma
                        continue
                    }

                    // Rule 2: pronoun+的, and char before comma is not a sentence-final particle
                    if afterPronoun.hasPrefix("的")
                        && !sentenceFinalParticles.contains(beforeComma)
                        && isCJK(beforeComma)
                    {
                        i += 1  // skip comma
                        continue
                    }
                }
            }

            result.append(chars[i])
            i += 1
        }

        return String(result)
    }

    // MARK: - Fixed phrase cleanup

    /// Remove commas that split known fixed phrases.
    /// e.g. "另外，一個" → "另外一個"
    private static func cleanFixedPhrases(_ text: String) -> String {
        var result = text
        for (before, after) in fixedPhrases {
            result = result.replacingOccurrences(of: "\(before)，\(after)", with: "\(before)\(after)")
        }
        return result
    }

    // MARK: - ZTEXT-guided comparison cleanup

    /// Remove commas that the LLM inserted between characters that were adjacent
    /// (no punctuation between them) in the ASR original text (ZTEXT).
    ///
    /// Key insight: check the original text directly WITHOUT stripping punctuation.
    /// If the original has "結果出來", the pair "果出" exists as a substring → remove.
    /// If the original also has "想說，我們", the pair "說我" is NOT adjacent → keep.
    ///
    /// Safety valve: if the CJK run since the last punctuation exceeds 12 characters,
    /// the comma is kept (long sentences need pauses).
    private static func cleanByOriginalComparison(_ text: String, originalText: String) -> String {
        guard !originalText.isEmpty else { return text }

        let chars = Array(text)
        var result: [Character] = []
        var cjkRunSinceLastPunct = 0

        for i in 0..<chars.count {
            let ch = chars[i]

            if ch == "，"
                && i >= 1
                && i + 1 < chars.count
                && isCJK(chars[i - 1])
                && isCJK(chars[i + 1])
            {
                let pair = String([chars[i - 1], chars[i + 1]])

                if originalText.contains(pair) {
                    // Compute forward CJK run to next punctuation
                    var cjkRunRight = 0
                    for j in (i + 1)..<chars.count {
                        if isCJK(chars[j]) {
                            cjkRunRight += 1
                        } else if boundaryPunctuation.contains(chars[j]) {
                            break
                        }
                    }

                    // Safety valve: keep comma if BOTH sides are long
                    // (provides needed structure in long sentences).
                    // But if right side is very short (<= 3), it's a word split — always remove.
                    if cjkRunSinceLastPunct <= 12 || cjkRunRight <= 3 {
                        continue
                    }
                }
            }

            result.append(ch)

            // Track CJK run length for safety valve
            if isCJK(ch) {
                cjkRunSinceLastPunct += 1
            } else if boundaryPunctuation.contains(ch) {
                cjkRunSinceLastPunct = 0
            }
        }

        return String(result)
    }

    // MARK: - Helpers

    private static func isCJK(_ c: Character) -> Bool {
        c.unicodeScalars.first.map {
            (0x4E00...0x9FFF).contains($0.value) || (0x3400...0x4DBF).contains($0.value)
        } ?? false
    }

}
