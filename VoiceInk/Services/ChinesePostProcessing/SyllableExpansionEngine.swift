import Foundation
import NaturalLanguage
import os

/// Syllable expansion engine for correcting compressed syllable errors.
///
/// Whisper sometimes compresses two syllables into one character,
/// e.g., "語音" (yǔ yīn) → "硬" (yìng). This engine detects such
/// errors using bigram context and expands them back to 2-char words.
///
/// Pipeline position: Step 3, Layer 3 (after HomophoneCorrectionEngine).
final class SyllableExpansionEngine {
    static let shared = SyllableExpansionEngine()

    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "SyllableExpansion")
    private let db = PinyinDatabase.shared

    // MARK: - Tuning Constants

    /// Weight for bigram context improvement
    private let bigramWeight: Double = 0.5

    /// Weight for internal bigram of the 2-char candidate word
    private let internalBigramWeight: Double = 0.3

    /// Penalty per unit of merge edit distance
    private let distancePenalty: Double = 2.0

    /// Minimum context improvement required (hard gate)
    private let minContextImprovement: Double = 3.0

    /// Minimum total score required to apply correction
    private let minTotalScore: Double = 7.0

    /// Maximum bigram frequency for a character to be considered "out of context"
    private let suspiciousBigramThreshold: Int = 50

    /// Common function words to skip (same set as HomophoneCorrectionEngine)
    private static let skipChars: Set<Character> = [
        "的", "了", "嗎", "呢", "吧", "啊", "哦", "喔", "嗯", "呀",
        "是", "在", "有", "和", "也", "都", "就", "不", "我", "你",
        "他", "她", "它", "們", "這", "那", "個", "把", "被", "讓",
        "會", "能", "可", "要", "得", "地", "著", "過", "到", "從",
        "與", "及", "或", "而", "但", "因", "為", "所", "以", "如",
    ]

    // MARK: - Cache

    /// Lazy cache: toneless pinyin → [ExpansionCandidate]
    private var expansionCache: [String: [ExpansionCandidate]] = [:]
    private let cacheLock = NSLock()

    /// Pre-computed merge index: mergePinyin → [(word, freq)]
    /// Built lazily on first use.
    private var mergeIndex: [String: [(word: String, freq: Int)]]?
    private let mergeIndexLock = NSLock()

    private struct ExpansionCandidate {
        let word: String
        let freq: Int
        let mergeDist: Int
    }

    // MARK: - Result Type

    struct CorrectionResult {
        let text: String
        let corrections: [Correction]

        struct Correction {
            let original: String
            let corrected: String
            let score: Double
        }
    }

    // MARK: - Pinyin Helpers

    /// All Mandarin initials, ordered longest first for greedy matching
    private static let initials = [
        "zh", "ch", "sh",
        "b", "p", "m", "f",
        "d", "t", "n", "l",
        "g", "k", "h",
        "j", "q", "x",
        "r", "z", "c", "s",
        "y", "w",
    ]

    /// Extract the initial of a pinyin syllable
    static func pinyinInitial(_ pinyin: String) -> String {
        for ini in initials {
            if pinyin.hasPrefix(ini) {
                return ini
            }
        }
        return "" // zero initial (e.g., "a", "e", "o", "an")
    }

    /// Extract the final of a pinyin syllable (everything after the initial)
    static func pinyinFinal(_ pinyin: String) -> String {
        let ini = pinyinInitial(pinyin)
        return String(pinyin.dropFirst(ini.count))
    }

    /// Compute the "merge form" of two pinyin syllables:
    /// initial(p1) + final(p2), simulating syllable compression in fast speech.
    static func mergePinyin(_ p1: String, _ p2: String) -> String {
        pinyinInitial(p1) + pinyinFinal(p2)
    }

    /// Levenshtein edit distance between two strings
    static func editDistance(_ a: String, _ b: String) -> Int {
        let a = Array(a), b = Array(b)
        let m = a.count, n = b.count
        if m == 0 { return n }
        if n == 0 { return m }

        var prev = Array(0...n)
        var curr = [Int](repeating: 0, count: n + 1)

        for i in 1...m {
            curr[0] = i
            for j in 1...n {
                if a[i - 1] == b[j - 1] {
                    curr[j] = prev[j - 1]
                } else {
                    curr[j] = min(prev[j - 1], prev[j], curr[j - 1]) + 1
                }
            }
            swap(&prev, &curr)
        }
        return prev[n]
    }

    // MARK: - Merge Index

    /// Build merge index: [mergePinyin → [(word, freq)]] for all 2-char words in wordFreq.
    private func buildMergeIndex() -> [String: [(word: String, freq: Int)]] {
        var index: [String: [(word: String, freq: Int)]] = [:]

        for (word, freq) in db.wordFreq {
            let chars = Array(word)
            guard chars.count == 2 else { continue }

            guard let p1readings = db.charToPinyin[chars[0]],
                  let p1 = p1readings.first,
                  let p2readings = db.charToPinyin[chars[1]],
                  let p2 = p2readings.first else { continue }

            let p1toneless = PinyinDatabase.stripTone(p1)
            let p2toneless = PinyinDatabase.stripTone(p2)
            let merge = Self.mergePinyin(p1toneless, p2toneless)

            guard !merge.isEmpty else { continue }

            index[merge, default: []].append((word: word, freq: freq))
        }

        return index
    }

    /// Get or build the merge index (thread-safe, lazy)
    private func getMergeIndex() -> [String: [(word: String, freq: Int)]] {
        mergeIndexLock.lock()
        defer { mergeIndexLock.unlock() }

        if let existing = mergeIndex {
            return existing
        }

        let built = buildMergeIndex()
        mergeIndex = built
        logger.info("SyllableExpansion merge index built: \(built.count) merge forms")
        return built
    }

    // MARK: - Main Entry Point

    /// Correct syllable compression errors in the given text.
    func correct(_ text: String) -> CorrectionResult {
        guard db.isLoaded else {
            return CorrectionResult(text: text, corrections: [])
        }

        let suspects = detectSuspicious(text)
        guard !suspects.isEmpty else {
            return CorrectionResult(text: text, corrections: [])
        }

        var result = text
        var corrections: [CorrectionResult.Correction] = []
        let textChars = Array(text)

        // Process from end to start so character offsets remain valid
        for suspect in suspects.reversed() {
            let leftContext: Character? = suspect.offset > 0 ? textChars[suspect.offset - 1] : nil
            let rightContext: Character? = suspect.offset + 1 < textChars.count ? textChars[suspect.offset + 1] : nil

            if let best = findBestExpansion(
                for: suspect.char,
                leftContext: leftContext,
                rightContext: rightContext
            ) {
                let charStr = String(suspect.char)
                if let range = findRange(of: charStr, in: result, near: suspect.offset) {
                    result = result.replacingCharacters(in: range, with: best.word)
                    corrections.append(.init(
                        original: charStr,
                        corrected: best.word,
                        score: best.score
                    ))
                }
            }
        }

        corrections.reverse() // restore original order
        return CorrectionResult(text: result, corrections: corrections)
    }

    // MARK: - Detection

    private struct SuspiciousChar {
        let char: Character
        let offset: Int
    }

    /// Find single-character tokens that don't fit their bigram context.
    private func detectSuspicious(_ text: String) -> [SuspiciousChar] {
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        tokenizer.setLanguage(.traditionalChinese)

        var suspects: [SuspiciousChar] = []
        let textChars = Array(text)

        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let word = String(text[range])

            // Only look at single-character tokens
            guard word.count == 1, let char = word.first else { return true }

            // Skip non-CJK
            let v = char.unicodeScalars.first!.value
            guard (0x4E00...0x9FFF).contains(v) || (0x3400...0x4DBF).contains(v) else { return true }

            // Skip common function words
            guard !Self.skipChars.contains(char) else { return true }

            // Character offset
            let offset = text.distance(from: text.startIndex, to: range.lowerBound)

            // Check left and right bigram frequencies — both must be low
            let leftFreq: Int
            if offset > 0 {
                let bigram = String(textChars[offset - 1]) + String(char)
                leftFreq = self.db.bigramFrequency(of: bigram)
            } else {
                leftFreq = 0
            }

            let rightFreq: Int
            if offset + 1 < textChars.count {
                let bigram = String(char) + String(textChars[offset + 1])
                rightFreq = self.db.bigramFrequency(of: bigram)
            } else {
                rightFreq = 0
            }

            if leftFreq <= self.suspiciousBigramThreshold && rightFreq <= self.suspiciousBigramThreshold {
                suspects.append(SuspiciousChar(char: char, offset: offset))
            }

            return true
        }

        return suspects
    }

    // MARK: - Candidate Search

    private struct ScoredExpansion {
        let word: String
        let score: Double
    }

    /// Find the best 2-char expansion for a suspicious character.
    private func findBestExpansion(
        for char: Character,
        leftContext: Character?,
        rightContext: Character?
    ) -> ScoredExpansion? {
        let pinyins = db.tonelessPinyin(of: char)
        guard !pinyins.isEmpty else { return nil }

        var best: ScoredExpansion?

        for targetPinyin in pinyins {
            let candidates = getCandidates(for: targetPinyin)

            for candidate in candidates {
                if let score = scoreCandidate(
                    candidate,
                    originalChar: char,
                    leftContext: leftContext,
                    rightContext: rightContext
                ) {
                    if best == nil || score > best!.score {
                        best = ScoredExpansion(word: candidate.word, score: score)
                    }
                }
            }
        }

        return best
    }

    /// Get or compute expansion candidates for a given target pinyin (cached).
    private func getCandidates(for targetPinyin: String) -> [ExpansionCandidate] {
        cacheLock.lock()
        if let cached = expansionCache[targetPinyin] {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()

        let index = getMergeIndex()
        var candidates: [ExpansionCandidate] = []

        // Check all merge forms within edit distance ≤ 1 of targetPinyin
        for (mergePinyin, words) in index {
            let dist = Self.editDistance(mergePinyin, targetPinyin)
            guard dist <= 1 else { continue }

            for entry in words {
                candidates.append(ExpansionCandidate(
                    word: entry.word,
                    freq: entry.freq,
                    mergeDist: dist
                ))
            }
        }

        cacheLock.lock()
        expansionCache[targetPinyin] = candidates
        cacheLock.unlock()

        return candidates
    }

    // MARK: - Scoring

    /// Score a candidate expansion. Returns nil if it doesn't pass thresholds.
    private func scoreCandidate(
        _ candidate: ExpansionCandidate,
        originalChar: Character,
        leftContext: Character?,
        rightContext: Character?
    ) -> Double? {
        let candidateChars = Array(candidate.word)
        guard candidateChars.count == 2 else { return nil }

        // Original bigram context
        let origLeftBigram: Double
        if let left = leftContext {
            origLeftBigram = log(Double(db.bigramFrequency(of: String(left) + String(originalChar)) + 1))
        } else {
            origLeftBigram = 0
        }

        let origRightBigram: Double
        if let right = rightContext {
            origRightBigram = log(Double(db.bigramFrequency(of: String(originalChar) + String(right)) + 1))
        } else {
            origRightBigram = 0
        }

        // New bigram context (first char connects to left, last char connects to right)
        let newLeftBigram: Double
        if let left = leftContext {
            newLeftBigram = log(Double(db.bigramFrequency(of: String(left) + String(candidateChars[0])) + 1))
        } else {
            newLeftBigram = 0
        }

        let newRightBigram: Double
        if let right = rightContext {
            newRightBigram = log(Double(db.bigramFrequency(of: String(candidateChars[1]) + String(right)) + 1))
        } else {
            newRightBigram = 0
        }

        // Context improvement (hard gate)
        let contextImprovement = bigramWeight * (
            (newLeftBigram - origLeftBigram) + (newRightBigram - origRightBigram)
        )

        guard contextImprovement >= minContextImprovement else { return nil }

        // Internal bigram bonus
        let internalBigram = String(candidateChars[0]) + String(candidateChars[1])
        let internalBigramFreq = db.bigramFrequency(of: internalBigram)

        // Total score
        let score = log(Double(candidate.freq + 1))
            + contextImprovement
            + internalBigramWeight * log(Double(internalBigramFreq + 1))
            - distancePenalty * Double(candidate.mergeDist)

        guard score >= minTotalScore else { return nil }

        return score
    }

    // MARK: - Helpers

    /// Find the range of `word` in `text`, preferring occurrence near `approximateOffset`.
    private func findRange(of word: String, in text: String, near offset: Int) -> Range<String.Index>? {
        var bestRange: Range<String.Index>?
        var bestDistance = Int.max
        var searchStart = text.startIndex

        while let range = text.range(of: word, range: searchStart..<text.endIndex) {
            let rangeOffset = text.distance(from: text.startIndex, to: range.lowerBound)
            let distance = abs(rangeOffset - offset)
            if distance < bestDistance {
                bestDistance = distance
                bestRange = range
            }
            searchStart = range.upperBound
        }
        return bestRange
    }
}
