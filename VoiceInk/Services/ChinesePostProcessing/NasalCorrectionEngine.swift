import Foundation
import NaturalLanguage
import os

/// Nasal ending correction engine (-n/-ng swap).
///
/// Whisper sometimes confuses front/back nasal endings, e.g.,
/// "螢幕" (yíng mù) → "銀幕" (yín mù). This engine scans all
/// 1-4 character words, tries nasal swaps, and applies corrections
/// when the candidate has overwhelmingly higher word frequency.
///
/// Pipeline position: Step 3, Layer 2.5 (after HomophoneCorrectionEngine,
/// before SyllableExpansionEngine).
///
/// Key differences from HomophoneCorrectionEngine:
/// - No suspicious detection — scans ALL words (nasal errors produce valid words)
/// - Uses `nasalVariants(of:)` instead of `homophones(of:)`
/// - Higher frequency threshold (minScoreDelta = 3.0, ~20x) for safety
/// - No dual-character simultaneous replacement
final class NasalCorrectionEngine {
    static let shared = NasalCorrectionEngine()

    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "NasalCorrection")
    private let db = PinyinDatabase.shared

    /// Minimum log-frequency difference required to accept a correction.
    /// 3.0 ≈ candidate must be ~20x more frequent than original.
    private let minScoreDelta: Double = 3.0

    /// Minimum absolute frequency for a candidate word.
    /// Prevents low-freq words from replacing unknown proper nouns (freq=0).
    private let minCandidateFreq: Int = 100

    /// Weight for bigram context score.
    private let bigramWeight: Double = 0.3

    /// Maximum word length to attempt correction on.
    private let maxWordLength = 4

    /// Common function words to skip.
    private static let skipChars: Set<Character> = [
        "的", "了", "嗎", "呢", "吧", "啊", "哦", "喔", "嗯", "呀",
        "是", "在", "有", "和", "也", "都", "就", "不", "我", "你",
        "他", "她", "它", "們", "這", "那", "個", "把", "被", "讓",
        "會", "能", "可", "要", "得", "地", "著", "過", "到", "從",
        "與", "及", "或", "而", "但", "因", "為", "所", "以", "如",
    ]

    // MARK: - Main Entry Point

    /// Correct nasal ending errors in the given text.
    func correct(_ text: String) -> CorrectionResult {
        guard db.isLoaded else {
            return CorrectionResult(text: text, corrections: [])
        }

        let segments = tokenize(text)
        guard !segments.isEmpty else {
            return CorrectionResult(text: text, corrections: [])
        }

        var corrections: [CorrectionResult.Correction] = []
        var result = text
        let textChars = Array(text)

        // Collect all candidate words to process
        var candidates: [CandidateWord] = []
        var charOffset = 0

        for seg in segments {
            defer { charOffset += seg.word.count }

            let word = seg.word

            // Skip non-CJK
            guard isCJK(word) else { continue }

            // Skip single function words
            if word.count == 1, let c = word.first, Self.skipChars.contains(c) { continue }

            // Skip words too long
            if word.count > maxWordLength { continue }

            candidates.append(CandidateWord(word: word, approximateOffset: charOffset))
        }

        guard !candidates.isEmpty else {
            return CorrectionResult(text: text, corrections: [])
        }

        // Process from end to start so indices remain valid
        for candidate in candidates.reversed() {
            let offset = candidate.approximateOffset
            let leftContext: Character? = offset > 0 ? textChars[offset - 1] : nil
            let rightEnd = offset + candidate.word.count
            let rightContext: Character? = rightEnd < textChars.count ? textChars[rightEnd] : nil

            if let best = findBestNasalCandidate(for: candidate.word, leftContext: leftContext, rightContext: rightContext) {
                if let range = findRange(of: candidate.word, in: result, near: candidate.approximateOffset) {
                    result = result.replacingCharacters(in: range, with: best.candidate)
                    corrections.append(.init(
                        original: candidate.word,
                        corrected: best.candidate,
                        score: best.score
                    ))
                    logger.debug("Nasal [data]: \(candidate.word)(\(best.originalFreq)) → \(best.candidate)(\(best.candidateFreq)) score=\(String(format: "%.1f", best.score))")
                }
            }
        }

        corrections.reverse()
        return CorrectionResult(text: result, corrections: corrections)
    }

    // MARK: - Tokenization

    private struct Segment {
        let word: String
        let range: Range<String.Index>
    }

    private func tokenize(_ text: String) -> [Segment] {
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        tokenizer.setLanguage(.traditionalChinese)

        var segments: [Segment] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let word = String(text[range])
            segments.append(Segment(word: word, range: range))
            return true
        }
        return segments
    }

    // MARK: - Candidate Types

    private struct CandidateWord {
        let word: String
        let approximateOffset: Int
    }

    private struct ScoredCandidate {
        let candidate: String
        let originalFreq: Int
        let candidateFreq: Int
        let score: Double
    }

    // MARK: - Candidate Generation + Scoring

    private func findBestNasalCandidate(for word: String, leftContext: Character?, rightContext: Character?) -> ScoredCandidate? {
        let chars = Array(word)
        let originalFreq = db.frequency(of: word)
        var best: ScoredCandidate?

        // Pre-compute original bigram context score
        let origBigramScore = bigramContextScore(for: chars, leftContext: leftContext, rightContext: rightContext)

        // Try replacing each single character with its nasal variant
        for i in 0..<chars.count {
            let variants = db.nasalVariants(of: chars[i])
            for variant in variants {
                var candidate = chars
                candidate[i] = variant
                let candidateWord = String(candidate)
                let candidateFreq = db.frequency(of: candidateWord)

                guard candidateFreq >= minCandidateFreq else { continue }

                let baseScore = log(Double(candidateFreq + 1)) - log(Double(originalFreq + 1))
                let candBigramScore = bigramContextScore(for: candidate, leftContext: leftContext, rightContext: rightContext)
                let score = baseScore + bigramWeight * (candBigramScore - origBigramScore)

                if score > minScoreDelta {
                    if best == nil || score > best!.score {
                        best = ScoredCandidate(candidate: candidateWord, originalFreq: originalFreq, candidateFreq: candidateFreq, score: score)
                    }
                }
            }
        }

        return best
    }

    /// Compute bigram context score for a word given its surrounding characters.
    private func bigramContextScore(for chars: [Character], leftContext: Character?, rightContext: Character?) -> Double {
        var score: Double = 0
        if let left = leftContext {
            let bigram = String(left) + String(chars.first!)
            score += log(Double(db.bigramFrequency(of: bigram) + 1))
        }
        if let right = rightContext {
            let bigram = String(chars.last!) + String(right)
            score += log(Double(db.bigramFrequency(of: bigram) + 1))
        }
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

    private func isCJK(_ text: String) -> Bool {
        guard let scalar = text.unicodeScalars.first else { return false }
        let v = scalar.value
        return (0x4E00...0x9FFF).contains(v) || (0x3400...0x4DBF).contains(v)
    }
}

// MARK: - CorrectionEngine Conformance

extension NasalCorrectionEngine: CorrectionEngine {
    var name: String { "NasalCorrection" }
    var logPrefix: String { "[nasal]" }
}
