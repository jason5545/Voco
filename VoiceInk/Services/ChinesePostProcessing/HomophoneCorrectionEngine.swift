import Foundation
import NaturalLanguage
import os

/// Data-driven homophone correction engine.
/// Uses pinyin lookup + word frequency scoring to automatically fix
/// same-sound character substitution errors from speech recognition.
final class HomophoneCorrectionEngine {
    static let shared = HomophoneCorrectionEngine()

    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "HomophoneEngine")
    private let db = PinyinDatabase.shared

    /// Minimum log-frequency difference required to accept a correction.
    /// 2.0 ≈ candidate must be ~7x more frequent than original.
    private let minScoreDelta: Double = 2.0

    /// Weight for bigram context score (conservative to avoid noise).
    private let bigramWeight: Double = 0.3

    /// Words below this frequency threshold are considered suspicious.
    private let lowFreqThreshold: Int = 5

    /// Maximum word length to attempt correction on (longer words are less likely speech errors).
    private let maxWordLength = 4

    /// Common function words to skip (too short, too common, not worth checking).
    private static let skipChars: Set<Character> = [
        "的", "了", "嗎", "呢", "吧", "啊", "哦", "喔", "嗯", "呀",
        "是", "在", "有", "和", "也", "都", "就", "不", "我", "你",
        "他", "她", "它", "們", "這", "那", "個", "把", "被", "讓",
        "會", "能", "可", "要", "得", "地", "著", "過", "到", "從",
        "與", "及", "或", "而", "但", "因", "為", "所", "以", "如",
    ]

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

    // MARK: - Main Entry Point

    /// Correct homophone errors in the given text.
    func correct(_ text: String) -> CorrectionResult {
        guard db.isLoaded else {
            return CorrectionResult(text: text, corrections: [])
        }

        // Step 1: Segment text with NLTokenizer
        let segments = tokenize(text)

        // Step 2: Find suspicious segments
        let suspicious = findSuspicious(segments)

        guard !suspicious.isEmpty else {
            return CorrectionResult(text: text, corrections: [])
        }

        // Steps 3-4: Generate candidates and score
        var corrections: [CorrectionResult.Correction] = []
        var result = text

        // Build character array for context lookup
        let textChars = Array(text)

        // Process from end to start so indices remain valid
        for seg in suspicious.reversed() {
            // Extract left/right context characters from original text
            let offset = seg.approximateOffset
            let leftContext: Character? = offset > 0 ? textChars[offset - 1] : nil
            let rightEnd = offset + seg.word.count
            let rightContext: Character? = rightEnd < textChars.count ? textChars[rightEnd] : nil

            if let best = findBestCandidate(for: seg.word, leftContext: leftContext, rightContext: rightContext) {
                // Replace the specific occurrence at the known range
                if let range = findRange(of: seg.word, in: result, near: seg.approximateOffset) {
                    result = result.replacingCharacters(in: range, with: best.candidate)
                    corrections.append(.init(
                        original: seg.word,
                        corrected: best.candidate,
                        score: best.score
                    ))
                    logger.debug("Homophone [data]: \(seg.word)(\(seg.freq)) → \(best.candidate)(\(best.candidateFreq)) score=\(String(format: "%.1f", best.score))")
                }
            }
        }

        corrections.reverse() // restore original order
        return CorrectionResult(text: result, corrections: corrections)
    }

    // MARK: - Step 1: Tokenization

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

    // MARK: - Step 2: Suspicious Detection

    private struct SuspiciousWord {
        let word: String
        let freq: Int
        let approximateOffset: Int // character offset for disambiguation
    }

    private func findSuspicious(_ segments: [Segment]) -> [SuspiciousWord] {
        var suspicious: [SuspiciousWord] = []
        var charOffset = 0

        for seg in segments {
            defer { charOffset += seg.word.count }

            let word = seg.word

            // Skip non-CJK content
            guard isCJK(word) else { continue }

            // Skip single function words
            if word.count == 1, let c = word.first, Self.skipChars.contains(c) {
                continue
            }

            // Skip words that are too long (unlikely speech errors)
            if word.count > maxWordLength { continue }

            let freq = db.frequency(of: word)

            // Suspicious if: not in dictionary, very low freq, or single-char (tokenizer failed to form a word)
            let isSuspicious: Bool
            if freq == 0 {
                // Unknown word — suspicious
                isSuspicious = true
            } else if freq <= lowFreqThreshold {
                // Very low frequency — suspicious
                isSuspicious = true
            } else if word.count == 1 {
                // Single character from tokenizer could mean it failed to form a word.
                // Only suspicious if there are adjacent single chars (broken segmentation).
                // We'll handle this via sliding window below, skip here.
                isSuspicious = false
            } else {
                isSuspicious = false
            }

            if isSuspicious {
                suspicious.append(SuspiciousWord(word: word, freq: freq, approximateOffset: charOffset))
            }
        }

        // Also try sliding window for 2-char combinations that the tokenizer split into singles
        let slidingWindowCandidates = findSlidingWindowSuspicious(segments)
        suspicious.append(contentsOf: slidingWindowCandidates)

        return suspicious
    }

    /// Find suspicious 2-char sequences from adjacent single-char tokens.
    private func findSlidingWindowSuspicious(_ segments: [Segment]) -> [SuspiciousWord] {
        var result: [SuspiciousWord] = []
        var charOffset = 0

        for i in 0..<segments.count {
            defer { charOffset += segments[i].word.count }

            guard segments[i].word.count == 1, isCJK(segments[i].word) else { continue }
            guard i + 1 < segments.count, segments[i + 1].word.count == 1, isCJK(segments[i + 1].word) else { continue }

            // Skip if either char is a function word
            if let c1 = segments[i].word.first, Self.skipChars.contains(c1) { continue }
            if let c2 = segments[i + 1].word.first, Self.skipChars.contains(c2) { continue }

            let combined = segments[i].word + segments[i + 1].word
            let freq = db.frequency(of: combined)

            // If the combined 2-char word is also not in dict or low freq, it's suspicious
            if freq <= lowFreqThreshold {
                result.append(SuspiciousWord(word: combined, freq: freq, approximateOffset: charOffset))
            }
        }
        return result
    }

    // MARK: - Steps 3-4: Candidate Generation + Scoring

    private struct ScoredCandidate {
        let candidate: String
        let candidateFreq: Int
        let score: Double
    }

    private func findBestCandidate(for word: String, leftContext: Character?, rightContext: Character?) -> ScoredCandidate? {
        let chars = Array(word)
        let originalFreq = db.frequency(of: word)
        var best: ScoredCandidate?

        // Pre-compute original bigram context score
        let origBigramScore = bigramContextScore(for: chars, leftContext: leftContext, rightContext: rightContext)

        // Try replacing each single character
        for i in 0..<chars.count {
            let homophones = db.homophones(of: chars[i])
            for homo in homophones {
                var candidate = chars
                candidate[i] = homo
                let candidateWord = String(candidate)
                let candidateFreq = db.frequency(of: candidateWord)

                guard candidateFreq > 0 else { continue }

                let baseScore = log(Double(candidateFreq)) - log(Double(originalFreq + 1))
                let candBigramScore = bigramContextScore(for: candidate, leftContext: leftContext, rightContext: rightContext)
                let score = baseScore + bigramWeight * (candBigramScore - origBigramScore)

                if score > minScoreDelta {
                    if best == nil || score > best!.score {
                        best = ScoredCandidate(candidate: candidateWord, candidateFreq: candidateFreq, score: score)
                    }
                }
            }
        }

        // For 2-char words: also try replacing both characters simultaneously
        if chars.count == 2 {
            let homos0 = db.homophones(of: chars[0])
            let homos1 = db.homophones(of: chars[1])

            // Limit combinatorial explosion: only try top homophones
            let limit = 30
            let h0 = homos0.prefix(limit)
            let h1 = homos1.prefix(limit)

            for c0 in h0 {
                for c1 in h1 {
                    let candidate = [c0, c1]
                    let candidateWord = String(candidate)
                    let candidateFreq = db.frequency(of: candidateWord)
                    guard candidateFreq > 0 else { continue }

                    let baseScore = log(Double(candidateFreq)) - log(Double(originalFreq + 1))
                    let candBigramScore = bigramContextScore(for: candidate, leftContext: leftContext, rightContext: rightContext)
                    let score = baseScore + bigramWeight * (candBigramScore - origBigramScore)

                    if score > minScoreDelta {
                        if best == nil || score > best!.score {
                            best = ScoredCandidate(candidate: candidateWord, candidateFreq: candidateFreq, score: score)
                        }
                    }
                }
            }
        }

        return best
    }

    /// Compute bigram context score for a word given its surrounding characters.
    /// Score = log(leftBigram+1) + log(rightBigram+1)
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

    // MARK: - Step 5: Apply Corrections

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

    // MARK: - Helpers

    private func isCJK(_ text: String) -> Bool {
        guard let scalar = text.unicodeScalars.first else { return false }
        let v = scalar.value
        return (0x4E00...0x9FFF).contains(v) || (0x3400...0x4DBF).contains(v)
    }
}
