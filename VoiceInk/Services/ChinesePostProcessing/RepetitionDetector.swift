import Foundation

/// Detects repetitive patterns in text (Whisper hallucination filtering)
/// Reference: xvoice/src/pipeline.py lines 172-248
class RepetitionDetector {
    static let shared = RepetitionDetector()

    private init() {}

    /// Information about detected repetition
    struct RepetitionInfo {
        let hasRepetition: Bool
        let repetitionRatio: Double  // 0.0 ~ 1.0
        let pattern: String?
        let details: String

        /// Whether this is severe repetition (should discard output)
        var isSevere: Bool {
            return hasRepetition && repetitionRatio >= 0.5
        }
    }

    /// Detect repetitive patterns in text
    /// - Parameters:
    ///   - text: Input text to analyze
    ///   - charThreshold: Single character repeat threshold (default: 4)
    ///   - word2Threshold: 2-char word repeat threshold (default: 3)
    ///   - word3Threshold: 3-char word repeat threshold (default: 2)
    /// - Returns: RepetitionInfo with detection results
    func detect(
        _ text: String,
        charThreshold: Int = 4,
        word2Threshold: Int = 3,
        word3Threshold: Int = 2
    ) -> RepetitionInfo {
        guard text.count >= 3 else {
            return RepetitionInfo(
                hasRepetition: false,
                repetitionRatio: 0.0,
                pattern: nil,
                details: "Text too short"
            )
        }

        let chars = Array(text)
        var maxRatio: Double = 0.0
        var maxPattern: String?
        var maxDetails = ""

        // Check 1-3 character repetition patterns
        for patternLen in 1...3 {
            let thresholds = [1: charThreshold, 2: word2Threshold, 3: word3Threshold]
            guard let threshold = thresholds[patternLen] else { continue }

            for i in 0...(chars.count - patternLen) {
                let pattern = String(chars[i..<min(i + patternLen, chars.count)])
                guard pattern.count == patternLen else { continue }

                // Count consecutive repetitions of this pattern
                var count = 0
                var pos = i
                while pos + patternLen <= chars.count {
                    let segment = String(chars[pos..<pos + patternLen])
                    if segment == pattern {
                        count += 1
                        pos += patternLen
                    } else {
                        break
                    }
                }

                if count >= threshold {
                    let repetitionLength = count * patternLen
                    let ratio = Double(repetitionLength) / Double(chars.count)
                    if ratio > maxRatio {
                        maxRatio = ratio
                        maxPattern = pattern
                        let patternType = patternLen == 1 ? "single char" : "word"
                        maxDetails = "\(patternType) \"\(pattern)\" repeated \(count) times"
                    }
                }
            }
        }

        return RepetitionInfo(
            hasRepetition: maxRatio > 0,
            repetitionRatio: maxRatio,
            pattern: maxPattern,
            details: maxDetails.isEmpty ? "No repetition" : maxDetails
        )
    }
}
