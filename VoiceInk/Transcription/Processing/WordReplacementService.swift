import Foundation
import SwiftData

class WordReplacementService {
    static let shared = WordReplacementService()

    private init() {}

    func applyReplacements(to text: String, using context: ModelContext) -> String {
        let descriptor = FetchDescriptor<WordReplacement>(
            predicate: #Predicate { $0.isEnabled }
        )

        guard let replacements = try? context.fetch(descriptor), !replacements.isEmpty else {
            return text // No replacements to apply
        }

        var modifiedText = text

        // Longest-first so specific triggers match before shorter overlapping ones
        let sortedReplacements = replacements.sorted {
            $0.originalText.count > $1.originalText.count
        }

        // Apply replacements (case-insensitive)
        for replacement in sortedReplacements {
            let originalGroup = replacement.originalText
            let replacementText = replacement.replacementText

            let variants = originalGroup
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
                .sorted { $0.count > $1.count }

            for original in variants {
                let usesBoundaries = usesWordBoundaries(for: original)

                if usesBoundaries {
                    // Lookarounds instead of \b so punctuation acts as a word boundary
                    let pattern = smartBoundaryPattern(for: original)
                    if let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) {
                        let range = NSRange(modifiedText.startIndex..., in: modifiedText)
                        modifiedText = regex.stringByReplacingMatches(
                            in: modifiedText,
                            options: [],
                            range: range,
                            withTemplate: replacementText
                        )
                    }
                } else {
                    // Fallback substring replace for non-spaced scripts
                    modifiedText = modifiedText.replacingOccurrences(of: original, with: replacementText, options: .caseInsensitive)
                }
            }
        }

        return modifiedText
    }

    /// Enforces the exact casing of VocabularyWords on the given text.
    /// E.g. if "LiSA" is in the vocabulary, "Lisa" in the text becomes "LiSA".
    func enforceVocabularyCasing(text: String, using context: ModelContext) -> String {
        let descriptor = FetchDescriptor<VocabularyWord>()
        guard let words = try? context.fetch(descriptor), !words.isEmpty else {
            return text
        }

        var modifiedText = text

        for vocabWord in words {
            let vocab = vocabWord.word
            guard !vocab.isEmpty else { continue }

            let usesBoundaries = usesWordBoundaries(for: vocab)

            if usesBoundaries {
                let pattern = smartBoundaryPattern(for: vocab)
                guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else { continue }
                // Iterate matches in reverse to preserve indices during replacement
                let nsString = modifiedText as NSString
                let fullRange = NSRange(location: 0, length: nsString.length)
                let matches = regex.matches(in: modifiedText, range: fullRange)
                for match in matches.reversed() {
                    let matchedString = nsString.substring(with: match.range)
                    if matchedString != vocab {
                        modifiedText = (modifiedText as NSString).replacingCharacters(in: match.range, with: vocab)
                    }
                }
            } else {
                // CJK / non-spaced: simple case-insensitive scan
                let searchRange = modifiedText.startIndex..<modifiedText.endIndex
                var cursor = searchRange.lowerBound
                while cursor < modifiedText.endIndex {
                    guard let range = modifiedText.range(of: vocab, options: .caseInsensitive, range: cursor..<modifiedText.endIndex) else { break }
                    let matched = String(modifiedText[range])
                    if matched != vocab {
                        modifiedText.replaceSubrange(range, with: vocab)
                    }
                    cursor = modifiedText.index(range.lowerBound, offsetBy: vocab.count)
                }
            }
        }

        return modifiedText
    }

    private func smartBoundaryPattern(for word: String) -> String {
        let escaped = NSRegularExpression.escapedPattern(for: word)
        return "(?<![a-zA-Z0-9])\(escaped)(?![a-zA-Z0-9])"
    }

    private func usesWordBoundaries(for text: String) -> Bool {
        // Returns false for languages without spaces (CJK, Thai), true for spaced languages
        let nonSpacedScripts: [ClosedRange<UInt32>] = [
            0x3040...0x309F, // Hiragana
            0x30A0...0x30FF, // Katakana
            0x4E00...0x9FFF, // CJK Unified Ideographs
            0xAC00...0xD7AF, // Hangul Syllables
            0x0E00...0x0E7F, // Thai
        ]

        for scalar in text.unicodeScalars {
            for range in nonSpacedScripts {
                if range.contains(scalar.value) {
                    return false
                }
            }
        }

        return true
    }
}
