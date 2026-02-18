import Foundation

struct WordSubstitution {
    let original: String
    let replacement: String
}

class EditModeDiffService {
    /// Compare original text with edited result; return the substitution if the diff is a single word-level swap (≤ 20 chars each side).
    /// The diff is expanded to word boundaries so that e.g. "BOCO"→"VOCO" yields "BOCO→VOCO", not "B→V".
    static func extractSubstitution(original: String, edited: String) -> WordSubstitution? {
        guard original != edited else { return nil }

        let commonPrefix = original.commonPrefix(with: edited)
        var prefixLen = commonPrefix.count

        let origSuffix = String(original.reversed())
        let editSuffix = String(edited.reversed())
        let commonSuffix = origSuffix.commonPrefix(with: editSuffix)
        var suffixLen = min(commonSuffix.count, min(original.count, edited.count) - prefixLen)

        // Expand left to nearest word boundary (whitespace, punctuation, or CJK).
        // For CJK: include one extra character for word context (e.g. 全綠 not just 綠).
        while prefixLen > 0 {
            let idx = original.index(original.startIndex, offsetBy: prefixLen - 1)
            let ch = original[idx]
            if ch.isWhitespace || ch.isPunctuation { break }
            if ch.isCJK {
                prefixLen -= 1  // include this CJK character as word context
                break
            }
            prefixLen -= 1
        }

        // Expand right to nearest word boundary (stop at CJK — don't pull in the next word)
        while suffixLen > 0 {
            let idx = original.index(original.endIndex, offsetBy: -suffixLen)
            let ch = original[idx]
            if ch.isWhitespace || ch.isPunctuation || ch.isCJK { break }
            suffixLen -= 1
        }

        let origStart = original.index(original.startIndex, offsetBy: prefixLen)
        let origEnd = original.index(original.endIndex, offsetBy: -suffixLen)
        guard origStart <= origEnd else { return nil }
        let origSegment = String(original[origStart..<origEnd]).trimmingCharacters(in: .whitespacesAndNewlines)

        let editStart = edited.index(edited.startIndex, offsetBy: prefixLen)
        let editEnd = edited.index(edited.endIndex, offsetBy: -suffixLen)
        guard editStart <= editEnd else { return nil }
        let editSegment = String(edited[editStart..<editEnd]).trimmingCharacters(in: .whitespacesAndNewlines)

        // Both sides must have content (substitution, not pure insertion/deletion)
        guard !origSegment.isEmpty, !editSegment.isEmpty else { return nil }
        // Minimum length: single Latin-letter replacements (e.g. B→V) are too generic.
        // CJK single-character substitutions are valid — each character is a word.
        let isCJKSubstitution = origSegment.first?.isCJK == true
        if !isCJKSubstitution {
            guard origSegment.count >= 2, editSegment.count >= 2 else { return nil }
        }
        // Length limit: not a simple word swap if too long
        guard origSegment.count <= 20, editSegment.count <= 20 else { return nil }

        return WordSubstitution(original: origSegment, replacement: editSegment)
    }
}

private extension Character {
    /// Whether this character is a CJK ideograph, kana, or hangul — languages without inter-word spaces.
    var isCJK: Bool {
        guard let scalar = unicodeScalars.first else { return false }
        let v = scalar.value
        return (0x4E00...0x9FFF).contains(v)     // CJK Unified Ideographs
            || (0x3400...0x4DBF).contains(v)     // Extension A
            || (0x20000...0x2A6DF).contains(v)   // Extension B
            || (0xF900...0xFAFF).contains(v)     // Compatibility Ideographs
            || (0x3040...0x309F).contains(v)     // Hiragana
            || (0x30A0...0x30FF).contains(v)     // Katakana
            || (0xAC00...0xD7AF).contains(v)     // Hangul
    }
}
