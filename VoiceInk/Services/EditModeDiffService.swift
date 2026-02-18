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

        // Expand left to nearest word boundary (whitespace or string start)
        while prefixLen > 0 {
            let idx = original.index(original.startIndex, offsetBy: prefixLen - 1)
            if original[idx].isWhitespace { break }
            prefixLen -= 1
        }

        // Expand right to nearest word boundary (whitespace or string end)
        while suffixLen > 0 {
            let idx = original.index(original.endIndex, offsetBy: -suffixLen)
            if original[idx].isWhitespace { break }
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
        // Minimum length: single-char replacements are too generic for dictionary rules
        guard origSegment.count >= 2, editSegment.count >= 2 else { return nil }
        // Length limit: not a simple word swap if too long
        guard origSegment.count <= 20, editSegment.count <= 20 else { return nil }

        return WordSubstitution(original: origSegment, replacement: editSegment)
    }
}
