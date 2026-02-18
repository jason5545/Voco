import Foundation

struct WordSubstitution {
    let original: String
    let replacement: String
}

class EditModeDiffService {
    /// Compare original text with edited result; return the substitution if the diff is a single short word swap (≤ 20 chars each side).
    static func extractSubstitution(original: String, edited: String) -> WordSubstitution? {
        guard original != edited else { return nil }

        let commonPrefix = original.commonPrefix(with: edited)
        let prefixLen = commonPrefix.count

        let origSuffix = String(original.reversed())
        let editSuffix = String(edited.reversed())
        let commonSuffix = origSuffix.commonPrefix(with: editSuffix)
        let suffixLen = min(commonSuffix.count, min(original.count, edited.count) - prefixLen)

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
        // Length limit: not a simple word swap if too long
        guard origSegment.count <= 20, editSegment.count <= 20 else { return nil }

        return WordSubstitution(original: origSegment, replacement: editSegment)
    }
}
