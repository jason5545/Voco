import Foundation

struct LLMValidationResult {
    let isValid: Bool
    let reasons: [String]

    /// Failures worth retrying with a conservative prompt.
    /// Blacklist/empty are fundamental failures; content-drift, short-edit-budget,
    /// and dropped-term may succeed with a more conservative approach.
    var isRetryable: Bool {
        guard !isValid else { return false }
        let retryablePrefixes = ["content-drift", "short-edit-budget", "dropped-term"]
        return reasons.allSatisfy { reason in
            retryablePrefixes.contains { reason.hasPrefix($0) }
        }
    }
}

/// Validates LLM responses to reject aggressive rewrites that are more likely
/// to introduce new recognition errors than fix them.
class LLMResponseValidator {
    static let shared = LLMResponseValidator()

    /// Phrases that should never appear in LLM output (they come from the system prompt)
    private let blacklistPhrases: [String] = [
        "使用臺灣語音輸入",
        "正體中文語音輸入",
        "這是正體中文",
        "臺灣正體中文",
        "語音辨識結果",
    ]

    /// Maximum ratio of response length to original length
    private let maxLengthRatio: Double = 3.0

    /// Short utterances should only receive very small content edits.
    private let shortContentLengthThreshold = 8
    private let shortContentEditBudget = 2

    /// Medium utterances can change more, but not drift into a rewrite.
    private let mediumContentLengthThreshold = 24
    private let mediumContentEditRatioThreshold = 0.55

    private let listMarkers = ["第一", "第二", "第三", "首先", "其次", "最後", "1.", "2.", "3.", "（1）", "(1)"]

    private init() {}

    func isValid(response: String, original: String, protectedTerms: [String] = []) -> Bool {
        validate(response: response, original: original, protectedTerms: protectedTerms).isValid
    }

    func validate(response: String, original: String, protectedTerms: [String] = []) -> LLMValidationResult {
        let trimmedResponse = response.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedOriginal = original.trimmingCharacters(in: .whitespacesAndNewlines)
        var reasons: [String] = []

        if trimmedOriginal.isEmpty {
            return LLMValidationResult(isValid: !trimmedResponse.isEmpty, reasons: trimmedResponse.isEmpty ? ["empty-response"] : [])
        }

        if trimmedResponse.isEmpty {
            reasons.append("empty-response")
        }

        for phrase in blacklistPhrases where trimmedResponse.contains(phrase) {
            reasons.append("blacklist:\(phrase)")
        }

        if Double(trimmedResponse.count) > Double(trimmedOriginal.count) * maxLengthRatio {
            reasons.append("length-ratio")
        }

        let termsToPreserve = collectProtectedTerms(original: trimmedOriginal, extras: protectedTerms)
        for term in termsToPreserve where containsEquivalent(term, in: trimmedOriginal) && !containsEquivalent(term, in: trimmedResponse) {
            reasons.append("dropped-term:\(term)")
        }

        let originalContent = normalizedContent(trimmedOriginal)
        let responseContent = normalizedContent(trimmedResponse)
        if !originalContent.isEmpty && responseContent.isEmpty {
            reasons.append("empty-content")
        }

        if !originalContent.isEmpty && !responseContent.isEmpty && !looksLikeExplicitListFormatting(original: trimmedOriginal, response: trimmedResponse) {
            let distance = levenshteinDistance(Array(originalContent), Array(responseContent))
            let maxLength = max(originalContent.count, responseContent.count)
            let editRatio = maxLength > 0 ? Double(distance) / Double(maxLength) : 0.0

            if originalContent.count <= shortContentLengthThreshold && distance > shortContentEditBudget {
                reasons.append("short-edit-budget")
            } else if originalContent.count <= mediumContentLengthThreshold && editRatio > mediumContentEditRatioThreshold {
                reasons.append("content-drift")
            }
        }

        return LLMValidationResult(isValid: reasons.isEmpty, reasons: reasons)
    }

    private func collectProtectedTerms(original: String, extras: [String]) -> [String] {
        let candidates = extras + extractTechnicalTerms(from: original)
        var seen: Set<String> = []
        var terms: [String] = []

        for term in candidates {
            let trimmed = term.trimmingCharacters(in: .whitespacesAndNewlines)
            let normalized = normalizeEquivalentText(trimmed)
            guard normalized.count >= 3 else { continue }
            if seen.insert(normalized).inserted {
                terms.append(trimmed)
            }
        }

        return terms.sorted { $0.count > $1.count }
    }

    private func extractTechnicalTerms(from text: String) -> [String] {
        var results: [String] = []
        var buffer = ""
        var hasLatin = false

        func flushBuffer() {
            let term = buffer.trimmingCharacters(in: CharacterSet(charactersIn: " -_./+"))
            defer {
                buffer = ""
                hasLatin = false
            }
            guard hasLatin else { return }
            let normalized = normalizeEquivalentText(term)
            guard normalized.count >= 3 else { return }
            results.append(term)
        }

        for char in text {
            let isCJK = char.unicodeScalars.contains {
                (0x4E00...0x9FFF).contains($0.value) || (0x3400...0x4DBF).contains($0.value)
                    || (0x20000...0x2A6DF).contains($0.value)
                    || (0x3040...0x309F).contains($0.value) || (0x30A0...0x30FF).contains($0.value)
            }
            if isCJK {
                // CJK characters are term boundaries — flush any Latin term in progress
                if !buffer.isEmpty { flushBuffer() }
            } else if char.isLetter || char.isNumber || " -_./+".contains(char) {
                buffer.append(char)
                if char.unicodeScalars.contains(where: { $0.isASCII && CharacterSet.letters.contains($0) }) {
                    hasLatin = true
                }
            } else if !buffer.isEmpty {
                flushBuffer()
            }
        }

        if !buffer.isEmpty {
            flushBuffer()
        }

        return results
    }

    private func containsEquivalent(_ term: String, in text: String) -> Bool {
        let normalizedTerm = normalizeEquivalentText(term)
        guard !normalizedTerm.isEmpty else { return false }
        return normalizeEquivalentText(text).contains(normalizedTerm)
    }

    private func normalizeEquivalentText(_ text: String) -> String {
        let converted = OpenCCConverter.shared.convert(text).lowercased()
        return converted.filter { $0.isLetter || $0.isNumber }
    }

    private func normalizedContent(_ text: String) -> String {
        let converted = OpenCCConverter.shared.convert(text).lowercased()
        return converted.filter { $0.isLetter || $0.isNumber }
    }

    private func looksLikeExplicitListFormatting(original: String, response: String) -> Bool {
        let originalMarkerHits = listMarkers.filter { original.contains($0) }.count
        guard originalMarkerHits >= 2 else { return false }
        return response.contains("\n1.") || response.contains("\n2.") || response.contains("\n- ")
    }

    private func levenshteinDistance(_ lhs: [Character], _ rhs: [Character]) -> Int {
        guard !lhs.isEmpty else { return rhs.count }
        guard !rhs.isEmpty else { return lhs.count }

        var previous = Array(0...rhs.count)
        for (i, leftChar) in lhs.enumerated() {
            var current = [i + 1]
            current.reserveCapacity(rhs.count + 1)

            for (j, rightChar) in rhs.enumerated() {
                let substitutionCost = leftChar == rightChar ? 0 : 1
                let insertion = current[j] + 1
                let deletion = previous[j + 1] + 1
                let substitution = previous[j] + substitutionCost
                current.append(min(insertion, deletion, substitution))
            }

            previous = current
        }

        return previous[rhs.count]
    }
}
