import Foundation

struct LLMValidationResult {
    let isValid: Bool
    let reasons: [String]

    /// Failures worth retrying with a conservative prompt.
    /// Blacklist/empty are fundamental failures; content-drift, short-edit-budget,
    /// and dropped-term may succeed with a more conservative approach.
    var isRetryable: Bool {
        guard !isValid else { return false }
        let retryablePrefixes = ["content-drift", "short-edit-budget", "dropped-term", "cross-script-substitution"]
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
    private let shortContentEditBudget = 1

    /// Medium utterances can change more, but not drift into a rewrite.
    private let mediumContentLengthThreshold = 24
    private let mediumContentEditRatioThreshold = 0.55

    private let listMarkers = ["第一", "第二", "第三", "首先", "其次", "最後", "1.", "2.", "3.", "（1）", "(1)"]
    private let chineseDigitMap: [Character: Character] = [
        "零": "0", "〇": "0", "○": "0",
        "一": "1", "二": "2", "兩": "2", "两": "2",
        "三": "3", "四": "4", "五": "5", "六": "6",
        "七": "7", "八": "8", "九": "9",
    ]

    private init() {}

    func isValid(response: String, original: String, protectedTerms: [String] = []) -> Bool {
        validate(response: response, original: original, protectedTerms: protectedTerms).isValid
    }

    func validate(response: String, original: String, protectedTerms: [String] = []) -> LLMValidationResult {
        validate(response: response, original: original, protectedTerms: protectedTerms, wordReplacements: [], customVocabulary: [])
    }

    func validate(
        response: String,
        original: String,
        protectedTerms: [String] = [],
        wordReplacements: [(original: String, replacement: String)],
        customVocabulary: [String]
    ) -> LLMValidationResult {
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
        let allowedVariantTerms = protectedTerms + customVocabulary
        for term in termsToPreserve where containsEquivalent(term, in: trimmedOriginal)
            && !containsEquivalent(term, in: trimmedResponse)
            && !containsAllowedTechnicalVariant(of: term, in: trimmedResponse, allowedTerms: allowedVariantTerms) {
            reasons.append("dropped-term:\(term)")
        }

        let originalContent = normalizedNumericContent(trimmedOriginal)
        let responseContent = normalizedNumericContent(trimmedResponse)
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

        // Cross-script substitution detection
        let crossScriptViolations = detectCrossScriptSubstitutions(
            original: trimmedOriginal,
            response: trimmedResponse,
            wordReplacements: wordReplacements,
            customVocabulary: customVocabulary
        )
        if crossScriptViolations > 0 {
            reasons.append("cross-script-substitution:\(crossScriptViolations) violations")
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

    private func containsAllowedTechnicalVariant(of term: String, in text: String, allowedTerms: [String]) -> Bool {
        let normalizedTerm = normalizeEquivalentText(term)
        let termWithoutDigits = normalizedTerm.filter { !$0.isNumber }

        guard termWithoutDigits.count >= 2, normalizedTerm.contains(where: { $0.isLetter }) else {
            return false
        }

        for allowedTerm in allowedTerms where containsEquivalent(allowedTerm, in: text) {
            let normalizedAllowed = normalizeEquivalentText(allowedTerm)
            let allowedWithoutDigits = normalizedAllowed.filter { !$0.isNumber }

            if allowedWithoutDigits == termWithoutDigits, normalizedAllowed.count >= normalizedTerm.count {
                return true
            }
        }

        return false
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

    // MARK: - Cross-Script Substitution Detection

    /// Segment text into runs of CJK vs Latin characters, stripping punctuation.
    private func segmentByScript(_ text: String) -> [(script: ScriptType, text: String)] {
        enum State { case none, cjk, latin }
        var segments: [(script: ScriptType, text: String)] = []
        var buffer = ""
        var state: State = .none

        for char in text {
            let isCJK = char.unicodeScalars.contains {
                (0x4E00...0x9FFF).contains($0.value) || (0x3400...0x4DBF).contains($0.value)
                    || (0x20000...0x2A6DF).contains($0.value)
            }
            let isLatin = char.unicodeScalars.contains(where: {
                $0.isASCII && (CharacterSet.letters.contains($0) || CharacterSet.decimalDigits.contains($0))
            })

            if isCJK {
                if state != .cjk && !buffer.isEmpty {
                    let s: ScriptType = (state == .latin) ? .latin : .cjk
                    segments.append((script: s, text: buffer))
                    buffer = ""
                }
                state = .cjk
                buffer.append(char)
            } else if isLatin {
                if state != .latin && !buffer.isEmpty {
                    let s: ScriptType = (state == .cjk) ? .cjk : .latin
                    segments.append((script: s, text: buffer))
                    buffer = ""
                }
                state = .latin
                buffer.append(char)
            } else if char == " " || char == "-" || char == "_" {
                // Keep spaces/hyphens within Latin runs
                if state == .latin {
                    buffer.append(char)
                }
                // Ignore within CJK runs
            }
            // Skip punctuation and other characters
        }
        if !buffer.isEmpty {
            let s: ScriptType = (state == .cjk) ? .cjk : (state == .latin) ? .latin : .cjk
            segments.append((script: s, text: buffer.trimmingCharacters(in: .whitespaces)))
        }
        return segments.filter { !$0.text.isEmpty }
    }

    private enum ScriptType { case cjk, latin }

    /// Detect cross-script substitutions: CJK segments in original replaced by Latin segments in response.
    /// Returns the number of unverified violations (0 = clean).
    private func detectCrossScriptSubstitutions(
        original: String,
        response: String,
        wordReplacements: [(original: String, replacement: String)],
        customVocabulary: [String]
    ) -> Int {
        if normalizedNumericContent(original) == normalizedNumericContent(response) {
            return 0
        }

        let origSegments = segmentByScript(original)
        let respSegments = segmentByScript(response)

        // Find CJK segments present in original but missing in response
        let origCJKTexts = origSegments.filter { $0.script == .cjk }.map { $0.text }
        let respCJKTexts = Set(respSegments.filter { $0.script == .cjk }.map { $0.text })

        // Find Latin segments present in response but not in original
        let origLatinTexts = Set(origSegments.filter { $0.script == .latin }.map { normalizeLatinSegment($0.text) })
        let respLatinSegments = respSegments.filter { $0.script == .latin }
        let newLatinSegments = respLatinSegments.filter {
            !origLatinTexts.contains(normalizeLatinSegment($0.text))
        }

        guard !newLatinSegments.isEmpty else { return 0 }

        // Find which CJK segments were removed
        var removedCJK: [String] = []
        var remainingRespCJK = respCJKTexts
        for cjk in origCJKTexts {
            if remainingRespCJK.contains(cjk) {
                remainingRespCJK.remove(cjk)
            } else {
                // Check if it appears as substring in any response CJK segment
                let found = respSegments.contains { seg in
                    seg.script == .cjk && seg.text.contains(cjk)
                }
                if !found {
                    removedCJK.append(cjk)
                }
            }
        }

        guard !removedCJK.isEmpty else { return 0 }

        // Build whitelist from WordReplacements
        var whitelistPairs: [(cjk: String, latin: String)] = []
        for wr in wordReplacements {
            let origHasCJK = wr.original.unicodeScalars.contains { (0x4E00...0x9FFF).contains($0.value) || (0x3400...0x4DBF).contains($0.value) }
            let replHasLatin = wr.replacement.unicodeScalars.contains(where: { $0.isASCII && CharacterSet.letters.contains($0) })
            if origHasCJK && replHasLatin {
                whitelistPairs.append((cjk: wr.original, latin: wr.replacement))
            }
            let replHasCJK = wr.replacement.unicodeScalars.contains { (0x4E00...0x9FFF).contains($0.value) || (0x3400...0x4DBF).contains($0.value) }
            let origHasLatin = wr.original.unicodeScalars.contains(where: { $0.isASCII && CharacterSet.letters.contains($0) })
            if replHasCJK && origHasLatin {
                whitelistPairs.append((cjk: wr.replacement, latin: wr.original))
            }
        }

        // Custom vocabulary as allowed Latin terms
        let normalizedVocabulary = Set(customVocabulary.map { normalizeLatinSegment($0) })

        var violations = 0

        for newLatin in newLatinSegments {
            let latinLower = normalizeLatinSegment(newLatin.text)
            guard !latinLower.isEmpty else { continue }

            // Check WordReplacement whitelist
            let whitelisted = whitelistPairs.contains { pair in
                let normalizedPair = normalizeLatinSegment(pair.latin)
                return latinLower.contains(normalizedPair) || normalizedPair.contains(latinLower)
            }
            if whitelisted { continue }

            // Check CustomVocabulary whitelist
            if normalizedVocabulary.contains(latinLower) { continue }

            // Check phonetic plausibility against removed CJK segments
            var phoneticallyPlausible = false
            for cjk in removedCJK {
                if isPhoneticallyPlausible(cjk: cjk, latin: latinLower) {
                    phoneticallyPlausible = true
                    break
                }
            }
            if phoneticallyPlausible { continue }

            let violationCJK = removedCJK.joined(separator: ",")
            let violationLatin = newLatin.text
            DispatchQueue.main.async {
                ChinesePostProcessingService.debugLog(
                    "CROSS_SCRIPT_VIOLATION: '\(violationCJK)' → '\(violationLatin)' — not phonetically plausible"
                )
            }
            violations += 1
        }

        return violations
    }

    private func normalizeLatinSegment(_ text: String) -> String {
        text.lowercased().filter { $0.isLetter || $0.isNumber }
    }

    private func normalizedNumericContent(_ text: String) -> String {
        let converted = OpenCCConverter.shared.convert(text).lowercased()
        var result = ""
        let chars = Array(converted)

        for (index, char) in chars.enumerated() {
            if let digit = chineseDigitMap[char], shouldNormalizeChineseDigit(char, at: index, in: chars) {
                result.append(digit)
            } else if isChineseNumericPoint(char, at: index, in: chars) {
                continue
            } else if char.isLetter || char.isNumber {
                result.append(char)
            }
        }

        return result
    }

    private func shouldNormalizeChineseDigit(_ char: Character, at index: Int, in chars: [Character]) -> Bool {
        if char != "一" {
            return true
        }

        guard index + 1 < chars.count else {
            return true
        }

        let next = chars[index + 1]
        let nonNumericMeasureWords: Set<Character> = ["個", "个", "下", "些", "種", "种"]
        return !nonNumericMeasureWords.contains(next)
    }

    private func isChineseNumericPoint(_ char: Character, at index: Int, in chars: [Character]) -> Bool {
        guard char == "點" || char == "点" else {
            return false
        }
        guard index > 0, index + 1 < chars.count else {
            return false
        }

        return isNumericLike(chars[index - 1]) && isNumericLike(chars[index + 1])
    }

    private func isNumericLike(_ char: Character) -> Bool {
        chineseDigitMap[char] != nil || char.isNumber
    }

    /// Check if a CJK string and a Latin string are phonetically plausible substitutions.
    /// Uses PinyinDatabase for CJK→pinyin conversion, then compares with Levenshtein distance.
    private func isPhoneticallyPlausible(cjk: String, latin: String) -> Bool {
        let db = PinyinDatabase.shared
        guard db.isLoaded else { return true } // Conservative: allow if DB not loaded

        // Build pinyin string for CJK text
        var pinyinParts: [String] = []
        for char in cjk {
            let readings = db.tonelessPinyin(of: char)
            if let primary = readings.first {
                pinyinParts.append(primary)
            }
        }

        guard !pinyinParts.isEmpty else { return true } // No pinyin data → conservatively allow

        let pinyinJoined = pinyinParts.joined().lowercased()
        let latinClean = latin.replacingOccurrences(of: " ", with: "")
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: "_", with: "")
            .lowercased()

        guard !latinClean.isEmpty else { return true }

        let distance = levenshteinDistance(Array(pinyinJoined), Array(latinClean))
        let maxLen = max(pinyinJoined.count, latinClean.count)
        let similarity = maxLen > 0 ? 1.0 - Double(distance) / Double(maxLen) : 0.0

        let phoneticMsg = "PHONETIC_CHECK: cjk='\(cjk)' pinyin='\(pinyinJoined)' latin='\(latinClean)' distance=\(distance) similarity=\(String(format: "%.2f", similarity))"
        DispatchQueue.main.async {
            ChinesePostProcessingService.debugLog(phoneticMsg)
        }

        return similarity >= 0.30
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
