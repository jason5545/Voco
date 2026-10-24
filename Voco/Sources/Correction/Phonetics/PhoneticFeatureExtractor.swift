import Foundation

enum PhoneticScriptMode: String, Codable, Equatable {
    case zhOnly
    case enOnly
    case mixedZhEn
    case numericSymbol
    case unknown
}

enum PhoneticLanguageMode: String, Codable, Equatable {
    case mandarin
    case english
    case codeSwitch
    case crossScript
    case unknown
}

enum PhoneticLengthBucket: String, Codable, Equatable {
    case oneToFour = "1_4"
    case fiveToFifteen = "5_15"
    case sixteenPlus = "16_plus"
    case unknown
}

enum PhoneticEditOperation: String, Codable, Equatable {
    case substitution
    case insertion
    case deletion
}

struct PhoneticConfusionPair: Codable, Equatable {
    let raw: String
    let target: String
    let operation: PhoneticEditOperation
    let position: Int?
}

struct PhoneticTextFeatures: Codable, Equatable {
    let original: String
    let trimmed: String
    let normalized: String
    let lowercaseComparison: String
    let scriptMode: PhoneticScriptMode
    let languageMode: PhoneticLanguageMode
    let lengthBucket: PhoneticLengthBucket
    let unitCount: Int
    let isCommandLike: Bool
    let isTechnicalTermCandidate: Bool
    let phones: [String]
}

struct PhoneticComparison: Codable, Equatable {
    let raw: PhoneticTextFeatures
    let target: PhoneticTextFeatures
    let languageMode: PhoneticLanguageMode
    let isCrossScript: Bool
    let isPurePhoneticCandidate: Bool
    let weightedPhoneticEditDistance: Double?
    let pinyinSimilarity: Double?
    let confusionPairs: [PhoneticConfusionPair]
}

enum PhoneticFeatureExtractor {
    private static let technicalTerms: Set<String> = [
        "api", "asr", "auto", "cloudflare", "flight", "envelope", "github",
        "json", "jsonl", "llm", "markdown", "mlx", "openai", "qwen",
        "session", "sqlite", "swift", "swiftdata", "voco", "voiceink", "workaround",
        "xcode"
    ]

    private static let commandTerms: Set<String> = [
        "copy", "delete", "open", "paste", "redo", "run", "save", "select",
        "undo", "全部", "刪除", "複製", "貼上", "開啟", "關閉", "儲存",
        "修正", "排版", "辨識"
    ]

    static func extract(_ text: String) -> PhoneticTextFeatures {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalized = normalize(trimmed)
        let lowercase = normalized.lowercased()
        let scriptMode = classifyScript(normalized)
        let languageMode = classifyLanguage(scriptMode: scriptMode)
        let unitCount = countUnits(in: normalized, scriptMode: scriptMode)
        let lengthBucket = bucket(for: unitCount)
        let technical = isTechnicalTermCandidate(normalized)
        let commandLike = isCommandLikeText(
            normalized,
            lengthBucket: lengthBucket,
            isTechnicalTermCandidate: technical
        )

        return PhoneticTextFeatures(
            original: text,
            trimmed: trimmed,
            normalized: normalized,
            lowercaseComparison: lowercase,
            scriptMode: scriptMode,
            languageMode: languageMode,
            lengthBucket: lengthBucket,
            unitCount: unitCount,
            isCommandLike: commandLike,
            isTechnicalTermCandidate: technical,
            phones: phones(for: normalized, scriptMode: scriptMode)
        )
    }

    static func compare(raw rawText: String, target targetText: String) -> PhoneticComparison {
        let raw = extract(rawText)
        let target = extract(targetText)
        let crossScript = isCrossScript(raw.scriptMode, target.scriptMode)
        let languageMode: PhoneticLanguageMode = crossScript ? .crossScript : mergedLanguageMode(raw.languageMode, target.languageMode)
        let distance = weightedEditDistance(raw.phones, target.phones)
        let pinyin = pinyinSimilarity(raw.phones, target.phones, rawMode: raw.scriptMode, targetMode: target.scriptMode)
        let pairs = confusionPairs(raw: raw.normalized, target: target.normalized)
        let purePhonetic = isPurePhoneticCandidate(
            raw: raw,
            target: target,
            crossScript: crossScript,
            distance: distance,
            pinyinSimilarity: pinyin
        )

        return PhoneticComparison(
            raw: raw,
            target: target,
            languageMode: languageMode,
            isCrossScript: crossScript,
            isPurePhoneticCandidate: purePhonetic,
            weightedPhoneticEditDistance: distance,
            pinyinSimilarity: pinyin,
            confusionPairs: pairs
        )
    }

    static func normalize(_ text: String) -> String {
        let converted = OpenCCConverter.shared.convert(text)
        return converted
            .folding(options: [.widthInsensitive], locale: Locale(identifier: "zh_Hant_TW"))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func classifyScript(_ text: String) -> PhoneticScriptMode {
        guard !text.isEmpty else { return .unknown }

        let hasCJK = text.contains(where: \.isCJK)
        let hasLatin = text.unicodeScalars.contains { scalar in
            (65...90).contains(Int(scalar.value)) || (97...122).contains(Int(scalar.value))
        }
        let hasNumberOrSymbol = text.unicodeScalars.contains { scalar in
            CharacterSet.decimalDigits.contains(scalar) || CharacterSet.symbols.contains(scalar)
        }

        if hasCJK && hasLatin { return .mixedZhEn }
        if hasCJK { return .zhOnly }
        if hasLatin { return .enOnly }
        if hasNumberOrSymbol { return .numericSymbol }
        return .unknown
    }

    private static func classifyLanguage(scriptMode: PhoneticScriptMode) -> PhoneticLanguageMode {
        switch scriptMode {
        case .zhOnly:
            return .mandarin
        case .enOnly:
            return .english
        case .mixedZhEn:
            return .codeSwitch
        case .numericSymbol, .unknown:
            return .unknown
        }
    }

    private static func mergedLanguageMode(_ raw: PhoneticLanguageMode, _ target: PhoneticLanguageMode) -> PhoneticLanguageMode {
        if raw == target { return raw }
        if raw == .unknown { return target }
        if target == .unknown { return raw }
        if raw == .codeSwitch || target == .codeSwitch { return .codeSwitch }
        return .unknown
    }

    private static func isCrossScript(_ raw: PhoneticScriptMode, _ target: PhoneticScriptMode) -> Bool {
        switch (raw, target) {
        case (.zhOnly, .enOnly), (.enOnly, .zhOnly), (.zhOnly, .mixedZhEn), (.enOnly, .mixedZhEn), (.mixedZhEn, .zhOnly), (.mixedZhEn, .enOnly):
            return true
        default:
            return false
        }
    }

    private static func countUnits(in text: String, scriptMode: PhoneticScriptMode) -> Int {
        switch scriptMode {
        case .enOnly:
            return latinTokens(in: text).count
        case .mixedZhEn:
            return max(1, latinTokens(in: text).count + text.filter(\.isCJK).count)
        case .zhOnly:
            return text.filter { $0.isCJK || $0.isNumber }.count
        case .numericSymbol:
            return text.filter { !$0.isWhitespace }.count
        case .unknown:
            return text.isEmpty ? 0 : text.count
        }
    }

    private static func bucket(for count: Int) -> PhoneticLengthBucket {
        switch count {
        case 1...4:
            return .oneToFour
        case 5...15:
            return .fiveToFifteen
        case 16...:
            return .sixteenPlus
        default:
            return .unknown
        }
    }

    private static func isCommandLikeText(
        _ text: String,
        lengthBucket: PhoneticLengthBucket,
        isTechnicalTermCandidate: Bool
    ) -> Bool {
        let key = text.lowercased()
        let tokens = latinTokens(in: key)
        let containsCommand = commandTerms.contains(key)
            || tokens.contains(where: { commandTerms.contains($0) })
            || commandTerms.contains(where: { key.contains($0) })

        guard containsCommand || isTechnicalTermCandidate else { return false }
        return lengthBucket == .oneToFour || lengthBucket == .fiveToFifteen
    }

    private static func isTechnicalTermCandidate(_ text: String) -> Bool {
        let lower = text.lowercased()
        let tokens = latinTokens(in: lower)

        if tokens.contains(where: { technicalTerms.contains($0) }) {
            return true
        }

        if text.range(of: #"[A-Z]{2,}|\w+\.\w+|[A-Za-z]+[0-9]+|[0-9]+[A-Za-z]+"#, options: .regularExpression) != nil {
            return true
        }

        return technicalTerms.contains(where: { lower.contains($0) })
    }

    private static func phones(for text: String, scriptMode: PhoneticScriptMode) -> [String] {
        switch scriptMode {
        case .zhOnly:
            return mandarinPhones(for: text)
        case .enOnly:
            return englishPhones(for: text)
        case .mixedZhEn:
            return mixedPhones(for: text)
        case .numericSymbol:
            return text.filter { !$0.isWhitespace }.map { "sym:\($0)" }
        case .unknown:
            return []
        }
    }

    private static func mixedPhones(for text: String) -> [String] {
        var result: [String] = []
        var latinRun = ""

        func flushLatinRun() {
            guard !latinRun.isEmpty else { return }
            result.append(contentsOf: englishPhones(for: latinRun))
            latinRun.removeAll()
        }

        for char in text {
            if char.isCJK {
                flushLatinRun()
                result.append(contentsOf: mandarinPhones(for: String(char)))
            } else if char.isLetter || char.isNumber {
                latinRun.append(char)
            } else {
                flushLatinRun()
            }
        }
        flushLatinRun()
        return result
    }

    private static func mandarinPhones(for text: String) -> [String] {
        var phones: [String] = []
        for char in text where char.isCJK || char.isNumber {
            if char.isNumber {
                phones.append("num:\(char)")
                continue
            }
            let readings = PinyinDatabase.shared.tonelessPinyin(of: char).sorted()
            if let first = readings.first {
                phones.append("zh:\(first)")
            } else {
                phones.append("zh:\(char)")
            }
        }
        return phones
    }

    private static func englishPhones(for text: String) -> [String] {
        latinTokens(in: text).map { token in
            "en:\(englishApproximation(token))"
        }
    }

    private static func englishApproximation(_ token: String) -> String {
        var value = token.lowercased()
        let replacements: [(String, String)] = [
            ("ph", "f"),
            ("ght", "t"),
            ("ck", "k"),
            ("qu", "kw"),
            ("x", "ks"),
            ("c", "k"),
            ("z", "s")
        ]
        for (from, to) in replacements {
            value = value.replacingOccurrences(of: from, with: to)
        }

        var result = ""
        var last: Character?
        for char in value where char.isLetter || char.isNumber {
            if "aeiou".contains(char), !result.isEmpty {
                continue
            }
            if char != last {
                result.append(char)
                last = char
            }
        }
        return result.isEmpty ? value : result
    }

    private static func latinTokens(in text: String) -> [String] {
        let pattern = #"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.matches(in: text, range: range).compactMap { match in
            guard let tokenRange = Range(match.range, in: text) else { return nil }
            return String(text[tokenRange]).lowercased()
        }
    }

    private static func weightedEditDistance(_ lhs: [String], _ rhs: [String]) -> Double? {
        guard !lhs.isEmpty || !rhs.isEmpty else { return nil }
        guard !lhs.isEmpty else { return Double(rhs.count) }
        guard !rhs.isEmpty else { return Double(lhs.count) }

        var previous = Array(stride(from: 0.0, through: Double(rhs.count), by: 1.0))
        for i in 1...lhs.count {
            var current = [Double](repeating: 0, count: rhs.count + 1)
            current[0] = Double(i)
            for j in 1...rhs.count {
                let deletion = previous[j] + 1.0
                let insertion = current[j - 1] + 1.0
                let substitution = previous[j - 1] + substitutionCost(lhs[i - 1], rhs[j - 1])
                current[j] = min(deletion, insertion, substitution)
            }
            previous = current
        }
        return previous[rhs.count]
    }

    private static func substitutionCost(_ lhs: String, _ rhs: String) -> Double {
        if lhs == rhs { return 0 }
        let left = lhs.components(separatedBy: ":").last ?? lhs
        let right = rhs.components(separatedBy: ":").last ?? rhs

        if left.first == right.first {
            return 0.55
        }
        if PinyinDatabase.swapNasal(left) == right || PinyinDatabase.swapNasal(right) == left {
            return 0.4
        }
        return 1.0
    }

    private static func pinyinSimilarity(
        _ lhs: [String],
        _ rhs: [String],
        rawMode: PhoneticScriptMode,
        targetMode: PhoneticScriptMode
    ) -> Double? {
        guard rawMode == .zhOnly, targetMode == .zhOnly else { return nil }
        let left = lhs.map { $0.replacingOccurrences(of: "zh:", with: "") }.joined(separator: " ")
        let right = rhs.map { $0.replacingOccurrences(of: "zh:", with: "") }.joined(separator: " ")
        guard !left.isEmpty, !right.isEmpty else { return nil }
        return sequenceRatio(Array(left), Array(right))
    }

    private static func sequenceRatio(_ lhs: [Character], _ rhs: [Character]) -> Double {
        guard !lhs.isEmpty || !rhs.isEmpty else { return 1 }
        let distance = levenshteinDistance(lhs, rhs)
        let maxLength = max(lhs.count, rhs.count)
        guard maxLength > 0 else { return 1 }
        return 1.0 - (Double(distance) / Double(maxLength))
    }

    private static func levenshteinDistance(_ lhs: [Character], _ rhs: [Character]) -> Int {
        guard !lhs.isEmpty else { return rhs.count }
        guard !rhs.isEmpty else { return lhs.count }

        var previous = Array(0...rhs.count)
        for i in 1...lhs.count {
            var current = [Int](repeating: 0, count: rhs.count + 1)
            current[0] = i
            for j in 1...rhs.count {
                let cost = lhs[i - 1] == rhs[j - 1] ? 0 : 1
                current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            }
            previous = current
        }
        return previous[rhs.count]
    }

    private static func confusionPairs(raw: String, target: String) -> [PhoneticConfusionPair] {
        let rawUnits = comparisonUnits(raw)
        let targetUnits = comparisonUnits(target)
        var pairs: [PhoneticConfusionPair] = []
        var i = 0
        var j = 0
        var position = 0

        while i < rawUnits.count || j < targetUnits.count {
            if i < rawUnits.count, j < targetUnits.count, rawUnits[i].localizedCaseInsensitiveCompare(targetUnits[j]) == .orderedSame {
                i += 1
                j += 1
                position += 1
            } else if i < rawUnits.count, j < targetUnits.count {
                pairs.append(
                    PhoneticConfusionPair(
                        raw: rawUnits[i],
                        target: targetUnits[j],
                        operation: .substitution,
                        position: position
                    )
                )
                i += 1
                j += 1
                position += 1
            } else if i < rawUnits.count {
                pairs.append(
                    PhoneticConfusionPair(
                        raw: rawUnits[i],
                        target: "",
                        operation: .deletion,
                        position: position
                    )
                )
                i += 1
                position += 1
            } else if j < targetUnits.count {
                pairs.append(
                    PhoneticConfusionPair(
                        raw: "",
                        target: targetUnits[j],
                        operation: .insertion,
                        position: position
                    )
                )
                j += 1
                position += 1
            }
        }

        return pairs
    }

    private static func comparisonUnits(_ text: String) -> [String] {
        let tokens = latinTokens(in: text)
        if !tokens.isEmpty, text.contains(where: { $0.isWhitespace }) {
            return tokens
        }
        return text
            .filter { !$0.isWhitespace && !$0.isPunctuation }
            .map(String.init)
    }

    private static func isPurePhoneticCandidate(
        raw: PhoneticTextFeatures,
        target: PhoneticTextFeatures,
        crossScript: Bool,
        distance: Double?,
        pinyinSimilarity: Double?
    ) -> Bool {
        guard raw.normalized != target.normalized else { return false }
        if crossScript {
            return target.isTechnicalTermCandidate || raw.isTechnicalTermCandidate
        }
        if let pinyinSimilarity, pinyinSimilarity >= 0.5 {
            return true
        }
        guard let distance else { return false }
        let denominator = Double(max(raw.phones.count, target.phones.count, 1))
        return distance / denominator <= 0.65
    }
}
