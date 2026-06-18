import Foundation

struct VocoPhoneticCorrectionRule: Codable, Equatable, Identifiable {
    let id: String
    let sourcePhrases: [String]
    let targetText: String
    let enabled: Bool
    let requiresTargetInVocabulary: Bool

    init(
        id: String,
        sourcePhrases: [String],
        targetText: String,
        enabled: Bool = true,
        requiresTargetInVocabulary: Bool = true
    ) {
        self.id = id
        self.sourcePhrases = sourcePhrases
        self.targetText = targetText
        self.enabled = enabled
        self.requiresTargetInVocabulary = requiresTargetInVocabulary
    }

    enum CodingKeys: String, CodingKey {
        case id
        case sourcePhrases
        case targetText
        case enabled
        case requiresTargetInVocabulary
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            id: try container.decode(String.self, forKey: .id),
            sourcePhrases: try container.decode([String].self, forKey: .sourcePhrases),
            targetText: try container.decode(String.self, forKey: .targetText),
            enabled: try container.decodeIfPresent(Bool.self, forKey: .enabled) ?? true,
            requiresTargetInVocabulary: try container.decodeIfPresent(Bool.self, forKey: .requiresTargetInVocabulary) ?? true
        )
    }
}

struct VocoPhoneticCorrectionFire: Equatable {
    let ruleID: String
    let sourceText: String
    let targetText: String
    let range: NSRange
    let confidence: Double

    var autoApplyPolicyFire: VocoAutoApplyPolicyFire {
        VocoAutoApplyPolicyFire(
            policyId: ruleID,
            policyType: "phoneticCorrectionTerm",
            autoApplyMode: "apply",
            sourcePattern: sourceText,
            targetText: targetText,
            sourceSlices: ["phonetic-correction-term"]
        )
    }
}

struct VocoPhoneticCorrectionEvaluation: Equatable {
    let inputText: String
    let outputText: String
    let applied: [VocoPhoneticCorrectionFire]

    var changed: Bool {
        inputText != outputText
    }
}

final class VocoPhoneticCorrectionService {
    static let shared = VocoPhoneticCorrectionService()
    static let reason = "phonetic-correction-term"

    private let rulesURL: URL
    private let seedRules: [VocoPhoneticCorrectionRule]

    init(
        rulesURL: URL = AppIdentifiers.appSupportDirectory
            .appendingPathComponent("PhoneticCorrections", isDirectory: true)
            .appendingPathComponent("phonetic-correction-rules.json"),
        seedRules: [VocoPhoneticCorrectionRule] = VocoPhoneticCorrectionService.defaultSeedRules
    ) {
        self.rulesURL = rulesURL
        self.seedRules = seedRules
    }

    func evaluate(
        _ text: String,
        vocabularyWords: [String]
    ) -> VocoPhoneticCorrectionEvaluation {
        guard !text.isEmpty,
              VoiceCommandService.shared.detectCommand(in: text) == nil
        else {
            return VocoPhoneticCorrectionEvaluation(inputText: text, outputText: text, applied: [])
        }

        let vocabulary = Set(vocabularyWords.map(normalizedKey))
        let rules = loadedRules()
            .filter(\.enabled)
            .filter { rule in
                !rule.targetText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
                    (!rule.requiresTargetInVocabulary || vocabulary.contains(normalizedKey(rule.targetText)))
            }

        guard !rules.isEmpty else {
            return VocoPhoneticCorrectionEvaluation(inputText: text, outputText: text, applied: [])
        }

        let matches = nonOverlappingMatches(in: text, rules: rules)
        let output = applying(matches, to: text)
        return VocoPhoneticCorrectionEvaluation(inputText: text, outputText: output, applied: matches)
    }

    private func loadedRules() -> [VocoPhoneticCorrectionRule] {
        var rules = seedRules
        guard let data = try? Data(contentsOf: rulesURL),
              let userRules = try? JSONDecoder().decode([VocoPhoneticCorrectionRule].self, from: data)
        else {
            return rules
        }

        var seen = Set(rules.map(\.id))
        for rule in userRules where seen.insert(rule.id).inserted {
            rules.append(rule)
        }
        return rules
    }

    private func nonOverlappingMatches(
        in text: String,
        rules: [VocoPhoneticCorrectionRule]
    ) -> [VocoPhoneticCorrectionFire] {
        let nsText = text as NSString
        let fullRange = NSRange(location: 0, length: nsText.length)
        var matches: [VocoPhoneticCorrectionFire] = []

        for rule in rules {
            let target = rule.targetText.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !target.isEmpty else { continue }

            for source in normalizedSources(rule.sourcePhrases, target: target) {
                let pattern = NSRegularExpression.escapedPattern(for: source)
                guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }

                for match in regex.matches(in: text, range: fullRange) {
                    let original = nsText.substring(with: match.range)
                    guard original != target else { continue }
                    matches.append(
                        VocoPhoneticCorrectionFire(
                            ruleID: rule.id,
                            sourceText: original,
                            targetText: target,
                            range: match.range,
                            confidence: 0.995
                        )
                    )
                }
            }
        }

        return nonOverlapping(matches)
    }

    private func normalizedSources(_ sources: [String], target: String) -> [String] {
        var seen: Set<String> = []
        return sources
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && $0 != target }
            .filter { source in
                // Two-character CJK phonetic pairs are too collision-prone. Keep
                // short corrections in explicit Word Replacement instead.
                !isShortCJKSource(source)
            }
            .sorted { $0.count > $1.count }
            .filter { seen.insert($0).inserted }
    }

    private func isShortCJKSource(_ source: String) -> Bool {
        source.count <= 2 && source.allSatisfy(Self.isCJKCharacter)
    }

    private func nonOverlapping(_ matches: [VocoPhoneticCorrectionFire]) -> [VocoPhoneticCorrectionFire] {
        let sorted = matches.sorted {
            if $0.range.location != $1.range.location {
                return $0.range.location < $1.range.location
            }
            if $0.range.length != $1.range.length {
                return $0.range.length > $1.range.length
            }
            return $0.confidence > $1.confidence
        }

        var accepted: [VocoPhoneticCorrectionFire] = []
        var occupied: [NSRange] = []
        for match in sorted {
            guard !occupied.contains(where: { NSIntersectionRange($0, match.range).length > 0 }) else {
                continue
            }
            accepted.append(match)
            occupied.append(match.range)
        }
        return accepted
    }

    private func applying(_ matches: [VocoPhoneticCorrectionFire], to text: String) -> String {
        guard !matches.isEmpty else { return text }

        let nsText = text as NSString
        var result = ""
        var cursor = 0

        for match in matches.sorted(by: { $0.range.location < $1.range.location }) {
            if cursor < match.range.location {
                result += nsText.substring(with: NSRange(location: cursor, length: match.range.location - cursor))
            }
            result += match.targetText
            cursor = match.range.location + match.range.length
        }

        if cursor < nsText.length {
            result += nsText.substring(from: cursor)
        }

        return Self.removingSpacesBetweenCJKOrKana(result)
    }

    private func normalizedKey(_ value: String) -> String {
        value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    private static func removingSpacesBetweenCJKOrKana(_ text: String) -> String {
        let chars = Array(text)
        guard chars.count >= 3 else { return text }

        var result = ""
        for index in chars.indices {
            let char = chars[index]
            if char.isWhitespace,
               index > chars.startIndex,
               index < chars.index(before: chars.endIndex),
               isCJKOrKanaCharacter(chars[chars.index(before: index)]),
               isCJKOrKanaCharacter(chars[chars.index(after: index)]) {
                continue
            }
            result.append(char)
        }

        return result
    }

    private static func isCJKCharacter(_ character: Character) -> Bool {
        character.unicodeScalars.contains { scalar in
            switch scalar.value {
            case 0x4E00...0x9FFF, 0x3400...0x4DBF, 0x20000...0x2A6DF:
                return true
            default:
                return false
            }
        }
    }

    private static func isCJKOrKanaCharacter(_ character: Character) -> Bool {
        character.unicodeScalars.contains { scalar in
            switch scalar.value {
            case 0x3040...0x309F, 0x30A0...0x30FF, 0x4E00...0x9FFF, 0x3400...0x4DBF, 0x20000...0x2A6DF:
                return true
            default:
                return false
            }
        }
    }

    private static let defaultSeedRules: [VocoPhoneticCorrectionRule] = [
        VocoPhoneticCorrectionRule(
            id: "seed.name.li-sheng-ling",
            sourcePhrases: ["李勝林", "李聖林"],
            targetText: "李聖苓"
        ),
        VocoPhoneticCorrectionRule(
            id: "seed.name.jian-rui-yan",
            sourcePhrases: ["簡瑞燕"],
            targetText: "簡瑞彥"
        ),
        VocoPhoneticCorrectionRule(
            id: "seed.name.li-sheng-hong",
            sourcePhrases: ["李勝宏"],
            targetText: "李聖葒"
        ),
        VocoPhoneticCorrectionRule(
            id: "seed.name.jian-yue-xiong",
            sourcePhrases: ["簡越雄"],
            targetText: "簡岳雄"
        ),
        VocoPhoneticCorrectionRule(
            id: "seed.company.shiji-wind-power",
            sourcePhrases: ["四季風電"],
            targetText: "世紀風電"
        ),
    ]
}
