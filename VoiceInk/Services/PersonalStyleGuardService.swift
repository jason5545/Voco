import Foundation

struct PersonalStyleGuardResult: Equatable {
    let isValid: Bool
    let reasons: [String]
}

final class PersonalStyleGuardService {
    static let shared = PersonalStyleGuardService()
    static let enabledKey = "PersonalStyleGuardEnabled"
    static let defaultEnabled = true

    private let assistantOpeners = [
        "以下是",
        "總而言之",
        "總結來說",
        "值得注意的是",
        "核心在於",
        "整體而言",
        "換句話說",
    ]

    private let structureMarkers = [
        "\n- ",
        "\n• ",
        "\n1.",
        "\n2.",
        "\n3.",
        "\n一、",
        "\n二、",
        "\n三、",
        "###",
        "##",
    ]

    private let rewriteIntentMarkers = [
        "條列",
        "列表",
        "重點",
        "整理成",
        "摘要",
        "總結",
        "幫我寫",
        "改寫",
        "email",
        "Email",
        "信件",
        "提案",
    ]

    static func isEnabled(defaults: UserDefaults = .standard) -> Bool {
        defaults.object(forKey: enabledKey) as? Bool ?? defaultEnabled
    }

    static func setEnabled(_ isEnabled: Bool, defaults: UserDefaults = .standard) {
        defaults.set(isEnabled, forKey: enabledKey)
    }

    func validate(response: String, original: String) -> PersonalStyleGuardResult {
        let trimmedResponse = response.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedOriginal = original.trimmingCharacters(in: .whitespacesAndNewlines)
        var reasons: [String] = []

        guard !trimmedOriginal.isEmpty, !trimmedResponse.isEmpty else {
            return PersonalStyleGuardResult(isValid: true, reasons: [])
        }

        for opener in assistantOpeners
            where trimmedResponse.hasPrefix(opener) && !trimmedOriginal.hasPrefix(opener) {
            reasons.append("assistant-opener:\(opener)")
        }

        if introducesStructuredFormatting(response: trimmedResponse, original: trimmedOriginal),
           !hasRewriteIntent(trimmedOriginal) {
            reasons.append("introduced-structured-format")
        }

        let lengthRatio = Double(trimmedResponse.count) / Double(max(trimmedOriginal.count, 1))
        if trimmedOriginal.count < 80,
           lengthRatio > 1.8,
           !hasRewriteIntent(trimmedOriginal) {
            reasons.append("style-expansion")
        }

        let droppedTerms = droppedMixedLanguageTerms(response: trimmedResponse, original: trimmedOriginal)
        for term in droppedTerms.prefix(3) {
            reasons.append("dropped-mixed-language-term:\(term)")
        }

        return PersonalStyleGuardResult(isValid: reasons.isEmpty, reasons: reasons)
    }

    private func introducesStructuredFormatting(response: String, original: String) -> Bool {
        let responseHasStructure = structureMarkers.contains { response.contains($0) }
        guard responseHasStructure else { return false }
        return !structureMarkers.contains { original.contains($0) }
    }

    private func hasRewriteIntent(_ text: String) -> Bool {
        rewriteIntentMarkers.contains { text.localizedCaseInsensitiveContains($0) }
    }

    private func droppedMixedLanguageTerms(response: String, original: String) -> [String] {
        let responseNormalized = normalizeLatin(response)
        return latinTerms(in: original).filter { term in
            !responseNormalized.contains(normalizeLatin(term))
        }
    }

    private func latinTerms(in text: String) -> [String] {
        var terms: [String] = []
        var buffer = ""

        func flush() {
            let trimmed = buffer.trimmingCharacters(in: CharacterSet(charactersIn: " -_./+"))
            defer { buffer = "" }
            let normalized = normalizeLatin(trimmed)
            guard normalized.count >= 2,
                  normalized.contains(where: { $0.isLetter })
            else {
                return
            }
            terms.append(trimmed)
        }

        for char in text {
            if char.isASCIIAlphanumeric || " -_./+".contains(char) {
                buffer.append(char)
            } else if !buffer.isEmpty {
                flush()
            }
        }

        if !buffer.isEmpty {
            flush()
        }

        var seen: Set<String> = []
        return terms.filter { seen.insert(normalizeLatin($0)).inserted }
    }

    private func normalizeLatin(_ text: String) -> String {
        text.lowercased().filter { $0.isLetter || $0.isNumber }
    }
}

private extension Character {
    var isASCIIAlphanumeric: Bool {
        unicodeScalars.allSatisfy { scalar in
            scalar.isASCII &&
                (CharacterSet.letters.contains(scalar) || CharacterSet.decimalDigits.contains(scalar))
        }
    }
}
