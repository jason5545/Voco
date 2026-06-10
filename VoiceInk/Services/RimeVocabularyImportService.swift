import Foundation
import SwiftData

enum RimeVocabularySourceFile: String, CaseIterable, Hashable {
    case personalDictionary = "personal_dict.txt"
    case customPhrase = "custom_phrase.txt"
}

enum RimeVocabularyCategoryGuess: String, CaseIterable, Hashable {
    case personName
    case technicalTerm
    case generalTerm

    var label: String {
        switch self {
        case .personName:
            return "Person name"
        case .technicalTerm:
            return "Technical term"
        case .generalTerm:
            return "General term"
        }
    }
}

enum RimeVocabularyImportAction: String, CaseIterable, Hashable {
    case vocoVocabulary
    case correctionProtectedTerm

    var label: String {
        switch self {
        case .vocoVocabulary:
            return "Voco Vocabulary"
        case .correctionProtectedTerm:
            return "Correction protected term"
        }
    }
}

struct RimeVocabularyCandidate: Identifiable, Hashable {
    let id: String
    let term: String
    let sourceFile: String
    let section: String?
    let code: String
    let weight: Int?
    let categoryGuess: RimeVocabularyCategoryGuess
    let lineNumber: Int
}

struct RimeVocabularyPreviewItem: Identifiable, Hashable {
    let id: String
    let candidate: RimeVocabularyCandidate
    let suggestedActions: Set<RimeVocabularyImportAction>
    let skipReasons: [String]
    let alreadyExistsInVocabulary: Bool
    let alreadyExistsAsProtectedTerm: Bool
    let duplicateInPreview: Bool

    var isSkipped: Bool {
        !skipReasons.isEmpty
    }

    var isReviewOnly: Bool {
        !isSkipped && suggestedActions.isEmpty
    }

    var pendingActions: Set<RimeVocabularyImportAction> {
        suggestedActions.filter { action in
            switch action {
            case .vocoVocabulary:
                return !alreadyExistsInVocabulary
            case .correctionProtectedTerm:
                return !alreadyExistsAsProtectedTerm
            }
        }
    }

    var isImportable: Bool {
        !isSkipped && !pendingActions.isEmpty
    }

    var destinationLabel: String {
        guard !isSkipped else { return "Skip / review only" }
        guard !suggestedActions.isEmpty else { return "Skip / review only" }

        return RimeVocabularyImportAction.allCases
            .filter { suggestedActions.contains($0) }
            .map(\.label)
            .joined(separator: " + ")
    }

    var statusLabel: String {
        if isSkipped {
            return skipReasons.joined(separator: ", ")
        }
        if pendingActions.isEmpty, !suggestedActions.isEmpty {
            return "Already exists"
        }
        if isReviewOnly {
            return "Review only"
        }
        return "Ready to import"
    }
}

struct RimeVocabularyPreviewSummary: Hashable {
    let totalCount: Int
    let newCount: Int
    let existingCount: Int
    let skippedCount: Int
    let reviewOnlyCount: Int
    let vocabularySuggestionCount: Int
    let protectedTermSuggestionCount: Int
}

struct RimeLearnedUserDBImportPolicy: Hashable {
    let frequencyThresholdRule: String
    let uncommonTermDetectionRule: String
    let technicalTermPriorityRule: String
    let personNameRule: String
    let previewOnlyRule: String
    let lowConfidenceRule: String

    static let phase2PreviewOnly = RimeLearnedUserDBImportPolicy(
        frequencyThresholdRule: "Only consider learned phrases that pass a high frequency threshold and appear across multiple sessions.",
        uncommonTermDetectionRule: "Prefer uncommon tokens by filtering out common function words, short generic words, and sentence-like fragments.",
        technicalTermPriorityRule: "Prioritize mixed Latin/CJK, acronym, digit-bearing, brand, project, and domain vocabulary candidates.",
        personNameRule: "Never infer person names from userdb alone; require an explicit trusted source or user confirmation.",
        previewOnlyRule: "Learned userdb candidates must enter preview first and cannot write directly to Vocabulary or protected terms.",
        lowConfidenceRule: "Never auto-import low confidence learned phrases."
    )
}

struct RimeVocabularyPreview: Hashable {
    let items: [RimeVocabularyPreviewItem]
    let summary: RimeVocabularyPreviewSummary
    let sourceWarnings: [String]
    let learnedUserDBPolicy: RimeLearnedUserDBImportPolicy
}

struct RimeVocabularyImportResult: Hashable {
    let insertedVocabularyCount: Int
    let existingVocabularyCount: Int
    let insertedProtectedTermCount: Int
    let existingProtectedTermCount: Int
    let skippedCount: Int

    var importedCount: Int {
        insertedVocabularyCount + insertedProtectedTermCount
    }
}

protocol CorrectionProtectionManaging: AnyObject {
    func allWords() -> [String]
    func addSynchronously(_ word: String)
}

extension CorrectionProtectionList: CorrectionProtectionManaging {}

final class RimeVocabularyImportService {
    static let shared = RimeVocabularyImportService()

    private let fileManager: FileManager

    init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    var defaultRimeDirectory: URL {
        URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent("Library", isDirectory: true)
            .appendingPathComponent("Rime", isDirectory: true)
    }

    func parse(_ text: String, sourceFile: RimeVocabularySourceFile) -> [RimeVocabularyCandidate] {
        var candidates: [RimeVocabularyCandidate] = []
        var currentSection: String?

        for (offset, rawLine) in text.components(separatedBy: .newlines).enumerated() {
            let lineNumber = offset + 1
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)

            guard !line.isEmpty else { continue }

            if line.hasPrefix("#") {
                if let section = sectionName(fromCommentLine: line) {
                    currentSection = section
                }
                continue
            }

            let fields = rawLine.split(separator: "\t", omittingEmptySubsequences: false)
            guard fields.count >= 2 else { continue }

            let term = String(fields[0]).trimmingCharacters(in: .whitespacesAndNewlines)
            let code = String(fields[1]).trimmingCharacters(in: .whitespacesAndNewlines)
            let weight: Int?
            if fields.count >= 3 {
                let rawWeight = String(fields[2]).trimmingCharacters(in: .whitespacesAndNewlines)
                weight = rawWeight.isEmpty ? nil : Int(rawWeight)
            } else {
                weight = nil
            }

            guard !term.isEmpty, !code.isEmpty else { continue }

            let category = categoryGuess(for: term, sourceFile: sourceFile, section: currentSection)
            let id = "\(sourceFile.rawValue):\(lineNumber):\(term):\(code)"

            candidates.append(
                RimeVocabularyCandidate(
                    id: id,
                    term: term,
                    sourceFile: sourceFile.rawValue,
                    section: currentSection,
                    code: code,
                    weight: weight,
                    categoryGuess: category,
                    lineNumber: lineNumber
                )
            )
        }

        return candidates
    }

    func parseFiles(in rimeDirectory: URL) -> (candidates: [RimeVocabularyCandidate], warnings: [String]) {
        var candidates: [RimeVocabularyCandidate] = []
        var warnings: [String] = []

        for sourceFile in RimeVocabularySourceFile.allCases {
            let url = rimeDirectory.appendingPathComponent(sourceFile.rawValue)

            guard fileManager.fileExists(atPath: url.path) else {
                warnings.append("Missing \(sourceFile.rawValue)")
                continue
            }

            do {
                let text = try String(contentsOf: url, encoding: .utf8)
                candidates.append(contentsOf: parse(text, sourceFile: sourceFile))
            } catch {
                warnings.append("Could not read \(sourceFile.rawValue): \(error.localizedDescription)")
            }
        }

        return (candidates, warnings)
    }

    @MainActor
    func makePreview(
        rimeDirectory: URL? = nil,
        context: ModelContext,
        protectionList: CorrectionProtectionManaging = CorrectionProtectionList.shared
    ) -> RimeVocabularyPreview {
        let source = parseFiles(in: rimeDirectory ?? defaultRimeDirectory)
        let vocabularyWords = fetchVocabularyWords(from: context)

        return makePreview(
            candidates: source.candidates,
            existingVocabularyWords: vocabularyWords,
            existingProtectedTerms: protectionList.allWords(),
            sourceWarnings: source.warnings
        )
    }

    func makePreview(
        candidates: [RimeVocabularyCandidate],
        existingVocabularyWords: [String],
        existingProtectedTerms: [String],
        sourceWarnings: [String] = []
    ) -> RimeVocabularyPreview {
        let existingVocabulary = Set(existingVocabularyWords.map(normalizedKey))
        let existingProtection = Set(existingProtectedTerms.map(normalizedKey))
        var seenCandidates: Set<String> = []

        let items = candidates.map { candidate -> RimeVocabularyPreviewItem in
            let key = normalizedKey(candidate.term)
            let duplicateInPreview = !seenCandidates.insert(key).inserted
            var skipReasons = unsafeSkipReasons(for: candidate)

            if duplicateInPreview {
                skipReasons.append("duplicate in RIME preview")
            }

            let actions = skipReasons.isEmpty
                ? suggestedActions(for: candidate)
                : []

            return RimeVocabularyPreviewItem(
                id: candidate.id,
                candidate: candidate,
                suggestedActions: actions,
                skipReasons: skipReasons,
                alreadyExistsInVocabulary: existingVocabulary.contains(key),
                alreadyExistsAsProtectedTerm: existingProtection.contains(key),
                duplicateInPreview: duplicateInPreview
            )
        }

        return RimeVocabularyPreview(
            items: items,
            summary: summary(for: items),
            sourceWarnings: sourceWarnings,
            learnedUserDBPolicy: .phase2PreviewOnly
        )
    }

    @MainActor
    func importSelectedItems(
        _ selectedItems: [RimeVocabularyPreviewItem],
        context: ModelContext,
        protectionList: CorrectionProtectionManaging = CorrectionProtectionList.shared
    ) throws -> RimeVocabularyImportResult {
        let importableItems = selectedItems.filter(\.isImportable)
        let vocabularyTerms = importableItems
            .filter { $0.pendingActions.contains(.vocoVocabulary) }
            .map { $0.candidate.term }

        let insertedVocabularyCount = try DictionaryService.addVocabularyWords(vocabularyTerms, context: context)

        var insertedProtectedTerms = 0
        var existingProtectedTerms = 0
        var protectedKeys = Set(protectionList.allWords().map(normalizedKey))

        for item in importableItems where item.pendingActions.contains(.correctionProtectedTerm) {
            let term = item.candidate.term.trimmingCharacters(in: .whitespacesAndNewlines)
            let key = normalizedKey(term)

            guard !term.isEmpty else { continue }

            if protectedKeys.contains(key) {
                existingProtectedTerms += 1
                continue
            }

            protectionList.addSynchronously(term)
            protectedKeys.insert(key)
            insertedProtectedTerms += 1
        }

        return RimeVocabularyImportResult(
            insertedVocabularyCount: insertedVocabularyCount,
            existingVocabularyCount: selectedItems.filter { $0.alreadyExistsInVocabulary }.count,
            insertedProtectedTermCount: insertedProtectedTerms,
            existingProtectedTermCount: existingProtectedTerms + selectedItems.filter { $0.alreadyExistsAsProtectedTerm }.count,
            skippedCount: selectedItems.count - importableItems.count
        )
    }

    @MainActor
    private func fetchVocabularyWords(from context: ModelContext) -> [String] {
        let descriptor = FetchDescriptor<VocabularyWord>()
        guard let words = try? context.fetch(descriptor) else {
            return []
        }
        return words.map(\.word)
    }

    private func sectionName(fromCommentLine line: String) -> String? {
        var section = line
            .dropFirst()
            .trimmingCharacters(in: .whitespacesAndNewlines)

        section = section.replacingOccurrences(of: "=", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard !section.isEmpty else { return nil }

        let metadataPrefixes = ["Rime table", "coding:", "格式", "範例"]
        if metadataPrefixes.contains(where: { section.hasPrefix($0) }) {
            return nil
        }

        return section
    }

    private func categoryGuess(
        for term: String,
        sourceFile: RimeVocabularySourceFile,
        section: String?
    ) -> RimeVocabularyCategoryGuess {
        if sourceFile == .personalDictionary, section?.contains("人名") == true {
            return .personName
        }

        if isTechnicalSection(section), !isCommonEnglishTerm(term) {
            return .technicalTerm
        }

        if sourceFile == .customPhrase, isTechnicalToken(term), !isCommonEnglishTerm(term) {
            return .technicalTerm
        }

        return .generalTerm
    }

    private func isTechnicalSection(_ section: String?) -> Bool {
        guard let section else { return false }

        let excludedKeywords = ["常用英文", "個人資訊"]
        if excludedKeywords.contains(where: { section.contains($0) }) {
            return false
        }

        let keywords = [
            "技術", "縮寫", "硬體", "網路", "協定", "程式", "格式", "設計", "數位",
            "AI", "機器", "職位", "品牌", "社交", "媒體", "軟體", "服務",
            "作業系統", "瀏覽器", "工具", "工程", "用語", "Apple", "Microsoft",
            "Google", "Amazon"
        ]

        return keywords.contains { section.localizedCaseInsensitiveContains($0) }
    }

    private func isTechnicalToken(_ term: String) -> Bool {
        guard !isCommonEnglishTerm(term) else { return false }

        if containsLatin(term) {
            return true
        }

        return false
    }

    private func suggestedActions(for candidate: RimeVocabularyCandidate) -> Set<RimeVocabularyImportAction> {
        switch candidate.categoryGuess {
        case .personName:
            if isFullChineseName(candidate.term) {
                return [.vocoVocabulary, .correctionProtectedTerm]
            }
            return [.vocoVocabulary]

        case .technicalTerm:
            if isStrongProtectedTechnicalTerm(candidate.term, section: candidate.section) {
                return [.vocoVocabulary, .correctionProtectedTerm]
            }
            return [.vocoVocabulary]

        case .generalTerm:
            return []
        }
    }

    private func unsafeSkipReasons(for candidate: RimeVocabularyCandidate) -> [String] {
        let term = candidate.term.trimmingCharacters(in: .whitespacesAndNewlines)
        var reasons: [String] = []

        if term.isEmpty {
            reasons.append("empty term")
        }

        if isTooShort(term, category: candidate.categoryGuess) {
            reasons.append("too short")
        }

        if isCommonEnglishTerm(term) || isCommonChineseTerm(term) {
            reasons.append("common term")
        }

        if looksSentenceLike(term) {
            reasons.append("sentence-like")
        }

        if hasTooMuchPunctuation(term) {
            reasons.append("too much punctuation")
        }

        if term.contains("@") {
            reasons.append("email-like")
        }

        return reasons
    }

    private func summary(for items: [RimeVocabularyPreviewItem]) -> RimeVocabularyPreviewSummary {
        RimeVocabularyPreviewSummary(
            totalCount: items.count,
            newCount: items.filter(\.isImportable).count,
            existingCount: items.filter { $0.alreadyExistsInVocabulary || $0.alreadyExistsAsProtectedTerm }.count,
            skippedCount: items.filter(\.isSkipped).count,
            reviewOnlyCount: items.filter(\.isReviewOnly).count,
            vocabularySuggestionCount: items.filter { $0.pendingActions.contains(.vocoVocabulary) }.count,
            protectedTermSuggestionCount: items.filter { $0.pendingActions.contains(.correctionProtectedTerm) }.count
        )
    }

    private func normalizedKey(_ term: String) -> String {
        term.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    private func isTooShort(_ term: String, category: RimeVocabularyCategoryGuess) -> Bool {
        if category == .personName {
            return term.count < 2
        }

        if containsLatin(term) {
            return alphanumericCount(in: term) < 2
        }

        return term.count < 2
    }

    private func isFullChineseName(_ term: String) -> Bool {
        term.count >= 3 && term.count <= 4 && term.allSatisfy(\.isCJK)
    }

    private func isStrongProtectedTechnicalTerm(_ term: String, section: String?) -> Bool {
        guard containsLatin(term) else { return false }
        guard !isCommonEnglishTerm(term) else { return false }

        if term.contains(where: { $0.isUppercase }) { return true }
        if containsDigit(term) { return true }
        if section?.localizedCaseInsensitiveContains("品牌") == true { return true }

        return term.contains(" ") || term.contains("-") || term.count <= 4
    }

    private func isCommonEnglishTerm(_ term: String) -> Bool {
        let key = normalizedKey(term).trimmingCharacters(in: CharacterSet(charactersIn: ".,!?;:"))
        let common: Set<String> = [
            "ok", "okay", "thank", "thanks", "please", "sorry", "hello", "bye",
            "yes", "no", "maybe", "welcome", "ing", "comments", "updated", "update"
        ]
        return common.contains(key)
    }

    private func isCommonChineseTerm(_ term: String) -> Bool {
        let common: Set<String> = ["的", "了", "是", "在", "有", "和", "我", "你", "他", "她", "它"]
        return common.contains(term)
    }

    private func looksSentenceLike(_ term: String) -> Bool {
        if term.contains(where: { "。！？；，".contains($0) }) {
            return true
        }

        let words = term.split(separator: " ")
        if words.count >= 4 {
            return true
        }

        if !containsLatin(term), term.count > 18 {
            return true
        }

        return false
    }

    private func hasTooMuchPunctuation(_ term: String) -> Bool {
        let allowed = Set("-_+#./ ")
        var punctuationCount = 0

        for scalar in term.unicodeScalars {
            let character = Character(scalar)
            if allowed.contains(character) {
                continue
            }
            if CharacterSet.punctuationCharacters.contains(scalar) || CharacterSet.symbols.contains(scalar) {
                punctuationCount += 1
            }
        }

        if term.last.map({ ".,;:".contains($0) }) == true {
            return true
        }

        guard !term.isEmpty else { return false }
        return punctuationCount >= 2 || Double(punctuationCount) / Double(term.count) > 0.2
    }

    private func containsLatin(_ term: String) -> Bool {
        term.unicodeScalars.contains { scalar in
            (65...90).contains(Int(scalar.value)) || (97...122).contains(Int(scalar.value))
        }
    }

    private func containsDigit(_ term: String) -> Bool {
        term.unicodeScalars.contains { CharacterSet.decimalDigits.contains($0) }
    }

    private func alphanumericCount(in term: String) -> Int {
        term.unicodeScalars.filter {
            CharacterSet.alphanumerics.contains($0)
        }.count
    }
}
