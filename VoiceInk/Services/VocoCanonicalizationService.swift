import Foundation

final class VocoCanonicalizationService {
    static let shared = VocoCanonicalizationService()
    static let defaultContextPackID = "builtin.voco-development"
    static let defaultActiveContextIDs = [defaultContextPackID]
    static let enabledContextPackIDsKey = "VocoEnabledContextPackIDs"

    let contextPacks: [VocoContextPack]

    init(contextPacks: [VocoContextPack] = VocoCanonicalizationService.builtInContextPacks) {
        self.contextPacks = contextPacks
    }

    func normalize(
        _ text: String,
        activeContextIDs: [String] = VocoCanonicalizationService.defaultActiveContextIDs,
        additionalTerms: [VocoCanonicalTerm] = [],
        contextHints: [String] = []
    ) -> VocoNormalizationResult {
        guard !text.isEmpty else {
            return VocoNormalizationResult(
                originalText: text,
                normalizedText: text,
                activeContextIDs: activeContextIDs,
                replacements: [],
                suggestions: []
            )
        }

        let termSources = termSources(for: activeContextIDs, additionalTerms: additionalTerms)
        let candidates = replacementCandidates(
            in: text,
            termSources: termSources,
            activeContextIDs: activeContextIDs,
            contextHints: contextHints
        )
        let accepted = nonOverlapping(candidates.filter(\.isAutomatic), keepingBlockers: true)
        let suggestions = nonOverlapping(candidates.filter { !$0.isAutomatic && !$0.isNoop }, keepingBlockers: false)
            .map { replacementRecord(for: $0, in: text) }

        return VocoNormalizationResult(
            originalText: text,
            normalizedText: applying(accepted, to: text),
            activeContextIDs: activeContextIDs,
            replacements: accepted.map { replacementRecord(for: $0, in: text) },
            suggestions: suggestions
        )
    }

    static func vocabularyTerms(from words: [String]) -> [VocoCanonicalTerm] {
        words
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .map {
                VocoCanonicalTerm(
                    id: "vocabulary.\($0.lowercased())",
                    canonical: $0,
                    aliases: [],
                    type: "personal-vocabulary",
                    contexts: ["personal"],
                    caseSensitive: true,
                    autoReplaceThreshold: 0.98
                )
            }
    }

    static func wordReplacementTerms(from replacements: [WordReplacement]) -> [VocoCanonicalTerm] {
        replacements
            .filter(\.isEnabled)
            .compactMap { replacement in
                let canonical = replacement.replacementText.trimmingCharacters(in: .whitespacesAndNewlines)
                let aliases = replacement.originalText
                    .split(separator: ",")
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }

                guard !canonical.isEmpty,
                      !aliases.isEmpty
                else { return nil }

                guard !aliases.contains(where: containsProtectedWord) else { return nil }

                return VocoCanonicalTerm(
                    id: "word-replacement.\(replacement.id.uuidString.lowercased())",
                    canonical: canonical,
                    aliases: aliases,
                    type: "word-replacement",
                    contexts: ["personal-dictionary"],
                    caseSensitive: true,
                    autoReplaceThreshold: 0.97
                )
            }
    }

    private static func containsProtectedWord(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }

        return CorrectionProtectionList.shared.containsProtectedTerm(in: trimmed)
    }

    static func powerModeContextHints(from config: PowerModeConfig?) -> [String] {
        guard let config, config.isEnabled else { return [] }

        var hints: [String] = []
        appendHint(config.name, to: &hints)

        for appConfig in config.appConfigs ?? [] {
            appendHint(appConfig.appName, to: &hints)
            appendHint(appConfig.bundleIdentifier, to: &hints)
        }

        for urlConfig in config.urlConfigs ?? [] {
            appendHint(urlConfig.url, to: &hints)
        }

        return uniqueHints(hints)
    }

    static func contextHints(
        powerMode: PowerModeConfig?,
        appName: String?,
        windowTitle: String?
    ) -> [String] {
        var hints = powerModeContextHints(from: powerMode)
        appendHint(appName, to: &hints)
        appendHint(windowTitle, to: &hints)
        return uniqueHints(hints)
    }

    static func enabledContextPackIDs(defaults: UserDefaults = .standard) -> [String] {
        guard let storedIDs = defaults.array(forKey: enabledContextPackIDsKey) as? [String] else {
            return defaultActiveContextIDs
        }
        return storedIDs
    }

    static func setEnabledContextPackIDs(_ ids: [String], defaults: UserDefaults = .standard) {
        defaults.set(ids, forKey: enabledContextPackIDsKey)
    }

    static func contextDisplayName(for id: String, contextPacks: [VocoContextPack] = builtInContextPacks) -> String {
        if let pack = contextPacks.first(where: { $0.id == id }) {
            return pack.displayName
        }

        if id.hasPrefix("power-mode:") {
            return "Power Mode"
        }

        return id
    }

    static func contextDisplayNames(for ids: [String], contextPacks: [VocoContextPack] = builtInContextPacks) -> [String] {
        ids.map { contextDisplayName(for: $0, contextPacks: contextPacks) }
    }

    private func termSources(
        for activeContextIDs: [String],
        additionalTerms: [VocoCanonicalTerm]
    ) -> [TermCandidateSource] {
        let activeIDs = Set(activeContextIDs)
        var seen: Set<String> = []
        var sources: [TermCandidateSource] = []

        func append(_ term: VocoCanonicalTerm, allowsAutomaticReplacement: Bool) {
            guard seen.insert(term.id).inserted else { return }
            sources.append(
                TermCandidateSource(
                    term: term,
                    allowsAutomaticReplacement: allowsAutomaticReplacement
                )
            )
        }

        let activePacks = contextPacks.filter { activeIDs.contains($0.id) }
        let inactivePacks = contextPacks.filter { !activeIDs.contains($0.id) }

        for pack in activePacks {
            for term in pack.terms {
                append(term, allowsAutomaticReplacement: true)
            }
        }

        for pack in inactivePacks {
            for term in pack.terms {
                append(term, allowsAutomaticReplacement: false)
            }
        }

        for term in additionalTerms {
            append(term, allowsAutomaticReplacement: true)
        }

        return sources
    }

    private func replacementCandidates(
        in text: String,
        termSources: [TermCandidateSource],
        activeContextIDs: [String],
        contextHints: [String]
    ) -> [ReplacementCandidate] {
        var candidates: [ReplacementCandidate] = []
        let nsText = text as NSString

        for source in termSources {
            let term = source.term
            for alias in matchAliases(for: term) {
                guard !alias.isEmpty else { continue }
                let pattern = pattern(for: alias)
                guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else {
                    continue
                }

                let matches = regex.matches(in: text, range: NSRange(location: 0, length: nsText.length))
                for match in matches {
                    let original = nsText.substring(with: match.range)
                    if isAlreadyCanonical(original: original, term: term) ||
                        isInsideExistingCanonical(text: text, range: match.range, alias: alias, canonical: term.canonical) {
                        candidates.append(
                            ReplacementCandidate(
                                range: match.range,
                                originalText: original,
                                replacementText: original,
                                termID: term.id,
                                confidence: 1.0,
                                reason: "canonical-match",
                                isAutomatic: true,
                                isBlocker: true
                            )
                        )
                        continue
                    }

                    if shouldSuppressAmbiguousWordReplacement(original: original, replacement: term.canonical, term: term) {
                        continue
                    }

                    let hasContext = hasStrongContext(
                        for: term,
                        in: text,
                        matchRange: match.range,
                        activeContextIDs: activeContextIDs,
                        contextHints: contextHints
                    )
                    let confidence = confidence(for: term, original: original, alias: alias, hasContext: hasContext)
                    let isAutomatic = source.allowsAutomaticReplacement && confidence >= term.autoReplaceThreshold

                    candidates.append(
                        ReplacementCandidate(
                            range: match.range,
                            originalText: original,
                            replacementText: term.canonical,
                            termID: term.id,
                            confidence: confidence,
                            reason: reason(
                                for: term,
                                original: original,
                                alias: alias,
                                isAutomatic: isAutomatic,
                                allowsAutomaticReplacement: source.allowsAutomaticReplacement
                            ),
                            isAutomatic: isAutomatic,
                            isBlocker: false
                        )
                    )
                }
            }
        }

        return candidates.sorted {
            if $0.range.location != $1.range.location {
                return $0.range.location < $1.range.location
            }
            if $0.range.length != $1.range.length {
                return $0.range.length > $1.range.length
            }
            return $0.confidence > $1.confidence
        }
    }

    private func matchAliases(for term: VocoCanonicalTerm) -> [String] {
        var seen: Set<String> = []
        var aliases: [String] = []

        for alias in [term.canonical] + term.aliases {
            let key = alias.lowercased()
            guard seen.insert(key).inserted else { continue }
            aliases.append(alias)
        }

        return aliases.sorted { $0.count > $1.count }
    }

    private func pattern(for alias: String) -> String {
        let escaped = NSRegularExpression.escapedPattern(for: alias)
        guard usesWordBoundaries(for: alias) else {
            return escaped
        }
        return "(?<![A-Za-z0-9])\(escaped)(?![A-Za-z0-9])"
    }

    private func usesWordBoundaries(for text: String) -> Bool {
        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 0x3040...0x309F, 0x30A0...0x30FF, 0x4E00...0x9FFF, 0xAC00...0xD7AF, 0x0E00...0x0E7F:
                return false
            default:
                continue
            }
        }
        return true
    }

    private func isAlreadyCanonical(original: String, term: VocoCanonicalTerm) -> Bool {
        if term.caseSensitive {
            return original == term.canonical
        }
        return original.compare(term.canonical, options: [.caseInsensitive, .diacriticInsensitive]) == .orderedSame
    }

    private func isInsideExistingCanonical(text: String, range: NSRange, alias: String, canonical: String) -> Bool {
        guard canonical.localizedCaseInsensitiveContains(alias),
              let swiftRange = Range(range, in: text)
        else {
            return false
        }

        let aliasOffsetCandidates = canonical.ranges(of: alias, options: [.caseInsensitive, .diacriticInsensitive])
            .map { canonical.distance(from: canonical.startIndex, to: $0.lowerBound) }

        for aliasOffset in aliasOffsetCandidates {
            guard let canonicalStart = text.index(swiftRange.lowerBound, offsetBy: -aliasOffset, limitedBy: text.startIndex),
                  let canonicalEnd = text.index(canonicalStart, offsetBy: canonical.count, limitedBy: text.endIndex)
            else {
                continue
            }

            let window = String(text[canonicalStart..<canonicalEnd])
            if window == canonical {
                return true
            }
        }

        return false
    }

    private func shouldSuppressAmbiguousWordReplacement(
        original: String,
        replacement: String,
        term: VocoCanonicalTerm
    ) -> Bool {
        guard term.type == "word-replacement" else { return false }

        // Both words are common valid nouns. Recent retranscribe audits showed
        // this pair being learned too broadly and corrupting QR-code poster text.
        if ["圖案", "图案"].contains(original),
           replacement == "專案" {
            return true
        }

        return false
    }

    private func confidence(
        for term: VocoCanonicalTerm,
        original: String,
        alias: String,
        hasContext: Bool
    ) -> Double {
        if term.requiresContextForAutoReplace && !hasContext {
            return 0.55
        }

        if original.compare(term.canonical, options: [.caseInsensitive, .diacriticInsensitive]) == .orderedSame {
            return term.caseSensitive ? 0.99 : 1.0
        }

        if alias.compare(original, options: [.caseInsensitive, .diacriticInsensitive]) == .orderedSame {
            return 0.97
        }

        return 0.92
    }

    private func hasStrongContext(
        for term: VocoCanonicalTerm,
        in text: String,
        matchRange: NSRange,
        activeContextIDs: [String],
        contextHints: [String]
    ) -> Bool {
        guard term.requiresContextForAutoReplace else { return true }

        let activeContexts = Set(activeContextIDs)
        if !activeContexts.intersection(term.contexts).isEmpty {
            return true
        }

        let hintText = contextHints.joined(separator: " ").lowercased()
        if !hintText.isEmpty {
            if term.contexts.contains(where: { hintText.contains($0.lowercased()) }) {
                return true
            }
            if contextIndicators(for: term).contains(where: { hintText.contains($0.lowercased()) }) {
                return true
            }
        }

        let nearbyText = contextWindow(in: text, range: matchRange).lowercased()
        return contextIndicators(for: term).contains { indicator in
            nearbyText.contains(indicator.lowercased())
        }
    }

    private func contextIndicators(for term: VocoCanonicalTerm) -> [String] {
        var indicators: [String] = []
        let contexts = Set(term.contexts.map { $0.lowercased() })

        if contexts.contains("music") ||
            contexts.contains("anime") ||
            term.type == "song" ||
            term.type == "artist" ||
            term.type == "anime" {
            indicators.append(contentsOf: Self.musicContextIndicators)
        }

        if contexts.contains("voco") ||
            contexts.contains("development") ||
            contexts.contains("asr") ||
            contexts.contains("dictation") ||
            term.canonical == "VOCO" {
            indicators.append(contentsOf: Self.vocoDevelopmentContextIndicators)
        }

        if contexts.contains("image-generation") ||
            contexts.contains("node-graph") ||
            term.canonical == "ComfyUI" {
            indicators.append(contentsOf: Self.imageGenerationContextIndicators)
        }

        indicators.append(contentsOf: term.contexts)
        return Self.uniqueHints(indicators)
    }

    private func contextWindow(in text: String, range: NSRange, radius: Int = 18) -> String {
        guard let swiftRange = Range(range, in: text) else { return text }
        let start = text.index(swiftRange.lowerBound, offsetBy: -radius, limitedBy: text.startIndex) ?? text.startIndex
        let end = text.index(swiftRange.upperBound, offsetBy: radius, limitedBy: text.endIndex) ?? text.endIndex
        return String(text[start..<end])
    }

    private func reason(
        for term: VocoCanonicalTerm,
        original: String,
        alias: String,
        isAutomatic: Bool,
        allowsAutomaticReplacement: Bool
    ) -> String {
        if !allowsAutomaticReplacement {
            return "inactive-context-suggestion"
        }
        if !isAutomatic {
            return "context-required"
        }
        if original.compare(term.canonical, options: [.caseInsensitive, .diacriticInsensitive]) == .orderedSame {
            return "case-normalization"
        }
        if alias.compare(original, options: [.caseInsensitive, .diacriticInsensitive]) == .orderedSame {
            return "alias-match"
        }
        return "contextual-alias-match"
    }

    private func nonOverlapping(_ candidates: [ReplacementCandidate], keepingBlockers: Bool) -> [ReplacementCandidate] {
        var accepted: [ReplacementCandidate] = []
        var occupied: [NSRange] = []

        for candidate in candidates {
            if occupied.contains(where: { NSIntersectionRange($0, candidate.range).length > 0 }) {
                continue
            }
            if candidate.isBlocker {
                if keepingBlockers {
                    occupied.append(candidate.range)
                }
                continue
            }
            guard !candidate.isNoop else { continue }
            accepted.append(candidate)
            occupied.append(candidate.range)
        }

        return accepted.sorted { $0.range.location < $1.range.location }
    }

    private func applying(_ candidates: [ReplacementCandidate], to text: String) -> String {
        guard !candidates.isEmpty else { return text }

        let nsText = text as NSString
        var result = ""
        var cursor = 0

        for candidate in candidates {
            if cursor < candidate.range.location {
                result += nsText.substring(with: NSRange(location: cursor, length: candidate.range.location - cursor))
            }
            result += candidate.replacementText
            cursor = candidate.range.location + candidate.range.length
        }

        if cursor < nsText.length {
            result += nsText.substring(with: NSRange(location: cursor, length: nsText.length - cursor))
        }

        if candidates.contains(where: { $0.replacementText.containsCJKOrKana }) {
            return result.removingSpacesBetweenCJKOrKana()
        }

        return result
    }

    private func replacementRecord(for candidate: ReplacementCandidate, in text: String) -> VocoReplacement {
        let characterRange = characterRange(for: candidate.range, in: text)
        return VocoReplacement(
            originalText: candidate.originalText,
            replacementText: candidate.replacementText,
            termID: candidate.termID,
            confidence: candidate.confidence,
            reason: candidate.reason,
            rangeStart: characterRange?.start,
            rangeLength: characterRange?.length
        )
    }

    private func characterRange(for range: NSRange, in text: String) -> (start: Int, length: Int)? {
        guard let swiftRange = Range(range, in: text) else { return nil }
        let start = text.distance(from: text.startIndex, to: swiftRange.lowerBound)
        let length = text.distance(from: swiftRange.lowerBound, to: swiftRange.upperBound)
        return (start, length)
    }

    private static func appendHint(_ value: String?, to hints: inout [String]) {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty
        else { return }

        hints.append(trimmed)

        let separators = CharacterSet(charactersIn: ".:/-_")
            .union(.whitespacesAndNewlines)
        let parts = trimmed
            .components(separatedBy: separators)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { $0.count >= 3 }
        hints.append(contentsOf: parts)
    }

    private static func uniqueHints(_ hints: [String]) -> [String] {
        var seen: Set<String> = []
        return hints.filter { hint in
            seen.insert(hint.lowercased()).inserted
        }
    }
}

private struct TermCandidateSource {
    let term: VocoCanonicalTerm
    let allowsAutomaticReplacement: Bool
}

private struct ReplacementCandidate {
    let range: NSRange
    let originalText: String
    let replacementText: String
    let termID: String
    let confidence: Double
    let reason: String
    let isAutomatic: Bool
    let isBlocker: Bool

    var isNoop: Bool {
        originalText == replacementText
    }
}

private extension String {
    func ranges(of searchString: String, options: String.CompareOptions) -> [Range<String.Index>] {
        var ranges: [Range<String.Index>] = []
        var searchRange = startIndex..<endIndex

        while let range = range(of: searchString, options: options, range: searchRange) {
            ranges.append(range)
            searchRange = range.upperBound..<endIndex
        }

        return ranges
    }

    var containsCJKOrKana: Bool {
        contains { $0.isCJKOrKana }
    }

    func removingSpacesBetweenCJKOrKana() -> String {
        let chars = Array(self)
        guard chars.count >= 3 else { return self }

        var result = ""
        for index in chars.indices {
            let char = chars[index]
            if char.isWhitespace,
               index > chars.startIndex,
               index < chars.index(before: chars.endIndex),
               chars[chars.index(before: index)].isCJKOrKana,
               chars[chars.index(after: index)].isCJKOrKana {
                continue
            }
            result.append(char)
        }

        return result
    }
}

private extension Character {
    var isCJKOrKana: Bool {
        unicodeScalars.contains { scalar in
            switch scalar.value {
            case 0x3040...0x309F, 0x30A0...0x30FF, 0x4E00...0x9FFF, 0x3400...0x4DBF, 0x20000...0x2A6DF:
                return true
            default:
                return false
            }
        }
    }
}

extension VocoCanonicalizationService {
    static let builtInContextPacks: [VocoContextPack] = [
        VocoContextPack(
            id: defaultContextPackID,
            displayName: "VOCO Development",
            terms: [
                VocoCanonicalTerm(
                    id: "product.voiceink",
                    canonical: "VoiceInk",
                    aliases: ["Voice Ink", "voice ink", "Voice Inc", "Voice ANC", "Voice INK"],
                    type: "product",
                    contexts: ["voco", "development"],
                    caseSensitive: true,
                    autoReplaceThreshold: 0.95
                ),
                VocoCanonicalTerm(
                    id: "product.voco",
                    canonical: "VOCO",
                    aliases: ["Voco", "voco", "voice co"],
                    type: "product",
                    contexts: ["voco", "development"],
                    caseSensitive: true,
                    autoReplaceThreshold: 0.95
                ),
                VocoCanonicalTerm(
                    id: "product.voco.ambiguous-vocal",
                    canonical: "VOCO",
                    aliases: ["vocal"],
                    type: "product",
                    contexts: ["voco", "development", "asr", "dictation"],
                    caseSensitive: true,
                    autoReplaceThreshold: 0.9,
                    requiresContextForAutoReplace: true
                ),
                VocoCanonicalTerm(
                    id: "model.qwen3-asr",
                    canonical: "Qwen3-ASR",
                    aliases: ["Qwen ASR", "Qwen3 ASR", "qwen asr", "Q one three ASR", "Qwen three ASR", "qwen three asr"],
                    type: "model",
                    contexts: ["asr", "development"],
                    caseSensitive: true,
                    autoReplaceThreshold: 0.95
                ),
                VocoCanonicalTerm(
                    id: "framework.mlx",
                    canonical: "MLX",
                    aliases: ["mlx", "M L X"],
                    type: "framework",
                    contexts: ["asr", "development"],
                    caseSensitive: true,
                    autoReplaceThreshold: 0.95
                ),
                VocoCanonicalTerm(
                    id: "model.whisper",
                    canonical: "Whisper",
                    aliases: ["whisper"],
                    type: "model",
                    contexts: ["asr", "development"],
                    caseSensitive: true,
                    autoReplaceThreshold: 0.95
                ),
                VocoCanonicalTerm(
                    id: "project.whisper-cpp",
                    canonical: "whisper.cpp",
                    aliases: ["Whisper.cpp", "whisper cpp"],
                    type: "project",
                    contexts: ["asr", "development"],
                    caseSensitive: true,
                    autoReplaceThreshold: 0.95
                ),
                VocoCanonicalTerm(
                    id: "format.gguf",
                    canonical: "GGUF",
                    aliases: ["gguf", "G G U F"],
                    type: "format",
                    contexts: ["asr", "development"],
                    caseSensitive: true,
                    autoReplaceThreshold: 0.95
                ),
                VocoCanonicalTerm(
                    id: "product.typeless",
                    canonical: "Typeless",
                    aliases: ["typeless", "Type less", "TypeLess"],
                    type: "product",
                    contexts: ["dictation", "development"],
                    caseSensitive: true,
                    autoReplaceThreshold: 0.95
                ),
                VocoCanonicalTerm(
                    id: "term.edge-case",
                    canonical: "edge case",
                    aliases: ["H case", "H Case", "h case"],
                    type: "technical-term",
                    contexts: ["development", "debugging"],
                    autoReplaceThreshold: 0.95
                ),
                VocoCanonicalTerm(
                    id: "app.comfyui",
                    canonical: "ComfyUI",
                    aliases: ["Config UI", "config UI", "config ui", "ConfigUI", "Confi UI", "confi ui", "config.yml"],
                    type: "application",
                    contexts: ["image-generation", "workflow", "node-graph"],
                    caseSensitive: true,
                    autoReplaceThreshold: 0.95,
                    requiresContextForAutoReplace: true
                ),
                VocoCanonicalTerm(
                    id: "platform.macos",
                    canonical: "macOS",
                    aliases: ["Mac OS", "MacOS", "mac os"],
                    type: "platform",
                    contexts: ["development", "system"],
                    caseSensitive: true,
                    autoReplaceThreshold: 0.95
                ),
                VocoCanonicalTerm(
                    id: "artist.lisa",
                    canonical: "LiSA",
                    aliases: ["Lisa", "LISA", "lisa", "莉莎", "リサ"],
                    type: "artist",
                    contexts: ["music", "anime"],
                    caseSensitive: true,
                    autoReplaceThreshold: 0.95
                ),
                VocoCanonicalTerm(
                    id: "anime.kimetsu",
                    canonical: "鬼滅之刃",
                    aliases: ["鬼滅", "Demon Slayer", "きめつのやいば"],
                    type: "anime",
                    contexts: ["music", "anime"],
                    autoReplaceThreshold: 0.95
                ),
                VocoCanonicalTerm(
                    id: "song.gurenge",
                    canonical: "紅蓮華",
                    aliases: ["紅蓮花", "ぐれんげ", "Gurenge"],
                    type: "song",
                    contexts: ["music", "anime"],
                    autoReplaceThreshold: 0.95
                ),
                VocoCanonicalTerm(
                    id: "song.homura",
                    canonical: "炎",
                    aliases: ["焰", "ほむら", "Homura", "homura"],
                    type: "song",
                    contexts: ["music", "anime"],
                    autoReplaceThreshold: 0.9,
                    requiresContextForAutoReplace: true
                ),
                VocoCanonicalTerm(
                    id: "song.akeboshi",
                    canonical: "明け星",
                    aliases: ["Akeboshi", "akeboshi", "あけぼし"],
                    type: "song",
                    contexts: ["music", "anime"],
                    autoReplaceThreshold: 0.95
                ),
                VocoCanonicalTerm(
                    id: "song.shirogane",
                    canonical: "白銀",
                    aliases: ["Shirogane", "shirogane", "しろがね"],
                    type: "song",
                    contexts: ["music", "anime"],
                    autoReplaceThreshold: 0.95
                ),
            ]
        ),
    ]

    fileprivate static let musicContextIndicators = [
        "lisa",
        "LiSA",
        "鬼滅",
        "鬼滅之刃",
        "Demon Slayer",
        "紅蓮華",
        "明け星",
        "白銀",
        "歌曲",
        "歌",
        "唱",
        "難唱",
        "主題曲",
        "片頭",
        "片尾",
        "music",
        "song",
        "anime",
    ]

    fileprivate static let vocoDevelopmentContextIndicators = [
        "voco",
        "VoiceInk",
        "voice ink",
        "Qwen",
        "Qwen3",
        "ASR",
        "MLX",
        "Whisper",
        "whisper.cpp",
        "GGUF",
        "Typeless",
        "dictation",
        "transcription",
        "speech",
        "candidate",
        "confidence",
        "context aware",
        "context-aware",
        "fork",
        "development",
        "開發",
        "模型",
        "轉錄",
        "聽寫",
        "語音",
        "候選",
        "信心",
    ]

    fileprivate static let imageGenerationContextIndicators = [
        "ComfyUI",
        "comfyui",
        "Stable Diffusion",
        "Draw Things",
        "diffusion",
        "產圖",
        "生圖",
        "圖片生成",
        "AI 繪圖",
        "workflow",
        "流程圖",
        "節點",
        "node",
        "nodes",
        "連線",
        "連來連去",
        "node graph",
    ]
}
