import Foundation

struct VocoRimeHomophoneLearnedWord: Codable, Equatable {
    let word: String
    let syllables: [String]
    let commits: Int
}

struct VocoRimeHomophoneCharacterReading: Codable, Equatable {
    let syllable: String
    let weightPercent: Double
}

struct VocoRimeHomophoneData: Codable, Equatable {
    let format: String
    let schemaVersion: Int
    let generatedAt: String
    let readingWeightMinimumPercent: Double
    let learnedWords: [VocoRimeHomophoneLearnedWord]
    let characterReadings: [String: [VocoRimeHomophoneCharacterReading]]
    let essayFrequencies: [String: Int]
    let taiwanToOpenCC: [String: String]
    let openCCToTaiwan: [String: String]
    let deferredSourcePatterns: [String]
}

struct VocoRimeHomophoneFire: Equatable {
    let sourceText: String
    let targetText: String
    let range: NSRange
    let commits: Int

    var autoApplyPolicyFire: VocoAutoApplyPolicyFire {
        VocoAutoApplyPolicyFire(
            policyId: "rime-homophone.\(sourceText).\(targetText)",
            policyType: "rimeHomophone",
            autoApplyMode: "apply",
            sourcePattern: sourceText,
            targetText: targetText,
            sourceSlices: [VocoRimeHomophoneService.reason]
        )
    }
}

struct VocoRimeHomophoneEvaluation: Equatable {
    let inputText: String
    let outputText: String
    let applied: [VocoRimeHomophoneFire]

    var changed: Bool { inputText != outputText }
}

struct VocoRimeHomophoneDataStatus: Equatable {
    let path: String
    let isLoaded: Bool
    let learnedWordCount: Int
    let characterReadingCount: Int
    let essayEntryCount: Int
    let modificationDate: Date?
}

final class VocoRimeHomophoneService {
    static let shared = VocoRimeHomophoneService()
    static let reason = "rime-homophone"
    static let dataFileName = "rime-homophones.json"
    static let expectedFormat = "voco.rime-homophones.v1"

    private let dataURL: URL
    private let lock = NSLock()
    private var cachedModificationDate: Date?
    private var cachedEngine: VocoRimeHomophoneEngine?

    init(
        dataURL: URL = AppIdentifiers.appSupportDirectory
            .appendingPathComponent("RimeHomophones", isDirectory: true)
            .appendingPathComponent("rime-homophones.json")
    ) {
        self.dataURL = dataURL
    }

    func evaluate(
        _ text: String,
        deferredSourcePatterns: [String]? = nil
    ) -> VocoRimeHomophoneEvaluation {
        guard !text.isEmpty,
              let engine = loadedEngine()
        else {
            return VocoRimeHomophoneEvaluation(inputText: text, outputText: text, applied: [])
        }

        let deferred = deferredSourcePatterns ?? engine.deferredSourcePatterns
        let matches = engine.matches(in: text, deferredSourcePatterns: deferred)
        guard !matches.isEmpty else {
            return VocoRimeHomophoneEvaluation(inputText: text, outputText: text, applied: [])
        }

        let source = text as NSString
        var output = text
        for match in matches.sorted(by: { $0.range.location > $1.range.location }) {
            output = (output as NSString).replacingCharacters(in: match.range, with: match.targetTaiwan)
        }
        let fires = matches.map { match in
            VocoRimeHomophoneFire(
                sourceText: source.substring(with: match.range),
                targetText: match.targetTaiwan,
                range: match.range,
                commits: match.commits
            )
        }
        return VocoRimeHomophoneEvaluation(inputText: text, outputText: output, applied: fires)
    }

    func dataStatus() -> VocoRimeHomophoneDataStatus {
        let modificationDate = (try? FileManager.default.attributesOfItem(atPath: dataURL.path)[.modificationDate]) as? Date
        guard let engine = loadedEngine() else {
            return VocoRimeHomophoneDataStatus(
                path: dataURL.path,
                isLoaded: false,
                learnedWordCount: 0,
                characterReadingCount: 0,
                essayEntryCount: 0,
                modificationDate: modificationDate
            )
        }
        return VocoRimeHomophoneDataStatus(
            path: dataURL.path,
            isLoaded: true,
            learnedWordCount: engine.learnedWordCount,
            characterReadingCount: engine.characterReadingCount,
            essayEntryCount: engine.essayEntryCount,
            modificationDate: modificationDate
        )
    }

    private func loadedEngine() -> VocoRimeHomophoneEngine? {
        lock.lock()
        defer { lock.unlock() }

        guard let attributes = try? FileManager.default.attributesOfItem(atPath: dataURL.path),
              let modificationDate = attributes[.modificationDate] as? Date
        else {
            cachedEngine = nil
            cachedModificationDate = nil
            return nil
        }
        if cachedModificationDate == modificationDate {
            return cachedEngine
        }
        guard let bytes = try? Data(contentsOf: dataURL),
              let decoded = try? JSONDecoder().decode(VocoRimeHomophoneData.self, from: bytes),
              decoded.format == Self.expectedFormat,
              decoded.schemaVersion == 1
        else {
            cachedEngine = nil
            cachedModificationDate = modificationDate
            return nil
        }
        cachedEngine = VocoRimeHomophoneEngine(data: decoded)
        cachedModificationDate = modificationDate
        return cachedEngine
    }
}

private struct VocoRimeReadingKey: Hashable {
    let length: Int
    let syllables: [String]
}

private struct VocoRimeFirstReadingKey: Hashable {
    let length: Int
    let firstSyllable: String
}

private struct VocoRimeCandidateWord {
    let word: String
    let syllables: [String]
}

private struct VocoRimeCandidate {
    let start: Int
    let length: Int
    let sourceOpenCC: String
    let targetTaiwan: String
    let commits: Int
    let range: NSRange
    let order: Int
}

private struct VocoRimeHomophoneEngine {
    private let data: VocoRimeHomophoneData
    private let readings: [String: Set<String>]
    private let typedCounts: [String: Int]
    private let typedWords: Set<String>
    private let buckets: [VocoRimeReadingKey: [String: Int]]
    private let candidatesByFirst: [VocoRimeFirstReadingKey: [VocoRimeCandidateWord]]
    private let essayTotal: Double
    let deferredSourcePatterns: [String]

    var learnedWordCount: Int { data.learnedWords.count }
    var characterReadingCount: Int { data.characterReadings.count }
    var essayEntryCount: Int { data.essayFrequencies.count }

    private static let minimumLength = 3
    private static let maximumLength = 10
    private static let minimumCommits = 2
    private static let maximumSegmentationWordLength = 8
    private static let pronounGroups = [Set("你妳您"), Set("他她它祂牠")]
    private static let particles = Set("的地得底")
    private static let variants: Set<Set<Character>> = [
        Set("台臺"), Set("着著"), Set("裏裡"), Set("峯峰"), Set("綫線")
    ]

    init(data: VocoRimeHomophoneData) {
        self.data = data
        var readings: [String: Set<String>] = [:]
        for (character, values) in data.characterReadings {
            let common = values
                .filter { $0.weightPercent >= data.readingWeightMinimumPercent }
                .map(\.syllable)
            readings[character] = Set(common.isEmpty ? values.map(\.syllable) : common)
        }
        self.readings = readings

        var typedCounts: [String: Int] = [:]
        var buckets: [VocoRimeReadingKey: [String: Int]] = [:]
        var candidatesByFirst: [VocoRimeFirstReadingKey: [VocoRimeCandidateWord]] = [:]
        var seen = Set<String>()

        for entry in data.learnedWords where entry.commits > 0 {
            typedCounts[entry.word, default: 0] += entry.commits
            let wordCharacters = Array(entry.word.unicodeScalars).map(String.init)
            guard wordCharacters.allSatisfy(Self.isCJK),
                  (Self.minimumLength...Self.maximumLength).contains(wordCharacters.count),
                  entry.syllables.count == wordCharacters.count
            else { continue }

            let key = VocoRimeReadingKey(length: wordCharacters.count, syllables: entry.syllables)
            var wordCounts = buckets[key, default: [:]]
            wordCounts[entry.word, default: 0] += entry.commits
            buckets[key] = wordCounts

            let firstKey = VocoRimeFirstReadingKey(length: wordCharacters.count, firstSyllable: entry.syllables[0])
            let uniqueKey = "\(wordCharacters.count)\u{1f}\(entry.syllables.joined(separator: " "))\u{1f}\(entry.word)"
            if seen.insert(uniqueKey).inserted {
                candidatesByFirst[firstKey, default: []].append(
                    VocoRimeCandidateWord(word: entry.word, syllables: entry.syllables)
                )
            }
        }

        self.typedCounts = typedCounts
        self.typedWords = Set(typedCounts.keys.filter { $0.unicodeScalars.count >= 2 })
        self.buckets = buckets
        self.candidatesByFirst = candidatesByFirst
        self.essayTotal = Double(data.essayFrequencies.values.reduce(0, +)).nonzeroOrOne
        self.deferredSourcePatterns = data.deferredSourcePatterns
    }

    func matches(in text: String, deferredSourcePatterns: [String]) -> [VocoRimeCandidate] {
        let originalScalars = Array(text.unicodeScalars)
        let normalized = originalScalars.map { mapped(String($0), using: data.taiwanToOpenCC) }
        let textLength = normalized.count
        guard textLength >= Self.minimumLength else { return [] }

        let utf16Offsets = Self.utf16Offsets(for: originalScalars)
        let deferredRanges = ranges(for: deferredSourcePatterns, in: text)
        var candidates: [VocoRimeCandidate] = []
        var order = 0

        for start in normalized.indices {
            let firstCharacter = normalized[start]
            guard Self.isCJK(firstCharacter) else { continue }
            for firstSyllable in (readings[firstCharacter] ?? []).sorted() {
                for length in Self.minimumLength...Self.maximumLength {
                    guard start + length <= textLength else { break }
                    let sourceCharacters = Array(normalized[start..<(start + length)])
                    guard sourceCharacters.allSatisfy(Self.isCJK) else { continue }
                    let sourceOpenCC = sourceCharacters.joined()
                    let firstKey = VocoRimeFirstReadingKey(length: length, firstSyllable: firstSyllable)
                    guard let targetWords = candidatesByFirst[firstKey] else { continue }

                    for target in targetWords {
                        guard target.word != sourceOpenCC else { continue }
                        var matchesReadings = true
                        for index in 1..<length {
                            if !(readings[sourceCharacters[index]] ?? []).contains(target.syllables[index]) {
                                matchesReadings = false
                                break
                            }
                        }
                        guard matchesReadings else { continue }

                        let bucketKey = VocoRimeReadingKey(length: length, syllables: target.syllables)
                        guard let wordCounts = buckets[bucketKey],
                              let commits = wordCounts[target.word],
                              commits >= Self.minimumCommits,
                              !typedWords.contains(sourceOpenCC),
                              !wordCounts.contains(where: { $0.key != target.word && $0.value >= commits }),
                              (data.essayFrequencies[sourceOpenCC] ?? 0) == 0,
                              !excludedPair(sourceOpenCC, target.word)
                        else { continue }

                        let nsRange = NSRange(
                            location: utf16Offsets[start],
                            length: utf16Offsets[start + length] - utf16Offsets[start]
                        )
                        guard !deferredRanges.contains(where: { NSIntersectionRange($0, nsRange).length > 0 }) else {
                            continue
                        }

                        let targetTaiwan = mapped(target.word, using: data.openCCToTaiwan)
                        let sourceTaiwan = mapped(sourceOpenCC, using: data.openCCToTaiwan)
                        guard sourceTaiwan != targetTaiwan,
                              targetTaiwan != textSubstring(text, range: nsRange)
                        else { continue }
                        candidates.append(
                            VocoRimeCandidate(
                                start: start,
                                length: length,
                                sourceOpenCC: sourceOpenCC,
                                targetTaiwan: targetTaiwan,
                                commits: commits,
                                range: nsRange,
                                order: order
                            )
                        )
                        order += 1
                    }
                }
            }
        }

        let boundaries = segmentationBoundaries(for: normalized)
        candidates = candidates.filter { boundaries.contains($0.start) && boundaries.contains($0.start + $0.length) }
        candidates.sort {
            if $0.length != $1.length { return $0.length > $1.length }
            if $0.commits != $1.commits { return $0.commits > $1.commits }
            if $0.start != $1.start { return $0.start < $1.start }
            return $0.order < $1.order
        }

        var occupied = Array(repeating: false, count: textLength)
        var accepted: [VocoRimeCandidate] = []
        for candidate in candidates {
            let range = candidate.start..<(candidate.start + candidate.length)
            guard !occupied[range].contains(true) else { continue }
            for index in range { occupied[index] = true }
            accepted.append(candidate)
        }
        return accepted.sorted { $0.start < $1.start }
    }

    private func segmentationBoundaries(for text: [String]) -> Set<Int> {
        let count = text.count
        var best = Array(repeating: -1.0e18, count: count + 1)
        var lengths = Array(repeating: 0, count: count + 1)
        best[0] = 0
        let logTotal = log(essayTotal)

        for start in 0..<count where best[start] > -1.0e17 {
            for length in 1...min(Self.maximumSegmentationWordLength, count - start) {
                let word = text[start..<(start + length)].joined()
                var frequency = data.essayFrequencies[word] ?? 0
                if length == 1 {
                    frequency += 1
                } else {
                    frequency += (typedCounts[word] ?? 0) > 0 ? 1000 : 0
                    if frequency <= 0 { continue }
                }
                let score = best[start] + log(Double(frequency)) - logTotal
                if score > best[start + length] {
                    best[start + length] = score
                    lengths[start + length] = length
                }
            }
        }

        var boundaries: Set<Int> = [0, count]
        var end = count
        while end > 0 {
            let length = lengths[end] == 0 ? 1 : lengths[end]
            end -= length
            boundaries.insert(end)
        }
        return boundaries
    }

    private func excludedPair(_ source: String, _ target: String) -> Bool {
        let sourceCharacters = Array(mapped(source, using: data.openCCToTaiwan))
        let targetCharacters = Array(mapped(target, using: data.openCCToTaiwan))
        for (left, right) in zip(sourceCharacters, targetCharacters) where left != right {
            if Self.pronounGroups.contains(where: { $0.contains(left) && $0.contains(right) }) { return true }
            if Self.particles.contains(left) || Self.particles.contains(right) { return true }
            if Self.variants.contains(Set([left, right])) { return true }
        }
        return false
    }

    private func ranges(for patterns: [String], in text: String) -> [NSRange] {
        let nsText = text as NSString
        var matches: [NSRange] = []
        for pattern in Set(patterns) where !pattern.isEmpty {
            var searchStart = 0
            while searchStart < nsText.length {
                let searchRange = NSRange(location: searchStart, length: nsText.length - searchStart)
                let found = nsText.range(of: pattern, options: [], range: searchRange)
                guard found.location != NSNotFound else { break }
                matches.append(found)
                searchStart = found.location + max(found.length, 1)
            }
        }
        return matches
    }

    private static func utf16Offsets(for scalars: [Unicode.Scalar]) -> [Int] {
        var offsets = [0]
        var current = 0
        for scalar in scalars {
            current += String(scalar).utf16.count
            offsets.append(current)
        }
        return offsets
    }

    private static func isCJK(_ value: String) -> Bool {
        guard value.unicodeScalars.count == 1, let scalar = value.unicodeScalars.first else { return false }
        return (0x3400...0x4DBF).contains(scalar.value)
            || (0x4E00...0x9FFF).contains(scalar.value)
            || (0x20000...0x2A6DF).contains(scalar.value)
    }
}

private func mapped(_ value: String, using mapping: [String: String]) -> String {
    value.unicodeScalars.map { scalar in
        let original = String(scalar)
        return mapping[original] ?? original
    }.joined()
}

private func textSubstring(_ value: String, range: NSRange) -> String {
    (value as NSString).substring(with: range)
}

private extension Double {
    var nonzeroOrOne: Double { self == 0 ? 1 : self }
}
