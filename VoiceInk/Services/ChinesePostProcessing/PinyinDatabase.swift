import Foundation
import os

/// Singleton database for pinyin lookups and word frequency data.
/// Loads char_pinyin.json, pinyin_chars.json, and word_freq.tsv from the app bundle
/// on a background thread at app launch.
final class PinyinDatabase: @unchecked Sendable {
    static let shared = PinyinDatabase()

    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "PinyinDatabase")

    // MARK: - Data

    /// char → [pinyin with tone], e.g. "辨" → ["bian4"]
    private(set) var charToPinyin: [Character: [String]] = [:]

    /// toneless pinyin → [chars], e.g. "bian" → ["辨", "辯", "變", ...]
    private(set) var pinyinToChars: [String: [Character]] = [:]

    /// word → frequency, e.g. "程式" → 80000
    private(set) var wordFreq: [String: Int] = [:]

    /// bigram → frequency, e.g. "式碼" → 60000
    private(set) var bigramFreq: [String: Int] = [:]

    /// Whether all data has finished loading
    private(set) var isLoaded = false

    // MARK: - Init

    private init() {
        loadInBackground()
    }

    // MARK: - Public API

    /// Get all homophones of a character (same toneless pinyin).
    /// Returns empty array if char not found or data not loaded.
    ///
    /// To avoid false matches from archaic/rare readings in the pinyin database,
    /// candidates are only accepted if their **primary** (first-listed) reading
    /// matches the searched pinyin.  This filters out characters like 探 whose
    /// primary reading is tàn but that carry an obsolete xián reading in the data.
    func homophones(of char: Character) -> [Character] {
        guard isLoaded else { return [] }
        guard let readings = charToPinyin[char],
              let primaryReading = readings.first else { return [] }

        // Use primary reading only — secondary/archaic readings (e.g. 掉=nuo2)
        // cause false positives by matching unrelated characters (e.g. 諾=nuo4).
        let toneless = Self.stripTone(primaryReading)
        var result = Set<Character>()
        if let chars = pinyinToChars[toneless] {
            for candidate in chars {
                // Only accept candidates whose primary reading matches
                if let candidateReadings = charToPinyin[candidate],
                   let primary = candidateReadings.first,
                   Self.stripTone(primary) == toneless {
                    result.insert(candidate)
                }
            }
        }
        result.remove(char) // exclude self
        return Array(result)
    }

    /// Get frequency of a word. Returns 0 if not found.
    func frequency(of word: String) -> Int {
        wordFreq[word] ?? 0
    }

    /// Get frequency of a character bigram. Returns 0 if not found.
    func bigramFrequency(of bigram: String) -> Int {
        bigramFreq[bigram] ?? 0
    }

    /// Get toneless pinyin(s) for a character.
    func tonelessPinyin(of char: Character) -> [String] {
        guard let readings = charToPinyin[char] else { return [] }
        return Array(Set(readings.map { Self.stripTone($0) }))
    }

    /// Get all characters that share the given toneless pinyin.
    func characters(forPinyin pinyin: String) -> [Character] {
        pinyinToChars[pinyin] ?? []
    }

    // MARK: - Loading

    private func loadInBackground() {
        DispatchQueue.global(qos: .utility).async { [self] in
            let start = CFAbsoluteTimeGetCurrent()

            loadCharPinyin()
            loadPinyinChars()
            loadWordFreq()
            loadBigramFreq()

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            isLoaded = true
            logger.info("PinyinDatabase loaded in \(String(format: "%.2f", elapsed))s — chars: \(self.charToPinyin.count), pinyinGroups: \(self.pinyinToChars.count), words: \(self.wordFreq.count), bigrams: \(self.bigramFreq.count)")
        }
    }

    private func loadCharPinyin() {
        guard let url = Bundle.main.url(forResource: "char_pinyin", withExtension: "json") else {
            logger.error("char_pinyin.json not found in bundle")
            return
        }
        do {
            let data = try Data(contentsOf: url)
            let dict = try JSONDecoder().decode([String: [String]].self, from: data)
            var result: [Character: [String]] = [:]
            result.reserveCapacity(dict.count)
            for (key, value) in dict {
                if let char = key.first, key.count == 1 {
                    result[char] = value
                }
            }
            charToPinyin = result
        } catch {
            logger.error("Failed to load char_pinyin.json: \(error)")
        }
    }

    private func loadPinyinChars() {
        guard let url = Bundle.main.url(forResource: "pinyin_chars", withExtension: "json") else {
            logger.error("pinyin_chars.json not found in bundle")
            return
        }
        do {
            let data = try Data(contentsOf: url)
            let dict = try JSONDecoder().decode([String: [String]].self, from: data)
            var result: [String: [Character]] = [:]
            result.reserveCapacity(dict.count)
            for (pinyin, chars) in dict {
                result[pinyin] = chars.compactMap(\.first)
            }
            pinyinToChars = result
        } catch {
            logger.error("Failed to load pinyin_chars.json: \(error)")
        }
    }

    private func loadWordFreq() {
        guard let url = Bundle.main.url(forResource: "word_freq", withExtension: "tsv") else {
            logger.error("word_freq.tsv not found in bundle")
            return
        }
        do {
            let content = try String(contentsOf: url, encoding: .utf8)
            var result: [String: Int] = [:]
            result.reserveCapacity(400_000)
            for line in content.split(separator: "\n") {
                let parts = line.split(separator: "\t", maxSplits: 1)
                if parts.count == 2, let freq = Int(parts[1]) {
                    result[String(parts[0])] = freq
                }
            }
            wordFreq = result
        } catch {
            logger.error("Failed to load word_freq.tsv: \(error)")
        }
    }

    private func loadBigramFreq() {
        guard let url = Bundle.main.url(forResource: "bigram_freq", withExtension: "tsv") else {
            logger.warning("bigram_freq.tsv not found in bundle — bigram scoring disabled")
            return
        }
        do {
            let content = try String(contentsOf: url, encoding: .utf8)
            var result: [String: Int] = [:]
            result.reserveCapacity(300_000)
            for line in content.split(separator: "\n") {
                let parts = line.split(separator: "\t", maxSplits: 1)
                if parts.count == 2, let freq = Int(parts[1]) {
                    result[String(parts[0])] = freq
                }
            }
            bigramFreq = result
        } catch {
            logger.error("Failed to load bigram_freq.tsv: \(error)")
        }
    }

    // MARK: - Nasal Variants

    /// Swap nasal ending: -n ↔ -ng.
    /// "yin" → "ying", "ying" → "yin", "ba" → nil
    static func swapNasal(_ pinyin: String) -> String? {
        if pinyin.hasSuffix("ng") {
            return String(pinyin.dropLast()) // "ying" → "yin"
        } else if pinyin.hasSuffix("n") {
            return pinyin + "g"              // "yin" → "ying"
        }
        return nil
    }

    /// Get nasal-variant characters (-n/-ng swap) for a given character.
    /// Similar to `homophones(of:)` but searches the swapped nasal pinyin.
    func nasalVariants(of char: Character) -> [Character] {
        guard isLoaded else { return [] }
        guard let readings = charToPinyin[char] else { return [] }

        var result = Set<Character>()
        for reading in readings {
            let toneless = Self.stripTone(reading)
            guard let swapped = Self.swapNasal(toneless) else { continue }
            guard let chars = pinyinToChars[swapped] else { continue }
            for candidate in chars {
                // Only accept candidates whose primary reading matches the swapped pinyin
                if let candidateReadings = charToPinyin[candidate],
                   let primary = candidateReadings.first,
                   Self.stripTone(primary) == swapped {
                    result.insert(candidate)
                }
            }
        }
        result.remove(char)
        return Array(result)
    }

    // MARK: - Helpers

    /// Strip tone number from pinyin: "bian4" → "bian"
    static func stripTone(_ pinyin: String) -> String {
        if let last = pinyin.last, last.isNumber {
            return String(pinyin.dropLast())
        }
        return pinyin
    }
}
