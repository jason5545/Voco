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
    func homophones(of char: Character) -> [Character] {
        guard isLoaded else { return [] }
        guard let readings = charToPinyin[char] else { return [] }

        var result = Set<Character>()
        for reading in readings {
            let toneless = Self.stripTone(reading)
            if let chars = pinyinToChars[toneless] {
                result.formUnion(chars)
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

    // MARK: - Helpers

    /// Strip tone number from pinyin: "bian4" → "bian"
    static func stripTone(_ pinyin: String) -> String {
        if let last = pinyin.last, last.isNumber {
            return String(pinyin.dropLast())
        }
        return pinyin
    }
}
