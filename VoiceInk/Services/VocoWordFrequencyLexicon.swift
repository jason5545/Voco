import Foundation
import os

/// Word-frequency table (`word_freq.tsv`) loaded synchronously on first use.
///
/// The auto-apply runtime needs a deterministic lexicon: `PinyinDatabase` loads the
/// same file on a background queue and answers 0 until it finishes, which would make
/// `runtime.single-prefix-restart-collapse` fire differently on a cold start. This
/// class blocks the first caller until the table is in memory, then serves lookups
/// lock-free. Mac, Android (`assets/chinese_correction/word_freq.tsv`), and the
/// Python control CLI all read the identical file.
final class VocoWordFrequencyLexicon: @unchecked Sendable {
    static let shared = VocoWordFrequencyLexicon()

    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "VocoWordFrequencyLexicon")
    private let lock = NSLock()
    private var table: [String: Int]?
    private let loader: () -> [String: Int]

    init(loader: @escaping () -> [String: Int] = VocoWordFrequencyLexicon.loadBundledTable) {
        self.loader = loader
    }

    /// Frequency of `word`, or 0 when it is not in the lexicon.
    func frequency(of word: String) -> Int {
        loadedTable()[word] ?? 0
    }

    private func loadedTable() -> [String: Int] {
        lock.lock()
        defer { lock.unlock() }
        if let table { return table }
        let start = CFAbsoluteTimeGetCurrent()
        let loaded = loader()
        table = loaded
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        logger.info("word frequency lexicon loaded: \(loaded.count) words in \(String(format: "%.2f", elapsed))s")
        return loaded
    }

    static func loadBundledTable() -> [String: Int] {
        guard let url = Bundle.main.url(forResource: "word_freq", withExtension: "tsv") else {
            return [:]
        }
        return (try? loadTable(at: url)) ?? [:]
    }

    static func loadTable(at url: URL) throws -> [String: Int] {
        let content = try String(contentsOf: url, encoding: .utf8)
        var result: [String: Int] = [:]
        result.reserveCapacity(400_000)
        for line in content.split(separator: "\n") {
            let parts = line.split(separator: "\t", maxSplits: 1)
            if parts.count == 2, let frequency = Int(parts[1].trimmingCharacters(in: .whitespaces)) {
                result[String(parts[0])] = frequency
            }
        }
        return result
    }
}
