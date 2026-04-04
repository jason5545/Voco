import Foundation
import os
import SQLite3

/// Personal correction engine that learns phonetic error patterns from the user's
/// transcription history.
///
/// Mines (original_text, enhanced_text) pairs from the database, extracts character-level
/// replacements, filters by pinyin similarity, and applies recurring patterns as corrections.
///
/// Pipeline position: Step 3, Layer 3 (after HomophoneCorrectionEngine).
final class PersonalCorrectionEngine {
    static let shared = PersonalCorrectionEngine()

    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "PersonalCorrection")
    private let db = PinyinDatabase.shared

    // MARK: - Tuning Constants

    /// Minimum times a pattern must appear to become a rule
    private let minCount = 2

    /// Minimum pinyin similarity (0.0–1.0) to be considered a phonetic pattern
    private let minPinyinSimilarity: Double = 0.5

    /// Number of new transcriptions before re-mining
    private let remineThreshold = 50

    /// UserDefaults key for last mined record count
    private let lastMinedCountKey = "PersonalCorrectionLastMinedCount"

    /// Common function words to skip
    private static let skipChars: Set<Character> = [
        "的", "了", "嗎", "呢", "吧", "啊", "哦", "喔", "嗯", "呀",
        "是", "在", "有", "和", "也", "都", "就", "不", "我", "你",
        "他", "她", "它", "們", "這", "那", "個", "把", "被", "讓",
        "會", "能", "可", "要", "得", "地", "著", "過", "到", "從",
        "與", "及", "或", "而", "但", "因", "為", "所", "以", "如",
        "跟", "更", "再", "很", "才",
    ]

    // MARK: - State

    struct PersonalRule {
        let original: String
        let corrected: String
        let count: Int
        let pinyinSimilarity: Double
    }

    private var rules: [PersonalRule] = []
    private var isLoaded = false
    private var isLoading = false
    private let loadLock = NSLock()
    private let mineQueue = DispatchQueue(label: "com.jasonchien.Voco.personalCorrection", qos: .utility)

    // MARK: - Public API

    /// Trigger background loading/refresh of rules from the transcription database.
    func loadRulesIfNeeded() {
        loadLock.lock()
        let alreadyLoaded = isLoaded
        let currentlyLoading = isLoading
        loadLock.unlock()

        guard !currentlyLoading else { return }

        if alreadyLoaded {
            // Check if re-mining is needed
            let lastCount = UserDefaults.standard.integer(forKey: lastMinedCountKey)
            let currentCount = countTranscriptions()
            guard currentCount - lastCount >= remineThreshold else { return }
            logger.info("Re-mining: \(currentCount - lastCount) new records since last mine")
        }

        loadLock.lock()
        isLoading = true
        loadLock.unlock()

        mineQueue.async { [weak self] in
            self?.mine()
            self?.loadLock.lock()
            self?.isLoading = false
            self?.loadLock.unlock()
        }
    }

    // MARK: - Mining

    private func mine() {
        guard db.isLoaded else {
            logger.warning("PinyinDatabase not loaded, skipping mine")
            return
        }

        let dbPath = Self.databasePath()
        guard FileManager.default.fileExists(atPath: dbPath) else {
            logger.warning("Database not found at \(dbPath)")
            return
        }

        var dbHandle: OpaquePointer?
        guard sqlite3_open_v2(dbPath, &dbHandle, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else {
            logger.error("Failed to open database")
            return
        }
        defer { sqlite3_close(dbHandle) }

        // Fetch transcription pairs
        let pairs = fetchPairs(db: dbHandle!)
        logger.info("Mining \(pairs.count) transcription pairs")

        // Extract replacements and count
        var counter: [String: (corrected: String, count: Int)] = [:]
        for (original, enhanced) in pairs {
            let repls = extractReplacements(original: original, enhanced: enhanced)
            for (orig, enh) in repls {
                let key = "\(orig)→\(enh)"
                if let existing = counter[key] {
                    counter[key] = (corrected: existing.corrected, count: existing.count + 1)
                } else {
                    counter[key] = (corrected: enh, count: 1)
                }
            }
        }

        // Filter by count and pinyin similarity
        var newRules: [PersonalRule] = []
        for (key, value) in counter {
            guard value.count >= minCount else { continue }

            let parts = key.split(separator: "→", maxSplits: 1)
            guard parts.count == 2 else { continue }
            let orig = String(parts[0])
            let corr = String(parts[1])

            // Skip single skip-chars
            if orig.count == 1, let ch = orig.first, Self.skipChars.contains(ch) { continue }
            if corr.count == 1, let ch = corr.first, Self.skipChars.contains(ch) { continue }

            // Skip protected words
            if CorrectionProtectionList.shared.contains(orig) { continue }

            let sim = pinyinSimilarity(orig, corr)
            guard sim >= minPinyinSimilarity else { continue }

            newRules.append(PersonalRule(
                original: orig,
                corrected: corr,
                count: value.count,
                pinyinSimilarity: sim
            ))
        }

        // Sort by count descending
        newRules.sort { $0.count > $1.count }

        loadLock.lock()
        rules = newRules
        isLoaded = true
        loadLock.unlock()

        UserDefaults.standard.set(pairs.count, forKey: lastMinedCountKey)
        logger.info("Mined \(newRules.count) personal correction rules from \(pairs.count) records")
        for rule in newRules.prefix(10) {
            logger.info("  \(rule.original) → \(rule.corrected) (\(rule.count)x, sim=\(String(format: "%.2f", rule.pinyinSimilarity)))")
        }
    }

    // MARK: - Database Access

    private static func databasePath() -> String {
        AppIdentifiers.appSupportDirectory
            .appendingPathComponent("default.store")
            .path
    }

    private func countTranscriptions() -> Int {
        let dbPath = Self.databasePath()
        var dbHandle: OpaquePointer?
        guard sqlite3_open_v2(dbPath, &dbHandle, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return 0 }
        defer { sqlite3_close(dbHandle) }

        var stmt: OpaquePointer?
        let sql = "SELECT COUNT(*) FROM ZTRANSCRIPTION WHERE ZTEXT IS NOT NULL AND ZENHANCEDTEXT IS NOT NULL AND ZTEXT != ZENHANCEDTEXT"
        guard sqlite3_prepare_v2(dbHandle, sql, -1, &stmt, nil) == SQLITE_OK else { return 0 }
        defer { sqlite3_finalize(stmt) }

        if sqlite3_step(stmt) == SQLITE_ROW {
            return Int(sqlite3_column_int(stmt, 0))
        }
        return 0
    }

    private func fetchPairs(db dbHandle: OpaquePointer) -> [(original: String, enhanced: String)] {
        var stmt: OpaquePointer?
        let sql = """
            SELECT ZTEXT, ZENHANCEDTEXT FROM ZTRANSCRIPTION
            WHERE ZTEXT IS NOT NULL AND ZENHANCEDTEXT IS NOT NULL AND ZTEXT != ZENHANCEDTEXT
            ORDER BY ZTIMESTAMP DESC
            """
        guard sqlite3_prepare_v2(dbHandle, sql, -1, &stmt, nil) == SQLITE_OK else { return [] }
        defer { sqlite3_finalize(stmt) }

        var pairs: [(String, String)] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let textPtr = sqlite3_column_text(stmt, 0),
                  let enhPtr = sqlite3_column_text(stmt, 1) else { continue }
            let text = String(cString: textPtr).trimmingCharacters(in: .whitespaces)
            let enhanced = String(cString: enhPtr).trimmingCharacters(in: .whitespaces)
            guard !text.isEmpty, !enhanced.isEmpty, text != enhanced else { continue }
            pairs.append((text, enhanced))
        }
        return pairs
    }

    // MARK: - Diff Algorithm

    /// Extract CJK replacement pairs from two strings using LCS-based diffing.
    func extractReplacements(original: String, enhanced: String) -> [(String, String)] {
        let a = Array(original)
        let b = Array(enhanced)
        let blocks = matchingBlocks(a, b)

        var replacements: [(String, String)] = []
        var ai = 0, bi = 0

        for (aPos, bPos, size) in blocks {
            if ai < aPos || bi < bPos {
                let origSeg = String(a[ai..<aPos])
                let enhSeg = String(b[bi..<bPos])
                if !origSeg.isEmpty, !enhSeg.isEmpty,
                   origSeg.contains(where: isCJK), enhSeg.contains(where: isCJK) {
                    replacements.append((origSeg, enhSeg))
                }
            }
            ai = aPos + size
            bi = bPos + size
        }

        // Trailing segment
        if ai < a.count || bi < b.count {
            let origSeg = String(a[ai...])
            let enhSeg = String(b[bi...])
            if !origSeg.isEmpty, !enhSeg.isEmpty,
               origSeg.contains(where: isCJK), enhSeg.contains(where: isCJK) {
                replacements.append((origSeg, enhSeg))
            }
        }

        return replacements
    }

    /// Find matching blocks between two character arrays using LCS (DP).
    /// Returns sorted list of (aPos, bPos, size) triples + sentinel.
    private func matchingBlocks(_ a: [Character], _ b: [Character]) -> [(Int, Int, Int)] {
        let m = a.count, n = b.count
        guard m > 0, n > 0 else { return [(m, n, 0)] }

        // DP table for LCS length
        var dp = [[Int]](repeating: [Int](repeating: 0, count: n + 1), count: m + 1)
        for i in 1...m {
            for j in 1...n {
                if a[i - 1] == b[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }

        // Backtrack to find LCS positions
        var lcsA: [Int] = []
        var lcsB: [Int] = []
        var i = m, j = n
        while i > 0 && j > 0 {
            if a[i - 1] == b[j - 1] {
                lcsA.append(i - 1)
                lcsB.append(j - 1)
                i -= 1; j -= 1
            } else if dp[i - 1][j] > dp[i][j - 1] {
                i -= 1
            } else {
                j -= 1
            }
        }
        lcsA.reverse()
        lcsB.reverse()

        // Group consecutive LCS matches into blocks
        var blocks: [(Int, Int, Int)] = []
        var k = 0
        while k < lcsA.count {
            let startA = lcsA[k], startB = lcsB[k]
            var size = 1
            while k + size < lcsA.count
                    && lcsA[k + size] == startA + size
                    && lcsB[k + size] == startB + size {
                size += 1
            }
            blocks.append((startA, startB, size))
            k += size
        }

        // Sentinel
        blocks.append((m, n, 0))
        return blocks
    }

    // MARK: - Pinyin Similarity

    /// Compute pinyin similarity between two CJK strings (0.0–1.0).
    func pinyinSimilarity(_ s1: String, _ s2: String) -> Double {
        let py1 = pinyinString(s1)
        let py2 = pinyinString(s2)
        guard !py1.isEmpty, !py2.isEmpty else { return 0 }
        return sequenceRatio(Array(py1), Array(py2))
    }

    /// Get toneless pinyin string for CJK characters.
    private func pinyinString(_ text: String) -> String {
        var parts: [String] = []
        for ch in text where isCJK(ch) {
            if let pinyins = db.charToPinyin[ch], let first = pinyins.first {
                parts.append(PinyinDatabase.stripTone(first))
            }
        }
        return parts.joined(separator: " ")
    }

    /// SequenceMatcher.ratio() equivalent using LCS.
    private func sequenceRatio(_ a: [Character], _ b: [Character]) -> Double {
        let m = a.count, n = b.count
        guard m > 0 || n > 0 else { return 1.0 }

        // LCS length via DP
        var dp = [[Int]](repeating: [Int](repeating: 0, count: n + 1), count: m + 1)
        for i in 1...m {
            for j in 1...n {
                if a[i - 1] == b[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }

        let lcsLen = dp[m][n]
        return Double(2 * lcsLen) / Double(m + n)
    }

    // MARK: - Correction (Rule Application)

    /// Apply with CJK boundary protection (same pattern as PinyinCorrector).
    private func applyRule(_ text: String, rule: PersonalRule) -> String {
        let wrongChars = Array(rule.original)
        let needsBoundaryCheck = wrongChars.count <= 2
            && db.isLoaded
            && wrongChars.allSatisfy(isCJK)

        guard needsBoundaryCheck else {
            return text.replacingOccurrences(of: rule.original, with: rule.corrected)
        }

        var result = text
        var searchEnd = result.endIndex
        var ranges: [Range<String.Index>] = []
        while let range = result.range(of: rule.original, range: result.startIndex..<searchEnd) {
            ranges.append(range)
            searchEnd = range.lowerBound
        }

        for range in ranges {
            let matchStart = result.distance(from: result.startIndex, to: range.lowerBound)
            let matchEnd = matchStart + wrongChars.count
            let currentChars = Array(result)

            // Right boundary check
            if matchEnd < currentChars.count {
                let nextChar = currentChars[matchEnd]
                if isCJK(nextChar) {
                    let rightPair = String(wrongChars.last!) + String(nextChar)
                    if db.frequency(of: rightPair) > 0 { continue }
                }
            }

            // Left boundary check
            if matchStart > 0 {
                let prevChar = currentChars[matchStart - 1]
                if isCJK(prevChar) {
                    let leftPair = String(prevChar) + String(wrongChars.first!)
                    if db.frequency(of: leftPair) > 0 { continue }
                }
            }

            result = result.replacingCharacters(in: range, with: rule.corrected)
        }

        return result
    }

    // MARK: - Helpers

    private func isCJK(_ char: Character) -> Bool {
        guard let scalar = char.unicodeScalars.first else { return false }
        let v = scalar.value
        return (0x4E00...0x9FFF).contains(v) || (0x3400...0x4DBF).contains(v)
    }
}

// MARK: - CorrectionEngine Conformance

extension PersonalCorrectionEngine: CorrectionEngine {
    var name: String { "PersonalCorrection" }
    var logPrefix: String { "[personal]" }

    func correct(_ text: String) -> CorrectionResult {
        // Trigger background mining on first use or when new data available
        loadRulesIfNeeded()

        loadLock.lock()
        let currentRules = rules
        loadLock.unlock()

        guard !currentRules.isEmpty else {
            return CorrectionResult(text: text, corrections: [])
        }

        var result = text
        var corrections: [CorrectionResult.Correction] = []

        for rule in currentRules {
            guard result.contains(rule.original) else { continue }
            if CorrectionProtectionList.shared.containsSubstring(in: rule.original) { continue }

            let replaced = applyRule(result, rule: rule)
            if replaced != result {
                corrections.append(.init(
                    original: rule.original,
                    corrected: rule.corrected,
                    score: Double(rule.count) * rule.pinyinSimilarity
                ))
                result = replaced
            }
        }

        return CorrectionResult(text: result, corrections: corrections)
    }
}
