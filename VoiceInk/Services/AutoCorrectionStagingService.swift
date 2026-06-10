// AutoCorrectionStagingService.swift
// Stages edit mode corrections for explicit user approval as WordReplacement.
// Uses LCS diff + pinyin similarity to detect ASR errors from before/after text.

import Foundation
import SwiftData
import os

@MainActor
final class AutoCorrectionStagingService {

    static let shared = AutoCorrectionStagingService()

    private let logger = Logger(subsystem: "com.jasonchien.Voco", category: "AutoCorrectionStaging")

    // MARK: - Diff-Based Substitution Extraction

    /// Extract a single ASR-error substitution pair from edit mode before/after text.
    /// Returns nil if the diff doesn't look like an ASR error (rewrite, multi-change, etc.).
    func extractSubstitution(original: String, edited: String) -> WordSubstitution? {
        let a = Array(original)
        let b = Array(edited)

        guard !a.isEmpty, !b.isEmpty else { return nil }

        let blocks = matchingBlocks(a, b)

        // Overall similarity check: if texts differ by more than 50%, it's a rewrite
        let commonCount = blocks.dropLast().reduce(0) { $0 + $1.2 }
        let maxLen = max(a.count, b.count)
        guard maxLen > 0, Double(commonCount) / Double(maxLen) >= 0.5 else { return nil }

        // Extract changed segments from the diff
        var segments: [(String, String)] = []
        var ai = 0, bi = 0

        for (aPos, bPos, size) in blocks {
            if ai < aPos || bi < bPos {
                let origSeg = String(a[ai..<aPos]).trimmingCharacters(in: .whitespacesAndNewlines)
                let editSeg = String(b[bi..<bPos]).trimmingCharacters(in: .whitespacesAndNewlines)
                if !origSeg.isEmpty, !editSeg.isEmpty {
                    segments.append((origSeg, editSeg))
                }
            }
            ai = aPos + size
            bi = bPos + size
        }

        guard segments.count == 1 else { return nil }

        let (origSeg, editSeg) = segments[0]

        guard origSeg.count <= 20, editSeg.count <= 20 else { return nil }
        guard abs(origSeg.count - editSeg.count) <= 1 else { return nil }

        // Skip punctuation-only differences
        let origClean = origSeg.filter { !$0.isPunctuation && !$0.isWhitespace }
        let editClean = editSeg.filter { !$0.isPunctuation && !$0.isWhitespace }
        guard origClean != editClean else { return nil }

        // Phonetic similarity check
        let hasCJK = origSeg.contains(where: \.isCJK) || editSeg.contains(where: \.isCJK)
        if hasCJK {
            let similarity = PersonalCorrectionEngine.shared.pinyinSimilarity(origSeg, editSeg)
            guard similarity >= 0.5 else { return nil }
        } else {
            let distance = levenshteinDistance(origClean.lowercased(), editClean.lowercased())
            let similarity = 1.0 - Double(distance) / Double(max(origClean.count, editClean.count))
            guard similarity >= 0.5 else { return nil }
        }

        logger.info("Extracted ASR correction candidate: \(origSeg, privacy: .private) → \(editSeg, privacy: .private)")
        return WordSubstitution(original: origSeg, replacement: editSeg)
    }

    // MARK: - Staging

    /// Stage a correction pair: insert or increment hitCount, then ask for explicit approval.
    func stageCorrection(
        _ sub: WordSubstitution,
        in modelContext: ModelContext,
        source: String = WordReplacement.sourceEditMode
    ) {
        let coreOriginal = sub.original
        let coreReplacement = sub.replacement
        guard !CorrectionProtectionList.shared.containsProtectedTerm(in: coreOriginal) else {
            logger.info("Skipped staged correction touching protected source term: \(coreOriginal, privacy: .private)")
            return
        }

        let descriptor = FetchDescriptor<WordReplacement>(
            predicate: #Predicate<WordReplacement> {
                $0.originalText == coreOriginal && $0.replacementText == coreReplacement
            }
        )

        if let existing = (try? modelContext.fetch(descriptor))?.first {
            // Already user-confirmed — no need to stage
            if existing.source == WordReplacement.sourceUser {
                return
            }

            existing.hitCount += 1
            existing.lastSeenDate = Date()
            notifyLearningProgressIfNeeded(existing)
        } else {
            let entry = WordReplacement(
                originalText: coreOriginal,
                replacementText: coreReplacement,
                isEnabled: false,
                source: source
            )
            modelContext.insert(entry)
            logger.info("New staged correction: \(coreOriginal, privacy: .private) → \(coreReplacement, privacy: .private)")
        }

        try? modelContext.save()
    }

    // MARK: - Learning Review

    private func notifyLearningProgressIfNeeded(_ entry: WordReplacement) {
        guard !entry.isEnabled,
              entry.source == WordReplacement.sourceEditMode || entry.source == WordReplacement.sourceCorrectionFeedback
        else {
            return
        }

        switch entry.hitCount {
        case 2:
            NotificationManager.shared.showNotification(
                title: "「\(entry.originalText)」→「\(entry.replacementText)」已出現 2 次，確認後才會加入辭典",
                type: .info,
                duration: 4.0
            )
        case WordReplacement.learningReviewThreshold:
            NotificationManager.shared.showNotification(
                title: "「\(entry.originalText)」→「\(entry.replacementText)」已累積 3 次，請確認後加入辭典",
                type: .info,
                duration: 4.0
            )
        default:
            break
        }
    }

    // MARK: - LCS

    private func matchingBlocks(_ a: [Character], _ b: [Character]) -> [(Int, Int, Int)] {
        let m = a.count, n = b.count
        guard m > 0, n > 0 else { return [(m, n, 0)] }

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

        var lcsA: [Int] = [], lcsB: [Int] = []
        var i = m, j = n
        while i > 0 && j > 0 {
            if a[i - 1] == b[j - 1] {
                lcsA.append(i - 1); lcsB.append(j - 1)
                i -= 1; j -= 1
            } else if dp[i - 1][j] > dp[i][j - 1] {
                i -= 1
            } else {
                j -= 1
            }
        }
        lcsA.reverse(); lcsB.reverse()

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
        blocks.append((m, n, 0))
        return blocks
    }

    // MARK: - String Helpers

    private func levenshteinDistance(_ s1: String, _ s2: String) -> Int {
        let a = Array(s1), b = Array(s2)
        let m = a.count, n = b.count
        guard m > 0 else { return n }
        guard n > 0 else { return m }
        var prev = Array(0...n)
        var curr = [Int](repeating: 0, count: n + 1)
        for i in 1...m {
            curr[0] = i
            for j in 1...n {
                if a[i - 1] == b[j - 1] {
                    curr[j] = prev[j - 1]
                } else {
                    curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
                }
            }
            swap(&prev, &curr)
        }
        return prev[n]
    }
}
