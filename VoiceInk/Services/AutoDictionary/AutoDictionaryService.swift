import Foundation
import SwiftData
import os

/// Detects user corrections to Whisper transcriptions and automatically
/// promotes frequently corrected words into the WordReplacement dictionary.
@MainActor
final class AutoDictionaryService {
    static let shared = AutoDictionaryService()

    private let logger = Logger(subsystem: "com.jasonchien.voco", category: "AutoDictionary")

    /// Minimum number of identical corrections before auto-promoting.
    private let promotionThreshold = 2

    // MARK: - Snapshot State

    /// The text that was pasted into the text field.
    private var lastInsertedText: String?
    /// The text field content *before* the paste.
    private var lastPreFieldContent: String?
    /// The bundle ID of the app at the time of paste.
    private var lastAppBundleID: String?

    private init() {}

    // MARK: - Public API

    /// Call this **before** `CursorPaster.pasteAtCursor` to snapshot the current text field state.
    func capturePrePasteSnapshot(insertedText: String) {
        guard UserDefaults.standard.bool(forKey: "AutoDictionaryEnabled") else { return }

        let fieldContent = TextFieldReader.readFocusedTextFieldValue()
        let bundleID = TextFieldReader.frontmostAppBundleID()

        lastInsertedText = insertedText
        lastPreFieldContent = fieldContent
        lastAppBundleID = bundleID

        logger.debug("Snapshot captured – app=\(bundleID ?? "nil", privacy: .public), fieldLen=\(fieldContent?.count ?? -1), insertedLen=\(insertedText.count)")
    }

    /// Call this when the user starts a **new** recording to check whether the previous
    /// paste was manually corrected.
    func checkForUserCorrections(modelContext: ModelContext) {
        guard UserDefaults.standard.bool(forKey: "AutoDictionaryEnabled") else { return }

        defer { clearSnapshot() }

        guard let insertedText = lastInsertedText,
              let preFieldContent = lastPreFieldContent,
              let savedBundleID = lastAppBundleID else {
            logger.debug("No snapshot to compare")
            return
        }

        // Only compare within the same app
        guard let currentBundleID = TextFieldReader.frontmostAppBundleID(),
              currentBundleID == savedBundleID else {
            logger.debug("App changed since paste, skipping comparison")
            return
        }

        guard let currentFieldContent = TextFieldReader.readFocusedTextFieldValue() else {
            logger.debug("Cannot read current field content")
            return
        }

        let substitutions = extractSubstitutions(
            preFieldContent: preFieldContent,
            insertedText: insertedText,
            currentFieldContent: currentFieldContent
        )

        for sub in substitutions {
            recordCorrection(original: sub.original, corrected: sub.corrected, modelContext: modelContext)
        }
    }

    // MARK: - Diff Logic

    struct Substitution {
        let original: String
        let corrected: String
    }

    /// Uses prefix/suffix matching to find substitutions the user made inside the inserted region.
    func extractSubstitutions(preFieldContent: String, insertedText: String, currentFieldContent: String) -> [Substitution] {
        // Step 1: Find the inserted region in currentFieldContent by matching the prefix/suffix with preFieldContent
        let commonPrefixLen = commonPrefixLength(preFieldContent, currentFieldContent)
        let commonSuffixLen = commonSuffixLength(
            preFieldContent, currentFieldContent,
            maxA: preFieldContent.count - commonPrefixLen,
            maxB: currentFieldContent.count - commonPrefixLen
        )

        let currentStart = currentFieldContent.index(currentFieldContent.startIndex, offsetBy: commonPrefixLen)
        let currentEnd = currentFieldContent.index(currentFieldContent.endIndex, offsetBy: -commonSuffixLen)
        guard currentStart <= currentEnd else { return [] }
        let currentInsertedRegion = String(currentFieldContent[currentStart..<currentEnd])

        // If the inserted region matches the original text exactly, no correction was made
        if currentInsertedRegion == insertedText { return [] }

        // Step 2: Find differences within the inserted region
        let innerPrefixLen = commonPrefixLength(insertedText, currentInsertedRegion)
        let innerSuffixLen = commonSuffixLength(
            insertedText, currentInsertedRegion,
            maxA: insertedText.count - innerPrefixLen,
            maxB: currentInsertedRegion.count - innerPrefixLen
        )

        let origStart = insertedText.index(insertedText.startIndex, offsetBy: innerPrefixLen)
        let origEnd = insertedText.index(insertedText.endIndex, offsetBy: -innerSuffixLen)
        guard origStart <= origEnd else { return [] }
        let originalSegment = String(insertedText[origStart..<origEnd])

        let corrStart = currentInsertedRegion.index(currentInsertedRegion.startIndex, offsetBy: innerPrefixLen)
        let corrEnd = currentInsertedRegion.index(currentInsertedRegion.endIndex, offsetBy: -innerSuffixLen)
        guard corrStart <= corrEnd else { return [] }
        let correctedSegment = String(currentInsertedRegion[corrStart..<corrEnd])

        // Only record substitutions (both sides non-empty), skip pure insertions/deletions
        guard !originalSegment.isEmpty, !correctedSegment.isEmpty else { return [] }

        // Length guard: too short or too long might be noise
        guard originalSegment.count >= 1, originalSegment.count <= 50,
              correctedSegment.count >= 1, correctedSegment.count <= 50 else { return [] }

        // Ignore if the only change is whitespace
        let trimmedOrig = originalSegment.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedCorr = correctedSegment.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmedOrig != trimmedCorr else { return [] }

        return [Substitution(original: trimmedOrig, corrected: trimmedCorr)]
    }

    // MARK: - Candidate Management

    private func recordCorrection(original: String, corrected: String, modelContext: ModelContext) {
        logger.notice("Correction detected: \"\(original, privacy: .public)\" → \"\(corrected, privacy: .public)\"")

        // Check if this WordReplacement already exists
        let existingReplacements = (try? modelContext.fetch(
            FetchDescriptor<WordReplacement>(predicate: #Predicate {
                $0.originalText == original && $0.replacementText == corrected
            })
        )) ?? []

        if !existingReplacements.isEmpty {
            logger.debug("WordReplacement already exists for \"\(original, privacy: .public)\" → \"\(corrected, privacy: .public)\", skipping")
            return
        }

        // Find or create a CorrectionCandidate
        let candidates = (try? modelContext.fetch(
            FetchDescriptor<CorrectionCandidate>(predicate: #Predicate {
                $0.originalText == original && $0.correctedText == corrected && $0.isPromoted == false
            })
        )) ?? []

        if let existing = candidates.first {
            existing.count += 1
            existing.lastSeen = Date()
            logger.notice("Candidate count updated: \"\(original, privacy: .public)\" → \"\(corrected, privacy: .public)\" (count=\(existing.count))")

            if existing.count >= promotionThreshold {
                promoteCandidate(existing, modelContext: modelContext)
            }
        } else {
            let candidate = CorrectionCandidate(originalText: original, correctedText: corrected)
            modelContext.insert(candidate)
            logger.notice("New candidate created: \"\(original, privacy: .public)\" → \"\(corrected, privacy: .public)\"")
        }

        try? modelContext.save()
    }

    private func promoteCandidate(_ candidate: CorrectionCandidate, modelContext: ModelContext) {
        candidate.isPromoted = true

        let replacement = WordReplacement(
            originalText: candidate.originalText,
            replacementText: candidate.correctedText
        )
        modelContext.insert(replacement)
        try? modelContext.save()

        logger.notice("Promoted to WordReplacement: \"\(candidate.originalText, privacy: .public)\" → \"\(candidate.correctedText, privacy: .public)\"")

        // Show notification with undo action
        let originalText = candidate.originalText
        let correctedText = candidate.correctedText
        let replacementID = replacement.id

        let notificationTitle = "\(String(localized: "Auto-learned")): \(originalText) → \(correctedText)"
        NotificationManager.shared.showNotificationWithAction(
            title: notificationTitle,
            type: .success,
            actionTitle: String(localized: "Undo"),
            duration: 5.0
        ) {
            // Undo: delete the WordReplacement
            Task { @MainActor in
                let fetchDescriptor = FetchDescriptor<WordReplacement>(predicate: #Predicate {
                    $0.id == replacementID
                })
                if let toDelete = try? modelContext.fetch(fetchDescriptor).first {
                    modelContext.delete(toDelete)
                    try? modelContext.save()
                    NotificationManager.shared.showNotification(
                        title: String(localized: "Undone"),
                        type: .info,
                        duration: 2.0
                    )
                }
            }
        }
    }

    // MARK: - Helpers

    private func clearSnapshot() {
        lastInsertedText = nil
        lastPreFieldContent = nil
        lastAppBundleID = nil
    }

    /// Returns the number of characters in the common prefix of two strings.
    private func commonPrefixLength(_ a: String, _ b: String) -> Int {
        let aChars = Array(a)
        let bChars = Array(b)
        let minLen = min(aChars.count, bChars.count)
        var i = 0
        while i < minLen && aChars[i] == bChars[i] {
            i += 1
        }
        return i
    }

    /// Returns the number of characters in the common suffix of two strings,
    /// limited to `maxA` and `maxB` characters from the end to avoid overlapping with the prefix.
    private func commonSuffixLength(_ a: String, _ b: String, maxA: Int, maxB: Int) -> Int {
        let aChars = Array(a)
        let bChars = Array(b)
        let limit = min(maxA, maxB)
        var i = 0
        while i < limit && aChars[aChars.count - 1 - i] == bChars[bChars.count - 1 - i] {
            i += 1
        }
        return i
    }
}
