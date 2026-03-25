import Foundation

/// Shared protocol for data-driven correction engines.
///
/// Conforming engines are used in a pipeline loop by `ChinesePostProcessingService`.
/// Each engine takes a plain `String`, returns a `CorrectionResult` with the
/// corrected text and a list of individual corrections applied.
protocol CorrectionEngine {
    /// Human-readable name shown in pipeline trace (e.g. "HomophoneCorrection").
    var name: String { get }

    /// Log prefix for debug output (e.g. "[data]", "[nasal]", "[expand]").
    var logPrefix: String { get }

    /// Apply corrections to the given text.
    func correct(_ text: String) -> CorrectionResult
}

/// Result of a single correction engine pass.
struct CorrectionResult {
    let text: String
    let corrections: [Correction]

    struct Correction {
        let original: String
        let corrected: String
        let score: Double
    }
}

// MARK: - Protection List

/// Thread-safe set of words that should never be modified by correction engines.
/// Persisted via UserDefaults.
final class CorrectionProtectionList {
    static let shared = CorrectionProtectionList()

    private let key = "CorrectionProtectionWords"
    private let queue = DispatchQueue(label: "com.jasonchien.Voco.protectionList", attributes: .concurrent)
    private var words: Set<String>

    private init() {
        let stored = UserDefaults.standard.stringArray(forKey: key) ?? []
        self.words = Set(stored)
    }

    /// Check if a word (or any substring of it) is protected.
    func contains(_ word: String) -> Bool {
        queue.sync { words.contains(word) }
    }

    /// Check if any protected word appears as a substring in the given text.
    func containsSubstring(in text: String) -> Bool {
        queue.sync {
            for w in words {
                if text.contains(w) { return true }
            }
            return false
        }
    }

    func add(_ word: String) {
        queue.async(flags: .barrier) {
            self.words.insert(word)
            self.save()
        }
    }

    func remove(_ word: String) {
        queue.async(flags: .barrier) {
            self.words.remove(word)
            self.save()
        }
    }

    func allWords() -> [String] {
        queue.sync { Array(words).sorted() }
    }

    private func save() {
        UserDefaults.standard.set(Array(words), forKey: key)
    }
}
