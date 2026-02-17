import Foundation

/// Manages recent transcription context for LLM disambiguation
/// Reference: xvoice/src/context.py
class TranscriptionContextMemory {
    private struct Entry {
        let text: String
        let timestamp: Date
    }

    private var entries: [Entry] = []
    private let maxEntries: Int
    private let ttl: TimeInterval // seconds

    init(maxEntries: Int = 10, ttl: TimeInterval = 300) { // 5 minutes default
        self.maxEntries = maxEntries
        self.ttl = ttl
    }

    /// Add a new transcription to context memory
    func add(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        cleanupExpired()
        entries.append(Entry(text: trimmed, timestamp: Date()))

        // Remove oldest if over limit
        while entries.count > maxEntries {
            entries.removeFirst()
        }
    }

    /// Get recent transcriptions (oldest to newest)
    func getRecent(count: Int = 5) -> [String] {
        cleanupExpired()
        let slice = entries.suffix(count)
        return slice.map { $0.text }
    }

    /// Clear all context
    func clear() {
        entries.removeAll()
    }

    /// Current number of entries (after cleanup)
    var count: Int {
        cleanupExpired()
        return entries.count
    }

    var isEmpty: Bool {
        return count == 0
    }

    // MARK: - Private

    private func cleanupExpired() {
        guard ttl > 0 else { return }
        let now = Date()
        entries.removeAll { now.timeIntervalSince($0.timestamp) >= ttl }
    }
}
