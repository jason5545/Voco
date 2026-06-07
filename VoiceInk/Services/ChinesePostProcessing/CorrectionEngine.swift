import Foundation
import NaturalLanguage

// MARK: - Shared Utilities

/// Common function words that correction engines should never replace as single-character rules.
let correctionSkipChars: Set<Character> = [
    "的", "了", "嗎", "呢", "吧", "啊", "哦", "喔", "嗯", "呀",
    "是", "在", "有", "和", "也", "都", "就", "不", "我", "你",
    "他", "她", "它", "們", "這", "那", "個", "把", "被", "讓",
    "會", "能", "可", "要", "得", "地", "著", "過", "到", "從",
    "與", "及", "或", "而", "但", "因", "為", "所", "以", "如",
    "跟", "更", "再", "很", "才",
]

extension Character {
    /// Whether this character is a CJK ideograph (Unified, Extensions, Compatibility).
    var isCJK: Bool {
        guard let v = unicodeScalars.first?.value else { return false }
        return (0x4E00...0x9FFF).contains(v)       // CJK Unified Ideographs
            || (0x3400...0x4DBF).contains(v)       // CJK Extension A
            || (0x20000...0x2A6DF).contains(v)     // CJK Extension B
            || (0xF900...0xFAFF).contains(v)       // CJK Compatibility Ideographs
            || (0x2F800...0x2FA1F).contains(v)     // CJK Compatibility Supplement
    }
}

/// Tokenize text into word segments using NLTokenizer (Traditional Chinese).
struct CorrectionSegment {
    let word: String
    let range: Range<String.Index>
}

func correctionTokenize(_ text: String) -> [CorrectionSegment] {
    let tokenizer = NLTokenizer(unit: .word)
    tokenizer.string = text
    tokenizer.setLanguage(.traditionalChinese)

    var segments: [CorrectionSegment] = []
    tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
        segments.append(CorrectionSegment(word: String(text[range]), range: range))
        return true
    }
    return segments
}

/// Find the range of `word` in `text` closest to `approximateOffset` (character offset).
func correctionFindRange(of word: String, in text: String, near offset: Int) -> Range<String.Index>? {
    var bestRange: Range<String.Index>?
    var bestDistance = Int.max
    var searchStart = text.startIndex

    while let range = text.range(of: word, range: searchStart..<text.endIndex) {
        let rangeOffset = text.distance(from: text.startIndex, to: range.lowerBound)
        let distance = abs(rangeOffset - offset)
        if distance < bestDistance {
            bestDistance = distance
            bestRange = range
        }
        searchStart = range.upperBound
    }
    return bestRange
}

// MARK: - Protocol

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
    private let defaultWords: Set<String> = [
        "到家",
        "到家了",
        "回家",
        "在家",
        "我家",
        "大家",
        "自家",
        "M5 Max",
        "M5 Max 128GB",
        "鑑定",
        "身心障礙鑑定",
        "轉錄",
        "語音轉錄",
        "retranscribe",
        "Retranscribe",
    ]
    private var words: Set<String>

    private init() {
        let stored = UserDefaults.standard.stringArray(forKey: key) ?? []
        self.words = Set(stored).union(defaultWords)
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

    /// Check protected words after the same script normalization used by the ASR cleanup pipeline.
    func containsProtectedTerm(in text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }

        if containsSubstring(in: trimmed) { return true }

        let converted = OpenCCConverter.shared.convert(trimmed)
        return converted != trimmed && containsSubstring(in: converted)
    }

    /// Check whether the word at a known offset sits inside a protected phrase.
    func containsProtectedPhrase(in chars: [Character], covering offset: Int, length: Int, radius: Int = 2) -> Bool {
        guard !chars.isEmpty, offset >= 0, length > 0, offset < chars.count else {
            return false
        }

        let protectedStart = max(0, offset - radius)
        let protectedEnd = min(chars.count, offset + length + radius)
        let targetEnd = min(chars.count, offset + length)

        return queue.sync {
            for start in protectedStart...offset {
                guard start < targetEnd else { continue }
                for end in targetEnd...protectedEnd {
                    guard start < end else { continue }
                    let phrase = String(chars[start..<end])
                    if words.contains(phrase) {
                        return true
                    }
                }
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
