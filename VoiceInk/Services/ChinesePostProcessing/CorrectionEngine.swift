import Foundation

// MARK: - Shared Utilities

/// Common function words that hand-curated correction rules should never replace as single-character rules.
let correctionSkipChars: Set<Character> = [
    "的", "了", "嗎", "呢", "吧", "啊", "哦", "喔", "嗯", "呀",
    "是", "在", "有", "和", "也", "都", "就", "不", "我", "你",
    "他", "她", "它", "們", "這", "那", "個", "把", "被", "讓",
    "會", "能", "可", "要", "得", "地", "著", "過", "到", "從",
    "與", "及", "或", "而", "但", "因", "為", "所", "以", "如",
    "跟", "更", "再", "很", "才",
    // 唸 is the correct Traditional Chinese form of 念 (to read aloud).
    "唸",
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
        "轉路",
        "转路",
        "語音轉錄",
        "轉錄的技能",
        "retranscribe",
        "Retranscribe",
        "本地",
        "本地模型",
        "新的",
        "新的模型",
        "新對話",
        "新 prompt",
        "新 context prompt",
        "規則性",
        "規則性的",
        "規則性的模型",
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
        addSynchronously(word)
    }

    func addSynchronously(_ word: String) {
        let trimmed = word.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        queue.sync(flags: .barrier) {
            self.words.insert(trimmed)
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
