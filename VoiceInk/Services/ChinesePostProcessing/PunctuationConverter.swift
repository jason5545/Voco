import Foundation

/// Converts half-width punctuation to full-width and spoken punctuation names to symbols
/// Reference: xvoice/src/pipeline.py lines 42-140
class PunctuationConverter {
    static let shared = PunctuationConverter()

    // MARK: - Half-width to full-width punctuation

    private let halfWidthToFullWidth: [String: String] = [
        ",": "\u{FF0C}",   // ，
        "?": "\u{FF1F}",   // ？
        "!": "\u{FF01}",   // ！
        ":": "\u{FF1A}",   // ：
        ";": "\u{FF1B}",   // ；
        "(": "\u{FF08}",   // （
        ")": "\u{FF09}",   // ）
        "\u{FE50}": "\u{FF0C}", // ﹐ → ，
        "\u{FE51}": "\u{3001}", // ﹑ → 、
    ]

    // MARK: - Spoken punctuation map

    /// Maps spoken punctuation names to actual symbols.
    /// Sorted by key length descending at init time to prevent substring conflicts.
    private let spokenPunctuationMap: [(key: String, value: String)]

    // MARK: - Ambiguous punctuation

    /// Words that might be spoken punctuation or regular text - need LLM to decide
    let ambiguousPunctuation: [String: String] = [
        "都好": "\u{FF0C}",   // Could be "逗號" or "大家都好"
        "多好": "\u{FF0C}",   // Could be "逗號" or "這樣多好"
        "文化": "\u{FF1F}",   // Could be "問號" or "中華文化"
        "我好": "\u{FF1F}",   // Could be "問號" or "我好想你"
        "我很好": "\u{FF1F}", // Could be "問號" or a normal sentence
    ]

    private init() {
        // Build the spoken punctuation map
        let rawMap: [String: String] = [
            // Comma variants (including Whisper English misrecognitions)
            "逗號": "，",
            "重號": "，",
            "動號": "，",
            "鬥號": "，",
            "斗号": "，",
            "豆號": "，",
            "so-how": "，",
            "so how": "，",
            "So-how": "，",
            "So how": "，",
            "So-How": "，",
            "So How": "，",
            "SO-HOW": "，",
            "SO HOW": "，",
            "sohow": "，",
            "Sohow": "，",
            "SoHow": "，",
            "SOHOW": "，",
            "oh-how": "，",
            "oh how": "，",
            "Oh-how": "，",
            "Oh how": "，",
            "Oh-How": "，",
            "Oh How": "，",
            "OH-HOW": "，",
            "OH HOW": "，",
            "ohhow": "，",
            "Ohhow": "，",
            "OhHow": "，",
            "OHHOW": "，",
            "know-how": "，",
            "know how": "，",
            "Know-how": "，",
            "Know how": "，",
            "Know-How": "，",
            "Know How": "，",
            "KNOW-HOW": "，",
            "KNOW HOW": "，",
            "knowhow": "，",
            "Knowhow": "，",
            "KnowHow": "，",
            "KNOWHOW": "，",
            "how": "，",
            "How": "，",
            "HOW": "，",
            // Other punctuation (including common misrecognitions)
            "句號": "。",
            "G號": "。",
            "g號": "。",
            "巨號": "。",
            "具號": "。",
            "問號": "？",
            "驚嘆號": "！",
            "感嘆號": "！",
            "冒號": "：",
            "分號": "；",
            "頓號": "、",
            "省略號": "……",
            "刪節號": "……",
            "左引號": "「",
            "右引號": "」",
            "左括號": "（",
            "右括號": "）",
            "破折號": "——",
            "空格": " ",
            "換行": "\n",
        ]

        // Sort by key length descending to prevent substring conflicts
        spokenPunctuationMap = rawMap
            .sorted { $0.key.count > $1.key.count }
            .map { (key: $0.key, value: $0.value) }
    }

    // MARK: - Public API

    /// Convert half-width punctuation to full-width
    func convertHalfWidthToFullWidth(_ text: String) -> String {
        var result = text
        for (half, full) in halfWidthToFullWidth {
            result = result.replacingOccurrences(of: half, with: full)
        }
        return result
    }

    /// Convert spoken punctuation names to actual symbols
    /// Uses case-insensitive matching, processes longest keys first
    func convertSpokenPunctuation(_ text: String) -> String {
        var result = text
        for (key, value) in spokenPunctuationMap {
            // Case-insensitive replacement
            if let range = result.range(of: key, options: .caseInsensitive) {
                result = result.replacingOccurrences(of: key, with: value, options: .caseInsensitive)
            }
        }
        return result
    }

    /// Check if text contains ambiguous punctuation words
    func hasAmbiguousPunctuation(_ text: String) -> Bool {
        return ambiguousPunctuation.keys.contains { text.contains($0) }
    }
}
