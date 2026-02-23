import Foundation

// MARK: - Data Types

struct PinyinCorrectionRule {
    let wrong: String
    let correct: String
    let tier: CorrectionTier
    let contextKeywords: [String] // contextDependent 時使用，OR 邏輯

    enum CorrectionTier {
        case alwaysApply
        case contextDependent
    }
}

struct CorrectionContext {
    let recentTranscriptions: [String]
    let appName: String?
    let windowTitle: String?
}

struct PinyinCorrectionResult {
    let text: String
    let corrections: [AppliedCorrection]

    struct AppliedCorrection {
        let original: String
        let corrected: String
        let tier: PinyinCorrectionRule.CorrectionTier
    }
}

// MARK: - PinyinCorrector

/// Pinyin-based correction for common speech recognition errors
/// Reference: xvoice/src/pinyin.py lines 37-79
class PinyinCorrector {
    static let shared = PinyinCorrector()

    private let rules: [PinyinCorrectionRule]

    /// Sorted rules by wrong-word length (longest first) to avoid substring conflicts
    private let sortedRules: [PinyinCorrectionRule]

    /// Database for boundary word checks
    private let db = PinyinDatabase.shared

    private init() {
        var allRules: [PinyinCorrectionRule] = []

        // ── alwaysApply rules ──
        let alwaysCorrections: [(String, String)] = [
            ("耳度", "額度"),
            ("變色", "辨識"),       // biànsè vs biànshí
            ("邊視", "辨識"),       // biānshì vs biànshí
            ("邊是", "辨識"),
            ("變是", "辨識"),
            ("便是", "辨識"),
            ("病史", "辨識"),       // bìngshǐ vs biànshí
            ("北頭", "北投"),       // běitóu
            ("去永所", "區公所"),
            ("去公所", "區公所"),
            ("曲公所", "區公所"),
            ("大圓模型", "大語言模型"),
            ("大援模型", "大語言模型"),
            ("大宇", "大語言"),     // dàyǔ vs dàyǔyán
            ("雨停", "語音"),       // yǔtíng vs yǔyīn
            ("靜動人", "漸凍人"),   // jìngdòngrén vs jiàndòngrén
            ("近動人", "漸凍人"),
            ("克爾勃", "克勞德"),   // Claude
            ("配頌", "配送"),
            ("鬥號", "逗號"),
            ("斗號", "逗號"),
            ("精子座", "金子做"),   // jīngzǐzuò vs jīnzizuò
            ("精子做", "金子做"),
            ("魔女飛行", "模擬飛行"), // mónǚfēixíng vs mónǐfēixíng
            ("雲酸", "運算"),
            ("端雲酸", "端運算"),
            ("雲端雲", "雲端運"),
            ("雲端雲酸", "雲端運算"),
            ("我哪好", "我很好"),
            ("哪好", "很好"),
            ("城市嘛", "程式碼"),   // chéngshìma vs chéngshìmǎ
            ("硬輸入", "語音輸入"),   // yìng vs yǔyīn — 兩音節壓縮成一音節
            ("硬辨識", "語音辨識"),
            ("硬轉文字", "語音轉文字"),
            ("硬轉錄", "語音轉錄"),
            ("硬助手", "語音助手"),
            ("硬信箱", "語音信箱"),
            ("獨立時候", "的時候"),   // 的(de) → 獨立(dúlì) ASR/pipeline hallucination
            ("提bug", "debug"),     // de → 提(tí) ASR error; 使用者會說「提出 bug」而非「提bug」
            ("抵bug", "debug"),     // de → 抵(dǐ) ASR error
        ]
        for (wrong, correct) in alwaysCorrections {
            allRules.append(PinyinCorrectionRule(
                wrong: wrong, correct: correct,
                tier: .alwaysApply, contextKeywords: []
            ))
        }

        // ── contextDependent rules ──
        let programmingKeywords = ["程式", "程式碼", "開發", "寫", "code", "Xcode", "Terminal", "Claude", "編譯", "build"]

        let contextCorrections: [(String, String, [String])] = [
            ("清晰度", "信心度", ["信心", "模型", "辨識", "轉錄", "Whisper", "Voco", "語音", "confidence"]),
            ("城市", "程式", programmingKeywords),
            ("成事", "程式", programmingKeywords),
            ("日劇", "日誌", ["log", "日誌", "紀錄", "除錯", "debug", "系統", "伺服器"]),
            ("專欄", "專案", ["專案", "project", "開發", "GitHub", "repo", "資料夾"]),
            ("單字", "單指", ["手指", "輸入", "打字", "鍵盤", "操作"]),
            ("轉入", "轉錄", ["轉錄", "語音", "錄音", "transcri", "Whisper", "辨識"]),
            ("推測", "推送", ["推送", "通知", "notification", "push", "訊息"]),
        ]
        for (wrong, correct, keywords) in contextCorrections {
            allRules.append(PinyinCorrectionRule(
                wrong: wrong, correct: correct,
                tier: .contextDependent, contextKeywords: keywords
            ))
        }

        self.rules = allRules
        self.sortedRules = allRules.sorted { $0.wrong.count > $1.wrong.count }
    }

    /// Apply corrections to the input text
    /// - Parameters:
    ///   - text: Input text to correct
    ///   - context: Optional context for contextDependent rules. When nil, only alwaysApply rules run.
    /// - Returns: Corrected text and list of corrections made
    func correct(_ text: String, context: CorrectionContext? = nil) -> PinyinCorrectionResult {
        var result = text
        var corrections: [PinyinCorrectionResult.AppliedCorrection] = []

        // Build combined context string once (lowercased for case-insensitive matching)
        let contextString: String? = context.map { ctx in
            var parts: [String] = []
            parts.append(contentsOf: ctx.recentTranscriptions)
            if let app = ctx.appName { parts.append(app) }
            if let title = ctx.windowTitle { parts.append(title) }
            return parts.joined(separator: " ").lowercased()
        }

        for rule in sortedRules {
            guard result.contains(rule.wrong) else { continue }

            // Skip if the wrong word is in the protection list
            if CorrectionProtectionList.shared.contains(rule.wrong) { continue }

            switch rule.tier {
            case .alwaysApply:
                let replaced = applyRuleWithBoundaryCheck(result, rule: rule)
                if replaced != result {
                    corrections.append(.init(
                        original: rule.wrong, corrected: rule.correct, tier: .alwaysApply
                    ))
                    result = replaced
                }

            case .contextDependent:
                guard matchesContext(
                    keywords: rule.contextKeywords,
                    currentText: text,
                    contextString: contextString
                ) else { continue }

                let replaced = applyRuleWithBoundaryCheck(result, rule: rule)
                if replaced != result {
                    corrections.append(.init(
                        original: rule.wrong, corrected: rule.correct, tier: .contextDependent
                    ))
                    result = replaced
                }
            }
        }

        return PinyinCorrectionResult(text: result, corrections: corrections)
    }

    // MARK: - Private

    /// Apply a rule with CJK boundary protection for short rules (≤ 2 chars).
    ///
    /// For rules where `wrong` is 1-2 CJK characters, checks whether the match
    /// crosses into an adjacent known word. If the last char of `wrong` + the
    /// next char forms a known word (freq > 0), or the previous char + the first
    /// char of `wrong` forms a known word, the match is skipped.
    private func applyRuleWithBoundaryCheck(_ text: String, rule: PinyinCorrectionRule) -> String {
        let wrongChars = Array(rule.wrong)
        let needsBoundaryCheck = wrongChars.count <= 2
            && db.isLoaded
            && wrongChars.allSatisfy { isCJK($0) }

        guard needsBoundaryCheck else {
            return text.replacingOccurrences(of: rule.wrong, with: rule.correct)
        }

        var result = text
        let textChars = Array(result)

        // Process matches from end to start to keep indices valid
        var searchEnd = result.endIndex
        var ranges: [Range<String.Index>] = []
        while let range = result.range(of: rule.wrong, range: result.startIndex..<searchEnd) {
            ranges.append(range)
            searchEnd = range.lowerBound
        }

        for range in ranges {
            let matchStart = result.distance(from: result.startIndex, to: range.lowerBound)
            let matchEnd = matchStart + wrongChars.count
            let currentChars = Array(result)

            // Check right boundary: last char of wrong + next char
            if matchEnd < currentChars.count {
                let nextChar = currentChars[matchEnd]
                if isCJK(nextChar) {
                    let rightPair = String(wrongChars.last!) + String(nextChar)
                    if db.frequency(of: rightPair) > 0 {
                        continue // skip this match
                    }
                }
            }

            // Check left boundary: previous char + first char of wrong
            if matchStart > 0 {
                let prevChar = currentChars[matchStart - 1]
                if isCJK(prevChar) {
                    let leftPair = String(prevChar) + String(wrongChars.first!)
                    if db.frequency(of: leftPair) > 0 {
                        continue // skip this match
                    }
                }
            }

            result = result.replacingCharacters(in: range, with: rule.correct)
        }

        return result
    }

    private func isCJK(_ char: Character) -> Bool {
        guard let scalar = char.unicodeScalars.first else { return false }
        let v = scalar.value
        return (0x4E00...0x9FFF).contains(v) || (0x3400...0x4DBF).contains(v)
    }

    /// Check if any keyword matches in the current text or combined context (OR logic)
    private func matchesContext(keywords: [String], currentText: String, contextString: String?) -> Bool {
        let lowerText = currentText.lowercased()
        for keyword in keywords {
            let lowerKeyword = keyword.lowercased()
            if lowerText.contains(lowerKeyword) { return true }
            if let ctx = contextString, ctx.contains(lowerKeyword) { return true }
        }
        return false
    }
}
