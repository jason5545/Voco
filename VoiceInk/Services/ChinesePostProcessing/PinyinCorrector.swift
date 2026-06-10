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
            ("Chat GPT", "ChatGPT"),
            ("chat gpt", "ChatGPT"),
            ("Cloud Code", "Claude Code"),
            ("Cloud code", "Claude Code"),
            ("cloud code", "Claude Code"),
            ("可以推測了", "可以推送了"),
            ("心理智商", "心理諮商"),
            ("心理資商", "心理諮商"),
            ("回到夾", "回到家"),
            ("先跟我承認一下", "先跟我確認一下"),
            ("重重新辨析", "重新辨識"),
            ("重重新轉錄", "重新轉錄"),
            ("我在我在說", "我在說"),
            ("重新辨析", "重新辨識"),
            ("確定是often", "確定是 orphan"),
            ("確定是 often", "確定是 orphan"),
            ("很凹頭", "很 auto"),
            ("屬於自己的失重", "屬於自己的實作"),
            ("work on the resources", "Workaround 與實作"),
        ]
        for (wrong, correct) in alwaysCorrections {
            allRules.append(PinyinCorrectionRule(
                wrong: wrong, correct: correct,
                tier: .alwaysApply, contextKeywords: []
            ))
        }

        // ── contextDependent rules ──
        let programmingKeywords = ["程式", "程式碼", "開發", "寫", "code", "Xcode", "Terminal", "Claude", "編譯", "build"]
        let aiKeywords = ["AI", "助理", "模型", "語音", "ASR", "轉錄", "Prompt", "prompt", "Claude", "ChatGPT", "Whisper", "Qwen", "OpenAI"]
        let therapyKeywords = ["心理", "心理諮商", "心理智商", "諮商", "諮商師", "心理師", "焦慮", "療程", "治療", "下個禮拜帶給他"]
        let sessionKeywords = ["session", "Codex", "Claude", "十個小時", "開一個新的", "接過來", "爆掉"]
        let uiKeywords = ["UI", "U I", "Codex", "Server", "app server", "第三方 UI", "介面", "網頁版"]
        let imageGenerationKeywords = [
            "ComfyUI", "comfyui", "Stable Diffusion", "Draw Things",
            "產圖", "生圖", "圖片生成", "AI 繪圖", "workflow", "流程圖",
            "節點", "node", "nodes", "連線", "連來連去"
        ]
        let correctionKeywords = ["語音辨識", "語音", "辨識", "ASR", "轉錄", "錯誤", "修正", "校正", "更正", "Codex", "Claude", "AI", "文字", "用字遣詞"]
        let keyboardFreezeKeywords = ["鍵盤", "滑鼠", "不動", "卡死", "畫面", "Caps", "caps lock", "大寫鍵", "按鍵"]
        let systemResourceKeywords = ["macOS", "Mac OS", "Activity Monitor", "資源", "耗盡", "卡死", "提示窗", "記憶體", "Memory Pressure", "系統"]
        let appleHardwareKeywords = ["Mac", "macOS", "Mac OS", "Apple", "硬體", "配備", "最頂", "基本版", "Max", "M5", "晶片", "筆電"]
        let loadingKeywords = ["Loading", "loading", "load", "卡", "卡死", "Activity Monitor"] + systemResourceKeywords + appleHardwareKeywords
        let dataImportKeywords = ["資料", "檔案", "Excel", "demo", "欄位", "欄位格式", "名稱", "匯入", "去年", "今年", "中元節", "單位"]
        let templeDataKeywords = [
            "廟", "廟方", "中元節", "功德金", "香油錢",
            "春節", "資料", "檔案", "Excel", "匯入", "去年", "今年", "查詢", "單位"
        ]
        let cloudflareKeywords = ["Cloudflare", "Workers", "D1", "D 1", "Durable Object", "repo", "GitHub", "專案", "部署"]
        let repositoryKeywords = ["repo", "r e p o", "GitHub", "commit", "push", "推到", "推送", "遠端", "main", "origin"]
        let inputMethodKeywords = [
            "RIME", "rime", "鼠鬚管", "鼠須管", "輸入法", "詞庫", "人名",
            "快捷鍵", "shortcut", "custom_phrase", "personal_dict", "Voco"
        ]
        let fieldRecognitionKeywords = dataImportKeywords + correctionKeywords + ["表格", "格式", "藍位", "浪費", "狼狽", "辨識成", "變質成"]
        let virtualizationKeywords = [
            "Windows", "Virtual Machine", "VM", "V M", "B M", "BM",
            "虛擬機", "虛擬機器", "PVE", "Proxmox", "Apollo", "串流", "遠端"
        ]
        let aiWritingKeywords = [
            "AI", "Gemini", "Google", "人工", "味道", "文章", "這篇",
            "去 AI", "去AI", "de-AI", "檢測器", "偵測器"
        ]
        let appleFoundationModelKeywords = [
            "Apple", "Foundation Models", "Foundation Model", "foundation model",
            "Apple Intelligence", "LLM", "模型", "省電", "插電", "電池"
        ]
        let jobApplicationKeywords = [
            "招聘", "求職", "工作", "應徵", "履歷", "自傳", "PDF",
            "提供", "繳交", "提交", "寄出", "信箱", "電子郵件"
        ]
        let contextCollectionKeywords = [
            "收集", "context", "上下文", "原始句子", "原來的句子",
            "重新辨識", "重新辨析", "看不出來", "列出來", "raw",
            "ASR", "轉錄", "辨識", "shadow", "replay"
        ]
        let blockingKeywords = [
            "阻塞", "堵塞", "喚醒", "卡住", "卡死", "延遲",
            "調查模組", "模組", "原因", "成因", "具體", "merge",
            "prewarm", "warm", "engine", "instance"
        ]

        let contextCorrections: [(String, String, [String])] = [
            ("清晰度", "信心度", ["信心", "模型", "辨識", "轉錄", "Whisper", "Voco", "語音", "confidence"]),
            ("城市", "程式", programmingKeywords),
            ("成事", "程式", programmingKeywords),
            ("日劇", "日誌", ["log", "日誌", "紀錄", "除錯", "debug", "系統", "伺服器"]),
            ("專欄", "專案", ["專案", "project", "開發", "GitHub", "repo", "資料夾"]),
            ("單字", "單指", ["手指", "輸入", "打字", "鍵盤", "操作"]),
            ("轉入", "轉錄", ["轉錄", "語音", "錄音", "transcri", "Whisper", "辨識"]),
            ("轉怒", "轉錄", ["轉錄", "retranscribe", "技能", "語音", "辨識", "ASR", "Voco"]),
            ("轉路", "轉錄", ["轉錄", "retranscribe", "技能", "語音", "辨識", "ASR", "Voco"]),
            ("推測", "推送", ["推送", "通知", "notification", "push", "訊息"]),
            ("差值被統一", "ChatGPT", aiKeywords),
            ("差點被統一", "ChatGPT", aiKeywords),
            ("Swiffer", "Whisper", ["Whisper", "語音", "ASR", "轉錄", "模型"]),
            ("千萬的 ASR", "Qwen 的 ASR", ["Qwen", "ASR", "模型", "語音", "千問"]),
            ("做持倉", "做諮商", therapyKeywords),
            ("對智障是就", "對諮商師就", therapyKeywords),
            ("智障是就", "諮商師就", therapyKeywords),
            ("執商師", "諮商師", therapyKeywords),
            ("智商的過程", "諮商的過程", therapyKeywords),
            ("在智商", "在諮商", therapyKeywords),
            ("的氣聲接", "的 session 接", sessionKeywords),
            ("氣聲接", "session 接", sessionKeywords),
            ("個氣聲", "個 session", sessionKeywords),
            ("並非有微弱", "並非 UI 問題", uiKeywords),
            ("並非無微", "並非 UI 問題", uiKeywords),
            ("Config UI", "ComfyUI", imageGenerationKeywords),
            ("config UI", "ComfyUI", imageGenerationKeywords),
            ("config ui", "ComfyUI", imageGenerationKeywords),
            ("ConfigUI", "ComfyUI", imageGenerationKeywords),
            ("Confi UI", "ComfyUI", imageGenerationKeywords),
            ("confi ui", "ComfyUI", imageGenerationKeywords),
            ("config.yml", "ComfyUI", imageGenerationKeywords),
            ("小振", "修正", correctionKeywords),
            ("大小雪", "大寫鍵", keyboardFreezeKeywords),
            ("支援耗盡", "資源耗盡", systemResourceKeywords),
            ("漏頂對", "loading 對", loadingKeywords),
            ("漏頂太大", "loading 太大", loadingKeywords),
            ("漏頂", "loading", loadingKeywords),
            ("M 五的", "M5 的", appleHardwareKeywords),
            ("M 五來", "M5 來", appleHardwareKeywords),
            ("M 五", "M5", appleHardwareKeywords),
            ("開始說吧，然後照流程", "開始修正吧，然後照流程", correctionKeywords + ["流程", "部署"]),
            ("西成的總長", "session 的總長", sessionKeywords + ["SESSION", "總長", "輪", "push"]),
            ("資料的，新就", "資料的新舊", dataImportKeywords),
            ("新就不重要", "新舊不重要", dataImportKeywords),
            ("闌尾的名稱", "欄位的名稱", dataImportKeywords),
            ("一比而已", "一筆而已", dataImportKeywords),
            ("莊園前一星期", "中元節前一星期", templeDataKeywords),
            ("妙方", "廟方", templeDataKeywords),
            ("妙芳", "廟方", templeDataKeywords),
            ("藍位的", "欄位的", fieldRecognitionKeywords),
            ("藍位辨識", "欄位辨識", fieldRecognitionKeywords),
            ("藍位格式", "欄位格式", fieldRecognitionKeywords),
            ("藍位", "欄位", fieldRecognitionKeywords),
            ("浪費的名稱", "欄位的名稱", dataImportKeywords),
            ("狼狽的名稱", "欄位的名稱", dataImportKeywords),
            ("浪費格式", "欄位格式", dataImportKeywords),
            ("狼狽格式", "欄位格式", dataImportKeywords),
            ("變質成", "辨識成", fieldRecognitionKeywords),
            ("變吃成", "辨識成", correctionKeywords + virtualizationKeywords),
            ("Windows B M", "Windows VM", virtualizationKeywords),
            ("Windows BM", "Windows VM", virtualizationKeywords),
            ("Virtual Machine 的 B M", "Virtual Machine 的 VM", virtualizationKeywords),
            ("Virtual Machine 的 BM", "Virtual Machine 的 VM", virtualizationKeywords),
            ("virtual machine 的 B M", "virtual machine 的 VM", virtualizationKeywords),
            ("virtual machine 的 BM", "virtual machine 的 VM", virtualizationKeywords),
            ("虛擬機器的 B M", "虛擬機器的 VM", virtualizationKeywords),
            ("虛擬機器的 BM", "虛擬機器的 VM", virtualizationKeywords),
            ("Load Fail", "Cloudflare", cloudflareKeywords),
            ("D One", "D1", cloudflareKeywords),
            ("鼠須管", "鼠鬚管", inputMethodKeywords),
            ("i iM 輸入法", "RIME 輸入法", inputMethodKeywords),
            ("I I M 輸入法", "RIME 輸入法", inputMethodKeywords),
            ("把它帶進這個城市", "把它帶進這個程式", inputMethodKeywords),
            ("帶進這個城市", "帶進這個程式", inputMethodKeywords),
            ("推 Ripper", "推 repo", repositoryKeywords),
            ("推Ripper", "推 repo", repositoryKeywords),
            ("推 reaper", "推 repo", repositoryKeywords),
            ("推reaper", "推 repo", repositoryKeywords),
            ("reaper", "repo", repositoryKeywords),
            ("Ripper", "repo", repositoryKeywords),
            ("目標是整車漆", "目標是偵測器", aiWritingKeywords),
            ("這片AI的味道", "這篇AI的味道", aiWritingKeywords),
            ("這片 AI 的味道", "這篇 AI 的味道", aiWritingKeywords),
            ("去一下AI好了", "去 AI 化好了", aiWritingKeywords),
            ("去一下 AI 好了", "去 AI 化好了", aiWritingKeywords),
            ("交給居民", "交給 Gemini", aiWritingKeywords),
            ("去 DAI 為", "去 de-AI 化", aiWritingKeywords),
            ("去DAI為", "去 de-AI 化", aiWritingKeywords),
            ("的防雷圈 Moto", "的 Foundation Model", appleFoundationModelKeywords),
            ("的防雷圈 Model", "的 Foundation Model", appleFoundationModelKeywords),
            ("防雷圈 Moto", "Foundation Model", appleFoundationModelKeywords),
            ("防雷圈 Model", "Foundation Model", appleFoundationModelKeywords),
            ("Foundation motto", "Foundation model", appleFoundationModelKeywords),
            ("Foundation Moto", "Foundation Model", appleFoundationModelKeywords),
            ("個人自傳跟找教的方式", "個人自傳跟繳交的方式", jobApplicationKeywords),
            ("個人自傳跟找工作的方式", "個人自傳跟繳交的方式", jobApplicationKeywords),
            ("個人自傳跟找教的方法", "個人自傳跟繳交的方法", jobApplicationKeywords),
            ("個人自傳跟找工作的方法", "個人自傳跟繳交的方法", jobApplicationKeywords),
            ("教教的方式", "繳交的方式", jobApplicationKeywords),
            ("怎麼角標自轉", "怎麼繳交自傳", jobApplicationKeywords),
            ("要自轉", "要自傳", jobApplicationKeywords),
            ("用新相機出去", "用信箱寄出去", jobApplicationKeywords),
            ("先做手機", "先做收集", contextCollectionKeywords),
            ("做手機", "做收集", contextCollectionKeywords),
            ("堵塞點", "阻塞點", blockingKeywords),
            ("組賽", "阻塞", blockingKeywords),
            ("陳英感覺", "成因感覺", blockingKeywords),
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
            if CorrectionProtectionList.shared.contains(rule.wrong),
               !allowsProtectedOverride(rule) {
                continue
            }

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
                guard allowsContextualRuleInCurrentText(rule, currentText: result) else {
                    continue
                }
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

    private func allowsProtectedOverride(_ rule: PinyinCorrectionRule) -> Bool {
        rule.tier == .contextDependent &&
            rule.wrong == "轉路" &&
            rule.correct == "轉錄"
    }

    private func allowsContextualRuleInCurrentText(_ rule: PinyinCorrectionRule, currentText: String) -> Bool {
        if rule.wrong == "做手機",
           rule.correct == "做收集" {
            let lowerText = currentText.lowercased()
            let mobileProductCues = [
                "手機 app", "手機app", "手機版", "手機 ui", "手機ui",
                "手機應用", "手機軟體", "手機介面", "手機開發"
            ]
            if mobileProductCues.contains(where: { lowerText.contains($0.lowercased()) }) {
                return false
            }

            let localCues = ["先做手機", "做手機。", "做手機，", "做手機？", "做手機！", "做手機."]
            return localCues.contains { currentText.contains($0) }
        }

        guard rule.wrong == "用新相機出去",
              rule.correct == "用信箱寄出去" else {
            return true
        }

        let localCues = [
            "自傳", "自轉", "角標", "履歷", "應徵", "招聘",
            "繳交", "提交", "寄出", "信箱", "電子郵件", "PDF"
        ]
        let lowerText = currentText.lowercased()
        return localCues.contains { lowerText.contains($0.lowercased()) }
    }

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
            && wrongChars.allSatisfy(\.isCJK)

        guard needsBoundaryCheck else {
            return text.replacingOccurrences(of: rule.wrong, with: rule.correct)
        }

        var result = text
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
                if nextChar.isCJK {
                    let rightPair = String(wrongChars.last!) + String(nextChar)
                    if db.frequency(of: rightPair) > 0 {
                        continue // skip this match
                    }
                }
            }

            // Check left boundary: previous char + first char of wrong
            if matchStart > 0 {
                let prevChar = currentChars[matchStart - 1]
                if prevChar.isCJK {
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

    /// Current utterance gets priority. Cached context needs stronger evidence
    /// for broad keyword sets so generic app/window noise does not over-trigger.
    private func matchesContext(keywords: [String], currentText: String, contextString: String?) -> Bool {
        let lowerText = currentText.lowercased()
        let currentMatches = Set(
            keywords.map { $0.lowercased() }.filter { lowerText.contains($0) }
        )
        if !currentMatches.isEmpty {
            return true
        }

        guard let ctx = contextString else { return false }
        let contextMatches = Set(
            keywords.map { $0.lowercased() }.filter { ctx.contains($0) }
        )
        let requiredContextMatches = keywords.count >= 6 ? 2 : 1
        return contextMatches.count >= requiredContextMatches
    }
}
