import Foundation
import os

/// Result of the Chinese post-processing pipeline
struct PostProcessingResult {
    let processedText: String
    let appliedSteps: [String]
    let repetitionInfo: RepetitionDetector.RepetitionInfo?
    let needsLLMCorrection: Bool
}

/// Main controller for Chinese post-processing pipeline
/// Coordinates all sub-services: OpenCC, pinyin correction, punctuation conversion,
/// repetition detection, confidence routing, context memory, and LLM validation.
///
/// Reference: xvoice/src/pipeline.py
@MainActor
class ChinesePostProcessingService: ObservableObject {
    static let shared = ChinesePostProcessingService()

    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "ChinesePostProcessing")

    // MARK: - Sub-services

    let openCCConverter = OpenCCConverter.shared
    let pinyinCorrector = PinyinCorrector.shared
    let homophoneEngine = HomophoneCorrectionEngine.shared
    let nasalCorrectionEngine = NasalCorrectionEngine.shared
    let syllableExpansionEngine = SyllableExpansionEngine.shared
    let punctuationConverter = PunctuationConverter.shared
    let repetitionDetector = RepetitionDetector.shared
    let contextMemory = TranscriptionContextMemory()
    let llmResponseValidator = LLMResponseValidator.shared

    // MARK: - Confidence data (set by transcription services)

    var lastAvgLogProb: Double = 0.0
    var lastModelProvider: ModelProvider?
    var lastAudioDuration: Double = 0.0

    // MARK: - Settings (UserDefaults backed)

    @Published var isEnabled: Bool {
        didSet { UserDefaults.standard.set(isEnabled, forKey: "ChinesePostProcessingEnabled") }
    }

    @Published var isOpenCCEnabled: Bool {
        didSet { UserDefaults.standard.set(isOpenCCEnabled, forKey: "ChinesePostProcessingOpenCC") }
    }

    @Published var isPinyinCorrectionEnabled: Bool {
        didSet { UserDefaults.standard.set(isPinyinCorrectionEnabled, forKey: "ChinesePostProcessingPinyin") }
    }

    @Published var isDataDrivenCorrectionEnabled: Bool {
        didSet { UserDefaults.standard.set(isDataDrivenCorrectionEnabled, forKey: "ChinesePostProcessingDataDriven") }
    }

    @Published var isSpokenPunctuationEnabled: Bool {
        didSet { UserDefaults.standard.set(isSpokenPunctuationEnabled, forKey: "ChinesePostProcessingSpokenPunctuation") }
    }

    @Published var isHalfWidthConversionEnabled: Bool {
        didSet { UserDefaults.standard.set(isHalfWidthConversionEnabled, forKey: "ChinesePostProcessingHalfWidth") }
    }

    @Published var isRepetitionDetectionEnabled: Bool {
        didSet { UserDefaults.standard.set(isRepetitionDetectionEnabled, forKey: "ChinesePostProcessingRepetition") }
    }

    @Published var isConfidenceRoutingEnabled: Bool {
        didSet { UserDefaults.standard.set(isConfidenceRoutingEnabled, forKey: "ChinesePostProcessingConfidence") }
    }

    @Published var isContextMemoryEnabled: Bool {
        didSet { UserDefaults.standard.set(isContextMemoryEnabled, forKey: "ChinesePostProcessingContextMemory") }
    }

    @Published var isLLMValidationEnabled: Bool {
        didSet { UserDefaults.standard.set(isLLMValidationEnabled, forKey: "ChinesePostProcessingLLMValidation") }
    }

    @Published var logProbThreshold: Double {
        didSet { UserDefaults.standard.set(logProbThreshold, forKey: "ChinesePostProcessingLogProbThreshold") }
    }

    @Published var qwen3SkipThreshold: Int {
        didSet { UserDefaults.standard.set(qwen3SkipThreshold, forKey: "ChinesePostProcessingQwen3SkipThreshold") }
    }


    // MARK: - Init

    private init() {
        self.isEnabled = UserDefaults.standard.bool(forKey: "ChinesePostProcessingEnabled")
        self.isOpenCCEnabled = UserDefaults.standard.object(forKey: "ChinesePostProcessingOpenCC") as? Bool ?? true
        self.isPinyinCorrectionEnabled = UserDefaults.standard.object(forKey: "ChinesePostProcessingPinyin") as? Bool ?? true
        self.isDataDrivenCorrectionEnabled = UserDefaults.standard.object(forKey: "ChinesePostProcessingDataDriven") as? Bool ?? true
        self.isSpokenPunctuationEnabled = UserDefaults.standard.object(forKey: "ChinesePostProcessingSpokenPunctuation") as? Bool ?? true
        self.isHalfWidthConversionEnabled = UserDefaults.standard.object(forKey: "ChinesePostProcessingHalfWidth") as? Bool ?? true
        self.isRepetitionDetectionEnabled = UserDefaults.standard.object(forKey: "ChinesePostProcessingRepetition") as? Bool ?? true
        self.isConfidenceRoutingEnabled = UserDefaults.standard.object(forKey: "ChinesePostProcessingConfidence") as? Bool ?? false
        self.isContextMemoryEnabled = UserDefaults.standard.object(forKey: "ChinesePostProcessingContextMemory") as? Bool ?? true
        self.isLLMValidationEnabled = UserDefaults.standard.object(forKey: "ChinesePostProcessingLLMValidation") as? Bool ?? true
        self.logProbThreshold = UserDefaults.standard.object(forKey: "ChinesePostProcessingLogProbThreshold") as? Double ?? -0.3
        self.qwen3SkipThreshold = UserDefaults.standard.object(forKey: "ChinesePostProcessingQwen3SkipThreshold") as? Int ?? 30
    }

    // MARK: - Main Processing Pipeline

    /// Process text through the Chinese post-processing pipeline
    func process(_ text: String) -> PostProcessingResult {
        var result = text
        var steps: [String] = []
        var repetitionInfo: RepetitionDetector.RepetitionInfo?

        // Step 1: OpenCC s2twp conversion (segment-aware: skips Japanese parts)
        if isOpenCCEnabled {
            let converted = openCCConvertSkippingJapanese(result)
            if converted != result {
                steps.append("OpenCC")
                logger.debug("OpenCC: \(result) → \(converted)")
                result = converted
            }
        }

        // Step 2: Half-width → full-width punctuation
        if isHalfWidthConversionEnabled {
            let converted = punctuationConverter.convertHalfWidthToFullWidth(result)
            if converted != result {
                steps.append("HalfWidthPunctuation")
                result = converted
            }
        }

        // Step 3: Pinyin correction (rule-based + data-driven)
        if isPinyinCorrectionEnabled {
            // Layer 1: Rule-based (hand-curated, highest priority)
            let editCache = EditModeCacheService.shared
            let correctionContext = CorrectionContext(
                recentTranscriptions: contextMemory.getRecent(count: 5),
                appName: editCache.cachedAppName,
                windowTitle: editCache.cachedWindowTitle
            )
            let correctionResult = pinyinCorrector.correct(result, context: correctionContext)
            if !correctionResult.corrections.isEmpty {
                steps.append("PinyinCorrection")
                for c in correctionResult.corrections {
                    let tier = c.tier == .contextDependent ? "[ctx]" : "[always]"
                    logger.debug("Pinyin \(tier): \(c.original) → \(c.corrected)")
                }
                result = correctionResult.text
            }

            // Layer 2: Data-driven homophone correction (automatic, catches remaining errors)
            if isDataDrivenCorrectionEnabled, PinyinDatabase.shared.isLoaded {
                let engineResult = homophoneEngine.correct(result)
                if !engineResult.corrections.isEmpty {
                    steps.append("HomophoneCorrection")
                    for c in engineResult.corrections {
                        logger.debug("Pinyin [data]: \(c.original) → \(c.corrected) (score=\(String(format: "%.1f", c.score)))")
                    }
                    result = engineResult.text
                }

                // Layer 2.5: Nasal ending correction (-n/-ng swap)
                let nasalResult = nasalCorrectionEngine.correct(result)
                if !nasalResult.corrections.isEmpty {
                    steps.append("NasalCorrection")
                    for c in nasalResult.corrections {
                        logger.debug("Pinyin [nasal]: \(c.original) → \(c.corrected) (score=\(String(format: "%.1f", c.score)))")
                    }
                    result = nasalResult.text
                }

                // Layer 3: Syllable expansion (1→2 char, compressed syllable recovery)
                let expandResult = syllableExpansionEngine.correct(result)
                if !expandResult.corrections.isEmpty {
                    steps.append("SyllableExpansion")
                    for c in expandResult.corrections {
                        logger.debug("Pinyin [expand]: \(c.original) → \(c.corrected) (score=\(String(format: "%.1f", c.score)))")
                    }
                    result = expandResult.text
                }

                // Layer 1 re-scan: catch patterns introduced by data-driven layers
                let reCheckResult = pinyinCorrector.correct(result, context: correctionContext)
                if !reCheckResult.corrections.isEmpty {
                    steps.append("PinyinReCheck")
                    for c in reCheckResult.corrections {
                        logger.debug("Pinyin [recheck]: \(c.original) → \(c.corrected)")
                    }
                    result = reCheckResult.text
                }
            }
        }

        // Step 4: Spoken punctuation conversion
        if isSpokenPunctuationEnabled {
            let converted = punctuationConverter.convertSpokenPunctuation(result)
            if converted != result {
                steps.append("SpokenPunctuation")
                result = converted
            }
        }

        // Step 5: Repetition detection
        if isRepetitionDetectionEnabled {
            let info = repetitionDetector.detect(result)
            repetitionInfo = info
            if info.hasRepetition {
                steps.append("RepetitionDetected")
                logger.debug("Repetition: \(info.details), ratio: \(String(format: "%.1f%%", info.repetitionRatio * 100))")
            }
        }

        // Determine if LLM enhancement should be skipped
        let needsLLM = !shouldSkipLLMEnhancement(text: result, repetitionInfo: repetitionInfo)

        return PostProcessingResult(
            processedText: result,
            appliedSteps: steps,
            repetitionInfo: repetitionInfo,
            needsLLMCorrection: needsLLM
        )
    }

    // MARK: - Confidence-Based LLM Routing

    /// Determine if LLM enhancement should be skipped based on confidence metrics
    func shouldSkipLLMEnhancement(text: String, repetitionInfo: RepetitionDetector.RepetitionInfo? = nil) -> Bool {
        guard isConfidenceRoutingEnabled else {
            Self.debugLog("ROUTING: disabled → use LLM | text(\(text.count)): \(text)")
            return false
        }

        // Pure English/numbers → skip LLM
        if isMainlyEnglish(text) {
            Self.debugLog("SKIP: mainly English | text(\(text.count)): \(text)")
            return true
        }

        // Simple short responses → skip LLM
        if isSimpleText(text) {
            Self.debugLog("SKIP: simple text | text(\(text.count)): \(text)")
            return true
        }

        // Long sentence without punctuation → force LLM
        let cjkPunct: Set<Character> = ["，", "。", "？", "！", "、", "；", "：", "「", "」", "『", "』", "（", "）"]
        let foundPunct = text.filter { cjkPunct.contains($0) }
        let punctNeeded = needsPunctuation(text)
        if punctNeeded {
            Self.debugLog("FORCE LLM: needs punctuation (len=\(text.count), foundCJKPunct=\"\(foundPunct)\") | text: \(text)")
            return false
        }

        // CJK repeated character detection (e.g. 偶偶然, 狀狀況, 沒沒有)
        // Must run BEFORE confidence check — overrides high confidence
        if hasRepeatedCJKCharacters(text) {
            Self.debugLog("FORCE LLM: repeated CJK characters | text(\(text.count)): \(text)")
            return false
        }

        // Provider-specific confidence check
        switch lastModelProvider {
        case .local:
            // Whisper: use avgLogProb (unchanged)
            if lastAvgLogProb > logProbThreshold {
                Self.debugLog("SKIP: high confidence (avgLogProb=\(String(format: "%.3f", lastAvgLogProb)), threshold=\(String(format: "%.3f", logProbThreshold)), foundCJKPunct=\"\(foundPunct)\") | text(\(text.count)): \(text)")
                return true
            }
        case .qwen3:
            // Qwen3: use logprob when available, fallback to text heuristic
            if lastAvgLogProb != 0.0 {
                // Has logprob data → use it (same logic as Whisper)
                if lastAvgLogProb > logProbThreshold {
                    Self.debugLog("SKIP: Qwen3 high confidence (avgLogProb=\(String(format: "%.3f", lastAvgLogProb)), threshold=\(String(format: "%.3f", logProbThreshold))) | text(\(text.count)): \(text)")
                    return true
                }
            } else if qwen3TextQualityCheck(text) {
                // No logprob data (fallback) → text heuristic
                Self.debugLog("SKIP: Qwen3 text quality OK (cjk≤\(qwen3SkipThreshold) or rate normal) | text(\(text.count)): \(text)")
                return true
            }
        default:
            break // other providers: fall through to default (use LLM)
        }

        // Has ambiguous punctuation or repetition → need LLM
        if punctuationConverter.hasAmbiguousPunctuation(text) {
            Self.debugLog("FORCE LLM: ambiguous punctuation | text(\(text.count)): \(text)")
            return false
        }
        if let info = repetitionInfo, info.hasRepetition {
            Self.debugLog("FORCE LLM: repetition | text(\(text.count)): \(text)")
            return false
        }

        Self.debugLog("DEFAULT: use LLM | text(\(text.count)): \(text)")
        return false
    }

    // MARK: - Debug File Logging

    private static let debugLogURL: URL = {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/Voco", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("confidence-routing.log")
    }()

    static func debugLog(_ message: String) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let entry = "[\(timestamp)] \(message)\n"
        if let data = entry.data(using: .utf8) {
            if FileManager.default.fileExists(atPath: debugLogURL.path) {
                if let handle = try? FileHandle(forWritingTo: debugLogURL) {
                    handle.seekToEndOfFile()
                    handle.write(data)
                    handle.closeFile()
                }
            } else {
                try? data.write(to: debugLogURL)
            }
        }
    }

    // MARK: - Private Helpers

    /// OpenCC 轉換，自動跳過含假名的日文段落
    /// 將文字分為 CJK 段（漢字+假名）和非 CJK 段（標點、空格、ASCII），
    /// 只對不含假名的 CJK 段執行 OpenCC。
    private func openCCConvertSkippingJapanese(_ text: String) -> String {
        var segments: [(text: String, isCJKRun: Bool, hasKana: Bool)] = []
        var currentRun = ""
        var inCJKRun = false
        var runHasKana = false

        for char in text {
            let v = char.unicodeScalars.first!.value
            let isKana = (0x3040...0x309F).contains(v) || (0x30A0...0x30FF).contains(v)
            let isCJK = (0x4E00...0x9FFF).contains(v) || (0x3400...0x4DBF).contains(v)
                || (0x20000...0x2A6DF).contains(v)

            if isKana || isCJK {
                if !inCJKRun && !currentRun.isEmpty {
                    segments.append((currentRun, false, false))
                    currentRun = ""
                }
                inCJKRun = true
                if isKana { runHasKana = true }
                currentRun.append(char)
            } else {
                if inCJKRun && !currentRun.isEmpty {
                    segments.append((currentRun, true, runHasKana))
                    currentRun = ""
                    runHasKana = false
                }
                inCJKRun = false
                currentRun.append(char)
            }
        }
        if !currentRun.isEmpty {
            segments.append((currentRun, inCJKRun, runHasKana))
        }

        // 無假名 → 整段 OpenCC（最常見路徑，避免不必要的分段開銷）
        if !segments.contains(where: { $0.hasKana }) {
            return openCCConverter.convert(text)
        }

        return segments.map { seg in
            if seg.isCJKRun && !seg.hasKana {
                return openCCConverter.convert(seg.text)
            }
            return seg.text
        }.joined()
    }

    private func containsKana(_ text: String) -> Bool {
        text.unicodeScalars.contains { v in
            (0x3040...0x309F).contains(v.value) || (0x30A0...0x30FF).contains(v.value)
        }
    }

    /// Check if text is mainly English/numbers (no CJK characters)
    /// Detect repeated CJK characters like 偶偶然, 狀狀況, 沒沒有
    /// Returns true if any non-legitimate CJK character doubling is found
    private func hasRepeatedCJKCharacters(_ text: String) -> Bool {
        // Common legitimate reduplicated words (疊字)
        let legitimateDoubles: Set<Character> = [
            // Family
            "媽", "爸", "哥", "姐", "弟", "妹", "奶", "爺", "叔", "伯", "婆", "娃", "寶",
            // Adverbs/adjectives
            "慢", "常", "漸", "剛", "偷", "悄", "靜", "默", "輕", "淡", "深", "久", "僅",
            "大", "多", "好", "天", "人", "處", "時", "事", "往", "稍", "略", "微", "早",
            "乖", "快", "高", "長", "滿", "緊",
            // Onomatopoeia / other
            "哈", "嘻", "呵",
            // Common words
            "謝", "星",
        ]

        let chars = Array(text)
        var i = 0
        while i < chars.count - 1 {
            let c = chars[i]
            if c == chars[i + 1] {
                let v = c.unicodeScalars.first!.value
                let isCJK = (0x4E00...0x9FFF).contains(v) || (0x3400...0x4DBF).contains(v)
                    || (0x20000...0x2A6DF).contains(v)
                let isKana = (0x3040...0x309F).contains(v) || (0x30A0...0x30FF).contains(v)
                if (isCJK || isKana) && !legitimateDoubles.contains(c) {
                    return true
                }
                i += 2 // skip the pair
            } else {
                i += 1
            }
        }
        return false
    }

    private func isMainlyEnglish(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }
        let nonASCII = text.unicodeScalars.filter { $0.value > 127 }.count
        return nonASCII == 0
    }

    /// Check if text is a simple definitive response
    private func isSimpleText(_ text: String) -> Bool {
        // Pure digits
        if text.replacingOccurrences(of: " ", with: "").allSatisfy(\.isNumber) {
            return true
        }

        let simpleResponses: Set<String> = [
            "好", "是", "對", "不", "沒", "有", "嗯", "喔", "啊",
            "好的", "是的", "對的", "不是", "沒有", "知道", "了解",
            "謝謝", "感謝", "抱歉", "不好意思",
        ]
        return simpleResponses.contains(text)
    }

    /// Check if Qwen3 transcription text quality is good enough to skip LLM
    private func qwen3TextQualityCheck(_ text: String) -> Bool {
        let cjkCount = text.unicodeScalars.filter {
            (0x4E00...0x9FFF).contains($0.value) ||
            (0x3400...0x4DBF).contains($0.value)
        }.count

        // Excessive filler words → force LLM (redundant/repeated particles)
        if hasExcessiveFillers(text) {
            Self.debugLog("Qwen3: excessive filler words detected | text(\(text.count)): \(text)")
            return false
        }

        // List-like content → force LLM (prompt supports bullet formatting)
        if hasListContent(text) {
            Self.debugLog("Qwen3: list content detected → force LLM for formatting | text(\(text.count)): \(text)")
            return false
        }

        // Short text → likely clean
        if cjkCount <= qwen3SkipThreshold {
            return true
        }

        // Speech rate check: abnormal rate → force LLM
        if lastAudioDuration > 0 {
            let rate = Double(cjkCount) / lastAudioDuration
            if rate < 1.5 || rate > 8.0 {
                Self.debugLog("Qwen3: abnormal speech rate \(String(format: "%.1f", rate)) chars/sec (cjk=\(cjkCount), dur=\(String(format: "%.1f", lastAudioDuration))s)")
                return false
            }
        }

        return false // long text → use LLM
    }

    /// Check if text contains list-like / enumeration content that benefits from LLM formatting
    private func hasListContent(_ text: String) -> Bool {
        // 第一、第二… / 第一個、第二個…
        let ordinalPattern = /第[一二三四五六七八九十百]\S{0,1}/
        let ordinalMatches = text.matches(of: ordinalPattern).count
        if ordinalMatches >= 2 { return true }

        // 首先…然後/接著/再來/最後
        let sequenceWords: [String] = ["首先", "然後", "接著", "再來", "最後", "其次", "另外", "還有"]
        let seqCount = sequenceWords.reduce(0) { count, word in
            count + text.components(separatedBy: word).count - 1
        }
        if seqCount >= 2 { return true }

        // 1. 2. 3. or (1) (2) or ① ②
        let numberListPattern = /(?:\d+[.、]|[（(]\d+[)）]|[①②③④⑤⑥⑦⑧⑨⑩])/
        let numMatches = text.matches(of: numberListPattern).count
        if numMatches >= 2 { return true }

        return false
    }

    /// Check if text has excessive or repeated filler words (語助詞)
    private func hasExcessiveFillers(_ text: String) -> Bool {
        let fillers: Set<Character> = ["啊", "嗯", "呢", "吧", "啦", "喔", "欸", "齁", "嘛", "呀", "哦", "噢", "唉", "哎", "嘿", "蛤"]
        let chars = Array(text)
        var fillerCount = 0

        for (i, ch) in chars.enumerated() {
            guard fillers.contains(ch) else { continue }
            fillerCount += 1
            // Consecutive repeated filler (e.g. 啊啊啊) → definitely excessive
            if i + 1 < chars.count && chars[i + 1] == ch {
                return true
            }
        }

        // Filler density: >15% of text characters are fillers → excessive
        guard text.count > 0 else { return false }
        let ratio = Double(fillerCount) / Double(text.count)
        return ratio > 0.15
    }

    /// Check if text needs punctuation added by LLM (density-based)
    private func needsPunctuation(_ text: String, minLength: Int = 10) -> Bool {
        guard text.count >= minLength else { return false }
        let punctuationMarks: Set<Character> = ["，", "。", "？", "！", "、", "；", "：", "「", "」", "『", "』", "（", "）"]
        let punctCount = text.filter { punctuationMarks.contains($0) }.count
        if punctCount == 0 { return true }

        // Overall density check: expect at least 1 punctuation per 20 characters
        let expectedPunct = text.count / 20
        if punctCount < max(expectedPunct, 1) { return true }

        // Long span check: any segment between punctuation with >20 CJK chars → needs LLM
        let maxCJKSpan = 20
        var cjkCount = 0
        for char in text {
            if punctuationMarks.contains(char) {
                cjkCount = 0
            } else if char.unicodeScalars.first.map({ (0x4E00...0x9FFF).contains($0.value) || (0x3400...0x4DBF).contains($0.value) }) == true {
                cjkCount += 1
                if cjkCount > maxCJKSpan { return true }
            }
        }

        return false
    }
}
