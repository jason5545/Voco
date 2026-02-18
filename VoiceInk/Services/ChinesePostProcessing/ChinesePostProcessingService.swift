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

    private let logger = Logger(subsystem: "com.jasonchien.voco", category: "ChinesePostProcessing")

    // MARK: - Sub-services

    let openCCConverter = OpenCCConverter.shared
    let pinyinCorrector = PinyinCorrector.shared
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

        // Step 1: OpenCC s2twp conversion
        if isOpenCCEnabled {
            let converted = openCCConverter.convert(result)
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

        // Step 3: Pinyin correction
        if isPinyinCorrectionEnabled {
            let (corrected, corrections) = pinyinCorrector.correct(result)
            if !corrections.isEmpty {
                steps.append("PinyinCorrection")
                for c in corrections {
                    logger.debug("Pinyin: \(c.original) → \(c.corrected)")
                }
                result = corrected
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

        // Provider-specific confidence check
        switch lastModelProvider {
        case .local:
            // Whisper: use avgLogProb (unchanged)
            if lastAvgLogProb > logProbThreshold {
                Self.debugLog("SKIP: high confidence (avgLogProb=\(String(format: "%.3f", lastAvgLogProb)), threshold=\(String(format: "%.3f", logProbThreshold)), foundCJKPunct=\"\(foundPunct)\") | text(\(text.count)): \(text)")
                return true
            }
        case .qwen3:
            // Qwen3: text heuristic — short clean text can skip
            if qwen3TextQualityCheck(text) {
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

    // MARK: - Japanese Sentence Drift Detection

    /// Detect if text looks like a complete Japanese sentence (indicates Qwen3 misdetected Chinese as Japanese)
    /// Users only produce single Japanese words or song titles — never full sentences.
    static func detectsJapaneseSentenceDrift(_ text: String) -> Bool {
        // Must contain at least one hiragana/katakana to be considered Japanese
        let hasKana = text.unicodeScalars.contains { scalar in
            let v = scalar.value
            return (0x3040...0x309F).contains(v) || (0x30A0...0x30FF).contains(v)
        }
        guard hasKana else { return false }

        let particles = countJapaneseParticles(in: text)

        // Path A: Polite ending + >= 2 particles
        let politeEndings = ["です", "ます", "ました", "ません", "ました", "でした", "ましょう", "ください"]
        let hasPoliteEnding = politeEndings.contains { text.hasSuffix($0) }
        if hasPoliteEnding && particles >= 2 {
            debugLog("[JP_DRIFT] Path A: polite ending + \(particles) particles | text: \(text)")
            return true
        }

        // Path B: >= 3 particles + text length >= 10
        if particles >= 3 && text.count >= 10 {
            debugLog("[JP_DRIFT] Path B: \(particles) particles + len \(text.count) | text: \(text)")
            return true
        }

        // Path C: >= 2 multi-char grammatical particles
        let multiCharParticles = ["から", "まで", "より", "けど", "ので", "のに", "だけ", "ばかり", "ながら", "たり"]
        let multiCount = multiCharParticles.reduce(0) { count, particle in
            count + text.components(separatedBy: particle).count - 1
        }
        if multiCount >= 2 {
            debugLog("[JP_DRIFT] Path C: \(multiCount) multi-char particles | text: \(text)")
            return true
        }

        return false
    }

    /// Count single-char Japanese grammatical particles that follow CJK/kana characters.
    /// Excludes の (commonly used in song titles like 空の椅子).
    /// Only counts when preceded by kanji/hiragana/katakana to avoid false positives
    /// (e.g. こんにちは — the は is word-final, not a particle).
    private static func countJapaneseParticles(in text: String) -> Int {
        let particles: Set<Character> = ["は", "が", "を", "に", "で", "と", "も", "へ"]
        let chars = Array(text)
        var count = 0

        for i in 1..<chars.count {
            guard particles.contains(chars[i]) else { continue }
            // Check if previous character is CJK/hiragana/katakana
            let prev = chars[i - 1].unicodeScalars.first?.value ?? 0
            let isCJK = (0x4E00...0x9FFF).contains(prev) || (0x3400...0x4DBF).contains(prev)
            let isHiragana = (0x3040...0x309F).contains(prev)
            let isKatakana = (0x30A0...0x30FF).contains(prev)
            if isCJK || isHiragana || isKatakana {
                count += 1
            }
        }

        return count
    }

    // MARK: - Private Helpers

    /// Check if text is mainly English/numbers (no CJK characters)
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
        // Density check: expect at least 1 punctuation per 20 characters
        let expectedPunct = text.count / 20
        return punctCount < max(expectedPunct, 1)
    }
}
