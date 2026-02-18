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

    // MARK: - Confidence data (set by LocalTranscriptionService)

    var lastAvgLogProb: Double = 0.0

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

        // High confidence → skip LLM
        if lastAvgLogProb > logProbThreshold {
            Self.debugLog("SKIP: high confidence (avgLogProb=\(String(format: "%.3f", lastAvgLogProb)), threshold=\(String(format: "%.3f", logProbThreshold)), foundCJKPunct=\"\(foundPunct)\") | text(\(text.count)): \(text)")
            return true
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
