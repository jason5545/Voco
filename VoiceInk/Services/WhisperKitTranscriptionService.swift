// WhisperKitTranscriptionService.swift
// CoreML + ANE transcription via WhisperKit
// [AI-Claude: 2026-02-27]

import Foundation
import WhisperKit
import os

class WhisperKitTranscriptionService: TranscriptionService {
    private var whisperKit: WhisperKit?
    private var loadedVariant: String?
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "WhisperKitTranscriptionService")

    /// Average log-probability from the last transcription (for confidence routing)
    var lastAvgLogProb: Double = 0.0

    /// Auto-mode fallback thresholds. These are language-agnostic.
    private static let autoFallbackConfidenceThreshold: Float = -0.65
    private static let lowLanguageDetectionThreshold: Float = -0.55

    private struct TranscriptionAttempt {
        let text: String
        let avgLogProb: Float
        let detectedLang: String
    }

    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String {
        guard let wkModel = model as? WhisperKitModel else {
            throw WhisperKitTranscriptionError.invalidModel
        }

        let kit = try await ensureModelLoaded(wkModel)

        // Build decoding options
        let language = UserDefaults.standard.string(forKey: "SelectedLanguage")
        let promptText = UserDefaults.standard.string(forKey: "TranscriptionPrompt") ?? ""
        let promptTokens: [Int]? = promptText.isEmpty ? nil : kit.tokenizer?.encode(text: promptText)

        let effectiveLanguage = (language == "auto" || language == nil) ? nil : language

        await MainActor.run {
            ChinesePostProcessingService.debugLog("WHISPERKIT_OPTIONS: task=transcribe, language=\(effectiveLanguage ?? "nil(auto)"), promptTokens=\(promptTokens?.count ?? 0)")
        }

        let finalAttempt: TranscriptionAttempt
        if effectiveLanguage == nil {
            finalAttempt = try await transcribeAutoLanguageAgnostic(
                kit: kit,
                audioURL: audioURL,
                promptTokens: promptTokens
            )
        } else {
            finalAttempt = try await runTranscription(
                kit: kit,
                audioURL: audioURL,
                language: effectiveLanguage,
                promptTokens: promptTokens,
                usePrefillPrompt: true,
                detectLanguage: false
            )
        }

        storeConfidenceData(avgLogProb: finalAttempt.avgLogProb)
        await MainActor.run {
            ChinesePostProcessingService.debugLog(
                "WHISPERKIT_FINAL: lang=\(finalAttempt.detectedLang), avgLogProb=\(String(format: "%.3f", finalAttempt.avgLogProb)) | text(\(finalAttempt.text.count)): \(finalAttempt.text)"
            )
        }
        return finalAttempt.text
    }

    // MARK: - Private

    private func ensureModelLoaded(_ wkModel: WhisperKitModel) async throws -> WhisperKit {
        if let existing = whisperKit, loadedVariant == wkModel.whisperKitVariant {
            return existing
        }

        logger.notice("Loading WhisperKit model: \(wkModel.whisperKitVariant)")
        let modelDir = WhisperKitModelManager.modelDirectory(for: wkModel.whisperKitVariant)
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            throw WhisperKitTranscriptionError.modelNotDownloaded
        }

        let config = WhisperKitConfig(
            model: wkModel.whisperKitVariant,
            modelFolder: modelDir.path,
            computeOptions: ModelComputeOptions(
                audioEncoderCompute: .cpuAndNeuralEngine,
                textDecoderCompute: .cpuAndNeuralEngine
            ),
            verbose: false,
            load: true,
            download: false
        )
        let kit = try await WhisperKit(config)
        whisperKit = kit
        loadedVariant = wkModel.whisperKitVariant
        logger.notice("WhisperKit model loaded successfully")
        return kit
    }

    private func runTranscription(
        kit: WhisperKit,
        audioURL: URL,
        language: String?,
        promptTokens: [Int]?,
        usePrefillPrompt: Bool,
        detectLanguage: Bool?
    ) async throws -> TranscriptionAttempt {
        let shouldDetectLanguage = detectLanguage ?? (language == nil)
        let options = DecodingOptions(
            task: .transcribe,
            language: language,
            temperature: 0.0,
            usePrefillPrompt: usePrefillPrompt,
            detectLanguage: shouldDetectLanguage,
            wordTimestamps: false,
            promptTokens: promptTokens
        )

        let results = try await kit.transcribe(audioPath: audioURL.path, decodeOptions: options)

        // Log each result
        let logEntries = results.enumerated().map { (i, result) in
            "WHISPERKIT_RESULT[\(i)]: detectedLang=\(result.language), segments=\(result.segments.count), text(\(result.text.count)): \(result.text)"
        }
        await MainActor.run {
            for entry in logEntries {
                ChinesePostProcessingService.debugLog(entry)
            }
        }

        let text = results.map { $0.text }.joined()
        let detectedLang = results.first?.language ?? "unknown"

        let allSegments = results.flatMap { $0.segments }
        let avgLogProb: Float = allSegments.isEmpty ? 0.0 :
            allSegments.map { $0.avgLogprob }.reduce(0, +) / Float(allSegments.count)

        return TranscriptionAttempt(text: text, avgLogProb: avgLogProb, detectedLang: detectedLang)
    }

    /// Auto mode, language-agnostic:
    /// 1) dedicated language detection,
    /// 2) pinned-language transcription,
    /// 3) optional neutral fallback pass when quality is suspicious.
    private func transcribeAutoLanguageAgnostic(
        kit: WhisperKit,
        audioURL: URL,
        promptTokens: [Int]?
    ) async throws -> TranscriptionAttempt {
        let detection = try? await kit.detectLanguage(audioPath: audioURL.path)
        let detectedLanguage = normalizedLanguageCode(detection?.language ?? "")
        let detectionLogProb = detectionLogProbability(from: detection, language: detectedLanguage)

        await MainActor.run {
            if let detection {
                ChinesePostProcessingService.debugLog(
                    "WHISPERKIT_LANG_DETECT: language=\(detection.language), logProb=\(String(format: "%.3f", detectionLogProb ?? 0.0))"
                )
            } else {
                ChinesePostProcessingService.debugLog("WHISPERKIT_LANG_DETECT: failed, falling back to decode-time auto detection")
            }
        }

        let primaryLanguage = detectedLanguage.isEmpty ? nil : detectedLanguage
        let primaryAttempt = try await runTranscription(
            kit: kit,
            audioURL: audioURL,
            language: primaryLanguage,
            promptTokens: promptTokens,
            usePrefillPrompt: true,
            detectLanguage: primaryLanguage == nil
        )

        guard shouldRunNeutralFallback(primaryAttempt: primaryAttempt, detectionLogProb: detectionLogProb) else {
            return primaryAttempt
        }

        await MainActor.run {
            ChinesePostProcessingService.debugLog(
                "WHISPERKIT_NEUTRAL_RETRY: primaryQuality=\(String(format: "%.3f", qualityScore(primaryAttempt))), primaryLang=\(primaryAttempt.detectedLang), retrying with neutral decoding"
            )
        }

        let neutralAttempt = try await runTranscription(
            kit: kit,
            audioURL: audioURL,
            language: nil,
            promptTokens: nil,
            usePrefillPrompt: false,
            detectLanguage: false
        )

        await MainActor.run {
            ChinesePostProcessingService.debugLog(
                "WHISPERKIT_NEUTRAL_RETRY_RESULT: lang=\(neutralAttempt.detectedLang), avgLogProb=\(String(format: "%.3f", neutralAttempt.avgLogProb)) | text(\(neutralAttempt.text.count)): \(neutralAttempt.text)"
            )
        }

        return qualityScore(neutralAttempt) > qualityScore(primaryAttempt) ? neutralAttempt : primaryAttempt
    }

    private func shouldRunNeutralFallback(
        primaryAttempt: TranscriptionAttempt,
        detectionLogProb: Float?
    ) -> Bool {
        let trimmed = primaryAttempt.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return true }
        if containsHyphenatedRomanizationPattern(trimmed) { return true }
        if primaryAttempt.avgLogProb <= Self.autoFallbackConfidenceThreshold { return true }
        if let detectionLogProb, detectionLogProb <= Self.lowLanguageDetectionThreshold { return true }
        return false
    }

    private func qualityScore(_ attempt: TranscriptionAttempt) -> Float {
        let trimmed = attempt.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return -10.0 }

        var score = attempt.avgLogProb
        if containsHyphenatedRomanizationPattern(trimmed) {
            score -= 0.35
        }
        return score
    }

    private func detectionLogProbability(
        from detection: (language: String, langProbs: [String: Float])?,
        language: String
    ) -> Float? {
        guard let detection else { return nil }
        if let direct = detection.langProbs[language] {
            return direct
        }
        if let raw = detection.langProbs[detection.language] {
            return raw
        }
        return nil
    }

    private func normalizedLanguageCode(_ detectedLang: String) -> String {
        let code = detectedLang.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if let base = code.split(separator: "-", maxSplits: 1).first, !base.isEmpty {
            return String(base)
        }
        return code
    }

    private func containsHyphenatedRomanizationPattern(_ text: String) -> Bool {
        text.range(of: #"[A-Za-z]+(?:-[A-Za-z]+){2,}"#, options: .regularExpression) != nil
    }

    private func storeConfidenceData(avgLogProb: Float) {
        lastAvgLogProb = Double(avgLogProb)
        Task { @MainActor in
            ChinesePostProcessingService.shared.lastAvgLogProb = Double(avgLogProb)
            ChinesePostProcessingService.shared.lastModelProvider = .whisperKit
        }
    }

    func cleanup() {
        whisperKit = nil
        loadedVariant = nil
        logger.notice("WhisperKit resources released")
    }
}

enum WhisperKitTranscriptionError: LocalizedError {
    case invalidModel
    case modelNotDownloaded
    case modelNotLoaded

    var errorDescription: String? {
        switch self {
        case .invalidModel:
            return "Invalid WhisperKit model type"
        case .modelNotDownloaded:
            return "WhisperKit model has not been downloaded"
        case .modelNotLoaded:
            return "WhisperKit model failed to load"
        }
    }
}
