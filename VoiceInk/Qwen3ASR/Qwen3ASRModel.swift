// Qwen3ASRModel.swift
// Adapted from qwen3-asr-swift Qwen3ASR.swift
// Removed: fromPretrained(), backward-compat extensions
// Added: load(from:modelSize:), transcribe() throws
// [AI-Claude: 2025-02-18]

import Foundation
import MLX
import MLXNN
import MLXFast
import os

enum Qwen3ASRModelError: Error, LocalizedError {
    case textDecoderNotLoaded
    case loadFailed(String)

    var errorDescription: String? {
        switch self {
        case .textDecoderNotLoaded:
            return "Qwen3-ASR text decoder not loaded"
        case .loadFailed(let reason):
            return "Failed to load Qwen3-ASR model: \(reason)"
        }
    }
}

/// Main Qwen3-ASR model for speech recognition
class Qwen3ASRModel {
    struct TranscriptionResult {
        let text: String
        let avgLogProb: Double
        let tokenCount: Int
        let detectedLanguage: String?  // auto 模式偵測到的語言（如 "Japanese"），手動指定時為 nil
    }

    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "Qwen3ASRModel")

    /// Language tags that cause English transliteration; remap to preserve code-switching
    private static let codeSwitchLanguageRemap: [String: String] = [
        "Chinese": "zh",
    ]

    let audioEncoder: Qwen3AudioEncoder
    let featureExtractor: Qwen3FeatureExtractor
    var textDecoder: Qwen3QuantizedTextModel?
    private var tokenizer: Qwen3Tokenizer?
    let textConfig: Qwen3TextDecoderConfig

    init(
        audioConfig: Qwen3AudioEncoderConfig = .default,
        textConfig: Qwen3TextDecoderConfig = .small
    ) throws {
        self.audioEncoder = Qwen3AudioEncoder(config: audioConfig)
        self.featureExtractor = try Qwen3FeatureExtractor()
        self.textConfig = textConfig
        self.textDecoder = nil
    }

    /// Load model weights from a directory
    func load(from directory: URL, modelSize: Qwen3ASRModelSize) throws {
        // Load tokenizer
        let vocabPath = directory.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabPath.path) {
            let tok = Qwen3Tokenizer()
            try tok.load(from: vocabPath)
            self.tokenizer = tok
        }

        Self.logger.info("Loading audio encoder weights...")
        try Qwen3WeightLoader.loadAudioEncoderWeights(into: audioEncoder, from: directory)

        Self.logger.info("Loading text decoder weights...")
        self.textDecoder = Qwen3QuantizedTextModel(config: textConfig)
        if let textDecoder = self.textDecoder {
            try Qwen3WeightLoader.loadTextDecoderWeights(into: textDecoder, from: directory)
        }

        Self.logger.info("Model loaded successfully")
    }

    /// Transcribe audio to text
    func transcribe(
        audio: [Float],
        sampleRate: Int = 16000,
        language: String? = nil,
        prompt: String? = nil,
        maxTokens: Int? = nil
    ) throws -> TranscriptionResult {
        // Scale maxTokens proportionally to audio duration (448 tokens per 30s baseline)
        let durationSeconds = Double(audio.count) / Double(sampleRate)
        let effectiveMaxTokens = maxTokens ?? min(max(448, Int(durationSeconds / 30.0 * 448.0)), 32768)

        let melFeatures = try featureExtractor.process(audio, sampleRate: sampleRate)
        let batchedFeatures = melFeatures.expandedDimensions(axis: 0)

        var audioEmbeds = audioEncoder(batchedFeatures)
        audioEmbeds = audioEmbeds.expandedDimensions(axis: 0)

        guard let textDecoder = textDecoder else {
            throw Qwen3ASRModelError.textDecoderNotLoaded
        }

        return try generateText(
            audioEmbeds: audioEmbeds,
            textDecoder: textDecoder,
            language: language,
            prompt: prompt,
            maxTokens: effectiveMaxTokens
        )
    }

    private func generateText(
        audioEmbeds: MLXArray,
        textDecoder: Qwen3QuantizedTextModel,
        language: String?,
        prompt: String? = nil,
        maxTokens: Int
    ) throws -> TranscriptionResult {
        let tokens = Qwen3ASRTokens.self
        let numAudioTokens = audioEmbeds.dim(1)

        var inputIds: [Int32] = []

        // <|im_start|>system\n{prompt}<|im_end|>\n
        if let prompt = prompt, !prompt.isEmpty, let tokenizer = tokenizer {
            inputIds.append(contentsOf: [tokens.imStartTokenId, tokens.systemId, tokens.newlineId].map { Int32($0) })
            let promptTokens = tokenizer.encode(prompt)
            inputIds.append(contentsOf: promptTokens.map { Int32($0) })
            inputIds.append(contentsOf: [tokens.imEndTokenId, tokens.newlineId].map { Int32($0) })
        } else {
            inputIds.append(contentsOf: [tokens.imStartTokenId, tokens.systemId, tokens.newlineId, tokens.imEndTokenId, tokens.newlineId].map { Int32($0) })
        }

        // <|im_start|>user\n<|audio_start|>
        inputIds.append(contentsOf: [tokens.imStartTokenId, tokens.userId, tokens.newlineId, tokens.audioStartTokenId].map { Int32($0) })

        // <|audio_pad|> * numAudioTokens
        let audioStartIndex = inputIds.count
        for _ in 0..<numAudioTokens {
            inputIds.append(Int32(tokens.audioTokenId))
        }
        let audioEndIndex = inputIds.count

        // <|audio_end|><|im_end|>\n
        inputIds.append(contentsOf: [tokens.audioEndTokenId, tokens.imEndTokenId, tokens.newlineId].map { Int32($0) })

        // <|im_start|>assistant\n
        inputIds.append(contentsOf: [tokens.imStartTokenId, tokens.assistantId, tokens.newlineId].map { Int32($0) })

        // Auto-detect mode: pre-fill "language" as prompt prefix so the model
        // only needs to predict the language NAME (e.g. " Chinese"), not the
        // entire "language Chinese<asr_text>..." sequence from scratch.
        // This dramatically stabilizes first-inference language detection.
        var autoDetectPrefillTokens: [Int32]? = nil
        if language == nil, let tokenizer = tokenizer {
            let prefillTokens = tokenizer.encode("language").map { Int32($0) }
            autoDetectPrefillTokens = prefillTokens
            inputIds.append(contentsOf: prefillTokens)
        }

        if let lang = language, let tokenizer = tokenizer {
            let langPrefix = "language \(lang)"
            let langTokens = tokenizer.encode(langPrefix)
            inputIds.append(contentsOf: langTokens.map { Int32($0) })
            inputIds.append(Int32(tokens.asrTextId))
        }

        let inputIdsTensor = MLXArray(inputIds).expandedDimensions(axis: 0)
        var inputEmbeds = textDecoder.embedTokens(inputIdsTensor)

        let audioEmbedsTyped = audioEmbeds.asType(inputEmbeds.dtype)
        let beforeAudio = inputEmbeds[0..., 0..<audioStartIndex, 0...]
        let afterAudio = inputEmbeds[0..., audioEndIndex..., 0...]

        inputEmbeds = concatenated([beforeAudio, audioEmbedsTyped, afterAudio], axis: 1)

        var cache: [(MLXArray, MLXArray)]? = nil
        var generatedTokens: [Int32] = []
        // Pre-seed "language" tokens into generatedTokens so downstream
        // parsing (which expects "language XXX" before <asr_text>) works unchanged.
        if let prefill = autoDetectPrefillTokens {
            generatedTokens.append(contentsOf: prefill)
        }
        let evalInterval = 50  // Force MLX evaluation every N tokens to prevent computation graph accumulation

        // Per-token logprob tracking
        var totalLogProb: Double = 0.0
        var logProbTokenCount: Int = 0
        // When language is specified, <asr_text> was appended to inputIds;
        // all generated tokens are text tokens → start counting immediately
        var isCountingLogProb = (language != nil)

        // Helper: extract language name from "language XXX" prefix
        func extractLanguageName(from rawText: String) -> String? {
            guard rawText.hasPrefix("language ") else { return nil }
            let afterLang = rawText.dropFirst("language ".count)
            // Language name is a single word (e.g. "Japanese", "Chinese", "English")
            if let spaceIdx = afterLang.firstIndex(of: " ") {
                return String(afterLang[afterLang.startIndex..<spaceIdx])
            }
            // If no space found, the entire remainder is the language name (edge case)
            return afterLang.isEmpty ? nil : String(afterLang)
        }

        var (hiddenStates, newCache) = try textDecoder(inputsEmbeds: inputEmbeds, cache: cache)
        cache = newCache

        let seqLen = hiddenStates.dim(1)
        let lastHidden = hiddenStates[0..., (seqLen-1)..<seqLen, 0...]
        var logits = textDecoder.embedTokens.asLinear(lastHidden)
        var nextToken = argMax(logits, axis: -1).squeezed().item(Int32.self)

        if nextToken == Int32(tokens.asrTextId) {
            // Code-switch remap: if auto-detect found a language that transliterates English,
            // re-run with the remapped tag to preserve code-switching
            if language == nil, let tok = tokenizer {
                let prefixText = tok.decode(tokens: generatedTokens.map { Int($0) })
                if let langName = extractLanguageName(from: prefixText),
                   let remappedLang = Self.codeSwitchLanguageRemap[langName] {
                    Self.logger.info("Code-switch remap: \(langName) → \(remappedLang)")
                    cache = nil
                    let result = try generateText(
                        audioEmbeds: audioEmbeds,
                        textDecoder: textDecoder,
                        language: remappedLang,
                        prompt: prompt,
                        maxTokens: maxTokens
                    )
                    return TranscriptionResult(
                        text: result.text,
                        avgLogProb: result.avgLogProb,
                        tokenCount: result.tokenCount,
                        detectedLanguage: langName
                    )
                }
            }
            isCountingLogProb = true
        } else if isCountingLogProb && nextToken != Int32(tokens.eosTokenId) {
            let tokenProb = softmax(logits, axis: -1).reshaped(-1)[Int(nextToken)].item(Float.self)
            totalLogProb += log(Double(max(tokenProb, 1e-30)))
            logProbTokenCount += 1
        }
        generatedTokens.append(nextToken)

        for _ in 1..<maxTokens {
            if nextToken == Int32(tokens.eosTokenId) {
                break
            }

            let tokenEmbeds = textDecoder.embedTokens(MLXArray([nextToken]).expandedDimensions(axis: 0))
            (hiddenStates, newCache) = try textDecoder(inputsEmbeds: tokenEmbeds, cache: cache)
            cache = newCache

            let lastHiddenNext = hiddenStates[0..., (-1)..., .ellipsis]
            logits = textDecoder.embedTokens.asLinear(lastHiddenNext)
            nextToken = argMax(logits, axis: -1).squeezed().item(Int32.self)

            if nextToken == Int32(tokens.asrTextId) {
                // Code-switch remap (same check as first-token path)
                if language == nil, let tok = tokenizer {
                    let prefixText = tok.decode(tokens: generatedTokens.map { Int($0) })
                    if let langName = extractLanguageName(from: prefixText),
                       let remappedLang = Self.codeSwitchLanguageRemap[langName] {
                        Self.logger.info("Code-switch remap: \(langName) → \(remappedLang)")
                        cache = nil
                        let result = try generateText(
                            audioEmbeds: audioEmbeds,
                            textDecoder: textDecoder,
                            language: remappedLang,
                            prompt: prompt,
                            maxTokens: maxTokens
                        )
                        return TranscriptionResult(
                            text: result.text,
                            avgLogProb: result.avgLogProb,
                            tokenCount: result.tokenCount,
                            detectedLanguage: langName
                        )
                    }
                }
                isCountingLogProb = true
            } else if isCountingLogProb && nextToken != Int32(tokens.eosTokenId) {
                let tokenProb = softmax(logits, axis: -1).reshaped(-1)[Int(nextToken)].item(Float.self)
                totalLogProb += log(Double(max(tokenProb, 1e-30)))
                logProbTokenCount += 1
            }
            generatedTokens.append(nextToken)

            // Periodically force-evaluate the KV cache to materialize computation graph
            // and release intermediate MLXArray nodes, preventing GPU memory accumulation
            if generatedTokens.count % evalInterval == 0, let currentCache = cache {
                eval(currentCache.map { [$0.0, $0.1] }.flatMap { $0 })
            }
        }

        // Final eval to ensure all cache tensors are materialized before they go out of scope
        if let finalCache = cache {
            eval(finalCache.map { [$0.0, $0.1] }.flatMap { $0 })
        }

        let avgLogProb = logProbTokenCount > 0 ? totalLogProb / Double(logProbTokenCount) : 0.0

        guard let tokenizer = tokenizer else {
            return TranscriptionResult(
                text: generatedTokens.map { String($0) }.joined(separator: " "),
                avgLogProb: avgLogProb,
                tokenCount: logProbTokenCount,
                detectedLanguage: nil
            )
        }

        // Find <asr_text> marker by token ID (more reliable than string matching)
        let asrTokenId = Int32(tokens.asrTextId)
        let textTokens: [Int32]
        var detectedLang: String? = nil

        if let asrIndex = generatedTokens.firstIndex(of: asrTokenId) {
            // Extract only tokens after <asr_text>
            textTokens = Array(generatedTokens[(asrIndex + 1)...])
            // In auto mode, tokens before <asr_text> contain "language XXX"
            if language == nil {
                let prefixTokens = Array(generatedTokens[0..<asrIndex])
                let prefixText = tokenizer.decode(tokens: prefixTokens.map { Int($0) })
                detectedLang = extractLanguageName(from: prefixText)
            }
        } else if language == nil {
            // Auto mode: model may have generated "language XXX" prefix without <asr_text>
            // Fall back to string-based extraction
            let rawText = tokenizer.decode(tokens: generatedTokens.map { Int($0) })
            detectedLang = extractLanguageName(from: rawText)
            if let range = rawText.range(of: "<asr_text>") {
                return TranscriptionResult(
                    text: String(rawText[range.upperBound...]).trimmingCharacters(in: .whitespaces),
                    avgLogProb: avgLogProb,
                    tokenCount: logProbTokenCount,
                    detectedLanguage: detectedLang
                )
            }
            // Strip "language XXX" prefix if present
            if rawText.hasPrefix("language ") {
                let afterLang = rawText.dropFirst("language ".count)
                if let spaceIdx = afterLang.firstIndex(of: " ") {
                    return TranscriptionResult(
                        text: String(afterLang[afterLang.index(after: spaceIdx)...])
                            .trimmingCharacters(in: .whitespaces),
                        avgLogProb: avgLogProb,
                        tokenCount: logProbTokenCount,
                        detectedLanguage: detectedLang
                    )
                }
            }
            return TranscriptionResult(
                text: rawText,
                avgLogProb: avgLogProb,
                tokenCount: logProbTokenCount,
                detectedLanguage: detectedLang
            )
        } else {
            // Manual language mode: no prefix, all tokens are transcription
            textTokens = generatedTokens
        }

        // Filter out EOS token before decoding
        let filtered = textTokens.filter { $0 != Int32(tokens.eosTokenId) }
        return TranscriptionResult(
            text: tokenizer.decode(tokens: filtered.map { Int($0) })
                .trimmingCharacters(in: .whitespaces),
            avgLogProb: avgLogProb,
            tokenCount: logProbTokenCount,
            detectedLanguage: detectedLang
        )
    }
}
