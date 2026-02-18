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
    private static let logger = Logger(subsystem: "com.jasonchien.voco", category: "Qwen3ASRModel")

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
    ) throws -> String {
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
    ) throws -> String {
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

        var (hiddenStates, newCache) = try textDecoder(inputsEmbeds: inputEmbeds, cache: cache)
        cache = newCache

        let seqLen = hiddenStates.dim(1)
        let lastHidden = hiddenStates[0..., (seqLen-1)..<seqLen, 0...]
        var logits = textDecoder.embedTokens.asLinear(lastHidden)
        var nextToken = argMax(logits, axis: -1).squeezed().item(Int32.self)
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
            generatedTokens.append(nextToken)
        }

        guard let tokenizer = tokenizer else {
            return generatedTokens.map { String($0) }.joined(separator: " ")
        }

        // Find <asr_text> marker by token ID (more reliable than string matching)
        let asrTokenId = Int32(tokens.asrTextId)
        let textTokens: [Int32]
        if let asrIndex = generatedTokens.firstIndex(of: asrTokenId) {
            // Extract only tokens after <asr_text>
            textTokens = Array(generatedTokens[(asrIndex + 1)...])
        } else if language == nil {
            // Auto mode: model may have generated "language XXX" prefix without <asr_text>
            // Fall back to string-based extraction
            let rawText = tokenizer.decode(tokens: generatedTokens.map { Int($0) })
            if let range = rawText.range(of: "<asr_text>") {
                return String(rawText[range.upperBound...]).trimmingCharacters(in: .whitespaces)
            }
            // Strip "language XXX" prefix if present
            if rawText.hasPrefix("language ") {
                // Find end of language name (it's a single word like "Japanese", "English")
                let afterLang = rawText.dropFirst("language ".count)
                if let spaceIdx = afterLang.firstIndex(of: " ") {
                    return String(afterLang[afterLang.index(after: spaceIdx)...])
                        .trimmingCharacters(in: .whitespaces)
                }
            }
            return rawText
        } else {
            // Manual language mode: no prefix, all tokens are transcription
            textTokens = generatedTokens
        }

        // Filter out EOS token before decoding
        let filtered = textTokens.filter { $0 != Int32(tokens.eosTokenId) }
        return tokenizer.decode(tokens: filtered.map { Int($0) })
            .trimmingCharacters(in: .whitespaces)
    }
}
