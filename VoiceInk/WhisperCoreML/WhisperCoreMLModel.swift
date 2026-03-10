// WhisperCoreMLModel.swift
// Complete CoreML Whisper model: encoder + decoder + language detection + greedy decoding
// [AI-Claude: 2026-03-02]

import Foundation
import CoreML
import Accelerate
import os

enum WhisperCoreMLModelError: Error, LocalizedError {
    case modelNotLoaded
    case loadFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "CoreML Whisper model not loaded"
        case .loadFailed(let reason):
            return "Failed to load CoreML Whisper model: \(reason)"
        }
    }
}

/// Complete CoreML Whisper model implementation
class WhisperCoreMLModelImpl {
    struct TranscriptionResult {
        let text: String
        let avgLogProb: Double
        let tokenCount: Int
        let detectedLanguage: String?
    }

    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "WhisperCoreMLModel")

    private var encoder: WhisperCoreMLEncoder?
    private var decoder: WhisperCoreMLDecoder?
    private var tokenizer: WhisperTokenizer?
    private var melProcessor: WhisperCoreMLMelSpectrogram?
    private var config: WhisperCoreMLConfig?

    func load(from directory: URL) throws {
        let config = try WhisperCoreMLConfig.load(from: directory)
        self.config = config

        self.melProcessor = WhisperCoreMLMelSpectrogram(nMels: config.numMelBins)

        let enc = WhisperCoreMLEncoder()
        try enc.load(from: directory)
        self.encoder = enc

        let dec = WhisperCoreMLDecoder()
        try dec.load(from: directory, config: config)
        self.decoder = dec

        let tok = WhisperTokenizer()
        try tok.load(from: directory)
        self.tokenizer = tok

        Self.logger.info("CoreML model loaded: dModel=\(config.dModel), enc=\(config.encoderLayers)L, dec=\(config.decoderLayers)L")
    }

    /// Transcribe audio samples
    func transcribe(
        audio: [Float],
        language: String? = nil,
        prompt: String? = nil
    ) throws -> TranscriptionResult {
        guard let encoder = encoder, let decoder = decoder,
              let tokenizer = tokenizer, let melProcessor = melProcessor,
              let config = config else {
            throw WhisperCoreMLModelError.modelNotLoaded
        }

        // 1. Audio → Mel spectrogram [1, nMels, 3000]
        let mel = try melProcessor.process(audio)

        // 2. Encode audio
        let encoderOutput = try encoder.encode(mel: mel)

        // 3. Language detection (if auto mode)
        let effectiveLanguage: String?
        let detectedLanguage: String?
        if language == nil || language == "auto" {
            let (lang, _) = try detectLanguage(
                encoderOutput: encoderOutput,
                decoder: decoder,
                config: config
            )
            effectiveLanguage = lang
            detectedLanguage = lang
        } else {
            effectiveLanguage = language
            detectedLanguage = nil
        }

        // 4. Build initial token sequence
        var initialTokens: [Int32] = [Int32(WhisperTokens.sotTokenId)]

        if let lang = effectiveLanguage,
           let langTokenId = WhisperTokens.languageTokenId(for: lang) {
            initialTokens.append(Int32(langTokenId))
        }

        initialTokens.append(Int32(WhisperTokens.transcribeTokenId))
        initialTokens.append(Int32(WhisperTokens.noTimestampsTokenId))

        if let prompt = prompt, !prompt.isEmpty {
            let promptTokens = tokenizer.encode(prompt)
            for t in promptTokens {
                initialTokens.append(Int32(t))
            }
        }

        // 5. Greedy decoding
        let maxTokens = config.maxTargetPositions - initialTokens.count
        let result = try greedyDecode(
            initialTokens: initialTokens,
            encoderOutput: encoderOutput,
            decoder: decoder,
            tokenizer: tokenizer,
            config: config,
            maxTokens: maxTokens
        )

        return result
    }

    // MARK: - Language Detection

    private func detectLanguage(
        encoderOutput: MLMultiArray,
        decoder: WhisperCoreMLDecoder,
        config: WhisperCoreMLConfig
    ) throws -> (String?, Float) {
        let sotToken = try makeTokenArray([Int32(WhisperTokens.sotTokenId)])
        var cache = try decoder.createEmptyCache()

        let (logits, _) = try decoder.decode(
            tokenIds: sotToken,
            encoderOutput: encoderOutput,
            cache: cache
        )

        // Extract language probabilities from last position logits
        let vocabSize = config.vocabSize
        let numLanguages = min(WhisperTokens.languageCount(nMels: config.numMelBins), vocabSize - WhisperTokens.firstLanguageTokenId)
        guard numLanguages > 0 else { return (nil, 0.0) }

        // Get logits for language tokens and compute softmax
        let langStart = WhisperTokens.firstLanguageTokenId
        var langLogits = [Float](repeating: 0, count: numLanguages)

        // logits shape: [1, 1, vocabSize]
        let logitsPtr = logits.dataPointer.bindMemory(to: Float16.self, capacity: vocabSize)
        for i in 0..<numLanguages {
            langLogits[i] = Float(logitsPtr[langStart + i])
        }

        // Softmax
        let probs = softmax(langLogits)

        var bestIdx = 0
        var bestProb: Float = probs[0]
        for i in 1..<numLanguages {
            if probs[i] > bestProb {
                bestProb = probs[i]
                bestIdx = i
            }
        }

        return (WhisperTokens.languageOrder[bestIdx], bestProb)
    }

    // MARK: - Greedy Decode

    private func greedyDecode(
        initialTokens: [Int32],
        encoderOutput: MLMultiArray,
        decoder: WhisperCoreMLDecoder,
        tokenizer: WhisperTokenizer,
        config: WhisperCoreMLConfig,
        maxTokens: Int
    ) throws -> TranscriptionResult {
        var cache = try decoder.createEmptyCache()

        // Prefill pass: all initial tokens at once
        let prefillTokens = try makeTokenArray(initialTokens)
        var (logits, updatedCache) = try decoder.decode(
            tokenIds: prefillTokens,
            encoderOutput: encoderOutput,
            cache: cache
        )
        cache = updatedCache

        var generatedTokens: [Int32] = []
        var totalLogProb: Double = 0.0
        var logProbTokenCount: Int = 0

        let vocabSize = config.vocabSize

        // Get first generated token from last logits position
        let seqLen = initialTokens.count
        var nextToken = extractArgmax(from: logits, position: seqLen - 1, vocabSize: vocabSize)

        if nextToken != Int32(WhisperTokens.eotTokenId) {
            let tokenProb = extractTokenProb(from: logits, position: seqLen - 1, tokenId: Int(nextToken), vocabSize: vocabSize)
            totalLogProb += log(Double(max(tokenProb, 1e-30)))
            logProbTokenCount += 1
        }
        generatedTokens.append(nextToken)

        // Autoregressive generation
        for _ in 1..<maxTokens {
            if nextToken == Int32(WhisperTokens.eotTokenId) { break }

            let tokenInput = try makeTokenArray([nextToken])
            (logits, cache) = try decoder.decode(
                tokenIds: tokenInput,
                encoderOutput: encoderOutput,
                cache: cache
            )

            nextToken = extractArgmax(from: logits, position: 0, vocabSize: vocabSize)

            if nextToken != Int32(WhisperTokens.eotTokenId) {
                let tokenProb = extractTokenProb(from: logits, position: 0, tokenId: Int(nextToken), vocabSize: vocabSize)
                totalLogProb += log(Double(max(tokenProb, 1e-30)))
                logProbTokenCount += 1
            }
            generatedTokens.append(nextToken)
        }

        // Filter out EOT and decode
        let filtered = generatedTokens.filter { $0 != Int32(WhisperTokens.eotTokenId) }
        let text = tokenizer.decode(tokens: filtered.map { Int($0) })
        let avgLogProb = logProbTokenCount > 0 ? totalLogProb / Double(logProbTokenCount) : 0.0

        // Detect language from initial tokens
        let detectedLang: String?
        if initialTokens.count >= 2 {
            let langTokenId = Int(initialTokens[1])
            detectedLang = WhisperTokens.languageCode(for: langTokenId)
        } else {
            detectedLang = nil
        }

        return TranscriptionResult(
            text: text,
            avgLogProb: avgLogProb,
            tokenCount: logProbTokenCount,
            detectedLanguage: detectedLang
        )
    }

    // MARK: - Helpers

    /// Create MLMultiArray [1, seqLen] Int32 from token array
    private func makeTokenArray(_ tokens: [Int32]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, tokens.count as NSNumber], dataType: .int32)
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: tokens.count)
        for i in 0..<tokens.count {
            ptr[i] = tokens[i]
        }
        return array
    }

    /// Extract argmax token from logits at a given position
    /// logits shape: [1, seqLen, vocabSize]
    private func extractArgmax(from logits: MLMultiArray, position: Int, vocabSize: Int) -> Int32 {
        let ptr = logits.dataPointer.bindMemory(to: Float16.self, capacity: logits.count)
        let offset = position * vocabSize

        var bestIdx: Int = 0
        var bestVal = Float(ptr[offset])
        for i in 1..<vocabSize {
            let val = Float(ptr[offset + i])
            if val > bestVal {
                bestVal = val
                bestIdx = i
            }
        }
        return Int32(bestIdx)
    }

    /// Extract probability for a specific token after softmax
    private func extractTokenProb(from logits: MLMultiArray, position: Int, tokenId: Int, vocabSize: Int) -> Float {
        let ptr = logits.dataPointer.bindMemory(to: Float16.self, capacity: logits.count)
        let offset = position * vocabSize

        // Copy logits to Float32 for numerical stability
        var floatLogits = [Float](repeating: 0, count: vocabSize)
        for i in 0..<vocabSize {
            floatLogits[i] = Float(ptr[offset + i])
        }

        let probs = softmax(floatLogits)
        return probs[tokenId]
    }

    /// Compute softmax over Float array
    private func softmax(_ x: [Float]) -> [Float] {
        let maxVal = x.max() ?? 0
        var expVals = [Float](repeating: 0, count: x.count)
        var sum: Float = 0

        for i in 0..<x.count {
            expVals[i] = exp(x[i] - maxVal)
            sum += expVals[i]
        }

        if sum > 0 {
            for i in 0..<x.count {
                expVals[i] /= sum
            }
        }

        return expVals
    }

    func unload() {
        encoder?.unload()
        decoder?.unload()
        encoder = nil
        decoder = nil
        tokenizer = nil
        melProcessor = nil
        config = nil
    }
}
