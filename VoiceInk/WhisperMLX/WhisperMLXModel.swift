// WhisperMLXModel.swift
// Complete Whisper model: encoder + decoder + language detection + greedy decoding
// [AI-Claude: 2026-02-27]

import Foundation
import MLX
import MLXNN
import os

enum WhisperMLXModelError: Error, LocalizedError {
    case modelNotLoaded
    case loadFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Whisper MLX model not loaded"
        case .loadFailed(let reason):
            return "Failed to load Whisper MLX model: \(reason)"
        }
    }
}

/// Complete Whisper MLX model
class WhisperMLXModelImpl {
    struct TranscriptionResult {
        let text: String
        let avgLogProb: Double
        let tokenCount: Int
        let detectedLanguage: String?
    }

    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "WhisperMLXModel")

    var encoder: WhisperAudioEncoder?
    var decoder: WhisperTextDecoder?
    var tokenizer: WhisperTokenizer?
    var melProcessor: WhisperMelSpectrogram?
    var config: WhisperMLXConfig?

    func load(from directory: URL) throws {
        let config: WhisperMLXConfig
        do {
            config = try WhisperMLXConfig.load(from: directory)
        } catch {
            Self.logger.error("Config load failed: \(error)")
            throw error
        }
        self.config = config

        do {
            self.melProcessor = try WhisperMelSpectrogram(nMels: config.nMels)
        } catch {
            Self.logger.error("Mel spectrogram init failed: \(error)")
            throw error
        }

        let encoder = WhisperAudioEncoder(config: config)
        let decoder = WhisperTextDecoder(config: config)

        let weights: [String: MLXArray]
        do {
            weights = try WhisperWeightLoader.loadWeights(from: directory)
        } catch {
            Self.logger.error("Weight loading failed: \(error)")
            throw error
        }

        WhisperWeightLoader.applyWeights(weights, to: encoder, config: config)
        WhisperWeightLoader.applyWeights(weights, to: decoder, config: config)

        self.encoder = encoder
        self.decoder = decoder

        let tokenizer = WhisperTokenizer()
        do {
            try tokenizer.load(from: directory)
        } catch {
            Self.logger.error("Tokenizer load failed: \(error)")
            throw error
        }
        self.tokenizer = tokenizer

        Self.logger.info("Model loaded: nMels=\(config.nMels), encoder=\(config.nAudioLayer)L, decoder=\(config.nTextLayer)L, quantization=\(config.quantization?.bits ?? 0)bit")
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
            throw WhisperMLXModelError.modelNotLoaded
        }

        // 1. Audio → Mel spectrogram (padded to 30s)
        let mel = try melProcessor.process(audio)
        let batchedMel = mel.expandedDimensions(axis: 0)  // [1, nMels, nFrames]

        // 2. Encode audio
        let encoderOutput = encoder(batchedMel)
        eval(encoderOutput)  // Materialize encoder output

        // 3. Language detection (if auto mode)
        let effectiveLanguage: String?
        let detectedLanguage: String?
        if language == nil || language == "auto" {
            let (lang, _) = detectLanguage(encoderOutput: encoderOutput, decoder: decoder, config: config)
            effectiveLanguage = lang
            detectedLanguage = lang
        } else {
            effectiveLanguage = language
            detectedLanguage = nil
        }

        // 4. Build initial token sequence
        var initialTokens: [Int32] = [
            Int32(WhisperTokens.sotTokenId),
        ]

        // Language token
        if let lang = effectiveLanguage,
           let langTokenId = WhisperTokens.languageTokenId(for: lang) {
            initialTokens.append(Int32(langTokenId))
        }

        // Task token (always transcribe, not translate)
        initialTokens.append(Int32(WhisperTokens.transcribeTokenId))

        // No timestamps
        initialTokens.append(Int32(WhisperTokens.noTimestampsTokenId))

        // Prompt tokens (if any)
        if let prompt = prompt, !prompt.isEmpty {
            let promptTokens = tokenizer.encode(prompt)
            for t in promptTokens {
                initialTokens.append(Int32(t))
            }
        }

        // 5. Greedy decoding
        let maxTokens = config.nTextCtx - initialTokens.count
        let result = greedyDecode(
            initialTokens: initialTokens,
            encoderOutput: encoderOutput,
            decoder: decoder,
            tokenizer: tokenizer,
            maxTokens: maxTokens
        )

        return result
    }

    // MARK: - Language Detection

    private func detectLanguage(
        encoderOutput: MLXArray,
        decoder: WhisperTextDecoder,
        config: WhisperMLXConfig
    ) -> (String?, Float) {
        // Feed SOT token, get logits, check language token probabilities
        let sotTokens = MLXArray([Int32(WhisperTokens.sotTokenId)]).expandedDimensions(axis: 0)

        let (logits, _) = decoder(tokenIds: sotTokens, encoderOutput: encoderOutput, cache: nil)
        let lastLogits = logits[0, -1]  // [nVocab]

        // Get probabilities for language tokens only
        let numLanguages = min(WhisperTokens.languageOrder.count, config.nVocab - WhisperTokens.firstLanguageTokenId)
        guard numLanguages > 0 else { return (nil, 0.0) }

        let langStart = WhisperTokens.firstLanguageTokenId
        let langEnd = langStart + numLanguages
        let langLogits = lastLogits[langStart..<langEnd]
        let langProbs = softmax(langLogits, axis: 0)

        let bestIdx = argMax(langProbs, axis: 0).item(Int.self)
        let bestProb = langProbs[bestIdx].item(Float.self)
        let bestLang = WhisperTokens.languageOrder[bestIdx]

        return (bestLang, bestProb)
    }

    // MARK: - Greedy Decode

    private func greedyDecode(
        initialTokens: [Int32],
        encoderOutput: MLXArray,
        decoder: WhisperTextDecoder,
        tokenizer: WhisperTokenizer,
        maxTokens: Int
    ) -> TranscriptionResult {
        let tokens = MLXArray(initialTokens).expandedDimensions(axis: 0)

        // Initial prefill pass with all initial tokens
        var (logits, cache) = decoder(tokenIds: tokens, encoderOutput: encoderOutput, cache: nil)

        var generatedTokens: [Int32] = []
        var totalLogProb: Double = 0.0
        var logProbTokenCount: Int = 0

        // Get first generated token from last logits position
        var nextToken = argMax(logits[0, -1], axis: 0).item(Int32.self)
        if nextToken != Int32(WhisperTokens.eotTokenId) {
            let tokenProb = softmax(logits[0, -1], axis: 0)[Int(nextToken)].item(Float.self)
            totalLogProb += log(Double(max(tokenProb, 1e-30)))
            logProbTokenCount += 1
        }
        generatedTokens.append(nextToken)

        let evalInterval = 50

        // Autoregressive generation
        for _ in 1..<maxTokens {
            if nextToken == Int32(WhisperTokens.eotTokenId) { break }

            let tokenInput = MLXArray([nextToken]).expandedDimensions(axis: 0)
            (logits, cache) = decoder(tokenIds: tokenInput, encoderOutput: encoderOutput, cache: cache)

            nextToken = argMax(logits[0, -1], axis: 0).item(Int32.self)

            if nextToken != Int32(WhisperTokens.eotTokenId) {
                let tokenProb = softmax(logits[0, -1], axis: 0)[Int(nextToken)].item(Float.self)
                totalLogProb += log(Double(max(tokenProb, 1e-30)))
                logProbTokenCount += 1
            }
            generatedTokens.append(nextToken)

            // Periodically force-evaluate to prevent computation graph accumulation
            if generatedTokens.count % evalInterval == 0 {
                var tensorsToEval: [MLXArray] = [logits]
                for blockCache in cache {
                    if let sa = blockCache.selfAttnCache {
                        tensorsToEval.append(contentsOf: [sa.0, sa.1])
                    }
                    if let ca = blockCache.crossAttnCache {
                        tensorsToEval.append(contentsOf: [ca.0, ca.1])
                    }
                }
                eval(tensorsToEval)
            }
        }

        // Final eval
        var tensorsToEval: [MLXArray] = [logits]
        for blockCache in cache {
            if let sa = blockCache.selfAttnCache {
                tensorsToEval.append(contentsOf: [sa.0, sa.1])
            }
            if let ca = blockCache.crossAttnCache {
                tensorsToEval.append(contentsOf: [ca.0, ca.1])
            }
        }
        eval(tensorsToEval)

        // Filter out EOT and decode
        let filtered = generatedTokens.filter { $0 != Int32(WhisperTokens.eotTokenId) }
        let text = tokenizer.decode(tokens: filtered.map { Int($0) })
        let avgLogProb = logProbTokenCount > 0 ? totalLogProb / Double(logProbTokenCount) : 0.0

        // Detect language from initial tokens if we set one
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
}
