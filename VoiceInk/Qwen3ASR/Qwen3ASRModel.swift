// Qwen3ASRModel.swift
// Adapted from qwen3-asr-swift Qwen3ASR.swift
// Removed: fromPretrained(), backward-compat extensions
// Added: load(from:modelSize:), transcribe() throws
// [AI-Claude: 2025-02-18]

import Foundation
import MLX
import MLXNN
import MLXFast
import NaturalLanguage
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

/// A word (or merged subtoken group) with low ASR confidence
struct UncertainWord {
    let text: String       // 解碼後的文字（已合併 subtoken）
    let logProb: Double    // 該詞彙的平均 logProb
}

/// Main Qwen3-ASR model for speech recognition
class Qwen3ASRModel {
    struct TranscriptionResult {
        let text: String
        let avgLogProb: Double
        let tokenCount: Int
        let detectedLanguage: String?  // auto 模式偵測到的語言（如 "Japanese"），手動指定時為 nil
        let uncertainWords: [UncertainWord]
    }

    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "Qwen3ASRModel")

    /// Map NLLanguage to the language names used for detected language reporting
    private static let nlLanguageToName: [NLLanguage: String] = [
        .simplifiedChinese: "Chinese",
        .traditionalChinese: "Chinese",
        .japanese: "Japanese",
        .english: "English",
        .korean: "Korean",
    ]

    /// Detect dominant language from transcription text using NLLanguageRecognizer
    private static func detectLanguage(from text: String) -> String? {
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        guard let lang = recognizer.dominantLanguage else { return nil }
        return nlLanguageToName[lang] ?? lang.rawValue
    }

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
        eval(audioEmbeds)  // Materialize audio encoder output, sever computation graph

        guard let textDecoder = textDecoder else {
            throw Qwen3ASRModelError.textDecoderNotLoaded
        }

        let result = try generateText(
            audioEmbeds: audioEmbeds,
            textDecoder: textDecoder,
            language: language,
            prompt: prompt,
            maxTokens: effectiveMaxTokens
        )

        // Auto-detect: report detected language via NLLanguageRecognizer
        if language == nil {
            let detectedLang = Self.detectLanguage(from: result.text)
            return TranscriptionResult(
                text: result.text,
                avgLogProb: result.avgLogProb,
                tokenCount: result.tokenCount,
                detectedLanguage: detectedLang,
                uncertainWords: result.uncertainWords
            )
        }

        return result
    }

    func generateText(
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

        // Add language hint if specified, then always add <asr_text> marker.
        // <asr_text> forces the model into transcription mode — without it,
        // the model may translate or produce non-ASR output.
        if let lang = language, let tokenizer = tokenizer {
            let langPrefix = "language \(lang)"
            let langTokens = tokenizer.encode(langPrefix)
            inputIds.append(contentsOf: langTokens.map { Int32($0) })
        }
        inputIds.append(Int32(tokens.asrTextId))

        let inputIdsTensor = MLXArray(inputIds).expandedDimensions(axis: 0)
        var inputEmbeds = textDecoder.embedTokens(inputIdsTensor)

        let audioEmbedsTyped = audioEmbeds.asType(inputEmbeds.dtype)
        let beforeAudio = inputEmbeds[0..., 0..<audioStartIndex, 0...]
        let afterAudio = inputEmbeds[0..., audioEndIndex..., 0...]

        inputEmbeds = concatenated([beforeAudio, audioEmbedsTyped, afterAudio], axis: 1)

        var cache: [(MLXArray, MLXArray)]? = nil
        var generatedTokens: [Int32] = []
        let evalInterval = 50  // Force MLX evaluation every N tokens to prevent computation graph accumulation

        // Per-token logprob tracking — <asr_text> is always in inputIds,
        // so all generated tokens are text tokens; start counting immediately.
        var totalLogProb: Double = 0.0
        var logProbTokenCount: Int = 0

        // Low-confidence token tracking
        let uncertaintyThreshold: Double = -1.0
        var tokenLogProbs: [(index: Int, tokenId: Int32, logProb: Double)] = []

        var (hiddenStates, newCache) = try textDecoder(inputsEmbeds: inputEmbeds, cache: cache)
        cache = newCache

        let seqLen = hiddenStates.dim(1)
        let lastHidden = hiddenStates[0..., (seqLen-1)..<seqLen, 0...]
        var logits = textDecoder.embedTokens.asLinear(lastHidden)
        var nextToken = argMax(logits, axis: -1).squeezed().item(Int32.self)

        if nextToken != Int32(tokens.eosTokenId) {
            let tokenProb = softmax(logits, axis: -1).reshaped(-1)[Int(nextToken)].item(Float.self)
            let tokenLogProb = log(Double(max(tokenProb, 1e-30)))
            totalLogProb += tokenLogProb
            logProbTokenCount += 1
            if tokenLogProb < uncertaintyThreshold {
                tokenLogProbs.append((index: logProbTokenCount - 1, tokenId: nextToken, logProb: tokenLogProb))
            }
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

            if nextToken != Int32(tokens.eosTokenId) {
                let tokenProb = softmax(logits, axis: -1).reshaped(-1)[Int(nextToken)].item(Float.self)
                let tokenLogProb = log(Double(max(tokenProb, 1e-30)))
                totalLogProb += tokenLogProb
                logProbTokenCount += 1
                if tokenLogProb < uncertaintyThreshold {
                    tokenLogProbs.append((index: logProbTokenCount - 1, tokenId: nextToken, logProb: tokenLogProb))
                }
            }
            generatedTokens.append(nextToken)

            // Periodically force-evaluate the KV cache to materialize computation graph
            // and release intermediate MLXArray nodes, preventing GPU memory accumulation
            if generatedTokens.count % evalInterval == 0, let currentCache = cache {
                eval(currentCache.map { [$0.0, $0.1] }.flatMap { $0 } + [logits])
            }
        }

        // Final eval to ensure all cache tensors are materialized before they go out of scope
        if let finalCache = cache {
            eval(finalCache.map { [$0.0, $0.1] }.flatMap { $0 } + [logits])
        }

        let avgLogProb = logProbTokenCount > 0 ? totalLogProb / Double(logProbTokenCount) : 0.0

        guard let tokenizer = tokenizer else {
            return TranscriptionResult(
                text: generatedTokens.map { String($0) }.joined(separator: " "),
                avgLogProb: avgLogProb,
                tokenCount: logProbTokenCount,
                detectedLanguage: nil,
                uncertainWords: []
            )
        }

        // All generated tokens are transcription text (no prefix to strip)
        // since <asr_text> is always in the input prompt.
        let filtered = generatedTokens.filter { $0 != Int32(tokens.eosTokenId) }
        let uncertainWords = buildUncertainWords(
            tokenLogProbs: tokenLogProbs,
            textTokens: filtered,
            tokenizer: tokenizer
        )
        return TranscriptionResult(
            text: tokenizer.decode(tokens: filtered.map { Int($0) })
                .trimmingCharacters(in: .whitespaces),
            avgLogProb: avgLogProb,
            tokenCount: logProbTokenCount,
            detectedLanguage: nil,  // Language detection done by caller via NLLanguageRecognizer
            uncertainWords: uncertainWords
        )
    }

    // MARK: - Uncertain Word Grouping

    /// Build UncertainWord list by grouping adjacent low-confidence tokens
    func buildUncertainWords(
        tokenLogProbs: [(index: Int, tokenId: Int32, logProb: Double)],
        textTokens: [Int32],
        tokenizer: Qwen3Tokenizer
    ) -> [UncertainWord] {
        guard !tokenLogProbs.isEmpty, !textTokens.isEmpty else { return [] }

        // Build decoded text for each text token
        let tokenTexts: [(index: Int, text: String, logProb: Double?)] = textTokens.enumerated().map { (i, tokenId) in
            let decoded = tokenizer.decode(tokens: [Int(tokenId)])
            let lp = tokenLogProbs.first(where: { $0.index == i })?.logProb
            return (i, decoded, lp)
        }

        // Collect only low-confidence tokens
        let lowConfTokens = tokenTexts.compactMap { entry -> (index: Int, text: String, logProb: Double)? in
            guard let lp = entry.logProb else { return nil }
            return (entry.index, entry.text, lp)
        }

        guard !lowConfTokens.isEmpty else { return [] }

        // Group adjacent low-confidence tokens
        var groups: [[(index: Int, text: String, logProb: Double)]] = []
        var currentGroup: [(index: Int, text: String, logProb: Double)] = []

        for token in lowConfTokens {
            if let last = currentGroup.last {
                if token.index == last.index + 1 {
                    currentGroup.append(token)
                } else {
                    groups.append(currentGroup)
                    currentGroup = [token]
                }
            } else {
                currentGroup = [token]
            }
        }
        if !currentGroup.isEmpty {
            groups.append(currentGroup)
        }

        // Convert groups to UncertainWord, applying CJK/Latin merge limits
        var words: [UncertainWord] = []
        for group in groups {
            let mergedText = group.map { $0.text }.joined()
            let trimmed = mergedText.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }

            // CJK: max 4 chars per group
            let cjkCount = trimmed.unicodeScalars.filter {
                (0x4E00...0x9FFF).contains($0.value) || (0x3400...0x4DBF).contains($0.value)
            }.count
            if cjkCount > 4 { continue }  // Skip overly long groups

            let avgLogProb = group.map { $0.logProb }.reduce(0, +) / Double(group.count)
            words.append(UncertainWord(text: trimmed, logProb: avgLogProb))
        }

        // Sort by logProb (lowest first) and take top 8
        return Array(words.sorted { $0.logProb < $1.logProb }.prefix(8))
    }
}
