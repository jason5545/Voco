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

/// A word with its confidence score (0.0–1.0), covering ALL words in the transcription
struct WordConfidence {
    let word: String
    /// Confidence score derived from exp(mean token log-probability), clamped to 0.0–1.0
    let confidence: Float
}

/// Main Qwen3-ASR model for speech recognition
class Qwen3ASRModel {
    struct TranscriptionResult {
        let text: String
        let avgLogProb: Double
        let tokenCount: Int
        let detectedLanguage: String?  // auto 模式偵測到的語言（如 "Japanese"），手動指定時為 nil
        let uncertainWords: [UncertainWord]
        let wordConfidences: [WordConfidence]
    }

    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "Qwen3ASRModel")

    /// Language tags that cause English transliteration; remap to preserve code-switching
    private static let codeSwitchLanguageRemap: [String: String] = [
        "Chinese": "English",
    ]

    /// Count CJK characters in text
    private static func cjkCount(in text: String) -> Int {
        text.unicodeScalars.filter {
            (0x4E00...0x9FFF).contains($0.value) ||
            (0x3400...0x4DBF).contains($0.value) ||
            (0x3000...0x303F).contains($0.value)  // CJK punctuation
        }.count
    }

    /// Count Latin letter characters in text
    private static func latinCount(in text: String) -> Int {
        text.unicodeScalars.filter {
            $0.isASCII && CharacterSet.letters.contains($0)
        }.count
    }

    /// Map NLLanguage to the language names used by codeSwitchLanguageRemap
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
    private(set) var adapterMetadata: Qwen3ASRAdapterMetadata = .unavailable
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
    func load(from directory: URL, modelSize: Qwen3ASRModelSize, usesAudioAdapter: Bool = true) throws {
        // Load tokenizer
        let vocabPath = directory.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabPath.path) {
            let tok = Qwen3Tokenizer()
            try tok.load(from: vocabPath)
            self.tokenizer = tok
        }

        Self.logger.info("Loading audio encoder weights...")
        try Qwen3WeightLoader.loadAudioEncoderWeights(into: audioEncoder, from: directory)
        if usesAudioAdapter {
            adapterMetadata = Qwen3ASRAudioAdapterLoader.loadAndApplyIfPresent(
                modelDirectory: directory,
                audioEncoder: audioEncoder
            )
        } else {
            adapterMetadata = .unavailable
            Self.logger.info("Qwen3-ASR audio LoRA adapter skipped for this transcription context")
        }

        Self.logger.info("Loading text decoder weights...")
        let decoder = Qwen3QuantizedTextModel(config: textConfig)
        try Qwen3WeightLoader.loadTextDecoderWeights(into: decoder, from: directory)
        self.textDecoder = decoder

        Self.logger.info("Model loaded successfully")
    }

    func reloadAudioAdapter(from directory: URL) throws {
        Self.logger.info("Reloading Qwen3-ASR audio encoder weights for adapter refresh...")
        audioEncoder.clearPosEmbeddingCache()
        try Qwen3WeightLoader.loadAudioEncoderWeights(into: audioEncoder, from: directory)
        adapterMetadata = Qwen3ASRAudioAdapterLoader.loadAndApplyIfPresent(
            modelDirectory: directory,
            audioEncoder: audioEncoder
        )
        Memory.clearCache()
        Self.logger.info("Qwen3-ASR audio adapter refresh complete")
    }

    func reloadAudioEncoderBaseOnly(from directory: URL) throws {
        Self.logger.info("Reloading Qwen3-ASR base audio encoder weights for adapter guard...")
        audioEncoder.clearPosEmbeddingCache()
        try Qwen3WeightLoader.loadAudioEncoderWeights(into: audioEncoder, from: directory)
        adapterMetadata = .unavailable
        Memory.clearCache()
        Self.logger.info("Qwen3-ASR base audio encoder reload complete")
    }

    /// Transcribe audio to text
    func transcribe(
        audio: [Float],
        sampleRate: Int = 16000,
        language: String? = nil,
        prompt: String? = nil,
        maxTokens: Int? = nil,
        decodingOptions: Qwen3DecodingOptions = Qwen3DecodingOptions()
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
            maxTokens: effectiveMaxTokens,
            decodingOptions: decodingOptions
        )

        // Code-switch remap: detect language from output text using NLLanguageRecognizer.
        // If the dominant language transliterates English (e.g. Chinese), re-run with
        // a remapped tag to preserve code-switching.
        if language == nil {
            let detectedLang = Self.detectLanguage(from: result.text)
            let resultWithLang = TranscriptionResult(
                text: result.text,
                avgLogProb: result.avgLogProb,
                tokenCount: result.tokenCount,
                detectedLanguage: detectedLang,
                uncertainWords: result.uncertainWords,
                wordConfidences: result.wordConfidences
            )
            if let detectedLang,
               let remappedLang = Self.codeSwitchLanguageRemap[detectedLang] {
                Memory.clearCache()  // Release GPU buffers from the first pass
                Self.logger.info("Code-switch remap: \(detectedLang) → \(remappedLang)")
                let remapped = try generateText(
                    audioEmbeds: audioEmbeds,
                    textDecoder: textDecoder,
                    language: remappedLang,
                    prompt: prompt,
                    maxTokens: effectiveMaxTokens,
                    decodingOptions: decodingOptions
                )
                // Guard against translation: require that the remapped result
                // retains at least 70% of CJK characters AND actually gained Latin content.
                let originalCJK = Self.cjkCount(in: result.text)
                let remappedCJK = Self.cjkCount(in: remapped.text)
                let originalLatin = Self.latinCount(in: result.text)
                let remappedLatin = Self.latinCount(in: remapped.text)
                let cjkRetained = originalCJK == 0 || Double(remappedCJK) >= Double(originalCJK) * 0.7
                let latinGained = remappedLatin > originalLatin
                let lengthReasonable = remapped.text.count <= result.text.count * 2

                if cjkRetained && latinGained && lengthReasonable {
                    Self.logger.info("Code-switch remap accepted (CJK \(originalCJK)→\(remappedCJK), Latin \(originalLatin)→\(remappedLatin))")
                    return TranscriptionResult(
                        text: remapped.text,
                        avgLogProb: remapped.avgLogProb,
                        tokenCount: remapped.tokenCount,
                        detectedLanguage: detectedLang,
                        uncertainWords: remapped.uncertainWords,
                        wordConfidences: remapped.wordConfidences
                    )
                } else {
                    Self.logger.warning("Code-switch remap rejected (CJK \(originalCJK)→\(remappedCJK), Latin \(originalLatin)→\(remappedLatin), len \(result.text.count)→\(remapped.text.count)), using original")
                    return resultWithLang
                }
            }
            return resultWithLang
        }

        return result
    }

    /// Pick the next token from logits, optionally applying decoding
    /// modifications (repetition penalty, no-repeat n-gram, temperature).
    ///
    /// Fast path: when all options are at their defaults the function
    /// falls through to a single `argMax` on the GPU — zero extra cost.
    private static func pickNextToken(
        logits: MLXArray,
        generatedSoFar: [Int32],
        options: Qwen3DecodingOptions,
        hotwordTokenSequences: [[Int32]] = []
    ) -> Int32 {
        // Fast path — pure greedy, no modifications.
        if options.repetitionPenalty == 1.0,
           options.noRepeatNgramSize == 0,
           options.temperature == 0,
           options.hotwordBiasBoost == 0,
           hotwordTokenSequences.isEmpty,
           options.repeatNgramSize == 0 {
            return argMax(logits, axis: -1).squeezed().item(Int32.self)
        }

        // Pull logits to CPU for modification.
        let flat = logits.squeezed().asType(.float32)
        let vocabSize = flat.size
        var scores: [Float] = flat.asArray(Float.self)

        if options.hotwordBiasBoost != 0, !hotwordTokenSequences.isEmpty {
            applyHotwordBias(
                to: &scores,
                generatedSoFar: generatedSoFar,
                hotwordTokenSequences: hotwordTokenSequences,
                boost: options.hotwordBiasBoost
            )
        }

        // Repetition penalty: divide logits for already-generated tokens.
        if options.repetitionPenalty > 1.0 && !generatedSoFar.isEmpty {
            let penalty = options.repetitionPenalty
            for token in Set(generatedSoFar) {
                let idx = Int(token)
                guard idx >= 0, idx < vocabSize else { continue }
                let v = scores[idx]
                // Positive logits divide; negative logits multiply — matches
                // HuggingFace's implementation so the penalty always reduces
                // the probability of the repeated token.
                scores[idx] = v > 0 ? v / penalty : v * penalty
            }
        }

        // No-repeat-ngram: any next token whose emission would form a
        // repeated n-gram of size N gets pushed to -infinity.
        let n = options.noRepeatNgramSize
        if n > 0 && generatedSoFar.count >= n {
            let lastPrefix = Array(generatedSoFar.suffix(n - 1))
            for i in 0...(generatedSoFar.count - n) {
                let window = Array(generatedSoFar[i..<(i + n - 1)])
                guard window == lastPrefix else { continue }
                let forbidden = Int(generatedSoFar[i + n - 1])
                if forbidden >= 0, forbidden < vocabSize {
                    scores[forbidden] = -.infinity
                }
            }
        }

        applyRepeatNgramGuard(
            to: &scores,
            generatedSoFar: generatedSoFar,
            ngramSize: options.repeatNgramSize,
            maxCount: options.repeatNgramMaxCount
        )

        // Temperature sampling via Gumbel-max trick:
        // argmax(logits/T + Gumbel(0,1)) ~ categorical(softmax(logits/T)).
        if options.temperature > 0 {
            let t = options.temperature
            for i in 0..<vocabSize {
                let u = Float.random(in: 1e-6...1.0)
                scores[i] = scores[i] / t - Float.log(-Float.log(u))
            }
            // Find argmax of modified scores
            var bestIdx = 0
            var bestScore = scores[0]
            for i in 1..<vocabSize {
                if scores[i] > bestScore {
                    bestScore = scores[i]
                    bestIdx = i
                }
            }
            return Int32(bestIdx)
        }

        // Greedy with penalties applied
        var bestIdx = 0
        var bestScore = scores[0]
        for i in 1..<vocabSize {
            if scores[i] > bestScore {
                bestScore = scores[i]
                bestIdx = i
            }
        }
        return Int32(bestIdx)
    }

    private static func applyHotwordBias(
        to scores: inout [Float],
        generatedSoFar: [Int32],
        hotwordTokenSequences: [[Int32]],
        boost: Float
    ) {
        guard boost != 0 else { return }
        for tokenIds in hotwordTokenSequences where !tokenIds.isEmpty {
            let maxMatched = min(generatedSoFar.count, max(tokenIds.count - 1, 0))
            for matched in stride(from: maxMatched, through: 0, by: -1) {
                if matched > 0 {
                    let suffix = generatedSoFar.suffix(matched)
                    let prefix = tokenIds.prefix(matched)
                    guard Array(suffix) == Array(prefix) else { continue }
                }
                let nextIndex = matched
                guard nextIndex < tokenIds.count else { break }
                let token = Int(tokenIds[nextIndex])
                if token >= 0, token < scores.count {
                    scores[token] += boost
                }
                break
            }
        }
    }

    private static func applyRepeatNgramGuard(
        to scores: inout [Float],
        generatedSoFar: [Int32],
        ngramSize: Int,
        maxCount: Int
    ) {
        guard ngramSize > 0, maxCount > 0 else { return }
        guard generatedSoFar.count + 1 >= ngramSize else { return }
        let topCount = min(64, scores.count)
        guard topCount > 0 else { return }
        var topCandidateIndices: [Int] = []
        topCandidateIndices.reserveCapacity(topCount)
        for index in scores.indices {
            if topCandidateIndices.count < topCount {
                topCandidateIndices.append(index)
                if topCandidateIndices.count == topCount {
                    topCandidateIndices.sort { scores[$0] > scores[$1] }
                }
                continue
            }
            if let last = topCandidateIndices.last, scores[index] > scores[last] {
                topCandidateIndices.removeLast()
                let insertionIndex = topCandidateIndices.firstIndex { scores[index] > scores[$0] } ?? topCandidateIndices.endIndex
                topCandidateIndices.insert(index, at: insertionIndex)
            }
        }
        for token in topCandidateIndices {
            if wouldExceedNgramRepeat(
                generatedSoFar: generatedSoFar,
                nextToken: Int32(token),
                ngramSize: ngramSize,
                maxCount: maxCount
            ) {
                scores[token] = -.infinity
            }
        }
    }

    private static func wouldExceedNgramRepeat(
        generatedSoFar: [Int32],
        nextToken: Int32,
        ngramSize: Int,
        maxCount: Int
    ) -> Bool {
        guard ngramSize > 0, maxCount > 0 else { return false }
        let candidate = generatedSoFar + [nextToken]
        guard candidate.count >= ngramSize else { return false }
        let target = Array(candidate.suffix(ngramSize))
        var count = 0
        for start in 0...(candidate.count - ngramSize) {
            if Array(candidate[start..<(start + ngramSize)]) == target {
                count += 1
                if count > maxCount {
                    return true
                }
            }
        }
        return false
    }

    func generateText(
        audioEmbeds: MLXArray,
        textDecoder: Qwen3QuantizedTextModel,
        language: String?,
        prompt: String? = nil,
        maxTokens: Int,
        decodingOptions: Qwen3DecodingOptions = Qwen3DecodingOptions()
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

        var totalLogProb: Double = 0.0
        var logProbTokenCount: Int = 0
        var allTokenLogProbs: [(index: Int, tokenId: Int32, logProb: Double)] = []
        let hotwordTokenSequences: [[Int32]] = {
            guard let tokenizer, decodingOptions.hotwordBiasBoost != 0 else { return [] }
            var seen = Set<[Int32]>()
            var sequences: [[Int32]] = []
            for term in decodingOptions.hotwordBiasTerms {
                let trimmed = term.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { continue }
                let tokenIds = tokenizer.encode(trimmed).map { Int32($0) }
                guard !tokenIds.isEmpty, !seen.contains(tokenIds) else { continue }
                seen.insert(tokenIds)
                sequences.append(tokenIds)
            }
            return sequences
        }()

        // Extract log-probability for a token using logSumExp (avoids full softmax over 152K vocab)
        func collectLogProb(from logits: MLXArray, token: Int32) {
            let flatLogits = logits.reshaped(-1)
            let tokenLogit = flatLogits[Int(token)].item(Float.self)
            let lse = logSumExp(flatLogits).item(Float.self)
            let tokenLogProb = Double(tokenLogit - lse)
            totalLogProb += tokenLogProb
            logProbTokenCount += 1
            allTokenLogProbs.append((index: logProbTokenCount - 1, tokenId: token, logProb: tokenLogProb))
        }

        var (hiddenStates, newCache) = try textDecoder(inputsEmbeds: inputEmbeds, cache: cache)
        cache = newCache

        let seqLen = hiddenStates.dim(1)
        let lastHidden = hiddenStates[0..., (seqLen-1)..<seqLen, 0...]
        var logits = textDecoder.embedTokens.asLinear(lastHidden)

        var nextToken = Self.pickNextToken(
            logits: logits,
            generatedSoFar: generatedTokens,
            options: decodingOptions,
            hotwordTokenSequences: hotwordTokenSequences
        )

        if nextToken != Int32(tokens.eosTokenId) {
            collectLogProb(from: logits, token: nextToken)
        }
        generatedTokens.append(nextToken)

        var tokenIndex = 1
        while tokenIndex < maxTokens {
            if nextToken == Int32(tokens.eosTokenId) {
                break
            }

            let tokenEmbeds = textDecoder.embedTokens(MLXArray([nextToken]).expandedDimensions(axis: 0))
            (hiddenStates, newCache) = try textDecoder(inputsEmbeds: tokenEmbeds, cache: cache)
            cache = newCache

            let lastHiddenNext = hiddenStates[0..., (-1)..., .ellipsis]
            logits = textDecoder.embedTokens.asLinear(lastHiddenNext)
            nextToken = Self.pickNextToken(
                logits: logits,
                generatedSoFar: generatedTokens,
                options: decodingOptions,
                hotwordTokenSequences: hotwordTokenSequences
            )

            if nextToken != Int32(tokens.eosTokenId) {
                collectLogProb(from: logits, token: nextToken)
            }
            generatedTokens.append(nextToken)

            // Periodically force-evaluate the KV cache to materialize computation graph
            // and release intermediate MLXArray nodes, preventing GPU memory accumulation
            if generatedTokens.count % evalInterval == 0, let currentCache = cache {
                eval(currentCache.map { [$0.0, $0.1] }.flatMap { $0 } + [logits])
            }
            tokenIndex += 1
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
                uncertainWords: [],
                wordConfidences: []
            )
        }

        // All generated tokens are transcription text (no prefix to strip)
        // since <asr_text> is always in the input prompt.
        let filtered = generatedTokens.filter { $0 != Int32(tokens.eosTokenId) }
        let decodedTokens = decodeAllTokens(
            allTokenLogProbs: allTokenLogProbs,
            textTokens: filtered,
            tokenizer: tokenizer
        )
        let uncertainWords = buildUncertainWords(decodedTokens: decodedTokens)
        let wordConfidences = buildAllWordConfidences(decodedTokens: decodedTokens)
        return TranscriptionResult(
            text: tokenizer.decode(tokens: filtered.map { Int($0) })
                .trimmingCharacters(in: .whitespaces),
            avgLogProb: avgLogProb,
            tokenCount: logProbTokenCount,
            detectedLanguage: nil,  // Language detection done by caller via NLLanguageRecognizer
            uncertainWords: uncertainWords,
            wordConfidences: wordConfidences
        )
    }

    // MARK: - Uncertain Word Grouping

    private static let uncertaintyThreshold: Double = -1.0

    private struct DecodedToken {
        let index: Int
        let text: String  // decoded text with ▁ prefix stripped
        let logProb: Double
        let startsNewWord: Bool  // SentencePiece ▁ boundary
    }

    /// Decode all tokens once, building a shared representation for both
    /// uncertain word detection and per-word confidence scoring.
    private func decodeAllTokens(
        allTokenLogProbs: [(index: Int, tokenId: Int32, logProb: Double)],
        textTokens: [Int32],
        tokenizer: Qwen3Tokenizer
    ) -> [DecodedToken] {
        var logProbByIndex: [Int: Double] = [:]
        logProbByIndex.reserveCapacity(allTokenLogProbs.count)
        for entry in allTokenLogProbs {
            logProbByIndex[entry.index] = entry.logProb
        }

        return textTokens.enumerated().map { (i, tokenId) in
            let decoded = tokenizer.decode(tokens: [Int(tokenId)])
            let startsNewWord = decoded.hasPrefix("\u{2581}")
            let text = startsNewWord ? String(decoded.dropFirst()) : decoded
            return DecodedToken(
                index: i,
                text: text,
                logProb: logProbByIndex[i] ?? 0.0,
                startsNewWord: startsNewWord
            )
        }
    }

    /// Build UncertainWord list by grouping adjacent low-confidence tokens
    private func buildUncertainWords(decodedTokens: [DecodedToken]) -> [UncertainWord] {
        let lowConfTokens = decodedTokens.filter { $0.logProb < Self.uncertaintyThreshold }
        guard !lowConfTokens.isEmpty else { return [] }

        // Group adjacent low-confidence tokens
        var groups: [[DecodedToken]] = []
        var currentGroup: [DecodedToken] = []

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

            if Self.cjkCount(in: trimmed) > 4 { continue }

            let avgLogProb = group.map { $0.logProb }.reduce(0, +) / Double(group.count)
            words.append(UncertainWord(text: trimmed, logProb: avgLogProb))
        }

        return Array(words.sorted { $0.logProb < $1.logProb }.prefix(8))
    }

    // MARK: - Per-Word Confidence

    /// Build confidence scores for ALL words by grouping tokens at SentencePiece boundaries.
    private func buildAllWordConfidences(decodedTokens: [DecodedToken]) -> [WordConfidence] {
        guard !decodedTokens.isEmpty else { return [] }

        var words: [WordConfidence] = []
        var currentWord = ""
        var currentLogProbs: [Double] = []

        for token in decodedTokens {
            if token.startsNewWord && !currentWord.isEmpty {
                let meanLP = currentLogProbs.reduce(0, +) / Double(currentLogProbs.count)
                words.append(WordConfidence(word: currentWord, confidence: min(1.0, Float(exp(meanLP)))))
                currentWord = ""
                currentLogProbs = []
            }

            currentWord += token.text
            currentLogProbs.append(token.logProb)
        }

        if !currentWord.isEmpty {
            let meanLP = currentLogProbs.reduce(0, +) / Double(currentLogProbs.count)
            words.append(WordConfidence(word: currentWord, confidence: min(1.0, Float(exp(meanLP)))))
        }

        return words
    }
}
