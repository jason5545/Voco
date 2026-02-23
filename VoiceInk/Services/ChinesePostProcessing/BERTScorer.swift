import CoreML
import Foundation
import os

/// Singleton responsible for BERT MLM inference.
///
/// Loads a Core ML `bert-base-chinese` model and `vocab.txt`,
/// provides character-level tokenization and masked-language-model scoring
/// for the Homophone / Nasal correction engines.
///
/// Usage:
/// ```
/// await BERTScorer.shared.loadModel()
/// let delta = BERTScorer.shared.scoreWordReplacement(
///     text: "我在銀幕上看電影",
///     wordOffset: 2, originalWord: "銀幕", candidateWord: "螢幕"
/// )
/// ```
final class BERTScorer: @unchecked Sendable {
    static let shared = BERTScorer()

    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "BERTScorer")

    // MARK: - Vocabulary

    private var tokenToId: [String: Int] = [:]
    private var idToToken: [Int: String] = [:]

    // Special token IDs (bert-base-chinese defaults)
    private let clsTokenId = 101    // [CLS]
    private let sepTokenId = 102    // [SEP]
    private let maskTokenId = 103   // [MASK]
    private let padTokenId = 0      // [PAD]
    private let unkTokenId = 100    // [UNK]

    // MARK: - Model

    private var model: MLModel?
    private let lock = NSLock()

    /// Whether the BERT model is loaded and ready for inference.
    var isLoaded: Bool {
        lock.lock()
        defer { lock.unlock() }
        return model != nil
    }

    // MARK: - Lifecycle

    /// Load vocabulary and Core ML model from the app bundle.
    func loadModel() async {
        guard let vocabURL = BERTModelManager.vocabURL,
              let modelcDir = BERTModelManager.modelURL else {
            logger.warning("BERT model files not found in app bundle")
            return
        }

        // Load vocabulary
        do {
            let vocabContent = try String(contentsOf: vocabURL, encoding: .utf8)
            let lines = vocabContent.components(separatedBy: .newlines)
            var t2i: [String: Int] = [:]
            var i2t: [Int: String] = [:]
            t2i.reserveCapacity(lines.count)
            i2t.reserveCapacity(lines.count)
            for (id, token) in lines.enumerated() {
                guard !token.isEmpty else { continue }
                t2i[token] = id
                i2t[id] = token
            }

            lock.lock()
            self.tokenToId = t2i
            self.idToToken = i2t
            lock.unlock()

            logger.info("Loaded BERT vocabulary: \(t2i.count) tokens")
        } catch {
            logger.error("Failed to load vocab.txt: \(error.localizedDescription)")
            return
        }

        // Load Core ML model
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all  // Prefer Neural Engine
            let loadedModel = try MLModel(contentsOf: modelcDir, configuration: config)

            lock.lock()
            self.model = loadedModel
            lock.unlock()

            logger.info("BERT Core ML model loaded successfully")
        } catch {
            logger.error("Failed to load BERT Core ML model: \(error.localizedDescription)")
        }
    }

    /// Unload model and free memory.
    func unloadModel() {
        lock.lock()
        model = nil
        tokenToId = [:]
        idToToken = [:]
        lock.unlock()
        logger.info("BERT model unloaded")
    }

    // MARK: - Tokenization

    /// Character-level tokenization for Chinese BERT.
    /// Each CJK character maps to one token. Non-CJK characters use UNK.
    /// Returns `[CLS] + char_tokens + [SEP]`.
    private func tokenize(_ text: String) -> [Int] {
        var ids = [clsTokenId]
        ids.reserveCapacity(text.count + 2)
        for char in text {
            let key = String(char)
            ids.append(tokenToId[key] ?? unkTokenId)
        }
        ids.append(sepTokenId)
        return ids
    }

    // MARK: - Single Position Scoring

    /// Score a single character replacement at a given position.
    ///
    /// Masks the position in the input, runs BERT forward pass,
    /// and returns `candidateLogit - originalLogit`.
    ///
    /// - Parameters:
    ///   - text: The full sentence text.
    ///   - position: Character offset in `text` (0-based).
    ///   - originalChar: The character currently at this position.
    ///   - candidateChar: The proposed replacement character.
    /// - Returns: Logit difference, or `nil` if inference fails.
    func score(text: String, position: Int, originalChar: Character, candidateChar: Character) -> Double? {
        lock.lock()
        let currentModel = model
        let t2i = tokenToId
        lock.unlock()

        guard let mlModel = currentModel else { return nil }

        // Build token IDs
        var tokenIds = tokenize(text)
        let maskedIndex = position + 1  // +1 for [CLS]

        guard maskedIndex > 0, maskedIndex < tokenIds.count - 1 else { return nil }

        // Get original and candidate token IDs
        let origKey = String(originalChar)
        let candKey = String(candidateChar)
        guard let origTokenId = t2i[origKey], let candTokenId = t2i[candKey] else { return nil }

        // Mask the position
        tokenIds[maskedIndex] = maskTokenId

        // Build attention mask (all 1s, no padding)
        let attentionMask = [Int32](repeating: 1, count: tokenIds.count)
        let inputIds = tokenIds.map { Int32($0) }

        // Run inference
        do {
            let inputIdsArray = try MLMultiArray(shape: [1, NSNumber(value: inputIds.count)], dataType: .int32)
            let attMaskArray = try MLMultiArray(shape: [1, NSNumber(value: attentionMask.count)], dataType: .int32)

            for (i, val) in inputIds.enumerated() {
                inputIdsArray[[0, NSNumber(value: i)]] = NSNumber(value: val)
            }
            for (i, val) in attentionMask.enumerated() {
                attMaskArray[[0, NSNumber(value: i)]] = NSNumber(value: val)
            }

            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputIdsArray),
                "attention_mask": MLFeatureValue(multiArray: attMaskArray),
            ])

            let output = try mlModel.prediction(from: provider)

            // Extract logits at the masked position
            guard let logitsValue = output.featureValue(for: "logits"),
                  let logits = logitsValue.multiArrayValue else {
                logger.warning("BERT output missing 'logits'")
                return nil
            }

            // logits shape: [1, seq_len, vocab_size]
            let vocabSize = logits.shape[2].intValue
            let baseOffset = maskedIndex * vocabSize

            let origLogit = logits[baseOffset + origTokenId].doubleValue
            let candLogit = logits[baseOffset + candTokenId].doubleValue

            return candLogit - origLogit
        } catch {
            logger.error("BERT inference failed: \(error.localizedDescription)")
            return nil
        }
    }

    // MARK: - Word Replacement Scoring

    /// Score a multi-character word replacement by summing per-character logit diffs.
    ///
    /// For each character position where original and candidate differ,
    /// runs a separate masked prediction and sums the logit differences.
    ///
    /// - Parameters:
    ///   - text: The full sentence text.
    ///   - wordOffset: Character offset of the word in `text`.
    ///   - originalWord: The original word.
    ///   - candidateWord: The proposed replacement word.
    /// - Returns: Sum of logit diffs for changed positions, or `nil` if any inference fails.
    func scoreWordReplacement(text: String, wordOffset: Int, originalWord: String, candidateWord: String) -> Double? {
        let origChars = Array(originalWord)
        let candChars = Array(candidateWord)

        guard origChars.count == candChars.count else { return nil }

        var totalScore: Double = 0
        var scoredPositions = 0

        for i in 0..<origChars.count {
            // Only score positions that actually change
            guard origChars[i] != candChars[i] else { continue }

            guard let posScore = score(
                text: text,
                position: wordOffset + i,
                originalChar: origChars[i],
                candidateChar: candChars[i]
            ) else {
                return nil  // If any position fails, return nil for fallback
            }

            totalScore += posScore
            scoredPositions += 1
        }

        // If no positions changed (shouldn't happen), return nil
        guard scoredPositions > 0 else { return nil }

        return totalScore
    }
}
