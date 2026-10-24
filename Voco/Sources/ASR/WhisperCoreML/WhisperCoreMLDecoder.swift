// WhisperCoreMLDecoder.swift
// CoreML wrapper for Whisper decoder model with explicit KV cache management
// [AI-Claude: 2026-03-02]

import Foundation
import CoreML
import Accelerate
import os

enum WhisperCoreMLDecoderError: Error, LocalizedError {
    case modelNotLoaded
    case predictionFailed(String)
    case invalidCacheShape(String)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "CoreML decoder model not loaded"
        case .predictionFailed(let reason):
            return "CoreML decoder prediction failed: \(reason)"
        case .invalidCacheShape(let reason):
            return "Invalid KV cache shape: \(reason)"
        }
    }
}

/// KV cache state for a single decoder step
struct WhisperCoreMLKVCache {
    /// Self-attention KV per layer: [1, nHeads, maxCacheLen, dHead] Float16
    var selfKeys: [MLMultiArray]
    var selfValues: [MLMultiArray]
    /// Cross-attention KV per layer: [1, nHeads, 1500, dHead] Float16 (computed once)
    var crossKeys: [MLMultiArray]
    var crossValues: [MLMultiArray]
    /// Current write offset into self-attention cache
    var offset: Int
}

/// Wraps the WhisperDecoder.mlpackage CoreML model
class WhisperCoreMLDecoder {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "WhisperCoreMLDecoder")

    private var model: MLModel?
    private var config: WhisperCoreMLConfig?

    /// Load the decoder model
    func load(from directory: URL, config: WhisperCoreMLConfig) throws {
        self.config = config
        let compiledPath = directory.appendingPathComponent("WhisperDecoder.mlmodelc")
        let packagePath = directory.appendingPathComponent("WhisperDecoder.mlpackage")

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .all

        if FileManager.default.fileExists(atPath: compiledPath.path) {
            Self.logger.info("Loading pre-compiled decoder from mlmodelc")
            model = try MLModel(contentsOf: compiledPath, configuration: mlConfig)
        } else if FileManager.default.fileExists(atPath: packagePath.path) {
            Self.logger.info("Compiling decoder from mlpackage (first launch)")
            let compiledURL = try MLModel.compileModel(at: packagePath)
            let destURL = compiledPath
            try? FileManager.default.removeItem(at: destURL)
            try FileManager.default.moveItem(at: compiledURL, to: destURL)
            model = try MLModel(contentsOf: destURL, configuration: mlConfig)
        } else {
            throw WhisperCoreMLDecoderError.modelNotLoaded
        }

        Self.logger.info("CoreML decoder loaded")
    }

    /// Create a fresh KV cache (all zeros for self-attn, placeholder for cross-attn)
    func createEmptyCache() throws -> WhisperCoreMLKVCache {
        guard let config = config else {
            throw WhisperCoreMLDecoderError.modelNotLoaded
        }

        let nLayers = config.decoderLayers
        let nHeads = config.nHeads
        let dHead = config.dHead
        let maxCache = config.maxCacheLength

        var selfKeys: [MLMultiArray] = []
        var selfValues: [MLMultiArray] = []
        var crossKeys: [MLMultiArray] = []
        var crossValues: [MLMultiArray] = []

        for _ in 0..<nLayers {
            selfKeys.append(try MLMultiArray(
                shape: [1, nHeads as NSNumber, maxCache as NSNumber, dHead as NSNumber],
                dataType: .float16
            ))
            selfValues.append(try MLMultiArray(
                shape: [1, nHeads as NSNumber, maxCache as NSNumber, dHead as NSNumber],
                dataType: .float16
            ))
            // Cross-attn KV will be populated after first decode step
            crossKeys.append(try MLMultiArray(
                shape: [1, nHeads as NSNumber, 1500 as NSNumber, dHead as NSNumber],
                dataType: .float16
            ))
            crossValues.append(try MLMultiArray(
                shape: [1, nHeads as NSNumber, 1500 as NSNumber, dHead as NSNumber],
                dataType: .float16
            ))
        }

        return WhisperCoreMLKVCache(
            selfKeys: selfKeys,
            selfValues: selfValues,
            crossKeys: crossKeys,
            crossValues: crossValues,
            offset: 0
        )
    }

    /// Run a single decoder step
    /// - Parameters:
    ///   - tokenIds: MLMultiArray [1, seqLen] Int32
    ///   - encoderOutput: MLMultiArray [1, 1500, dModel] Float16
    ///   - cache: Current KV cache state
    /// - Returns: (logits [1, seqLen, vocabSize], updated cache)
    func decode(
        tokenIds: MLMultiArray,
        encoderOutput: MLMultiArray,
        cache: WhisperCoreMLKVCache
    ) throws -> (MLMultiArray, WhisperCoreMLKVCache) {
        guard let model = model, let config = config else {
            throw WhisperCoreMLDecoderError.modelNotLoaded
        }

        let nLayers = config.decoderLayers

        // Build input dictionary
        var inputDict: [String: MLFeatureValue] = [
            "token_ids": MLFeatureValue(multiArray: tokenIds),
            "encoder_output": MLFeatureValue(multiArray: encoderOutput),
            "cache_offset": MLFeatureValue(multiArray: try makeScalarInt32(cache.offset)),
        ]

        for i in 0..<nLayers {
            inputDict["self_attn_key_\(i)"] = MLFeatureValue(multiArray: cache.selfKeys[i])
            inputDict["self_attn_value_\(i)"] = MLFeatureValue(multiArray: cache.selfValues[i])
            inputDict["cross_attn_key_\(i)"] = MLFeatureValue(multiArray: cache.crossKeys[i])
            inputDict["cross_attn_value_\(i)"] = MLFeatureValue(multiArray: cache.crossValues[i])
        }

        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let output = try model.prediction(from: input)

        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw WhisperCoreMLDecoderError.predictionFailed("Missing logits in output")
        }

        // Extract updated self-attn KV cache
        var updatedCache = cache
        let seqLen = tokenIds.shape[1].intValue
        updatedCache.offset += seqLen

        for i in 0..<nLayers {
            guard let newSelfKey = output.featureValue(for: "new_self_attn_key_\(i)")?.multiArrayValue,
                  let newSelfValue = output.featureValue(for: "new_self_attn_value_\(i)")?.multiArrayValue else {
                throw WhisperCoreMLDecoderError.predictionFailed("Missing cache output for layer \(i)")
            }
            updatedCache.selfKeys[i] = newSelfKey
            updatedCache.selfValues[i] = newSelfValue
        }

        return (logits, updatedCache)
    }

    /// Create MLMultiArray with a single Int32 value
    private func makeScalarInt32(_ value: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1], dataType: .int32)
        array[0] = NSNumber(value: Int32(value))
        return array
    }

    func unload() {
        model = nil
        config = nil
    }
}
