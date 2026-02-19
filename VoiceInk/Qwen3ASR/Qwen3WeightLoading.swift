// Qwen3WeightLoading.swift
// Merged from qwen3-asr-swift WeightLoading.swift (ASR) + WeightLoading.swift (Common)
// Fixed: print() → os.Logger
// [AI-Claude: 2025-02-18]

import Foundation
import MLX
import MLXNN
import os

/// Weight loading errors
enum Qwen3WeightLoadingError: Error, LocalizedError {
    case noWeightsFound(URL)
    case incompatibleWeights(String)
    case missingRequiredWeight(String)

    var errorDescription: String? {
        switch self {
        case .noWeightsFound(let url):
            return "No safetensors files found in: \(url.path)"
        case .incompatibleWeights(let reason):
            return "Incompatible weights: \(reason)"
        case .missingRequiredWeight(let key):
            return "Missing required weight: \(key)"
        }
    }
}

/// Weight loading utilities for Qwen3-ASR
enum Qwen3WeightLoader {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "Qwen3WeightLoader")

    // MARK: - Safetensors Loading

    static func loadSafetensors(url: URL) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: url)
    }

    static func loadAllSafetensors(
        from directory: URL,
        prefix: String? = nil,
        stripPrefix: Bool = true
    ) throws -> [String: MLXArray] {
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }

        guard !safetensorFiles.isEmpty else {
            throw Qwen3WeightLoadingError.noWeightsFound(directory)
        }

        var allWeights: [String: MLXArray] = [:]
        for file in safetensorFiles {
            let weights = try loadSafetensors(url: file)
            allWeights.merge(weights) { _, new in new }
        }

        guard let prefix = prefix else { return allWeights }

        var filtered: [String: MLXArray] = [:]
        for (key, value) in allWeights {
            if key.hasPrefix(prefix) {
                let strippedKey = stripPrefix ? String(key.dropFirst(prefix.count)) : key
                filtered[strippedKey] = value
            }
        }
        return filtered
    }

    // MARK: - Audio Encoder Weight Loading

    static func loadAudioEncoderWeights(
        into audioEncoder: Qwen3AudioEncoder,
        from directory: URL
    ) throws {
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }

        guard !safetensorFiles.isEmpty else {
            throw Qwen3WeightLoadingError.noWeightsFound(directory)
        }

        logger.info("Found \(safetensorFiles.count) safetensor files")

        var allWeights: [String: MLXArray] = [:]
        for file in safetensorFiles {
            logger.debug("Loading: \(file.lastPathComponent)")
            let weights = try loadSafetensors(url: file)
            allWeights.merge(weights) { _, new in new }
        }

        logger.info("Loaded \(allWeights.count) weight tensors from files")

        var audioTowerWeights: [String: MLXArray] = [:]
        for (key, value) in allWeights {
            if key.hasPrefix("audio_tower.") {
                let strippedKey = String(key.dropFirst("audio_tower.".count))
                audioTowerWeights[strippedKey] = value
            }
        }

        logger.info("Found \(audioTowerWeights.count) audio_tower weights")

        applyConv2dWeights(to: audioEncoder.conv2d1, prefix: "conv2d1", from: audioTowerWeights)
        applyConv2dWeights(to: audioEncoder.conv2d2, prefix: "conv2d2", from: audioTowerWeights)
        applyConv2dWeights(to: audioEncoder.conv2d3, prefix: "conv2d3", from: audioTowerWeights)

        applyLinearWeights(to: audioEncoder.convOut, prefix: "conv_out", from: audioTowerWeights)
        applyLayerNormWeights(to: audioEncoder.lnPost, prefix: "ln_post", from: audioTowerWeights)

        applyLinearWeights(to: audioEncoder.proj1, prefix: "proj1", from: audioTowerWeights)
        applyLinearWeights(to: audioEncoder.proj2, prefix: "proj2", from: audioTowerWeights)

        for (index, layer) in audioEncoder.layers.enumerated() {
            let prefix = "layers.\(index)"
            applyEncoderLayerWeights(to: layer, prefix: prefix, from: audioTowerWeights)
        }

        logger.info("Applied weights to audio encoder (\(audioEncoder.layers.count) layers)")
    }

    // MARK: - Text Decoder Weight Loading

    static func loadTextDecoderWeights(
        into textModel: Qwen3QuantizedTextModel,
        from directory: URL
    ) throws {
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }

        guard !safetensorFiles.isEmpty else {
            throw Qwen3WeightLoadingError.noWeightsFound(directory)
        }

        var allWeights: [String: MLXArray] = [:]
        for file in safetensorFiles {
            let weights = try loadSafetensors(url: file)
            allWeights.merge(weights) { _, new in new }
        }

        var textWeights: [String: MLXArray] = [:]
        for (key, value) in allWeights {
            if key.hasPrefix("model.") {
                let strippedKey = String(key.dropFirst("model.".count))
                textWeights[strippedKey] = value
            }
        }

        logger.info("Found \(textWeights.count) text decoder weights")

        applyQuantizedEmbeddingWeights(
            to: textModel.embedTokens,
            prefix: "embed_tokens",
            from: textWeights
        )

        applyRMSNormWeights(to: textModel.norm, prefix: "norm", from: textWeights)

        for (index, layer) in textModel.layers.enumerated() {
            let prefix = "layers.\(index)"
            applyQuantizedDecoderLayerWeights(to: layer, prefix: prefix, from: textWeights)
        }

        logger.info("Applied weights to text decoder (\(textModel.layers.count) layers)")
    }

    // MARK: - Weight Application Helpers

    private static func applyConv2dWeights(
        to conv: Conv2d,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]
        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }
        if !params.isEmpty {
            conv.update(parameters: ModuleParameters(values: params))
        }
    }

    private static func applyEncoderLayerWeights(
        to layer: Qwen3AudioEncoderLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        applyLinearWeights(to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        applyLinearWeights(to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        applyLinearWeights(to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        applyLinearWeights(to: layer.selfAttn.outProj, prefix: "\(prefix).self_attn.out_proj", from: weights)
        applyLayerNormWeights(to: layer.selfAttnLayerNorm, prefix: "\(prefix).self_attn_layer_norm", from: weights)
        applyLayerNormWeights(to: layer.finalLayerNorm, prefix: "\(prefix).final_layer_norm", from: weights)
        applyLinearWeights(to: layer.fc1, prefix: "\(prefix).fc1", from: weights)
        applyLinearWeights(to: layer.fc2, prefix: "\(prefix).fc2", from: weights)
    }

    private static func applyQuantizedDecoderLayerWeights(
        to layer: Qwen3QuantizedTextDecoderLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        applyQuantizedLinearWeights(to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        applyQuantizedLinearWeights(to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        applyQuantizedLinearWeights(to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        applyQuantizedLinearWeights(to: layer.selfAttn.oProj, prefix: "\(prefix).self_attn.o_proj", from: weights)
        applyRMSNormWeights(to: layer.selfAttn.qNorm, prefix: "\(prefix).self_attn.q_norm", from: weights)
        applyRMSNormWeights(to: layer.selfAttn.kNorm, prefix: "\(prefix).self_attn.k_norm", from: weights)
        applyRMSNormWeights(to: layer.inputLayerNorm, prefix: "\(prefix).input_layernorm", from: weights)
        applyRMSNormWeights(to: layer.postAttentionLayerNorm, prefix: "\(prefix).post_attention_layernorm", from: weights)
        applyQuantizedLinearWeights(to: layer.mlp.gateProj, prefix: "\(prefix).mlp.gate_proj", from: weights)
        applyQuantizedLinearWeights(to: layer.mlp.upProj, prefix: "\(prefix).mlp.up_proj", from: weights)
        applyQuantizedLinearWeights(to: layer.mlp.downProj, prefix: "\(prefix).mlp.down_proj", from: weights)
    }

    // MARK: - Primitive Weight Helpers

    static func applyLinearWeights(
        to linear: Linear,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]
        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }
        if !params.isEmpty {
            linear.update(parameters: ModuleParameters(values: params))
        }
    }

    static func applyLayerNormWeights(
        to layerNorm: LayerNorm,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]
        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }
        if !params.isEmpty {
            layerNorm.update(parameters: ModuleParameters(values: params))
        }
    }

    static func applyRMSNormWeights(
        to norm: RMSNorm,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]
        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if !params.isEmpty {
            norm.update(parameters: ModuleParameters(values: params))
        }
    }

    static func applyQuantizedLinearWeights(
        to linear: QuantizedLinear,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]
        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if let scales = weights["\(prefix).scales"] {
            params["scales"] = .value(scales)
        }
        if let biases = weights["\(prefix).biases"] {
            params["biases"] = .value(biases)
        }
        if !params.isEmpty {
            linear.update(parameters: ModuleParameters(values: params))
        }
    }

    static func applyQuantizedEmbeddingWeights(
        to embedding: Qwen3PreQuantizedEmbedding,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]
        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if let scales = weights["\(prefix).scales"] {
            params["scales"] = .value(scales)
        }
        if let biases = weights["\(prefix).biases"] {
            params["biases"] = .value(biases)
        }
        if !params.isEmpty {
            embedding.update(parameters: ModuleParameters(values: params))
        }
    }
}
