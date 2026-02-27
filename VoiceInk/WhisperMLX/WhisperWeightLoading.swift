// WhisperWeightLoading.swift
// Load safetensors weights into WhisperMLX model
// [AI-Claude: 2026-02-27]

import Foundation
import MLX
import MLXNN
import os

enum WhisperWeightLoadingError: Error, LocalizedError {
    case noWeightsFound(URL)
    case unsupportedFormat(String)

    var errorDescription: String? {
        switch self {
        case .noWeightsFound(let url):
            return "No weight files found in: \(url.path)"
        case .unsupportedFormat(let reason):
            return "Unsupported weight format: \(reason)"
        }
    }
}

enum WhisperWeightLoader {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "WhisperWeightLoader")

    /// Load all weights from model directory (safetensors only)
    static func loadWeights(from directory: URL) throws -> [String: MLXArray] {
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)

        // Load safetensors files
        let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }
        if !safetensorFiles.isEmpty {
            var allWeights: [String: MLXArray] = [:]
            for file in safetensorFiles {
                let weights = try MLX.loadArrays(url: file)
                allWeights.merge(weights) { _, new in new }
            }
            return allWeights
        }

        // Check for NPZ (unsupported by MLX Swift)
        let npzFiles = contents.filter { $0.pathExtension == "npz" }
        if !npzFiles.isEmpty {
            throw WhisperWeightLoadingError.unsupportedFormat("NPZ format is not supported by MLX Swift, model needs safetensors format")
        }

        throw WhisperWeightLoadingError.noWeightsFound(directory)
    }

    // MARK: - Apply Weights to Model

    static func applyWeights(_ weights: [String: MLXArray], to encoder: WhisperAudioEncoder, config: WhisperMLXConfig) {
        // Encoder weights can be prefixed with "encoder." (HF format) or not (MLX format)
        let prefix = weights.keys.contains(where: { $0.hasPrefix("encoder.") }) ? "encoder." : ""
        let isQuantized = config.quantization != nil

        applyConv1dWeights(to: encoder.conv1, prefix: "\(prefix)conv1", from: weights)
        applyConv1dWeights(to: encoder.conv2, prefix: "\(prefix)conv2", from: weights)
        applyLayerNormWeights(to: encoder.lnPost, prefix: "\(prefix)ln_post", from: weights)

        for (i, block) in encoder.blocks.enumerated() {
            let blockPrefix = "\(prefix)blocks.\(i)"
            applyMaybeQuantizedLinearWeights(to: block.selfAttn.qProj, prefix: "\(blockPrefix).attn.query", alt: "\(blockPrefix).self_attn.q_proj", from: weights, quantized: isQuantized)
            applyMaybeQuantizedLinearWeights(to: block.selfAttn.kProj, prefix: "\(blockPrefix).attn.key", alt: "\(blockPrefix).self_attn.k_proj", from: weights, quantized: isQuantized)
            applyMaybeQuantizedLinearWeights(to: block.selfAttn.vProj, prefix: "\(blockPrefix).attn.value", alt: "\(blockPrefix).self_attn.v_proj", from: weights, quantized: isQuantized)
            applyMaybeQuantizedLinearWeights(to: block.selfAttn.outProj, prefix: "\(blockPrefix).attn.out", alt: "\(blockPrefix).self_attn.out_proj", from: weights, quantized: isQuantized)
            applyLayerNormWeights(to: block.selfAttnLayerNorm, prefix: "\(blockPrefix).attn_ln", alt: "\(blockPrefix).self_attn_layer_norm", from: weights)
            applyMaybeQuantizedLinearWeights(to: block.fc1, prefix: "\(blockPrefix).mlp1", alt: "\(blockPrefix).mlp.0", from: weights, quantized: isQuantized)
            applyMaybeQuantizedLinearWeights(to: block.fc2, prefix: "\(blockPrefix).mlp2", alt: "\(blockPrefix).mlp.2", from: weights, quantized: isQuantized)
            applyLayerNormWeights(to: block.finalLayerNorm, prefix: "\(blockPrefix).mlp_ln", alt: "\(blockPrefix).final_layer_norm", from: weights)
        }
    }

    static func applyWeights(_ weights: [String: MLXArray], to decoder: WhisperTextDecoder, config: WhisperMLXConfig) {
        let prefix = weights.keys.contains(where: { $0.hasPrefix("decoder.") }) ? "decoder." : ""
        let isQuantized = config.quantization != nil

        // Token embedding (including scales/biases for quantized)
        var embParams: [String: NestedItem<String, MLXArray>] = [:]
        if let w = weights["\(prefix)token_embedding.weight"] { embParams["weight"] = .value(w) }
        if let s = weights["\(prefix)token_embedding.scales"] { embParams["scales"] = .value(s) }
        if let b = weights["\(prefix)token_embedding.biases"] { embParams["biases"] = .value(b) }
        if !embParams.isEmpty {
            decoder.tokenEmbedding.update(parameters: ModuleParameters(values: embParams))
        }

        // Positional embedding — must use update(parameters:) for @ParameterInfo
        if let pe = weights["\(prefix)positional_embedding"] {
            decoder.update(parameters: ModuleParameters(values: ["positional_embedding": .value(pe)]))
        }

        // Layer norm
        applyLayerNormWeights(to: decoder.ln, prefix: "\(prefix)ln", from: weights)

        for (i, block) in decoder.blocks.enumerated() {
            let blockPrefix = "\(prefix)blocks.\(i)"

            // Self-attention
            applyMaybeQuantizedLinearWeights(to: block.selfAttn.qProj, prefix: "\(blockPrefix).attn.query", alt: "\(blockPrefix).self_attn.q_proj", from: weights, quantized: isQuantized)
            applyMaybeQuantizedLinearWeights(to: block.selfAttn.kProj, prefix: "\(blockPrefix).attn.key", alt: "\(blockPrefix).self_attn.k_proj", from: weights, quantized: isQuantized)
            applyMaybeQuantizedLinearWeights(to: block.selfAttn.vProj, prefix: "\(blockPrefix).attn.value", alt: "\(blockPrefix).self_attn.v_proj", from: weights, quantized: isQuantized)
            applyMaybeQuantizedLinearWeights(to: block.selfAttn.outProj, prefix: "\(blockPrefix).attn.out", alt: "\(blockPrefix).self_attn.out_proj", from: weights, quantized: isQuantized)
            applyLayerNormWeights(to: block.selfAttnLayerNorm, prefix: "\(blockPrefix).attn_ln", alt: "\(blockPrefix).self_attn_layer_norm", from: weights)

            // Cross-attention
            applyMaybeQuantizedLinearWeights(to: block.crossAttn.qProj, prefix: "\(blockPrefix).cross_attn.query", alt: "\(blockPrefix).encoder_attn.q_proj", from: weights, quantized: isQuantized)
            applyMaybeQuantizedLinearWeights(to: block.crossAttn.kProj, prefix: "\(blockPrefix).cross_attn.key", alt: "\(blockPrefix).encoder_attn.k_proj", from: weights, quantized: isQuantized)
            applyMaybeQuantizedLinearWeights(to: block.crossAttn.vProj, prefix: "\(blockPrefix).cross_attn.value", alt: "\(blockPrefix).encoder_attn.v_proj", from: weights, quantized: isQuantized)
            applyMaybeQuantizedLinearWeights(to: block.crossAttn.outProj, prefix: "\(blockPrefix).cross_attn.out", alt: "\(blockPrefix).encoder_attn.out_proj", from: weights, quantized: isQuantized)
            applyLayerNormWeights(to: block.crossAttnLayerNorm, prefix: "\(blockPrefix).cross_attn_ln", alt: "\(blockPrefix).encoder_attn_layer_norm", from: weights)

            // FFN
            applyMaybeQuantizedLinearWeights(to: block.fc1, prefix: "\(blockPrefix).mlp1", alt: "\(blockPrefix).mlp.0", from: weights, quantized: isQuantized)
            applyMaybeQuantizedLinearWeights(to: block.fc2, prefix: "\(blockPrefix).mlp2", alt: "\(blockPrefix).mlp.2", from: weights, quantized: isQuantized)
            applyLayerNormWeights(to: block.finalLayerNorm, prefix: "\(blockPrefix).mlp_ln", alt: "\(blockPrefix).final_layer_norm", from: weights)
        }
    }

    // MARK: - Primitive Helpers

    private static func applyConv1dWeights(to conv: Conv1d, prefix: String, from weights: [String: MLXArray]) {
        var params: [String: NestedItem<String, MLXArray>] = [:]
        if let w = weights["\(prefix).weight"] { params["weight"] = .value(w) }
        if let b = weights["\(prefix).bias"] { params["bias"] = .value(b) }
        if !params.isEmpty { conv.update(parameters: ModuleParameters(values: params)) }
    }

    private static func applyMaybeQuantizedLinearWeights(to linear: Linear, prefix: String, alt: String? = nil, from weights: [String: MLXArray], quantized: Bool) {
        let p = alt.flatMap { weights["\($0).weight"] != nil ? $0 : nil } ?? prefix
        var params: [String: NestedItem<String, MLXArray>] = [:]
        if let w = weights["\(p).weight"] { params["weight"] = .value(w) }
        if let b = weights["\(p).bias"] { params["bias"] = .value(b) }
        if quantized {
            if let s = weights["\(p).scales"] { params["scales"] = .value(s) }
            if let b = weights["\(p).biases"] { params["biases"] = .value(b) }
        }
        if !params.isEmpty { linear.update(parameters: ModuleParameters(values: params)) }
    }

    private static func applyLayerNormWeights(to ln: LayerNorm, prefix: String, alt: String? = nil, from weights: [String: MLXArray]) {
        let p = alt.flatMap { weights["\($0).weight"] != nil ? $0 : nil } ?? prefix
        var params: [String: NestedItem<String, MLXArray>] = [:]
        if let w = weights["\(p).weight"] { params["weight"] = .value(w) }
        if let b = weights["\(p).bias"] { params["bias"] = .value(b) }
        if !params.isEmpty { ln.update(parameters: ModuleParameters(values: params)) }
    }
}
