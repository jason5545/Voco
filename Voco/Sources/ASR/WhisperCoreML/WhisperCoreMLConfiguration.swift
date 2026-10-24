// WhisperCoreMLConfiguration.swift
// Configuration for CoreML Whisper models
// [AI-Claude: 2026-03-02]

import Foundation

/// Configuration loaded from coreml_config.json alongside the .mlpackage files
struct WhisperCoreMLConfig: Codable {
    let dModel: Int
    let encoderLayers: Int
    let decoderLayers: Int
    let encoderAttentionHeads: Int
    let decoderAttentionHeads: Int
    let vocabSize: Int
    let numMelBins: Int
    let maxSourcePositions: Int
    let maxTargetPositions: Int
    let maxCacheLength: Int

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case encoderLayers = "encoder_layers"
        case decoderLayers = "decoder_layers"
        case encoderAttentionHeads = "encoder_attention_heads"
        case decoderAttentionHeads = "decoder_attention_heads"
        case vocabSize = "vocab_size"
        case numMelBins = "num_mel_bins"
        case maxSourcePositions = "max_source_positions"
        case maxTargetPositions = "max_target_positions"
        case maxCacheLength = "max_cache_length"
    }

    var nHeads: Int { decoderAttentionHeads }
    var dHead: Int { dModel / decoderAttentionHeads }

    static func load(from directory: URL) throws -> WhisperCoreMLConfig {
        let configURL = directory.appendingPathComponent("coreml_config.json")
        let data = try Data(contentsOf: configURL)
        return try JSONDecoder().decode(WhisperCoreMLConfig.self, from: data)
    }
}
