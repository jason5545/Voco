// Qwen3ASRConfiguration.swift
// Adapted from qwen3-asr-swift (AudioEncoder.swift config + Configuration.swift decoder config + Qwen3ASR.swift model size)
// [AI-Claude: 2025-02-18]

import Foundation

// MARK: - Audio Encoder Configuration

/// Audio encoder configuration matching Qwen3-ASR HuggingFace model
struct Qwen3AudioEncoderConfig {
    let dModel: Int
    let encoderAttentionHeads: Int
    let encoderFFNDim: Int
    let encoderLayers: Int
    let numMelBins: Int
    let maxSourcePositions: Int
    let outputDim: Int
    let downsampleHiddenSize: Int
    let convChunksize: Int
    let nWindow: Int
    let nWindowInfer: Int
    let dropout: Float
    let attentionDropout: Float
    let activationDropout: Float
    let layerNormEps: Float
    let convOutInputDim: Int

    /// Config for Qwen3-ASR-0.6B (default)
    static let `default` = Qwen3AudioEncoderConfig(
        dModel: 896,
        encoderAttentionHeads: 14,
        encoderFFNDim: 3584,
        encoderLayers: 18,
        numMelBins: 128,
        maxSourcePositions: 1500,
        outputDim: 1024,
        downsampleHiddenSize: 480,
        convChunksize: 500,
        nWindow: 50,
        nWindowInfer: 800,
        dropout: 0.0,
        attentionDropout: 0.0,
        activationDropout: 0.0,
        layerNormEps: 1e-5,
        convOutInputDim: 7680
    )

    static let small = `default`

    /// Config for Qwen3-ASR-1.7B
    static let large = Qwen3AudioEncoderConfig(
        dModel: 1024,
        encoderAttentionHeads: 16,
        encoderFFNDim: 4096,
        encoderLayers: 24,
        numMelBins: 128,
        maxSourcePositions: 1500,
        outputDim: 2048,
        downsampleHiddenSize: 480,
        convChunksize: 500,
        nWindow: 50,
        nWindowInfer: 800,
        dropout: 0.0,
        attentionDropout: 0.0,
        activationDropout: 0.0,
        layerNormEps: 1e-5,
        convOutInputDim: 7680
    )
}

// MARK: - Text Decoder Configuration

/// Configuration for Qwen3 text decoder
struct Qwen3TextDecoderConfig: Sendable {
    var vocabSize: Int = 151936
    var hiddenSize: Int = 1024
    var numLayers: Int = 28
    var numHeads: Int = 16
    var numKVHeads: Int = 8
    var headDim: Int = 64
    var intermediateSize: Int = 3072
    var maxPositionEmbeddings: Int = 65536
    var rmsNormEps: Float = 1e-6
    var ropeTheta: Float = 1000000.0
    var tieWordEmbeddings: Bool = true
    var groupSize: Int = 64
    var bits: Int = 4

    /// Config for Qwen3-ASR-0.6B decoder
    static var small: Qwen3TextDecoderConfig {
        var config = Qwen3TextDecoderConfig()
        config.hiddenSize = 1024
        config.numLayers = 28
        config.numHeads = 16
        config.numKVHeads = 8
        config.headDim = 128
        config.intermediateSize = 3072
        config.groupSize = 64
        config.bits = 4
        return config
    }

    /// Config for Qwen3-ASR-1.7B decoder
    static var large: Qwen3TextDecoderConfig {
        var config = Qwen3TextDecoderConfig()
        config.hiddenSize = 2048
        config.numLayers = 28
        config.numHeads = 16
        config.numKVHeads = 8
        config.headDim = 128
        config.intermediateSize = 6144
        config.groupSize = 64
        config.bits = 8
        return config
    }
}

// MARK: - Model Size

/// Supported ASR model sizes
enum Qwen3ASRModelSize {
    case small  // 0.6B
    case large  // 1.7B

    var defaultModelId: String {
        switch self {
        case .small: return "mlx-community/Qwen3-ASR-0.6B-4bit"
        case .large: return "mlx-community/Qwen3-ASR-1.7B-8bit"
        }
    }

    var audioConfig: Qwen3AudioEncoderConfig {
        switch self {
        case .small: return .small
        case .large: return .large
        }
    }

    var textConfig: Qwen3TextDecoderConfig {
        switch self {
        case .small: return .small
        case .large: return .large
        }
    }

    static func detect(from modelId: String) -> Qwen3ASRModelSize {
        if modelId.contains("1.7B") || modelId.contains("1.7b") {
            return .large
        }
        return .small
    }
}

// MARK: - Special Tokens

/// Special token IDs for Qwen3-ASR
struct Qwen3ASRTokens {
    static let audioTokenId = 151676        // <|audio_pad|>
    static let audioStartTokenId = 151669   // <|audio_start|>
    static let audioEndTokenId = 151670     // <|audio_end|>
    static let eosTokenId = 151645          // <|im_end|>
    static let padTokenId = 151643          // <|endoftext|>
    static let imStartTokenId = 151644      // <|im_start|>
    static let imEndTokenId = 151645        // <|im_end|>
    static let asrTextId = 151704           // <asr_text>
    static let newlineId = 198
    static let systemId = 8948
    static let userId = 872
    static let assistantId = 77091
}
