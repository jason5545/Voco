// WhisperMLXConfiguration.swift
// Model configuration for Whisper MLX models
// [AI-Claude: 2026-02-27]

import Foundation

/// Whisper model configuration loaded from config.json
struct WhisperMLXConfig: Codable {
    let nMels: Int
    let nAudioCtx: Int
    let nAudioState: Int
    let nAudioHead: Int
    let nAudioLayer: Int
    let nVocab: Int
    let nTextCtx: Int
    let nTextState: Int
    let nTextHead: Int
    let nTextLayer: Int
    let quantization: WhisperQuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case nMels = "n_mels"
        case nAudioCtx = "n_audio_ctx"
        case nAudioState = "n_audio_state"
        case nAudioHead = "n_audio_head"
        case nAudioLayer = "n_audio_layer"
        case nVocab = "n_vocab"
        case nTextCtx = "n_text_ctx"
        case nTextState = "n_text_state"
        case nTextHead = "n_text_head"
        case nTextLayer = "n_text_layer"
        case quantization
    }

    // Also support HuggingFace Transformers-style config (e.g. whisper-large-asr-4bit)
    enum AltCodingKeys: String, CodingKey {
        case dModel = "d_model"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderLayers = "encoder_layers"
        case decoderAttentionHeads = "decoder_attention_heads"
        case decoderLayers = "decoder_layers"
        case decoderFFNDim = "decoder_ffn_dim"
        case numMelBins = "num_mel_bins"
        case maxSourcePositions = "max_source_positions"
        case maxTargetPositions = "max_target_positions"
        case vocabSize = "vocab_size"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    init(from decoder: Decoder) throws {
        // Try original MLX-style keys first
        if let container = try? decoder.container(keyedBy: CodingKeys.self),
           let nMels = try? container.decode(Int.self, forKey: .nMels) {
            self.nMels = nMels
            self.nAudioCtx = try container.decode(Int.self, forKey: .nAudioCtx)
            self.nAudioState = try container.decode(Int.self, forKey: .nAudioState)
            self.nAudioHead = try container.decode(Int.self, forKey: .nAudioHead)
            self.nAudioLayer = try container.decode(Int.self, forKey: .nAudioLayer)
            self.nVocab = try container.decode(Int.self, forKey: .nVocab)
            self.nTextCtx = try container.decode(Int.self, forKey: .nTextCtx)
            self.nTextState = try container.decode(Int.self, forKey: .nTextState)
            self.nTextHead = try container.decode(Int.self, forKey: .nTextHead)
            self.nTextLayer = try container.decode(Int.self, forKey: .nTextLayer)
            self.quantization = try container.decodeIfPresent(WhisperQuantizationConfig.self, forKey: .quantization)
        } else {
            // Fall back to HuggingFace Transformers-style keys
            let container = try decoder.container(keyedBy: AltCodingKeys.self)
            self.nAudioState = try container.decode(Int.self, forKey: .dModel)
            self.nTextState = self.nAudioState  // Same d_model for both
            self.nAudioHead = try container.decode(Int.self, forKey: .encoderAttentionHeads)
            self.nAudioLayer = try container.decode(Int.self, forKey: .encoderLayers)
            self.nTextHead = try container.decode(Int.self, forKey: .decoderAttentionHeads)
            self.nTextLayer = try container.decode(Int.self, forKey: .decoderLayers)
            self.nMels = try container.decode(Int.self, forKey: .numMelBins)
            self.nAudioCtx = try container.decode(Int.self, forKey: .maxSourcePositions)
            self.nTextCtx = try container.decode(Int.self, forKey: .maxTargetPositions)
            self.nVocab = try container.decode(Int.self, forKey: .vocabSize)
            self.quantization = try container.decodeIfPresent(WhisperQuantizationConfig.self, forKey: .quantization)
                ?? (try container.decodeIfPresent(WhisperQuantizationConfig.self, forKey: .quantizationConfig))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(nMels, forKey: .nMels)
        try container.encode(nAudioCtx, forKey: .nAudioCtx)
        try container.encode(nAudioState, forKey: .nAudioState)
        try container.encode(nAudioHead, forKey: .nAudioHead)
        try container.encode(nAudioLayer, forKey: .nAudioLayer)
        try container.encode(nVocab, forKey: .nVocab)
        try container.encode(nTextCtx, forKey: .nTextCtx)
        try container.encode(nTextState, forKey: .nTextState)
        try container.encode(nTextHead, forKey: .nTextHead)
        try container.encode(nTextLayer, forKey: .nTextLayer)
        try container.encodeIfPresent(quantization, forKey: .quantization)
    }

    static func load(from directory: URL) throws -> WhisperMLXConfig {
        let configURL = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        return try JSONDecoder().decode(WhisperMLXConfig.self, from: data)
    }
}

struct WhisperQuantizationConfig: Codable {
    let groupSize: Int
    let bits: Int

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }
}

/// Special token IDs for Whisper
struct WhisperTokens {
    // Base tokens
    static let eotTokenId = 50257         // <|endoftext|> (also used as EOS/PAD/BOS)
    static let sotTokenId = 50258         // <|startoftranscript|>
    static let translateTokenId = 50358   // <|translate|>
    static let transcribeTokenId = 50359  // <|transcribe|>
    static let noSpeechTokenId = 50362    // <|nospeech|>
    static let noTimestampsTokenId = 50363 // <|notimestamps|>

    // Language token range: 50259..50358 (99 languages for v1/v2, 100 for v3)
    static let firstLanguageTokenId = 50259

    /// Map language code to token ID
    static func languageTokenId(for code: String) -> Int? {
        guard let index = languageOrder.firstIndex(of: code) else { return nil }
        return firstLanguageTokenId + index
    }

    /// Map token ID to language code
    static func languageCode(for tokenId: Int) -> String? {
        let index = tokenId - firstLanguageTokenId
        guard index >= 0, index < languageOrder.count else { return nil }
        return languageOrder[index]
    }

    /// Whisper language order (same as tiktoken/openai-whisper)
    static let languageOrder: [String] = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
        "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
        "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
        "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
        "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
        "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
        "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
        "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
    ]
}

/// HuggingFace model definitions for Whisper MLX
enum WhisperMLXModelDefinition {
    case largeV2_4bit
    case largeV3Turbo_4bit
    case largeASR_4bit

    var huggingFaceRepo: String {
        switch self {
        case .largeV2_4bit: return "mlx-community/whisper-large-v2-mlx-4bit"
        case .largeV3Turbo_4bit: return "mlx-community/whisper-large-v3-turbo-4bit"
        case .largeASR_4bit: return "mlx-community/whisper-large-asr-4bit"
        }
    }

    /// Whether this model uses NPZ format (vs safetensors)
    var usesNPZ: Bool {
        switch self {
        case .largeV2_4bit: return true
        case .largeV3Turbo_4bit, .largeASR_4bit: return false
        }
    }
}
