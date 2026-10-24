// Qwen3CoreMLHybridModel.swift
// Hybrid model: CoreML encoder (ANE) + MLX text decoder (GPU)
// [AI-Claude: 2026-03-13]

import Foundation
import MLX
import os

enum Qwen3CoreMLHybridModelError: Error, LocalizedError {
    case encoderNotLoaded
    case decoderNotLoaded
    case loadFailed(String)

    var errorDescription: String? {
        switch self {
        case .encoderNotLoaded:
            return "Qwen3 CoreML encoder not loaded"
        case .decoderNotLoaded:
            return "Qwen3 MLX text decoder not loaded"
        case .loadFailed(let reason):
            return "Failed to load Qwen3 CoreML hybrid model: \(reason)"
        }
    }
}

/// Qwen3-ASR Hybrid: CoreML encoder on ANE + MLX text decoder on GPU
class Qwen3CoreMLHybridModel {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "Qwen3CoreMLHybridModel")

    private let coremlEncoder = Qwen3CoreMLEncoder()
    private var mlxModel: Qwen3ASRModel?

    /// Load both CoreML encoder and MLX decoder
    /// - Parameters:
    ///   - coremlDir: Directory containing encoder.mlmodelc
    ///   - mlxDir: Directory containing MLX model weights (safetensors, vocab.json, etc.)
    ///   - modelSize: Model size configuration
    func load(coremlDir: URL, mlxDir: URL, modelSize: Qwen3ASRModelSize) throws {
        Self.logger.info("Loading Qwen3 CoreML hybrid model...")

        // Load CoreML encoder
        try coremlEncoder.load(from: coremlDir)

        // Load MLX model (tokenizer + audio encoder + text decoder)
        // Note: MLX audio encoder weights are loaded but unused in hybrid mode
        let model = try Qwen3ASRModel(
            audioConfig: modelSize.audioConfig,
            textConfig: modelSize.textConfig
        )
        try model.load(from: mlxDir, modelSize: modelSize)
        self.mlxModel = model

        Self.logger.info("Qwen3 CoreML hybrid model loaded successfully")
    }

    /// Transcribe audio to text using CoreML encoder + MLX decoder
    func transcribe(
        audio: [Float],
        sampleRate: Int = 16000,
        language: String? = nil,
        prompt: String? = nil,
        maxTokens: Int? = nil
    ) throws -> Qwen3ASRModel.TranscriptionResult {
        guard let mlxModel = mlxModel else {
            throw Qwen3CoreMLHybridModelError.decoderNotLoaded
        }
        guard let textDecoder = mlxModel.textDecoder else {
            throw Qwen3ASRModelError.textDecoderNotLoaded
        }

        // Scale maxTokens proportionally to audio duration
        let durationSeconds = Double(audio.count) / Double(sampleRate)
        let effectiveMaxTokens = maxTokens ?? min(max(448, Int(durationSeconds / 30.0 * 448.0)), 32768)

        // Step 1: Extract mel features using shared feature extractor
        let melFeatures = try mlxModel.featureExtractor.process(audio, sampleRate: sampleRate)

        // Step 2: Encode with CoreML encoder (runs on ANE)
        var audioEmbeds = try coremlEncoder.encode(melFeatures: melFeatures)
        audioEmbeds = audioEmbeds.expandedDimensions(axis: 0)
        eval(audioEmbeds)

        // Step 3: Decode with MLX text decoder (runs on GPU)
        // 單趟解碼；code-switch remap 二次解碼已於 2026-09-05 移除
        // （依據 ReplayLab eval493 證據，詳見 Qwen3ASRModel.transcribe()）。
        return try mlxModel.generateText(
            audioEmbeds: audioEmbeds,
            textDecoder: textDecoder,
            language: language,
            prompt: prompt,
            maxTokens: effectiveMaxTokens
        )
    }

    func unload() {
        coremlEncoder.unload()
        mlxModel?.audioEncoder.clearPosEmbeddingCache()
        mlxModel = nil
        Memory.clearCache()
    }
}
