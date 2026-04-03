// Qwen3CoreMLTranscriptionService.swift
// TranscriptionService implementation for Qwen3-ASR CoreML Hybrid
// [AI-Claude: 2026-03-13]

import Foundation
import os

class Qwen3CoreMLTranscriptionService: TranscriptionService {
    private let engine = Qwen3CoreMLEngine()
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "Qwen3CoreMLTranscriptionService")

    /// Temporary language override for retry
    var languageOverride: String?

    /// Average log-probability from the last transcription (for confidence routing)
    var lastAvgLogProb: Double = 0.0

    /// Detected language from the last auto-mode transcription
    var lastDetectedLanguage: String?

    /// Low-confidence words from the last transcription (for LLM prompt injection)
    var lastUncertainWords: [UncertainWord] = []

    /// Per-word confidence scores from the last transcription (for post-processing routing)
    var lastWordConfidences: [WordConfidence] = []

    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String {
        guard let qwen3CoreMLModel = model as? Qwen3CoreMLModel else {
            throw Qwen3ServiceError.invalidModel
        }

        // Ensure model is loaded
        let coremlDir = Qwen3CoreMLModelManager.modelDirectory(for: qwen3CoreMLModel.coremlModelId)
        let mlxDir = Qwen3ModelManager.modelDirectory(for: qwen3CoreMLModel.mlxModelId)
        try await engine.loadModel(coremlDir: coremlDir, mlxDir: mlxDir, modelSize: qwen3CoreMLModel.modelSize)

        // Read audio samples from WAV file
        let audioSamples = try readWAVSamples(from: audioURL)

        // Language: override takes priority, then UserDefaults
        let selectedLanguage = languageOverride ?? UserDefaults.standard.string(forKey: "SelectedLanguage")
        let prompt = UserDefaults.standard.string(forKey: "TranscriptionPrompt")

        logger.info("Transcribing with Qwen3 CoreML Hybrid, samples: \(audioSamples.count), language: \(selectedLanguage ?? "auto")")

        let result = try await engine.transcribe(samples: audioSamples, language: selectedLanguage, prompt: prompt)
        self.lastAvgLogProb = result.avgLogProb
        self.lastDetectedLanguage = result.detectedLanguage
        self.lastUncertainWords = result.uncertainWords
        self.lastWordConfidences = result.wordConfidences
        await MainActor.run {
            ChinesePostProcessingService.shared.lastAvgLogProb = result.avgLogProb
            ChinesePostProcessingService.shared.lastUncertainWords = result.uncertainWords
            ChinesePostProcessingService.shared.lastWordConfidences = result.wordConfidences
        }

        logger.info("Qwen3 CoreML transcription complete (avgLogProb: \(String(format: "%.3f", result.avgLogProb)), tokens: \(result.tokenCount)): \(result.text.prefix(100))")
        return result.text
    }

    func cleanup() async {
        await engine.unloadModel()
    }
}
