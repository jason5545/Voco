// Qwen3TranscriptionService.swift
// TranscriptionService implementation for Qwen3-ASR
// [AI-Claude: 2025-02-18]

import Foundation
import os

enum Qwen3ServiceError: Error {
    case invalidModel
    case invalidAudioData
}

class Qwen3TranscriptionService: TranscriptionService {
    private let engine = Qwen3ASREngine()
    private let logger = Logger(subsystem: "com.jasonchien.voco", category: "Qwen3TranscriptionService")

    /// Temporary language override for retry (e.g. "Japanese"). Takes priority over UserDefaults.
    var languageOverride: String?

    /// Average log-probability from the last transcription (for confidence routing / low-confidence retry)
    var lastAvgLogProb: Double = 0.0

    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String {
        guard let qwen3Model = model as? Qwen3Model else {
            throw Qwen3ServiceError.invalidModel
        }

        // Ensure model is loaded
        let modelDir = Qwen3ModelManager.modelDirectory(for: qwen3Model.modelId)
        try await engine.loadModel(from: modelDir, modelSize: qwen3Model.modelSize)

        // Read audio samples from WAV file
        let audioSamples = try readAudioSamples(from: audioURL)

        // Language: override takes priority, then UserDefaults
        let selectedLanguage = languageOverride ?? UserDefaults.standard.string(forKey: "SelectedLanguage")
        let prompt = UserDefaults.standard.string(forKey: "TranscriptionPrompt")

        logger.info("Transcribing with Qwen3-ASR, samples: \(audioSamples.count), language: \(selectedLanguage ?? "auto"), prompt: \(prompt?.prefix(50) ?? "none")")

        let result = try await engine.transcribe(samples: audioSamples, language: selectedLanguage, prompt: prompt)
        self.lastAvgLogProb = result.avgLogProb
        await MainActor.run {
            ChinesePostProcessingService.shared.lastAvgLogProb = result.avgLogProb
        }

        logger.info("Qwen3-ASR transcription complete (avgLogProb: \(String(format: "%.3f", result.avgLogProb)), tokens: \(result.tokenCount)): \(result.text.prefix(100))")
        return result.text
    }

    private func readAudioSamples(from url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count > 44 else {
            throw Qwen3ServiceError.invalidAudioData
        }

        // Skip 44-byte WAV header, convert Int16 PCM to Float
        let floats = stride(from: 44, to: data.count, by: 2).map {
            return data[$0..<$0 + 2].withUnsafeBytes {
                let short = Int16(littleEndian: $0.load(as: Int16.self))
                return max(-1.0, min(Float(short) / 32767.0, 1.0))
            }
        }

        return floats
    }

    func cleanup() {
        Task {
            await engine.unloadModel()
        }
    }
}
