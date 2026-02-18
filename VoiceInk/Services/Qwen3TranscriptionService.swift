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

    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String {
        guard let qwen3Model = model as? Qwen3Model else {
            throw Qwen3ServiceError.invalidModel
        }

        // Ensure model is loaded
        let modelDir = Qwen3ModelManager.modelDirectory(for: qwen3Model.modelId)
        try await engine.loadModel(from: modelDir, modelSize: qwen3Model.modelSize)

        // Read audio samples from WAV file
        let audioSamples = try readAudioSamples(from: audioURL)

        // Get language and prompt from UserDefaults
        let selectedLanguage = UserDefaults.standard.string(forKey: "SelectedLanguage")
        let prompt = UserDefaults.standard.string(forKey: "TranscriptionPrompt")

        logger.info("Transcribing with Qwen3-ASR, samples: \(audioSamples.count), language: \(selectedLanguage ?? "auto"), prompt: \(prompt?.prefix(50) ?? "none")")

        let text = try await engine.transcribe(samples: audioSamples, language: selectedLanguage, prompt: prompt)

        logger.info("Qwen3-ASR transcription complete: \(text.prefix(100))")
        return text
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
