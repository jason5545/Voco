// WhisperMLXTranscriptionService.swift
// TranscriptionService implementation for Whisper MLX
// [AI-Claude: 2026-02-27]

import Foundation
import os

enum WhisperMLXTranscriptionError: LocalizedError {
    case invalidModel
    case modelNotDownloaded

    var errorDescription: String? {
        switch self {
        case .invalidModel:
            return "Invalid Whisper MLX model type"
        case .modelNotDownloaded:
            return "Whisper MLX model has not been downloaded"
        }
    }
}

class WhisperMLXTranscriptionService: TranscriptionService {
    private let engine = WhisperMLXEngine()
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "WhisperMLXTranscriptionService")

    /// Average log-probability from the last transcription (for confidence routing)
    var lastAvgLogProb: Double = 0.0

    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String {
        guard let whisperModel = model as? WhisperMLXModel else {
            logger.error("Invalid model type: \(String(describing: type(of: model)))")
            throw WhisperMLXTranscriptionError.invalidModel
        }

        // Ensure model is loaded
        let modelDir = WhisperMLXModelManager.modelDirectory(for: whisperModel.huggingFaceRepo)
        logger.warning("Model dir: \(modelDir.path)")
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            logger.error("Model directory not found: \(modelDir.path)")
            throw WhisperMLXTranscriptionError.modelNotDownloaded
        }
        do {
            try await engine.loadModel(from: modelDir)
        } catch {
            logger.error("Engine loadModel failed: \(error)")
            throw error
        }
        logger.warning("Model loaded successfully")

        // Read audio samples
        let audioSamples = try readWAVSamples(from: audioURL)

        // Language
        let selectedLanguage = UserDefaults.standard.string(forKey: "SelectedLanguage")
        let prompt = UserDefaults.standard.string(forKey: "TranscriptionPrompt")

        logger.warning("Transcribing with Whisper MLX, samples: \(audioSamples.count), language: \(selectedLanguage ?? "auto")")

        let result: WhisperMLXModelImpl.TranscriptionResult
        do {
            result = try await engine.transcribe(samples: audioSamples, language: selectedLanguage)
        } catch {
            logger.error("Engine transcribe failed: \(error)")
            throw error
        }

        self.lastAvgLogProb = result.avgLogProb
        await MainActor.run {
            ChinesePostProcessingService.shared.lastAvgLogProb = result.avgLogProb
            ChinesePostProcessingService.shared.lastModelProvider = .whisperMLX
        }

        logger.warning("Whisper MLX transcription complete (avgLogProb: \(String(format: "%.3f", result.avgLogProb)), tokens: \(result.tokenCount)): \(result.text.prefix(100))")
        return result.text
    }

    /// Preload model to avoid first-transcription latency
    func preloadModel(for model: WhisperMLXModel) async throws {
        let modelDir = WhisperMLXModelManager.modelDirectory(for: model.huggingFaceRepo)
        try await engine.loadModel(from: modelDir)
    }

    func cleanup() async {
        await engine.unloadModel()
    }
}
