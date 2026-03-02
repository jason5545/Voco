// WhisperCoreMLTranscriptionService.swift
// TranscriptionService implementation for Whisper CoreML
// [AI-Claude: 2026-03-02]

import Foundation
import os

enum WhisperCoreMLTranscriptionError: LocalizedError {
    case invalidModel
    case modelNotDownloaded

    var errorDescription: String? {
        switch self {
        case .invalidModel:
            return "Invalid Whisper CoreML model type"
        case .modelNotDownloaded:
            return "Whisper CoreML model has not been downloaded"
        }
    }
}

class WhisperCoreMLTranscriptionService: TranscriptionService {
    private let engine = WhisperCoreMLEngine()
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "WhisperCoreMLTranscriptionService")

    /// Average log-probability from the last transcription (for confidence routing)
    var lastAvgLogProb: Double = 0.0

    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String {
        guard let coremlModel = model as? WhisperCoreMLModel else {
            logger.error("Invalid model type: \(String(describing: type(of: model)))")
            throw WhisperCoreMLTranscriptionError.invalidModel
        }

        let modelDir = WhisperCoreMLModelManager.modelDirectory(for: coremlModel.coremlModelId)
        logger.warning("Model dir: \(modelDir.path)")
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            logger.error("Model directory not found: \(modelDir.path)")
            throw WhisperCoreMLTranscriptionError.modelNotDownloaded
        }

        do {
            try await engine.loadModel(from: modelDir)
        } catch {
            logger.error("Engine loadModel failed: \(error)")
            throw error
        }
        logger.warning("Model loaded successfully")

        let audioSamples = try readWAVSamples(from: audioURL)

        let selectedLanguage = UserDefaults.standard.string(forKey: "SelectedLanguage")

        logger.warning("Transcribing with Whisper CoreML, samples: \(audioSamples.count), language: \(selectedLanguage ?? "auto")")

        let result: WhisperCoreMLModelImpl.TranscriptionResult
        do {
            result = try await engine.transcribe(samples: audioSamples, language: selectedLanguage)
        } catch {
            logger.error("Engine transcribe failed: \(error)")
            throw error
        }

        self.lastAvgLogProb = result.avgLogProb
        await MainActor.run {
            ChinesePostProcessingService.shared.lastAvgLogProb = result.avgLogProb
            ChinesePostProcessingService.shared.lastModelProvider = .whisperCoreML
        }

        logger.warning("Whisper CoreML transcription complete (avgLogProb: \(String(format: "%.3f", result.avgLogProb)), tokens: \(result.tokenCount)): \(result.text.prefix(100))")
        return result.text
    }

    func preloadModel(for model: WhisperCoreMLModel) async throws {
        let modelDir = WhisperCoreMLModelManager.modelDirectory(for: model.coremlModelId)
        try await engine.loadModel(from: modelDir)
    }

    func cleanup() async {
        await engine.unloadModel()
    }
}
