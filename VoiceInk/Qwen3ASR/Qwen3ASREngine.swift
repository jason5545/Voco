// Qwen3ASREngine.swift
// Thread-safe actor wrapper for Qwen3ASRModel
// [AI-Claude: 2025-02-18]

import Foundation
import os

actor Qwen3ASREngine {
    private static let logger = Logger(subsystem: "com.jasonchien.voco", category: "Qwen3ASREngine")

    private var model: Qwen3ASRModel?
    private var loadedModelId: String?

    func loadModel(from directory: URL, modelSize: Qwen3ASRModelSize) throws {
        let modelId = modelSize.defaultModelId

        if loadedModelId == modelId {
            Self.logger.info("Model \(modelId) already loaded, skipping")
            return
        }

        unloadModel()

        Self.logger.info("Loading Qwen3-ASR model: \(modelId)")
        let newModel = try Qwen3ASRModel(
            audioConfig: modelSize.audioConfig,
            textConfig: modelSize.textConfig
        )
        try newModel.load(from: directory, modelSize: modelSize)

        self.model = newModel
        self.loadedModelId = modelId
        Self.logger.info("Qwen3-ASR model loaded successfully")
    }

    func transcribe(samples: [Float], language: String?, prompt: String? = nil) throws -> String {
        guard let model = model else {
            throw Qwen3ASRModelError.textDecoderNotLoaded
        }

        // Map "auto" or empty language to nil (let model auto-detect)
        let lang: String?
        if let language = language, language != "auto", !language.isEmpty {
            lang = language
        } else {
            lang = nil
        }

        return try model.transcribe(audio: samples, sampleRate: 16000, language: lang, prompt: prompt)
    }

    func isModelLoaded(modelId: String) -> Bool {
        return loadedModelId == modelId
    }

    func unloadModel() {
        model = nil
        loadedModelId = nil
        Self.logger.info("Qwen3-ASR model unloaded")
    }
}
