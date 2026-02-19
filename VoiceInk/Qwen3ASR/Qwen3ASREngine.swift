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

        // Warmup: first MLX inference compiles Metal shaders and stabilises GPU state.
        // Without this the very first real transcription may hallucinate in a wrong language.
        Self.logger.info("Running Qwen3 warmup inference…")
        let warmupSamples = [Float](repeating: 0, count: 16000) // 1 s of silence
        let _ = try? newModel.transcribe(audio: warmupSamples, sampleRate: 16000, language: "en")
        Self.logger.info("Qwen3 warmup complete")
    }

    private static let sampleRate = 16000
    /// Maximum samples per chunk: 20 minutes at 16kHz
    private static let maxSamplesPerChunk = 20 * 60 * sampleRate  // 19,200,000
    /// Search window for silence detection: ±30 seconds around the target cut point
    private static let silenceSearchWindow = 30 * sampleRate  // 480,000
    /// RMS analysis window: 0.5 seconds
    private static let rmsWindowSize = sampleRate / 2  // 0.5s at 16kHz

    func transcribe(samples: [Float], language: String?, prompt: String? = nil) throws -> Qwen3ASRModel.TranscriptionResult {
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

        // Audio within 20 minutes: single pass
        if samples.count <= Self.maxSamplesPerChunk {
            return try model.transcribe(audio: samples, sampleRate: 16000, language: lang, prompt: prompt)
        }

        // Audio over 20 minutes: segment at silence points
        let sr = Self.sampleRate
        Self.logger.info("Audio exceeds 20 minutes (\(samples.count / sr)s), segmenting at silence points...")
        var chunkResults: [Qwen3ASRModel.TranscriptionResult] = []
        var offset = 0
        while offset < samples.count {
            let remaining = samples.count - offset
            if remaining <= Self.maxSamplesPerChunk {
                // Last chunk: take everything
                let chunk = Array(samples[offset...])
                let result = try model.transcribe(audio: chunk, sampleRate: 16000, language: lang, prompt: prompt)
                if !result.text.isEmpty { chunkResults.append(result) }
                break
            }

            // Find the best silence point near the 20-minute mark
            let cutPoint = Self.findSilenceCutPoint(in: samples, targetCut: offset + Self.maxSamplesPerChunk)
            let chunk = Array(samples[offset..<cutPoint])
            Self.logger.info("Chunk: \(offset / sr)s - \(cutPoint / sr)s (\(chunk.count / sr)s)")
            let result = try model.transcribe(audio: chunk, sampleRate: 16000, language: lang, prompt: prompt)
            if !result.text.isEmpty { chunkResults.append(result) }
            offset = cutPoint
        }

        // Merge: concatenate text, weighted average logprob by token count, take first chunk's detected language
        let mergedText = chunkResults.map { $0.text }.joined(separator: " ")
        let totalTokens = chunkResults.reduce(0) { $0 + $1.tokenCount }
        let weightedLogProb = totalTokens > 0
            ? chunkResults.reduce(0.0) { $0 + $1.avgLogProb * Double($1.tokenCount) } / Double(totalTokens)
            : 0.0
        return Qwen3ASRModel.TranscriptionResult(text: mergedText, avgLogProb: weightedLogProb, tokenCount: totalTokens, detectedLanguage: chunkResults.first?.detectedLanguage)
    }

    /// Find the quietest point (lowest RMS energy) within ±30s of the target cut position
    private static func findSilenceCutPoint(in samples: [Float], targetCut: Int) -> Int {
        let searchStart = max(0, targetCut - silenceSearchWindow)
        let searchEnd = min(samples.count, targetCut + silenceSearchWindow)

        // Slide a 0.5s RMS window and find the position with minimum energy
        var minRMS: Float = .infinity
        var bestPos = targetCut

        var pos = searchStart
        while pos + rmsWindowSize <= searchEnd {
            var sumSquares: Float = 0
            for i in pos..<(pos + rmsWindowSize) {
                sumSquares += samples[i] * samples[i]
            }
            let rms = sumSquares / Float(rmsWindowSize)
            if rms < minRMS {
                minRMS = rms
                bestPos = pos + rmsWindowSize / 2  // Cut at center of the quiet window
            }
            pos += rmsWindowSize / 2  // Step by half window for overlap
        }

        let sr = sampleRate
        logger.info("Silence cut: target \(targetCut / sr)s → actual \(bestPos / sr)s (RMS: \(minRMS))")
        return bestPos
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
