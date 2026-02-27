// Qwen3ASREngine.swift
// Thread-safe actor wrapper for Qwen3ASRModel
// [AI-Claude: 2025-02-18]

import Foundation
import MLX
import os

enum Qwen3ASREngineError: LocalizedError {
    case warmupFailed(modelId: String, underlying: Error)

    var errorDescription: String? {
        switch self {
        case .warmupFailed(let modelId, let underlying):
            return "Qwen3 warmup failed for \(modelId): \(underlying.localizedDescription)"
        }
    }
}

actor Qwen3ASREngine {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "Qwen3ASREngine")

    private var model: Qwen3ASRModel?
    private var loadedModelId: String?
    private var hasCompletedWarmup = false

    func loadModel(from directory: URL, modelSize: Qwen3ASRModelSize) throws {
        let modelId = modelSize.defaultModelId

        if loadedModelId == modelId {
            if let model = model, !hasCompletedWarmup {
                Self.logger.warning("Model \(modelId) loaded but warmup not completed, retrying warmup")
                try ensureWarmup(using: model, modelId: modelId, reason: "loadModel(reuse)")
            }
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
        self.hasCompletedWarmup = false
        Self.logger.info("Qwen3-ASR model loaded successfully")

        try ensureWarmup(using: newModel, modelId: modelId, reason: "loadModel(new)")
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
        guard let loadedModelId = loadedModelId else {
            throw Qwen3ASRModelError.textDecoderNotLoaded
        }

        // Hard gate: the first real transcription must not run before warmup succeeds.
        if !hasCompletedWarmup {
            Self.logger.warning("Warmup not completed before transcription, retrying now")
            try ensureWarmup(using: model, modelId: loadedModelId, reason: "transcribe")
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
            let result = try model.transcribe(audio: samples, sampleRate: 16000, language: lang, prompt: prompt)
            Memory.clearCache()
            return result
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
                Memory.clearCache()
                if !result.text.isEmpty { chunkResults.append(result) }
                break
            }

            // Find the best silence point near the 20-minute mark
            let cutPoint = Self.findSilenceCutPoint(in: samples, targetCut: offset + Self.maxSamplesPerChunk)
            let chunk = Array(samples[offset..<cutPoint])
            Self.logger.info("Chunk: \(offset / sr)s - \(cutPoint / sr)s (\(chunk.count / sr)s)")
            let result = try model.transcribe(audio: chunk, sampleRate: 16000, language: lang, prompt: prompt)
            Memory.clearCache()
            if !result.text.isEmpty { chunkResults.append(result) }
            offset = cutPoint
        }

        // Release GPU cache after all chunks are processed
        Memory.clearCache()

        // Merge: concatenate text, weighted average logprob by token count, take first chunk's detected language
        let mergedText = chunkResults.map { $0.text }.joined(separator: " ")
        let totalTokens = chunkResults.reduce(0) { $0 + $1.tokenCount }
        let weightedLogProb = totalTokens > 0
            ? chunkResults.reduce(0.0) { $0 + $1.avgLogProb * Double($1.tokenCount) } / Double(totalTokens)
            : 0.0
        // Merge uncertain words from all chunks, keep top 8 by lowest logProb
        let allUncertainWords = chunkResults.flatMap { $0.uncertainWords }
        let mergedUncertainWords = Array(allUncertainWords.sorted { $0.logProb < $1.logProb }.prefix(8))
        return Qwen3ASRModel.TranscriptionResult(text: mergedText, avgLogProb: weightedLogProb, tokenCount: totalTokens, detectedLanguage: chunkResults.first?.detectedLanguage, uncertainWords: mergedUncertainWords)
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
        model?.audioEncoder.clearPosEmbeddingCache()
        model = nil
        loadedModelId = nil
        hasCompletedWarmup = false
        Memory.clearCache()
        Self.logger.info("Qwen3-ASR model unloaded, GPU cache cleared")
    }

    private func ensureWarmup(using model: Qwen3ASRModel, modelId: String, reason: String) throws {
        guard !hasCompletedWarmup else { return }

        // Warmup compiles Metal kernels and stabilizes first-pass MLX execution.
        // Require success before allowing user-facing transcription.
        let warmupSamples = [Float](repeating: 0, count: 16000) // 1 s of silence
        var lastError: Error?
        let maxAttempts = 3

        for attempt in 1...maxAttempts {
            do {
                Self.logger.info("Running Qwen3 warmup inference (\(reason), attempt \(attempt)/\(maxAttempts))…")
                let _ = try model.transcribe(audio: warmupSamples, sampleRate: 16000, language: nil)
                // Clear GPU buffer cache left by silence inference so the first real
                // transcription starts with clean state (prevents garbage output).
                Memory.clearCache()
                hasCompletedWarmup = true
                Self.logger.info("Qwen3 warmup complete (\(reason), attempt \(attempt))")
                return
            } catch {
                lastError = error
                Self.logger.error("⚠️ Qwen3 warmup attempt \(attempt) failed (\(reason)): \(error)")
            }
        }

        throw Qwen3ASREngineError.warmupFailed(
            modelId: modelId,
            underlying: lastError ?? Qwen3ASRModelError.loadFailed("unknown warmup error")
        )
    }
}
