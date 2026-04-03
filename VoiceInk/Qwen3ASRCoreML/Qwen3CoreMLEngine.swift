// Qwen3CoreMLEngine.swift
// Thread-safe actor wrapper for Qwen3CoreMLHybridModel
// [AI-Claude: 2026-03-13]

import Foundation
import MLX
import os

enum Qwen3CoreMLEngineError: LocalizedError {
    case warmupFailed(underlying: Error)

    var errorDescription: String? {
        switch self {
        case .warmupFailed(let underlying):
            return "Qwen3 CoreML warmup failed: \(underlying.localizedDescription)"
        }
    }
}

actor Qwen3CoreMLEngine {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "Qwen3CoreMLEngine")

    private var model: Qwen3CoreMLHybridModel?
    private var loadedModelKey: String?
    private var hasCompletedWarmup = false

    func loadModel(coremlDir: URL, mlxDir: URL, modelSize: Qwen3ASRModelSize) throws {
        let modelKey = "\(coremlDir.lastPathComponent)+\(mlxDir.lastPathComponent)"

        if loadedModelKey == modelKey {
            if model != nil && !hasCompletedWarmup {
                Self.logger.warning("Model loaded but warmup not completed, retrying warmup")
                try ensureWarmup(reason: "loadModel(reuse)")
            }
            Self.logger.info("Qwen3 CoreML hybrid model already loaded, skipping")
            return
        }

        unloadModel()

        Self.logger.info("Loading Qwen3 CoreML hybrid model...")
        let newModel = Qwen3CoreMLHybridModel()
        try newModel.load(coremlDir: coremlDir, mlxDir: mlxDir, modelSize: modelSize)

        self.model = newModel
        self.loadedModelKey = modelKey
        self.hasCompletedWarmup = false
        Self.logger.info("Qwen3 CoreML hybrid model loaded successfully")

        try ensureWarmup(reason: "loadModel(new)")
    }

    private static let sampleRate = 16000
    private static let maxSamplesPerChunk = 20 * 60 * sampleRate
    private static let silenceSearchWindow = 30 * sampleRate
    private static let rmsWindowSize = sampleRate / 2

    func transcribe(samples: [Float], language: String?, prompt: String? = nil) throws -> Qwen3ASRModel.TranscriptionResult {
        guard let model = model else {
            throw Qwen3CoreMLHybridModelError.decoderNotLoaded
        }

        if !hasCompletedWarmup {
            Self.logger.warning("Warmup not completed before transcription, retrying now")
            try ensureWarmup(reason: "transcribe")
        }

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
        Self.logger.info("Audio exceeds 20 minutes (\(samples.count / sr)s), segmenting...")
        var chunkResults: [Qwen3ASRModel.TranscriptionResult] = []
        var offset = 0
        while offset < samples.count {
            let remaining = samples.count - offset
            if remaining <= Self.maxSamplesPerChunk {
                let chunk = Array(samples[offset...])
                let result = try model.transcribe(audio: chunk, sampleRate: 16000, language: lang, prompt: prompt)
                Memory.clearCache()
                if !result.text.isEmpty { chunkResults.append(result) }
                break
            }

            let cutPoint = Self.findSilenceCutPoint(in: samples, targetCut: offset + Self.maxSamplesPerChunk)
            let chunk = Array(samples[offset..<cutPoint])
            Self.logger.info("Chunk: \(offset / sr)s - \(cutPoint / sr)s (\(chunk.count / sr)s)")
            let result = try model.transcribe(audio: chunk, sampleRate: 16000, language: lang, prompt: prompt)
            Memory.clearCache()
            if !result.text.isEmpty { chunkResults.append(result) }
            offset = cutPoint
        }

        Memory.clearCache()

        let mergedText = chunkResults.map { $0.text }.joined(separator: " ")
        let totalTokens = chunkResults.reduce(0) { $0 + $1.tokenCount }
        let weightedLogProb = totalTokens > 0
            ? chunkResults.reduce(0.0) { $0 + $1.avgLogProb * Double($1.tokenCount) } / Double(totalTokens)
            : 0.0
        let allUncertainWords = chunkResults.flatMap { $0.uncertainWords }
        let mergedUncertainWords = Array(allUncertainWords.sorted { $0.logProb < $1.logProb }.prefix(8))
        let mergedWordConfidences = chunkResults.flatMap { $0.wordConfidences }
        return Qwen3ASRModel.TranscriptionResult(
            text: mergedText,
            avgLogProb: weightedLogProb,
            tokenCount: totalTokens,
            detectedLanguage: chunkResults.first?.detectedLanguage,
            uncertainWords: mergedUncertainWords,
            wordConfidences: mergedWordConfidences
        )
    }

    private static func findSilenceCutPoint(in samples: [Float], targetCut: Int) -> Int {
        let searchStart = max(0, targetCut - silenceSearchWindow)
        let searchEnd = min(samples.count, targetCut + silenceSearchWindow)

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
                bestPos = pos + rmsWindowSize / 2
            }
            pos += rmsWindowSize / 2
        }

        let sr = sampleRate
        logger.info("Silence cut: target \(targetCut / sr)s → actual \(bestPos / sr)s (RMS: \(minRMS))")
        return bestPos
    }

    func isModelLoaded(modelKey: String) -> Bool {
        return loadedModelKey == modelKey
    }

    func unloadModel() {
        model?.unload()
        model = nil
        loadedModelKey = nil
        hasCompletedWarmup = false
        Memory.clearCache()
        Self.logger.info("Qwen3 CoreML hybrid model unloaded")
    }

    private func ensureWarmup(reason: String) throws {
        guard !hasCompletedWarmup else { return }
        guard let model = model else { return }

        let warmupSamples = [Float](repeating: 0, count: 16000)
        var lastError: Error?
        let maxAttempts = 3

        for attempt in 1...maxAttempts {
            do {
                Self.logger.info("Running Qwen3 CoreML warmup (\(reason), attempt \(attempt)/\(maxAttempts))...")
                let _ = try model.transcribe(audio: warmupSamples, sampleRate: 16000, language: nil)
                Memory.clearCache()
                hasCompletedWarmup = true
                Self.logger.info("Qwen3 CoreML warmup complete (\(reason), attempt \(attempt))")
                return
            } catch {
                lastError = error
                Self.logger.error("Qwen3 CoreML warmup attempt \(attempt) failed (\(reason)): \(error)")
            }
        }

        throw Qwen3CoreMLEngineError.warmupFailed(
            underlying: lastError ?? Qwen3CoreMLHybridModelError.loadFailed("unknown warmup error")
        )
    }
}
