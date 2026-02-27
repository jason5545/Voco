// WhisperMLXEngine.swift
// Thread-safe actor wrapper for WhisperMLXModel
// [AI-Claude: 2026-02-27]

import Foundation
import MLX
import os

enum WhisperMLXEngineError: LocalizedError {
    case warmupFailed(underlying: Error)

    var errorDescription: String? {
        switch self {
        case .warmupFailed(let underlying):
            return "Whisper MLX warmup failed: \(underlying.localizedDescription)"
        }
    }
}

actor WhisperMLXEngine {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "WhisperMLXEngine")

    private var model: WhisperMLXModelImpl?
    private var loadedModelDir: String?
    private var hasCompletedWarmup = false

    private static let sampleRate = 16000

    func loadModel(from directory: URL) throws {
        let dirPath = directory.path

        if loadedModelDir == dirPath {
            if model != nil && !hasCompletedWarmup {
                Self.logger.warning("Model loaded but warmup not completed, retrying warmup")
                try ensureWarmup(reason: "loadModel(reuse)")
            }
            Self.logger.info("Model already loaded, skipping")
            return
        }

        unloadModel()

        Self.logger.warning("Loading Whisper MLX model from: \(directory.lastPathComponent)")
        let newModel = WhisperMLXModelImpl()
        do {
            try newModel.load(from: directory)
        } catch {
            Self.logger.error("WhisperMLXEngine: model load failed: \(error)")
            throw error
        }

        self.model = newModel
        self.loadedModelDir = dirPath
        self.hasCompletedWarmup = false
        Self.logger.warning("Whisper MLX model loaded successfully")

        try ensureWarmup(reason: "loadModel(new)")
    }

    func transcribe(samples: [Float], language: String?) throws -> WhisperMLXModelImpl.TranscriptionResult {
        guard let model = model else {
            throw WhisperMLXModelError.modelNotLoaded
        }

        if !hasCompletedWarmup {
            Self.logger.warning("Warmup not completed before transcription, retrying now")
            try ensureWarmup(reason: "transcribe")
        }

        // Map "auto" or empty language to nil
        let lang: String?
        if let language = language, language != "auto", !language.isEmpty {
            lang = language
        } else {
            lang = nil
        }

        // Whisper processes max 30 seconds at a time
        // For longer audio, segment at silence points
        let maxSamples = 30 * Self.sampleRate
        if samples.count <= maxSamples {
            let result = try model.transcribe(audio: samples, language: lang)
            Memory.clearCache()
            return result
        }

        // Segment long audio (similar to Qwen3ASREngine pattern)
        Self.logger.info("Audio exceeds 30s (\(samples.count / Self.sampleRate)s), segmenting...")
        var chunkResults: [WhisperMLXModelImpl.TranscriptionResult] = []
        var offset = 0
        let searchWindow = 5 * Self.sampleRate  // ±5s silence search for 30s chunks

        while offset < samples.count {
            let remaining = samples.count - offset
            if remaining <= maxSamples {
                let chunk = Array(samples[offset...])
                let result = try model.transcribe(audio: chunk, language: lang)
                Memory.clearCache()
                if !result.text.isEmpty { chunkResults.append(result) }
                break
            }

            let targetCut = offset + maxSamples
            let cutPoint = findSilenceCutPoint(in: samples, targetCut: targetCut, searchWindow: searchWindow)
            let chunk = Array(samples[offset..<cutPoint])
            let result = try model.transcribe(audio: chunk, language: lang)
            Memory.clearCache()
            if !result.text.isEmpty { chunkResults.append(result) }
            offset = cutPoint
        }

        Memory.clearCache()

        // Merge results
        let mergedText = chunkResults.map { $0.text }.joined(separator: " ")
        let totalTokens = chunkResults.reduce(0) { $0 + $1.tokenCount }
        let weightedLogProb = totalTokens > 0
            ? chunkResults.reduce(0.0) { $0 + $1.avgLogProb * Double($1.tokenCount) } / Double(totalTokens)
            : 0.0

        return WhisperMLXModelImpl.TranscriptionResult(
            text: mergedText,
            avgLogProb: weightedLogProb,
            tokenCount: totalTokens,
            detectedLanguage: chunkResults.first?.detectedLanguage
        )
    }

    private func findSilenceCutPoint(in samples: [Float], targetCut: Int, searchWindow: Int) -> Int {
        let searchStart = max(0, targetCut - searchWindow)
        let searchEnd = min(samples.count, targetCut + searchWindow)
        let rmsWindowSize = Self.sampleRate / 2  // 0.5s

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

        return bestPos
    }

    func isModelLoaded(directory: URL) -> Bool {
        return loadedModelDir == directory.path
    }

    func unloadModel() {
        model = nil
        loadedModelDir = nil
        hasCompletedWarmup = false
        Memory.clearCache()
        Self.logger.info("Whisper MLX model unloaded, GPU cache cleared")
    }

    private func ensureWarmup(reason: String) throws {
        guard !hasCompletedWarmup else { return }
        guard let model = model else { return }

        let warmupSamples = [Float](repeating: 0, count: 16000)  // 1s silence
        var lastError: Error?
        let maxAttempts = 3

        for attempt in 1...maxAttempts {
            do {
                Self.logger.warning("Running Whisper MLX warmup (\(reason), attempt \(attempt)/\(maxAttempts))...")
                let _ = try model.transcribe(audio: warmupSamples, language: nil)
                Memory.clearCache()
                hasCompletedWarmup = true
                Self.logger.warning("Whisper MLX warmup complete (\(reason), attempt \(attempt))")
                return
            } catch {
                lastError = error
                Self.logger.error("Whisper MLX warmup attempt \(attempt) failed (\(reason)): \(error)")
            }
        }

        throw WhisperMLXEngineError.warmupFailed(
            underlying: lastError ?? WhisperMLXModelError.loadFailed("unknown warmup error")
        )
    }
}
