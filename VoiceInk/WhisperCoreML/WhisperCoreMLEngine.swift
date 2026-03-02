// WhisperCoreMLEngine.swift
// Thread-safe actor wrapper for WhisperCoreMLModel
// [AI-Claude: 2026-03-02]

import Foundation
import os

enum WhisperCoreMLEngineError: LocalizedError {
    case warmupFailed(underlying: Error)

    var errorDescription: String? {
        switch self {
        case .warmupFailed(let underlying):
            return "Whisper CoreML warmup failed: \(underlying.localizedDescription)"
        }
    }
}

actor WhisperCoreMLEngine {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "WhisperCoreMLEngine")

    private var model: WhisperCoreMLModelImpl?
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

        Self.logger.warning("Loading Whisper CoreML model from: \(directory.lastPathComponent)")
        let newModel = WhisperCoreMLModelImpl()
        do {
            try newModel.load(from: directory)
        } catch {
            Self.logger.error("WhisperCoreMLEngine: model load failed: \(error)")
            throw error
        }

        self.model = newModel
        self.loadedModelDir = dirPath
        self.hasCompletedWarmup = false
        Self.logger.warning("Whisper CoreML model loaded successfully")

        try ensureWarmup(reason: "loadModel(new)")
    }

    func transcribe(samples: [Float], language: String?) throws -> WhisperCoreMLModelImpl.TranscriptionResult {
        guard let model = model else {
            throw WhisperCoreMLModelError.modelNotLoaded
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

        // Whisper processes max 30 seconds at a time
        let maxSamples = 30 * Self.sampleRate
        if samples.count <= maxSamples {
            return try model.transcribe(audio: samples, language: lang)
        }

        // Segment long audio (same pattern as WhisperMLXEngine)
        Self.logger.info("Audio exceeds 30s (\(samples.count / Self.sampleRate)s), segmenting...")
        var chunkResults: [WhisperCoreMLModelImpl.TranscriptionResult] = []
        var offset = 0
        let searchWindow = 5 * Self.sampleRate

        while offset < samples.count {
            let remaining = samples.count - offset
            if remaining <= maxSamples {
                let chunk = Array(samples[offset...])
                let result = try model.transcribe(audio: chunk, language: lang)
                if !result.text.isEmpty { chunkResults.append(result) }
                break
            }

            let targetCut = offset + maxSamples
            let cutPoint = findSilenceCutPoint(in: samples, targetCut: targetCut, searchWindow: searchWindow)
            let chunk = Array(samples[offset..<cutPoint])
            let result = try model.transcribe(audio: chunk, language: lang)
            if !result.text.isEmpty { chunkResults.append(result) }
            offset = cutPoint
        }

        // Merge results
        let mergedText = chunkResults.map { $0.text }.joined(separator: " ")
        let totalTokens = chunkResults.reduce(0) { $0 + $1.tokenCount }
        let weightedLogProb = totalTokens > 0
            ? chunkResults.reduce(0.0) { $0 + $1.avgLogProb * Double($1.tokenCount) } / Double(totalTokens)
            : 0.0

        return WhisperCoreMLModelImpl.TranscriptionResult(
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
        model?.unload()
        model = nil
        loadedModelDir = nil
        hasCompletedWarmup = false
        Self.logger.info("Whisper CoreML model unloaded")
    }

    private func ensureWarmup(reason: String) throws {
        guard !hasCompletedWarmup else { return }
        guard let model = model else { return }

        let warmupSamples = [Float](repeating: 0, count: 16000)  // 1s silence
        var lastError: Error?
        let maxAttempts = 3

        for attempt in 1...maxAttempts {
            do {
                Self.logger.warning("Running Whisper CoreML warmup (\(reason), attempt \(attempt)/\(maxAttempts))...")
                let _ = try model.transcribe(audio: warmupSamples, language: nil)
                hasCompletedWarmup = true
                Self.logger.warning("Whisper CoreML warmup complete (\(reason), attempt \(attempt))")
                return
            } catch {
                lastError = error
                Self.logger.error("Whisper CoreML warmup attempt \(attempt) failed (\(reason)): \(error)")
            }
        }

        throw WhisperCoreMLEngineError.warmupFailed(
            underlying: lastError ?? WhisperCoreMLModelError.loadFailed("unknown warmup error")
        )
    }
}
