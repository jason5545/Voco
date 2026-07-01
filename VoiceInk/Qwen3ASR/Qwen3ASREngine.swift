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
    private var loadedModelDirectory: URL?
    private var loadedUsesAudioAdapter = false
    private var loadedAdapterFingerprint: Qwen3ASRAdapterFingerprint?
    private var hasCompletedWarmup = false
    private var adapterMetadata: Qwen3ASRAdapterMetadata = .unavailable

    func loadModel(from directory: URL, modelSize: Qwen3ASRModelSize, usesAudioAdapter: Bool = true) throws {
        let modelId = modelSize.defaultModelId
        let currentAdapterFingerprint = usesAudioAdapter
            ? Qwen3ASRAudioAdapterLoader.fingerprint(in: directory)
            : nil

        if loadedModelId == modelId {
            if loadedUsesAudioAdapter != usesAudioAdapter, let model = model {
                Self.logger.info("Qwen3-ASR adapter usage changed, refreshing audio encoder for \(modelId)")
                if usesAudioAdapter {
                    try model.reloadAudioAdapter(from: directory)
                } else {
                    try model.reloadAudioEncoderBaseOnly(from: directory)
                }
                adapterMetadata = model.adapterMetadata
                loadedModelDirectory = directory
                loadedAdapterFingerprint = currentAdapterFingerprint
                loadedUsesAudioAdapter = usesAudioAdapter
                hasCompletedWarmup = false
                try ensureWarmup(using: model, modelId: modelId, reason: "loadModel(adapter-usage-refresh)")
                return
            } else if loadedAdapterFingerprint != currentAdapterFingerprint, let model = model {
                Self.logger.info("Qwen3-ASR adapter changed, refreshing audio encoder for \(modelId)")
                try model.reloadAudioAdapter(from: directory)
                adapterMetadata = model.adapterMetadata
                loadedModelDirectory = directory
                loadedAdapterFingerprint = currentAdapterFingerprint
                loadedUsesAudioAdapter = usesAudioAdapter
                hasCompletedWarmup = false
                try ensureWarmup(using: model, modelId: modelId, reason: "loadModel(adapter-refresh)")
                return
            } else if let model = model, !hasCompletedWarmup {
                Self.logger.warning("Model \(modelId) loaded but warmup not completed, retrying warmup")
                try ensureWarmup(using: model, modelId: modelId, reason: "loadModel(reuse)")
                Self.logger.info("Model \(modelId) already loaded, skipping")
                return
            } else {
                Self.logger.info("Model \(modelId) already loaded, skipping")
                return
            }
        }

        unloadModel()

        Self.logger.info("Loading Qwen3-ASR model: \(modelId)")
        let newModel = try Qwen3ASRModel(
            audioConfig: modelSize.audioConfig,
            textConfig: modelSize.textConfig
        )
        try newModel.load(from: directory, modelSize: modelSize, usesAudioAdapter: usesAudioAdapter)

        self.model = newModel
        self.loadedModelId = modelId
        self.loadedModelDirectory = directory
        self.loadedAdapterFingerprint = currentAdapterFingerprint
        self.loadedUsesAudioAdapter = usesAudioAdapter
        self.adapterMetadata = newModel.adapterMetadata
        self.hasCompletedWarmup = false
        Self.logger.info("Qwen3-ASR model loaded successfully")

        MetalBudget.pinMemory()

        try ensureWarmup(using: newModel, modelId: modelId, reason: "loadModel(new)")
    }

    private static let sampleRate = 16000
    /// Maximum samples per chunk: 60 seconds at 16kHz.
    /// Qwen3-ASR memory grows steeply with audio context; 20-minute chunks can exhaust RAM.
    private static let maxChunkDurationSeconds = 60
    private static let maxSamplesPerChunk = maxChunkDurationSeconds * sampleRate
    /// Search window for silence detection: ±10 seconds around the target cut point.
    private static let silenceSearchWindow = 10 * sampleRate
    /// RMS analysis window: 0.5 seconds
    private static let rmsWindowSize = sampleRate / 2  // 0.5s at 16kHz

    func transcribe(samples: [Float], language: String?, prompt: String? = nil, decodingOptions: Qwen3DecodingOptions = Qwen3DecodingOptions()) throws -> Qwen3ASRModel.TranscriptionResult {
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

        // Audio within the safe Qwen3 context window: single pass
        if samples.count <= Self.maxSamplesPerChunk {
            let result = try model.transcribe(audio: samples, sampleRate: Self.sampleRate, language: lang, prompt: prompt, decodingOptions: decodingOptions)
            Memory.clearCache()
            return result
        }

        // Longer audio: segment at silence points
        let sr = Self.sampleRate
        Self.logger.info("Audio exceeds \(Self.maxChunkDurationSeconds)s (\(samples.count / sr)s), segmenting at silence points...")
        var chunkResults: [Qwen3ASRModel.TranscriptionResult] = []
        var offset = 0
        while offset < samples.count {
            let remaining = samples.count - offset
            if remaining <= Self.maxSamplesPerChunk {
                // Last chunk: take everything
                let chunk = Array(samples[offset...])
                let result = try model.transcribe(audio: chunk, sampleRate: Self.sampleRate, language: lang, prompt: prompt, decodingOptions: decodingOptions)
                Memory.clearCache()
                if !result.text.isEmpty { chunkResults.append(result) }
                break
            }

            // Find the best silence point near the chunk boundary
            let cutPoint = Self.findSilenceCutPoint(in: samples, targetCut: offset + Self.maxSamplesPerChunk)
            let chunk = Array(samples[offset..<cutPoint])
            Self.logger.info("Chunk: \(offset / sr)s - \(cutPoint / sr)s (\(chunk.count / sr)s)")
            let result = try model.transcribe(audio: chunk, sampleRate: Self.sampleRate, language: lang, prompt: prompt, decodingOptions: decodingOptions)
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
        let mergedWordConfidences = chunkResults.flatMap { $0.wordConfidences }
        return Qwen3ASRModel.TranscriptionResult(text: mergedText, avgLogProb: weightedLogProb, tokenCount: totalTokens, detectedLanguage: chunkResults.first?.detectedLanguage, uncertainWords: mergedUncertainWords, wordConfidences: mergedWordConfidences)
    }

    func transcribeBaseOnlyForAdapterGuard(
        samples: [Float],
        language: String?,
        prompt: String? = nil,
        decodingOptions: Qwen3DecodingOptions = Qwen3DecodingOptions()
    ) throws -> Qwen3ASRModel.TranscriptionResult {
        guard let model = model,
              let loadedModelId = loadedModelId,
              let loadedModelDirectory = loadedModelDirectory
        else {
            throw Qwen3ASRModelError.textDecoderNotLoaded
        }

        let originalUsesAudioAdapter = loadedUsesAudioAdapter
        try model.reloadAudioEncoderBaseOnly(from: loadedModelDirectory)
        adapterMetadata = model.adapterMetadata
        loadedUsesAudioAdapter = false
        loadedAdapterFingerprint = nil
        hasCompletedWarmup = false

        defer {
            do {
                if originalUsesAudioAdapter {
                    try model.reloadAudioAdapter(from: loadedModelDirectory)
                } else {
                    try model.reloadAudioEncoderBaseOnly(from: loadedModelDirectory)
                }
                adapterMetadata = model.adapterMetadata
                loadedUsesAudioAdapter = originalUsesAudioAdapter
                loadedAdapterFingerprint = originalUsesAudioAdapter
                    ? Qwen3ASRAudioAdapterLoader.fingerprint(in: loadedModelDirectory)
                    : nil
                hasCompletedWarmup = false
            } catch {
                adapterMetadata = model.adapterMetadata
                loadedUsesAudioAdapter = model.adapterMetadata.adapterApplied
                loadedAdapterFingerprint = loadedUsesAudioAdapter
                    ? Qwen3ASRAudioAdapterLoader.fingerprint(in: loadedModelDirectory)
                    : nil
                hasCompletedWarmup = false
                Self.logger.error("Failed to restore Qwen3-ASR adapter after base guard retry for \(loadedModelId): \(error.localizedDescription, privacy: .public)")
            }
        }

        return try transcribe(
            samples: samples,
            language: language,
            prompt: prompt,
            decodingOptions: decodingOptions
        )
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

    func currentAdapterMetadata() -> Qwen3ASRAdapterMetadata {
        adapterMetadata
    }

    func unloadModel() {
        model?.audioEncoder.clearPosEmbeddingCache()
        model = nil
        loadedModelId = nil
        loadedModelDirectory = nil
        loadedUsesAudioAdapter = false
        loadedAdapterFingerprint = nil
        adapterMetadata = .unavailable
        hasCompletedWarmup = false
        MetalBudget.unpinMemory()
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
                let _ = try model.transcribe(audio: warmupSamples, sampleRate: Self.sampleRate, language: nil)
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
