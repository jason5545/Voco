// Qwen3TranscriptionService.swift
// TranscriptionService implementation for Qwen3-ASR
// [AI-Claude: 2025-02-18]

import Foundation
import os

enum Qwen3ServiceError: Error, LocalizedError {
    case invalidModel
    case invalidAudioData
    case emptyAudioData

    var errorDescription: String? {
        switch self {
        case .invalidModel:
            return "Invalid Qwen3 model."
        case .invalidAudioData:
            return "Invalid audio data."
        case .emptyAudioData:
            return "No audio samples were captured."
        }
    }
}

enum WAVSampleReaderError: Error, LocalizedError {
    case invalidAudioData
    case emptyAudioData
    case unsupportedAudioFormat

    var errorDescription: String? {
        switch self {
        case .invalidAudioData:
            return "Invalid WAV audio data."
        case .emptyAudioData:
            return "No audio samples were captured."
        case .unsupportedAudioFormat:
            return "Unsupported WAV audio format."
        }
    }
}

/// Read Int16 PCM samples from a WAV file, correctly parsing chunk structure.
/// macOS Core Audio writes a FLLR padding chunk between fmt and data,
/// so the data chunk typically starts at byte 4096, not the naive 44.
func readWAVSamples(from url: URL) throws -> [Float] {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }

    func readExact(_ count: Int) throws -> Data {
        guard let data = try handle.read(upToCount: count), data.count == count else {
            throw WAVSampleReaderError.invalidAudioData
        }
        return data
    }

    func readUInt16(_ d: Data, at offset: Int) -> UInt16 {
        UInt16(d[offset])
            | (UInt16(d[offset + 1]) << 8)
    }

    func readUInt32(_ d: Data, at offset: Int) -> UInt32 {
        UInt32(d[offset])
            | (UInt32(d[offset + 1]) << 8)
            | (UInt32(d[offset + 2]) << 16)
            | (UInt32(d[offset + 3]) << 24)
    }

    let riffHeader = try readExact(12)
    guard riffHeader.starts(with: [0x52, 0x49, 0x46, 0x46]), // RIFF
          riffHeader[8] == 0x57, riffHeader[9] == 0x41,
          riffHeader[10] == 0x56, riffHeader[11] == 0x45 else { // WAVE
        throw WAVSampleReaderError.invalidAudioData
    }

    var audioFormat: UInt16?
    var channelCount: UInt16?
    var bitsPerSample: UInt16?
    var dataOffset: UInt64?
    var dataSize: UInt32?

    while true {
        guard let chunkHeader = try handle.read(upToCount: 8), !chunkHeader.isEmpty else {
            break
        }
        guard chunkHeader.count == 8 else {
            throw WAVSampleReaderError.invalidAudioData
        }

        let chunkID = Array(chunkHeader[0..<4])
        let chunkSize = readUInt32(chunkHeader, at: 4)
        let chunkDataOffset = try handle.offset()
        let paddedSize = UInt64(chunkSize) + UInt64(chunkSize % 2)

        switch chunkID {
        case [0x66, 0x6d, 0x74, 0x20]: // fmt
            guard chunkSize >= 16 else {
                throw WAVSampleReaderError.invalidAudioData
            }

            let bytesToRead = min(Int(chunkSize), 40)
            let formatData = try readExact(bytesToRead)
            audioFormat = readUInt16(formatData, at: 0)
            channelCount = readUInt16(formatData, at: 2)
            bitsPerSample = readUInt16(formatData, at: 14)
            try handle.seek(toOffset: chunkDataOffset + paddedSize)

        case [0x64, 0x61, 0x74, 0x61]: // data
            dataOffset = chunkDataOffset
            dataSize = chunkSize
            break

        default:
            try handle.seek(toOffset: chunkDataOffset + paddedSize)
        }

        if dataOffset != nil {
            break
        }
    }

    guard let offset = dataOffset, let size = dataSize else {
        throw WAVSampleReaderError.invalidAudioData
    }
    guard size > 0 else {
        throw WAVSampleReaderError.emptyAudioData
    }
    guard size >= 2, size.isMultiple(of: 2) else {
        throw WAVSampleReaderError.invalidAudioData
    }

    if let audioFormat, let channelCount, let bitsPerSample {
        let isPCM = audioFormat == 1 || audioFormat == 0xFFFE
        guard isPCM, channelCount == 1, bitsPerSample == 16 else {
            throw WAVSampleReaderError.unsupportedAudioFormat
        }
    }

    try handle.seek(toOffset: offset)

    var samples: [Float] = []
    samples.reserveCapacity(Int(size / 2))

    var remainingBytes = Int(size)
    let readChunkSize = 1_048_576
    while remainingBytes > 0 {
        let byteCount = min(remainingBytes, readChunkSize)
        guard let data = try handle.read(upToCount: byteCount), !data.isEmpty else {
            throw WAVSampleReaderError.invalidAudioData
        }

        data.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.bindMemory(to: UInt8.self).baseAddress else {
                return
            }

            var offset = 0
            while offset + 1 < data.count {
                let sample = UInt16(baseAddress[offset])
                    | (UInt16(baseAddress[offset + 1]) << 8)
                let value = Int16(bitPattern: sample)
                samples.append(max(-1.0, min(Float(value) / 32767.0, 1.0)))
                offset += 2
            }
        }

        remainingBytes -= data.count
    }

    return samples
}

class Qwen3TranscriptionService: TranscriptionService {
    private let engine = Qwen3ASREngine()
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "Qwen3TranscriptionService")

    /// Temporary language override for retry (e.g. "Japanese"). Takes priority over request context.
    var languageOverride: String?

    /// Average log-probability from the last transcription (for confidence routing)
    var lastAvgLogProb: Double = 0.0

    /// Detected language from the last auto-mode transcription (e.g. "Japanese", "Chinese")
    var lastDetectedLanguage: String?

    /// Low-confidence words from the last transcription (for LLM prompt injection)
    var lastUncertainWords: [UncertainWord] = []

    /// Per-word confidence scores from the last transcription (for post-processing routing)
    var lastWordConfidences: [WordConfidence] = []

    /// Audio-side LoRA adapter status from the last Qwen3-ASR model load.
    private(set) var lastAdapterMetadata: Qwen3ASRAdapterMetadata = .unavailable

    func transcribe(
        audioURL: URL,
        model: any TranscriptionModel,
        context: TranscriptionRequestContext
    ) async throws -> String {
        guard let qwen3Model = model as? Qwen3Model else {
            throw Qwen3ServiceError.invalidModel
        }

        // Ensure model is loaded
        let modelDir = Qwen3ModelManager.modelDirectory(for: qwen3Model.modelId)
        try await engine.loadModel(
            from: modelDir,
            modelSize: qwen3Model.modelSize,
            usesAudioAdapter: context.usesQwen3AudioAdapter
        )
        self.lastAdapterMetadata = await engine.currentAdapterMetadata()

        // Read audio samples from WAV file
        let audioSamples = try readAudioSamples(from: audioURL)

        let selectedLanguage = languageOverride ?? context.language
        let prompt = context.prompt

        logger.info("Transcribing with Qwen3-ASR, samples: \(audioSamples.count), language: \(selectedLanguage ?? "auto"), prompt: \(prompt?.prefix(50) ?? "none")")

        var result = try await engine.transcribe(samples: audioSamples, language: selectedLanguage, prompt: prompt)
        let audioDurationSeconds = Double(audioSamples.count) / 16_000.0
        if Qwen3ASRAdapterRuntimeGuard.shouldProbeBaseFallback(
            adapterTranscript: result.text,
            adapterMetadata: lastAdapterMetadata,
            audioDurationSeconds: audioDurationSeconds
        ) {
            do {
                let baseResult = try await engine.transcribeBaseOnlyForAdapterGuard(
                    samples: audioSamples,
                    language: selectedLanguage,
                    prompt: prompt
                )
                self.lastAdapterMetadata = await engine.currentAdapterMetadata()
                if Qwen3ASRAdapterRuntimeGuard.shouldUseBaseFallback(
                    adapterTranscript: result.text,
                    baseTranscript: baseResult.text
                ) {
                    logger.warning("Qwen3-ASR adapter guard used base fallback for long action-command transcript (duration: \(audioDurationSeconds, format: .fixed(precision: 3), privacy: .public)s)")
                    result = baseResult
                } else {
                    logger.info("Qwen3-ASR adapter guard kept adapter action-command transcript after base probe")
                }
            } catch {
                self.lastAdapterMetadata = await engine.currentAdapterMetadata()
                logger.error("Qwen3-ASR adapter guard base probe failed: \(error.localizedDescription, privacy: .public)")
            }
        }
        self.lastAvgLogProb = result.avgLogProb
        self.lastDetectedLanguage = result.detectedLanguage
        self.lastUncertainWords = result.uncertainWords
        self.lastWordConfidences = result.wordConfidences
        await MainActor.run {
            ChinesePostProcessingService.shared.lastAvgLogProb = result.avgLogProb
            ChinesePostProcessingService.shared.lastUncertainWords = result.uncertainWords
            ChinesePostProcessingService.shared.lastWordConfidences = result.wordConfidences
        }

        logger.info("Qwen3-ASR transcription complete (avgLogProb: \(String(format: "%.3f", result.avgLogProb)), tokens: \(result.tokenCount)): \(result.text.prefix(100))")
        return result.text
    }

    private func readAudioSamples(from url: URL) throws -> [Float] {
        return try readWAVSamples(from: url)
    }

    func cleanup() async {
        await engine.unloadModel()
    }
}
