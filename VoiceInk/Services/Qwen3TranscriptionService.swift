// Qwen3TranscriptionService.swift
// TranscriptionService implementation for Qwen3-ASR
// [AI-Claude: 2025-02-18]

import Foundation
import os

enum Qwen3ServiceError: Error {
    case invalidModel
    case invalidAudioData
}

/// Read Int16 PCM samples from a WAV file, correctly parsing chunk structure.
/// macOS Core Audio writes a FLLR padding chunk between fmt and data,
/// so the data chunk typically starts at byte 4096, not the naive 44.
func readWAVSamples(from url: URL) throws -> [Float] {
    let data = try Data(contentsOf: url)
    guard data.count > 12 else {
        throw Qwen3ServiceError.invalidAudioData
    }

    // Read a little-endian UInt32 from Data at the given offset
    func readUInt32(_ d: Data, at offset: Int) -> UInt32 {
        return UInt32(d[offset])
            | (UInt32(d[offset + 1]) << 8)
            | (UInt32(d[offset + 2]) << 16)
            | (UInt32(d[offset + 3]) << 24)
    }

    // Parse WAV chunks to find actual "data" chunk offset and size.
    // macOS Core Audio writes a FLLR padding chunk between fmt and data,
    // so the data chunk typically starts at byte 4096, not the naive 44.
    var dataOffset: Int?
    var dataSize: Int?
    var pos = 12 // skip RIFF header (12 bytes)
    while pos + 8 <= data.count {
        let isDataChunk = data[pos] == 0x64      // 'd'
            && data[pos + 1] == 0x61             // 'a'
            && data[pos + 2] == 0x74             // 't'
            && data[pos + 3] == 0x61             // 'a'
        let chunkSize = Int(readUInt32(data, at: pos + 4))
        if isDataChunk {
            dataOffset = pos + 8
            dataSize = chunkSize
            break
        }
        pos += 8 + chunkSize
    }

    guard let offset = dataOffset, let size = dataSize, offset + size <= data.count else {
        throw Qwen3ServiceError.invalidAudioData
    }

    // Convert Int16 PCM to Float
    let endOffset = offset + size
    return stride(from: offset, to: endOffset, by: 2).map { i in
        let short = Int16(data[i]) | (Int16(data[i + 1]) << 8)
        return max(-1.0, min(Float(short) / 32767.0, 1.0))
    }
}

class Qwen3TranscriptionService: TranscriptionService {
    private let engine = Qwen3ASREngine()
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "Qwen3TranscriptionService")

    /// Temporary language override for retry (e.g. "Japanese"). Takes priority over UserDefaults.
    var languageOverride: String?

    /// Average log-probability from the last transcription (for confidence routing)
    var lastAvgLogProb: Double = 0.0

    /// Detected language from the last auto-mode transcription (e.g. "Japanese", "Chinese")
    var lastDetectedLanguage: String?

    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String {
        guard let qwen3Model = model as? Qwen3Model else {
            throw Qwen3ServiceError.invalidModel
        }

        // Ensure model is loaded
        let modelDir = Qwen3ModelManager.modelDirectory(for: qwen3Model.modelId)
        try await engine.loadModel(from: modelDir, modelSize: qwen3Model.modelSize)

        // Read audio samples from WAV file
        let audioSamples = try readAudioSamples(from: audioURL)

        // Language: override takes priority, then UserDefaults
        let selectedLanguage = languageOverride ?? UserDefaults.standard.string(forKey: "SelectedLanguage")
        let prompt = UserDefaults.standard.string(forKey: "TranscriptionPrompt")

        logger.info("Transcribing with Qwen3-ASR, samples: \(audioSamples.count), language: \(selectedLanguage ?? "auto"), prompt: \(prompt?.prefix(50) ?? "none")")

        let result = try await engine.transcribe(samples: audioSamples, language: selectedLanguage, prompt: prompt)
        self.lastAvgLogProb = result.avgLogProb
        self.lastDetectedLanguage = result.detectedLanguage
        await MainActor.run {
            ChinesePostProcessingService.shared.lastAvgLogProb = result.avgLogProb
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
