// WhisperMelSpectrogram.swift
// MLX wrapper around WhisperMelSpectrogramCore: [Float] → MLXArray
// [AI-Claude: 2026-03-02]

import Foundation
import MLX

/// Whisper mel spectrogram extractor that returns MLXArray
/// Delegates all DSP work to WhisperMelSpectrogramCore
class WhisperMelSpectrogram {
    private let core: WhisperMelSpectrogramCore

    var nMels: Int { core.nMels }

    init(nMels: Int = 80) {
        self.core = WhisperMelSpectrogramCore(nMels: nMels)
    }

    /// Process audio samples into padded 30-second mel spectrogram
    /// Returns MLXArray of shape [nMels, 3000] (30s at 16kHz with hop=160)
    func process(_ audio: [Float]) -> MLXArray {
        let features = core.process(audio)
        let nFrames = core.nFrames
        // features is already transposed to [nMels, nFrames]
        return MLXArray(features, [nMels, nFrames])
    }
}
