// WhisperCoreMLMelSpectrogram.swift
// CoreML wrapper around WhisperMelSpectrogramCore: [Float] → MLMultiArray
// [AI-Claude: 2026-03-02]

import Foundation
import CoreML

/// Whisper mel spectrogram extractor that returns MLMultiArray for CoreML
/// Delegates all DSP work to WhisperMelSpectrogramCore
class WhisperCoreMLMelSpectrogram {
    private let core: WhisperMelSpectrogramCore

    var nMels: Int { core.nMels }

    init(nMels: Int = 80) {
        self.core = WhisperMelSpectrogramCore(nMels: nMels)
    }

    /// Process audio samples into padded 30-second mel spectrogram
    /// Returns MLMultiArray of shape [1, nMels, 3000] Float16
    func process(_ audio: [Float]) throws -> MLMultiArray {
        let features = core.process(audio)
        let nFrames = core.nFrames

        // Create MLMultiArray [1, nMels, nFrames]
        let array = try MLMultiArray(shape: [1, nMels as NSNumber, nFrames as NSNumber], dataType: .float16)

        // Copy data: features is [nMels, nFrames] row-major
        let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: nMels * nFrames)
        for i in 0..<(nMels * nFrames) {
            ptr[i] = Float16(features[i])
        }

        return array
    }
}
