// WhisperMelSpectrogramCore.swift
// Pure Swift mel spectrogram extractor using Accelerate (no MLX dependency)
// Shared by WhisperMLX (macOS) and WhisperCoreML (iOS) engines
// [AI-Claude: 2026-03-02]

import Foundation
import Accelerate

enum WhisperMelError: Error, LocalizedError {
    case melFilterbankNotInitialized

    var errorDescription: String? {
        switch self {
        case .melFilterbankNotInitialized:
            return "Mel filterbank not initialized"
        }
    }
}

/// Whisper-style mel spectrogram extractor
/// Converts 16kHz audio to 80-mel or 128-mel log spectrogram
/// Returns raw [Float] arrays; framework-specific wrappers convert to MLXArray or MLMultiArray
class WhisperMelSpectrogramCore {
    let sampleRate: Int = 16000
    let nFFT: Int = 400
    let hopLength: Int = 160
    let nMels: Int
    /// Whisper processes 30-second chunks (or pads to 30s)
    static let chunkSamples = 16000 * 30  // 480,000 samples = 30 seconds

    private var melFilterbank: [Float]  // [nBins x nMels] row-major, ready for vDSP_mmul
    private var hannWindow: [Float]
    // Precomputed DFT basis matrices for 400-point DFT via matrix-vector multiply.
    // Whisper models are trained with 400-point FFT (201 bins); zero-padding to 512
    // changes frequency resolution and mel features, so we keep the exact DFT.
    private var cosBasis: [Float]
    private var sinBasis: [Float]

    init(nMels: Int = 80) {
        self.nMels = nMels
        let nBins = nFFT / 2 + 1  // 201

        hannWindow = [Float](repeating: 0, count: nFFT)
        for i in 0..<nFFT {
            hannWindow[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(nFFT)))
        }

        // Precompute DFT twiddle factors (400-point, not power of 2)
        cosBasis = [Float](repeating: 0, count: nBins * nFFT)
        sinBasis = [Float](repeating: 0, count: nBins * nFFT)
        let twoPiOverN = 2.0 * Float.pi / Float(nFFT)
        for k in 0..<nBins {
            for n in 0..<nFFT {
                let angle = twoPiOverN * Float(k) * Float(n)
                cosBasis[k * nFFT + n] = cos(angle)
                sinBasis[k * nFFT + n] = sin(angle)
            }
        }

        melFilterbank = WhisperMelSpectrogramCore.buildMelFilterbank(
            nMels: nMels, nFFT: nFFT, sampleRate: sampleRate
        )
    }

    /// Build mel filterbank matrix [nBins x nMels]
    private static func buildMelFilterbank(nMels: Int, nFFT: Int, sampleRate: Int) -> [Float] {
        let nBins = nFFT / 2 + 1
        let fMin: Float = 0.0
        let fMax: Float = Float(sampleRate) / 2.0
        let minLogHertz: Float = 1000.0
        let minLogMel: Float = 15.0
        let logstepHzToMel: Float = 27.0 / log(6.4)
        let logstepMelToHz: Float = log(6.4) / 27.0

        func hzToMel(_ hz: Float) -> Float {
            hz < minLogHertz ? 3.0 * hz / 200.0 : minLogMel + log(hz / minLogHertz) * logstepHzToMel
        }

        func melToHz(_ mel: Float) -> Float {
            mel < minLogMel ? 200.0 * mel / 3.0 : minLogHertz * exp((mel - minLogMel) * logstepMelToHz)
        }

        var fftFreqs = [Float](repeating: 0, count: nBins)
        for i in 0..<nBins {
            fftFreqs[i] = Float(i) * Float(sampleRate) / Float(nFFT)
        }

        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        let nMelPoints = nMels + 2
        var melPoints = [Float](repeating: 0, count: nMelPoints)
        for i in 0..<nMelPoints {
            melPoints[i] = melMin + Float(i) * (melMax - melMin) / Float(nMelPoints - 1)
        }
        let filterFreqs = melPoints.map { melToHz($0) }
        var filterDiff = [Float](repeating: 0, count: nMelPoints - 1)
        for i in 0..<(nMelPoints - 1) {
            filterDiff[i] = filterFreqs[i + 1] - filterFreqs[i]
        }

        var filterbank = [Float](repeating: 0, count: nBins * nMels)
        for bin in 0..<nBins {
            let fftFreq = fftFreqs[bin]
            for mel in 0..<nMels {
                let downSlope = (fftFreq - filterFreqs[mel]) / filterDiff[mel]
                let upSlope = (filterFreqs[mel + 2] - fftFreq) / filterDiff[mel + 1]
                filterbank[bin * nMels + mel] = max(0.0, min(downSlope, upSlope))
            }
        }

        // Slaney-style normalization
        for mel in 0..<nMels {
            let enorm = 2.0 / (filterFreqs[mel + 2] - filterFreqs[mel])
            for bin in 0..<nBins {
                filterbank[bin * nMels + mel] *= enorm
            }
        }

        // filterbank is already [nBins x nMels] row-major — the layout vDSP_mmul needs
        return filterbank
    }

    /// Process audio samples into 30-second mel spectrogram
    /// Returns [Float] of shape [nMels, nFrames] in row-major order (transposed)
    /// nFrames = chunkSamples / hopLength = 3000 for 30s audio
    func process(_ audio: [Float]) -> [Float] {
        // Pad or truncate to exactly 30 seconds
        let targetSamples = Self.chunkSamples
        var padded: [Float]
        if audio.count >= targetSamples {
            padded = Array(audio.prefix(targetSamples))
        } else {
            padded = audio + [Float](repeating: 0, count: targetSamples - audio.count)
        }

        return extractFeatures(padded)
    }

    /// Returns the number of frames for a 30-second chunk
    var nFrames: Int {
        let targetSamples = Self.chunkSamples
        let paddedLength = nFFT / 2 + targetSamples + nFFT / 2
        return (paddedLength - nFFT) / hopLength
    }

    private func extractFeatures(_ audio: [Float]) -> [Float] {
        let nBins = nFFT / 2 + 1

        // Reflect-pad by nFFT/2 on both sides
        let padLength = nFFT / 2
        var paddedAudio = [Float](repeating: 0, count: padLength + audio.count + padLength)
        for i in 0..<padLength {
            let srcIdx = min(padLength - i, audio.count - 1)
            paddedAudio[i] = audio[max(0, srcIdx)]
        }
        for i in 0..<audio.count {
            paddedAudio[padLength + i] = audio[i]
        }
        for i in 0..<padLength {
            let srcIdx = audio.count - 2 - i
            paddedAudio[padLength + audio.count + i] = audio[max(0, srcIdx)]
        }

        // Match OpenAI Whisper: STFT produces N+1 frames, drop the last one (stft[..., :-1])
        let nFrames = (paddedAudio.count - nFFT) / hopLength

        var windowedFrame = [Float](repeating: 0, count: nFFT)
        var realPart = [Float](repeating: 0, count: nBins)
        var imagPart = [Float](repeating: 0, count: nBins)
        var magnitude = [Float](repeating: 0, count: nFrames * nBins)

        paddedAudio.withUnsafeBufferPointer { src in
            magnitude.withUnsafeMutableBufferPointer { mag in
                for frame in 0..<nFrames {
                    let start = frame * hopLength
                    vDSP_vmul(src.baseAddress! + start, 1, hannWindow, 1,
                              &windowedFrame, 1, vDSP_Length(nFFT))

                    // 400-point DFT via matrix-vector multiply
                    vDSP_mmul(cosBasis, 1, windowedFrame, 1, &realPart, 1,
                              vDSP_Length(nBins), 1, vDSP_Length(nFFT))
                    vDSP_mmul(sinBasis, 1, windowedFrame, 1, &imagPart, 1,
                              vDSP_Length(nBins), 1, vDSP_Length(nFFT))

                    // Power spectrum: real² + imag² (square in-place, safe since next frame overwrites)
                    vDSP_vsq(realPart, 1, &realPart, 1, vDSP_Length(nBins))
                    vDSP_vsq(imagPart, 1, &imagPart, 1, vDSP_Length(nBins))
                    let baseIdx = frame * nBins
                    vDSP_vadd(realPart, 1, imagPart, 1, mag.baseAddress! + baseIdx, 1, vDSP_Length(nBins))
                }
            }
        }

        // Apply mel filterbank: [nFrames x nBins] @ [nBins x nMels] = [nFrames x nMels]
        var melSpec = [Float](repeating: 0, count: nFrames * nMels)
        vDSP_mmul(magnitude, 1, melFilterbank, 1, &melSpec, 1,
                  vDSP_Length(nFrames), vDSP_Length(nMels), vDSP_Length(nBins))

        // Log mel: log10(max(mel, 1e-10))
        let count = melSpec.count
        var countN = Int32(count)
        var epsilon: Float = 1e-10
        vDSP_vclip(melSpec, 1, &epsilon, [Float.greatestFiniteMagnitude], &melSpec, 1, vDSP_Length(count))
        vvlog10f(&melSpec, melSpec, &countN)

        // Clamp to max - 8.0
        var maxVal: Float = -Float.infinity
        vDSP_maxv(melSpec, 1, &maxVal, vDSP_Length(count))
        var minClamp = maxVal - 8.0
        var maxClamp = Float.greatestFiniteMagnitude
        vDSP_vclip(melSpec, 1, &minClamp, &maxClamp, &melSpec, 1, vDSP_Length(count))

        // Scale: (mel + 4.0) / 4.0
        var scale: Float = 0.25
        var offset: Float = 1.0
        vDSP_vsmsa(melSpec, 1, &scale, &offset, &melSpec, 1, vDSP_Length(count))

        // Transpose from [nFrames, nMels] to [nMels, nFrames]
        var transposed = [Float](repeating: 0, count: count)
        vDSP_mtrans(melSpec, 1, &transposed, 1, vDSP_Length(nMels), vDSP_Length(nFrames))

        return transposed
    }
}
