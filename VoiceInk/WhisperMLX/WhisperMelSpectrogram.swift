// WhisperMelSpectrogram.swift
// Audio preprocessing: 16kHz PCM → mel spectrogram for Whisper
// [AI-Claude: 2026-02-27]

import Foundation
import Accelerate
import MLX

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
class WhisperMelSpectrogram {
    let sampleRate: Int = 16000
    let nFFT: Int = 400
    let hopLength: Int = 160
    let nMels: Int
    /// Whisper processes 30-second chunks (or pads to 30s)
    static let chunkSamples = 16000 * 30  // 480,000 samples = 30 seconds

    private var melFilterbank: [Float]  // [nMels x nBins] row-major
    private var hannWindow: [Float]
    private var cosBasis: [Float]
    private var sinBasis: [Float]

    init(nMels: Int = 80) throws {
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

        melFilterbank = WhisperMelSpectrogram.buildMelFilterbank(
            nMels: nMels, nFFT: nFFT, sampleRate: sampleRate
        )
    }

    /// Build mel filterbank matrix [nMels x nBins]
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

        // Build filterbank [nBins x nMels] then transpose
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

        // Transpose to [nMels x nBins]
        var transposed = [Float](repeating: 0, count: nMels * nBins)
        for mel in 0..<nMels {
            for bin in 0..<nBins {
                transposed[mel * nBins + bin] = filterbank[bin * nMels + mel]
            }
        }
        return transposed
    }

    /// Process audio samples into padded 30-second mel spectrogram
    /// Returns MLXArray of shape [nMels, 3000] (30s at 16kHz with hop=160)
    func process(_ audio: [Float]) throws -> MLXArray {
        // Pad or truncate to exactly 30 seconds
        let targetSamples = Self.chunkSamples
        var padded: [Float]
        if audio.count >= targetSamples {
            padded = Array(audio.prefix(targetSamples))
        } else {
            padded = audio + [Float](repeating: 0, count: targetSamples - audio.count)
        }

        return try extractFeatures(padded)
    }

    private func extractFeatures(_ audio: [Float]) throws -> MLXArray {
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

        for frame in 0..<nFrames {
            let start = frame * hopLength
            paddedAudio.withUnsafeBufferPointer { buf in
                vDSP_vmul(buf.baseAddress! + start, 1, hannWindow, 1, &windowedFrame, 1, vDSP_Length(nFFT))
            }

            // 400-point DFT via matrix-vector multiply
            vDSP_mmul(cosBasis, 1, windowedFrame, 1, &realPart, 1,
                      vDSP_Length(nBins), 1, vDSP_Length(nFFT))
            vDSP_mmul(sinBasis, 1, windowedFrame, 1, &imagPart, 1,
                      vDSP_Length(nBins), 1, vDSP_Length(nFFT))

            let baseIdx = frame * nBins
            for k in 0..<nBins {
                magnitude[baseIdx + k] = realPart[k] * realPart[k] + imagPart[k] * imagPart[k]
            }
        }

        // Apply mel filterbank: [nFrames x nBins] @ [nBins x nMels] = [nFrames x nMels]
        var melSpec = [Float](repeating: 0, count: nFrames * nMels)
        var filterbankT = [Float](repeating: 0, count: nBins * nMels)
        vDSP_mtrans(melFilterbank, 1, &filterbankT, 1, vDSP_Length(nBins), vDSP_Length(nMels))
        vDSP_mmul(magnitude, 1, filterbankT, 1, &melSpec, 1,
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

        // Shape: [nMels, nFrames] (transposed)
        let array = MLXArray(melSpec, [nFrames, nMels])
        return array.transposed(1, 0)
    }
}
