// Qwen3AudioPreprocessing.swift
// Adapted from qwen3-asr-swift AudioPreprocessing.swift
// Fixed: fatalError → throws, class renamed to avoid whisper.cpp collision
// Fixed: Use exact 400-point DFT (201 bins) to match HuggingFace WhisperFeatureExtractor
// [AI-Claude: 2025-02-18] [AI-Claude: 2026-03-22]

import Foundation
import Accelerate
import MLX

enum Qwen3PreprocessingError: Error, LocalizedError {
    case melFilterbankNotInitialized

    var errorDescription: String? {
        switch self {
        case .melFilterbankNotInitialized:
            return "Mel filterbank not initialized"
        }
    }
}

/// Whisper-style feature extractor for Qwen3-ASR
/// Converts raw audio to mel spectrograms
class Qwen3FeatureExtractor {
    let sampleRate: Int = 16000
    let nFFT: Int = 400
    let hopLength: Int = 160
    let nMels: Int = 128
    let chunkLength: Int = 1200  // Qwen3-ASR supports up to 20 minutes (1200s) per inference

    private var melFilterbank: [Float]?
    private var hannWindow: [Float]

    // Exact 400-point DFT via precomputed twiddle matrices and vDSP_mmul.
    // vDSP_fft_zrip requires power-of-2 lengths, so the old code zero-padded
    // nFFT=400 → 512, producing 257 bins instead of 201. This caused mel
    // spectrograms to diverge from the HuggingFace WhisperFeatureExtractor
    // (which uses a true 400-point FFT with 201 bins), leading to different
    // transcription results from the same model weights.
    private let nBins: Int  // nFFT/2 + 1 = 201
    private var dftCosMatrix: [Float]  // (nFFT × nBins) row-major: cos(2πkn/N)
    private var dftSinMatrix: [Float]  // (nFFT × nBins) row-major: -sin(2πkn/N)

    init() throws {
        nBins = nFFT / 2 + 1  // 201

        hannWindow = [Float](repeating: 0, count: nFFT)
        for i in 0..<nFFT {
            hannWindow[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(nFFT)))
        }

        // Precompute DFT twiddle factor matrices for exact 400-point DFT.
        // X[k] = Σ x[n] * exp(-j2πkn/N) = Σ x[n] * (cos(2πkn/N) - j·sin(2πkn/N))
        // Layout: dftCosMatrix[n * nBins + k] for efficient (nFrames,nFFT)×(nFFT,nBins) multiply.
        dftCosMatrix = [Float](repeating: 0, count: nFFT * nBins)
        dftSinMatrix = [Float](repeating: 0, count: nFFT * nBins)
        let twoPiOverN = 2.0 * Float.pi / Float(nFFT)
        for k in 0..<nBins {
            for n in 0..<nFFT {
                let angle = twoPiOverN * Float(k) * Float(n)
                dftCosMatrix[n * nBins + k] = cos(angle)
                dftSinMatrix[n * nBins + k] = -sin(angle)
            }
        }

        setupMelFilterbank()
    }

    private func setupMelFilterbank() {
        let fMin: Float = 0.0
        let fMax: Float = Float(sampleRate) / 2.0
        let minLogHertz: Float = 1000.0
        let minLogMel: Float = 15.0
        let logstepHzToMel: Float = 27.0 / log(6.4)
        let logstepMelToHz: Float = log(6.4) / 27.0

        func hzToMel(_ hz: Float) -> Float {
            if hz < minLogHertz {
                return 3.0 * hz / 200.0
            } else {
                return minLogMel + log(hz / minLogHertz) * logstepHzToMel
            }
        }

        func melToHz(_ mel: Float) -> Float {
            if mel < minLogMel {
                return 200.0 * mel / 3.0
            } else {
                return minLogHertz * exp((mel - minLogMel) * logstepMelToHz)
            }
        }

        // Use nFFT-based frequency bins (201 bins at 40 Hz spacing)
        // to match HuggingFace WhisperFeatureExtractor
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
                let filterValue = max(0.0, min(downSlope, upSlope))
                filterbank[bin * nMels + mel] = filterValue
            }
        }

        for mel in 0..<nMels {
            let enorm = 2.0 / (filterFreqs[mel + 2] - filterFreqs[mel])
            for bin in 0..<nBins {
                filterbank[bin * nMels + mel] *= enorm
            }
        }

        var filterbankTransposed = [Float](repeating: 0, count: nMels * nBins)
        for mel in 0..<nMels {
            for bin in 0..<nBins {
                filterbankTransposed[mel * nBins + bin] = filterbank[bin * nMels + mel]
            }
        }

        self.melFilterbank = filterbankTransposed
    }

    func extractFeatures(_ audio: [Float]) throws -> MLXArray {
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

        let nFrames = (paddedAudio.count - nFFT) / hopLength + 1

        // Build windowed frames matrix: (nFrames, nFFT)
        var windowedFrames = [Float](repeating: 0, count: nFrames * nFFT)
        paddedAudio.withUnsafeBufferPointer { buf in
            for frame in 0..<nFrames {
                let start = frame * hopLength
                let destOffset = frame * nFFT
                windowedFrames.withUnsafeMutableBufferPointer { dest in
                    vDSP_vmul(buf.baseAddress! + start, 1, hannWindow, 1,
                              dest.baseAddress! + destOffset, 1, vDSP_Length(nFFT))
                }
            }
        }

        // Exact 400-point DFT via batched matrix multiply:
        // realParts = windowedFrames × dftCosMatrix  → (nFrames, nBins)
        // imagParts = windowedFrames × dftSinMatrix  → (nFrames, nBins)
        let totalBins = nFrames * nBins
        var realParts = [Float](repeating: 0, count: totalBins)
        var imagParts = [Float](repeating: 0, count: totalBins)

        vDSP_mmul(windowedFrames, 1, dftCosMatrix, 1, &realParts, 1,
                  vDSP_Length(nFrames), vDSP_Length(nBins), vDSP_Length(nFFT))
        vDSP_mmul(windowedFrames, 1, dftSinMatrix, 1, &imagParts, 1,
                  vDSP_Length(nFrames), vDSP_Length(nBins), vDSP_Length(nFFT))

        // Power spectrum: magnitude[i] = real[i]² + imag[i]²
        var magnitude = [Float](repeating: 0, count: totalBins)
        vDSP_vsq(realParts, 1, &magnitude, 1, vDSP_Length(totalBins))
        var imagSq = [Float](repeating: 0, count: totalBins)
        vDSP_vsq(imagParts, 1, &imagSq, 1, vDSP_Length(totalBins))
        vDSP_vadd(magnitude, 1, imagSq, 1, &magnitude, 1, vDSP_Length(totalBins))

        // Apply mel filterbank: melSpec = magnitude × filterbankT
        guard let filterbank = melFilterbank else {
            throw Qwen3PreprocessingError.melFilterbankNotInitialized
        }

        var melSpec = [Float](repeating: 0, count: nFrames * nMels)
        var filterbankT = [Float](repeating: 0, count: nBins * nMels)
        vDSP_mtrans(filterbank, 1, &filterbankT, 1, vDSP_Length(nBins), vDSP_Length(nMels))

        vDSP_mmul(magnitude, 1, filterbankT, 1, &melSpec, 1,
                  vDSP_Length(nFrames), vDSP_Length(nMels), vDSP_Length(nBins))

        // Log-mel: log10(max(mel, 1e-10)), clamp to max-8, then (x+4)/4
        let count = melSpec.count
        var countN = Int32(count)
        var epsilon: Float = 1e-10
        vDSP_vclip(melSpec, 1, &epsilon, [Float.greatestFiniteMagnitude], &melSpec, 1, vDSP_Length(count))
        vvlog10f(&melSpec, melSpec, &countN)

        var maxVal: Float = -Float.infinity
        vDSP_maxv(melSpec, 1, &maxVal, vDSP_Length(count))

        var minClamp = maxVal - 8.0
        var maxClamp = Float.greatestFiniteMagnitude
        vDSP_vclip(melSpec, 1, &minClamp, &maxClamp, &melSpec, 1, vDSP_Length(count))

        var scale: Float = 0.25
        var offset: Float = 1.0
        vDSP_vsmsa(melSpec, 1, &scale, &offset, &melSpec, 1, vDSP_Length(count))

        // Trim last frame (matches Python WhisperFeatureExtractor: log_spec[:, :-1])
        let trimmedFrames = nFrames - 1
        let trimmedMelSpec = Array(melSpec.prefix(trimmedFrames * nMels))

        let maxFrames = chunkLength * sampleRate / hopLength
        var finalMelSpec = trimmedMelSpec

        if trimmedFrames > maxFrames {
            finalMelSpec = Array(trimmedMelSpec.prefix(maxFrames * nMels))
        }

        let finalFrames = finalMelSpec.count / nMels
        let array = MLXArray(finalMelSpec, [finalFrames, nMels])
        return array.transposed(1, 0)
    }

    func process(_ audio: [Float], sampleRate inputSampleRate: Int) throws -> MLXArray {
        var processedAudio = audio

        if inputSampleRate != sampleRate {
            processedAudio = resample(audio, from: inputSampleRate, to: sampleRate)
        }

        return try extractFeatures(processedAudio)
    }

    private func resample(_ audio: [Float], from inputRate: Int, to outputRate: Int) -> [Float] {
        let ratio = Double(outputRate) / Double(inputRate)
        let outputLength = Int(Double(audio.count) * ratio)

        guard outputLength > 0 else { return [] }

        var output = [Float](repeating: 0, count: outputLength)
        for i in 0..<outputLength {
            let srcIndex = Double(i) / ratio
            let srcIndexFloor = Int(srcIndex)
            let srcIndexCeil = min(srcIndexFloor + 1, audio.count - 1)
            let fraction = Float(srcIndex - Double(srcIndexFloor))
            output[i] = audio[srcIndexFloor] * (1 - fraction) + audio[srcIndexCeil] * fraction
        }

        return output
    }
}
