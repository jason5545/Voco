// Qwen3AudioPreprocessing.swift
// Adapted from qwen3-asr-swift AudioPreprocessing.swift
// Fixed: fatalError → throws, class renamed to avoid whisper.cpp collision
// Fixed: Use exact 400-point DFT (201 bins) to match HuggingFace preprocessor_config.json (n_fft=400)
// Note: Ivan upstream zero-pads to 512 for vDSP_fft_zrip compatibility, but the model's
//       preprocessor_config.json specifies n_fft=400. Python mlx_audio uses np.fft.rfft(400)
//       producing 201 bins. Zero-padding to 512 (257 bins) changes mel filterbank mapping
//       and causes measurably different (worse) transcription results.
// [AI-Claude: 2025-02-18] [AI-Claude: 2026-03-22] [AI-Claude: 2026-04-03]

import Foundation
import Accelerate
import MLX

enum Qwen3PreprocessingError: Error, LocalizedError {
    case melFilterbankNotInitialized
    case emptyAudioInput

    var errorDescription: String? {
        switch self {
        case .melFilterbankNotInitialized:
            return "Mel filterbank not initialized"
        case .emptyAudioInput:
            return "No audio samples were captured"
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

    private var melFilterbankT: [Float]?  // (nBins × nMels) filterbank, precomputed for extractFeatures
    private var hannWindow: [Float]

    // Exact 400-point DFT via precomputed twiddle matrices and vDSP_mmul.
    // vDSP_fft_zrip requires power-of-2 lengths, so the old code zero-padded
    // nFFT=400 → 512, producing 257 bins instead of 201. This caused mel
    // spectrograms to diverge from the HuggingFace WhisperFeatureExtractor
    // (which uses a true 400-point FFT with 201 bins), leading to different
    // transcription results from the same model weights.
    private var nBins: Int { nFFT / 2 + 1 }  // 201
    private var dftCosMatrix: [Float]
    private var dftSinMatrix: [Float]

    init() throws {
        let bins = nFFT / 2 + 1

        hannWindow = [Float](repeating: 0, count: nFFT)
        for i in 0..<nFFT {
            hannWindow[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(nFFT)))
        }

        // Precompute DFT twiddle factor matrices for exact 400-point DFT.
        // X[k] = Σ x[n] * exp(-j2πkn/N) = Σ x[n] * (cos(2πkn/N) - j·sin(2πkn/N))
        // Layout: dftCosMatrix[n * bins + k] for efficient (nFrames,nFFT)×(nFFT,bins) multiply.
        dftCosMatrix = [Float](repeating: 0, count: nFFT * bins)
        dftSinMatrix = [Float](repeating: 0, count: nFFT * bins)
        let twoPiOverN = 2.0 * Float.pi / Float(nFFT)
        for k in 0..<bins {
            for n in 0..<nFFT {
                let angle = twoPiOverN * Float(k) * Float(n)
                dftCosMatrix[n * bins + k] = cos(angle)
                dftSinMatrix[n * bins + k] = -sin(angle)
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
        // to match HuggingFace WhisperFeatureExtractor (preprocessor_config.json: n_fft=400)
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

        // filterbank is already (nBins × nMels) row-major — the layout vDSP_mmul needs
        self.melFilterbankT = filterbank
    }

    func extractFeatures(_ audio: [Float]) throws -> MLXArray {
        guard !audio.isEmpty else {
            throw Qwen3PreprocessingError.emptyAudioInput
        }

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
        paddedAudio.withUnsafeBufferPointer { src in
            windowedFrames.withUnsafeMutableBufferPointer { dest in
                for frame in 0..<nFrames {
                    let start = frame * hopLength
                    let destOffset = frame * nFFT
                    vDSP_vmul(src.baseAddress! + start, 1, hannWindow, 1,
                              dest.baseAddress! + destOffset, 1, vDSP_Length(nFFT))
                }
            }
        }

        let totalBins = nFrames * nBins
        var realParts = [Float](repeating: 0, count: totalBins)
        var imagParts = [Float](repeating: 0, count: totalBins)

        vDSP_mmul(windowedFrames, 1, dftCosMatrix, 1, &realParts, 1,
                  vDSP_Length(nFrames), vDSP_Length(nBins), vDSP_Length(nFFT))
        vDSP_mmul(windowedFrames, 1, dftSinMatrix, 1, &imagParts, 1,
                  vDSP_Length(nFrames), vDSP_Length(nBins), vDSP_Length(nFFT))

        // Power spectrum: real² + imag² (square imagParts in-place to avoid extra allocation)
        var magnitude = [Float](repeating: 0, count: totalBins)
        vDSP_vsq(realParts, 1, &magnitude, 1, vDSP_Length(totalBins))
        vDSP_vsq(imagParts, 1, &imagParts, 1, vDSP_Length(totalBins))
        vDSP_vadd(magnitude, 1, imagParts, 1, &magnitude, 1, vDSP_Length(totalBins))

        guard let filterbankT = melFilterbankT else {
            throw Qwen3PreprocessingError.melFilterbankNotInitialized
        }

        var melSpec = [Float](repeating: 0, count: nFrames * nMels)
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
        // and clamp to max chunk length in a single copy
        let maxFrames = chunkLength * sampleRate / hopLength
        let finalFrames = min(nFrames - 1, maxFrames)
        let finalMelSpec = Array(melSpec.prefix(finalFrames * nMels))
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
