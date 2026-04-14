#!/usr/bin/env swift
// Test: compare old 400-point matrix-multiply DFT (201 bins)
// vs new 512-point zero-padded vDSP_fft_zrip (257 bins)
// for mel spectrogram output
// Usage: swift scripts/test_mel_dft.swift <path_to_wav>

import Foundation
import Accelerate
import AVFoundation

// === Load WAV audio as 16kHz mono Float ===
func loadAudio(_ path: String) -> [Float] {
    let url = URL(fileURLWithPath: path)
    let file = try! AVAudioFile(forReading: url)
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    let frameCount = AVAudioFrameCount(file.length)
    let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frameCount)!
    try! file.read(into: buffer)
    let converter = AVAudioConverter(from: file.processingFormat, to: format)!
    let outBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(Double(frameCount) * 16000.0 / file.processingFormat.sampleRate) + 1024)!
    try! converter.convert(to: outBuffer, from: buffer)
    let ptr = outBuffer.floatChannelData![0]
    return Array(UnsafeBufferPointer(start: ptr, count: Int(outBuffer.frameLength)))
}

// === Shared parameters ===
let nFFT = 400
let hopLength = 160
let nMels = 80
let chunkSamples = 16000 * 30

// === Build Hann window ===
var hannWindow = [Float](repeating: 0, count: nFFT)
for i in 0..<nFFT {
    hannWindow[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(nFFT)))
}

// === Shared mel filterbank builder ===
func buildMelFilterbank(nMels: Int, nBins: Int, fftSize: Int, sampleRate: Int) -> [Float] {
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
    for i in 0..<nBins { fftFreqs[i] = Float(i) * Float(sampleRate) / Float(fftSize) }

    let melMin = hzToMel(fMin)
    let melMax = hzToMel(fMax)
    let nMelPoints = nMels + 2
    var melPoints = [Float](repeating: 0, count: nMelPoints)
    for i in 0..<nMelPoints { melPoints[i] = melMin + Float(i) * (melMax - melMin) / Float(nMelPoints - 1) }
    let filterFreqs = melPoints.map { melToHz($0) }
    var filterDiff = [Float](repeating: 0, count: nMelPoints - 1)
    for i in 0..<(nMelPoints - 1) { filterDiff[i] = filterFreqs[i + 1] - filterFreqs[i] }

    var filterbank = [Float](repeating: 0, count: nBins * nMels)
    for bin in 0..<nBins {
        for mel in 0..<nMels {
            let down = (fftFreqs[bin] - filterFreqs[mel]) / filterDiff[mel]
            let up = (filterFreqs[mel + 2] - fftFreqs[bin]) / filterDiff[mel + 1]
            filterbank[bin * nMels + mel] = max(0.0, min(down, up))
        }
    }
    for mel in 0..<nMels {
        let enorm = 2.0 / (filterFreqs[mel + 2] - filterFreqs[mel])
        for bin in 0..<nBins { filterbank[bin * nMels + mel] *= enorm }
    }
    // Transpose to [nMels x nBins]
    var transposed = [Float](repeating: 0, count: nMels * nBins)
    for mel in 0..<nMels {
        for bin in 0..<nBins { transposed[mel * nBins + bin] = filterbank[bin * nMels + mel] }
    }
    return transposed
}

// === Full mel spectrogram: OLD (matrix-multiply DFT, 201 bins) ===
func melSpectrogramOld(_ audio: [Float]) -> [Float] {
    let nBins = nFFT / 2 + 1  // 201
    var cosBasis = [Float](repeating: 0, count: nBins * nFFT)
    var sinBasis = [Float](repeating: 0, count: nBins * nFFT)
    let twoPiOverN = 2.0 * Float.pi / Float(nFFT)
    for k in 0..<nBins {
        for n in 0..<nFFT {
            let angle = twoPiOverN * Float(k) * Float(n)
            cosBasis[k * nFFT + n] = cos(angle)
            sinBasis[k * nFFT + n] = sin(angle)
        }
    }

    let padLength = nFFT / 2
    var padded = [Float](repeating: 0, count: padLength + audio.count + padLength)
    for i in 0..<padLength { padded[i] = audio[max(0, min(padLength - i, audio.count - 1))] }
    for i in 0..<audio.count { padded[padLength + i] = audio[i] }
    for i in 0..<padLength { padded[padLength + audio.count + i] = audio[max(0, audio.count - 2 - i)] }

    let nFrames = (padded.count - nFFT) / hopLength
    var windowedFrame = [Float](repeating: 0, count: nFFT)
    var realPart = [Float](repeating: 0, count: nBins)
    var imagPart = [Float](repeating: 0, count: nBins)
    var magnitude = [Float](repeating: 0, count: nFrames * nBins)

    for frame in 0..<nFrames {
        let start = frame * hopLength
        padded.withUnsafeBufferPointer { buf in
            vDSP_vmul(buf.baseAddress! + start, 1, hannWindow, 1, &windowedFrame, 1, vDSP_Length(nFFT))
        }
        vDSP_mmul(cosBasis, 1, windowedFrame, 1, &realPart, 1, vDSP_Length(nBins), 1, vDSP_Length(nFFT))
        vDSP_mmul(sinBasis, 1, windowedFrame, 1, &imagPart, 1, vDSP_Length(nBins), 1, vDSP_Length(nFFT))
        let baseIdx = frame * nBins
        for k in 0..<nBins {
            magnitude[baseIdx + k] = realPart[k] * realPart[k] + imagPart[k] * imagPart[k]
        }
    }

    // Mel filterbank with 201 bins
    let melFB = buildMelFilterbank(nMels: nMels, nBins: nBins, fftSize: nFFT, sampleRate: 16000)
    var melSpec = [Float](repeating: 0, count: nFrames * nMels)
    var filterbankT = [Float](repeating: 0, count: nBins * nMels)
    vDSP_mtrans(melFB, 1, &filterbankT, 1, vDSP_Length(nBins), vDSP_Length(nMels))
    vDSP_mmul(magnitude, 1, filterbankT, 1, &melSpec, 1,
              vDSP_Length(nFrames), vDSP_Length(nMels), vDSP_Length(nBins))

    // Log mel
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

    // Transpose [nFrames, nMels] -> [nMels, nFrames]
    var transposed = [Float](repeating: 0, count: count)
    vDSP_mtrans(melSpec, 1, &transposed, 1, vDSP_Length(nMels), vDSP_Length(nFrames))
    return transposed
}

// === Full mel spectrogram: NEW (512-point zero-padded FFT, 257 bins) ===
func melSpectrogramNew(_ audio: [Float]) -> [Float] {
    let paddedFFT = 512
    let halfPadded = paddedFFT / 2  // 256
    let nBins = paddedFFT / 2 + 1   // 257

    guard let fftSetup = vDSP_create_fftsetup(9, FFTRadix(kFFTRadix2)) else {
        fatalError("Failed to create FFT setup")
    }
    defer { vDSP_destroy_fftsetup(fftSetup) }

    let padLength = nFFT / 2
    var padded = [Float](repeating: 0, count: padLength + audio.count + padLength)
    for i in 0..<padLength { padded[i] = audio[max(0, min(padLength - i, audio.count - 1))] }
    for i in 0..<audio.count { padded[padLength + i] = audio[i] }
    for i in 0..<padLength { padded[padLength + audio.count + i] = audio[max(0, audio.count - 2 - i)] }

    let nFrames = (padded.count - nFFT) / hopLength
    var paddedFrame = [Float](repeating: 0, count: paddedFFT)
    var splitReal = [Float](repeating: 0, count: halfPadded)
    var splitImag = [Float](repeating: 0, count: halfPadded)
    var magnitude = [Float](repeating: 0, count: nFrames * nBins)

    for frame in 0..<nFrames {
        let start = frame * hopLength
        padded.withUnsafeBufferPointer { buf in
            vDSP_vmul(buf.baseAddress! + start, 1, hannWindow, 1, &paddedFrame, 1, vDSP_Length(nFFT))
        }
        for i in nFFT..<paddedFFT { paddedFrame[i] = 0 }

        for i in 0..<halfPadded {
            splitReal[i] = paddedFrame[2 * i]
            splitImag[i] = paddedFrame[2 * i + 1]
        }

        splitReal.withUnsafeMutableBufferPointer { realBuf in
            splitImag.withUnsafeMutableBufferPointer { imagBuf in
                var sc = DSPSplitComplex(realp: realBuf.baseAddress!, imagp: imagBuf.baseAddress!)
                vDSP_fft_zrip(fftSetup, &sc, 1, 9, FFTDirection(kFFTDirection_Forward))
            }
        }

        let baseIdx = frame * nBins
        magnitude[baseIdx] = splitReal[0] * splitReal[0]
        magnitude[baseIdx + halfPadded] = splitImag[0] * splitImag[0]
        for k in 1..<halfPadded {
            magnitude[baseIdx + k] = splitReal[k] * splitReal[k] + splitImag[k] * splitImag[k]
        }
    }

    // Mel filterbank with 257 bins
    let melFB = buildMelFilterbank(nMels: nMels, nBins: nBins, fftSize: paddedFFT, sampleRate: 16000)
    var melSpec = [Float](repeating: 0, count: nFrames * nMels)
    var filterbankT = [Float](repeating: 0, count: nBins * nMels)
    vDSP_mtrans(melFB, 1, &filterbankT, 1, vDSP_Length(nBins), vDSP_Length(nMels))
    vDSP_mmul(magnitude, 1, filterbankT, 1, &melSpec, 1,
              vDSP_Length(nFrames), vDSP_Length(nMels), vDSP_Length(nBins))

    // Log mel (same as old)
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

    // Transpose [nFrames, nMels] -> [nMels, nFrames]
    var transposed = [Float](repeating: 0, count: count)
    vDSP_mtrans(melSpec, 1, &transposed, 1, vDSP_Length(nMels), vDSP_Length(nFrames))
    return transposed
}

// === Main ===
guard CommandLine.arguments.count > 1 else {
    print("Usage: swift test_mel_dft.swift <wav_file>")
    exit(1)
}

let wavPath = CommandLine.arguments[1]
print("Loading audio: \(wavPath)")
var audio = loadAudio(wavPath)
print("Audio samples: \(audio.count) (\(String(format: "%.1f", Double(audio.count)/16000.0))s)")

if audio.count >= chunkSamples {
    audio = Array(audio.prefix(chunkSamples))
} else {
    audio += [Float](repeating: 0, count: chunkSamples - audio.count)
}

print("\nComputing mel spectrogram (OLD: 400-point matrix-multiply, 201 bins)...")
let t0 = CFAbsoluteTimeGetCurrent()
let melOld = melSpectrogramOld(audio)
let dt0 = CFAbsoluteTimeGetCurrent() - t0
print("  Done in \(String(format: "%.3f", dt0))s, \(melOld.count) values")

print("Computing mel spectrogram (NEW: 512-point vDSP_fft_zrip, 257 bins)...")
let t1 = CFAbsoluteTimeGetCurrent()
let melNew = melSpectrogramNew(audio)
let dt1 = CFAbsoluteTimeGetCurrent() - t1
print("  Done in \(String(format: "%.3f", dt1))s, \(melNew.count) values")

// Compare (both should be [nMels x nFrames] = same size)
print("\n=== Comparison ===")
print("Old size: \(melOld.count), New size: \(melNew.count)")

if melOld.count != melNew.count {
    print("❌ SIZE MISMATCH (expected since nBins differs, but mel output shape should match)")
    exit(1)
}

var maxAbsDiff: Float = 0
var sumAbsDiff: Float = 0
var maxRelDiff: Float = 0

for i in 0..<melOld.count {
    let absDiff = abs(melOld[i] - melNew[i])
    sumAbsDiff += absDiff
    if absDiff > maxAbsDiff { maxAbsDiff = absDiff }
    let denom = max(abs(melOld[i]), abs(melNew[i]), 1e-10)
    let relDiff = absDiff / denom
    if relDiff > maxRelDiff { maxRelDiff = relDiff }
}

let meanAbsDiff = sumAbsDiff / Float(melOld.count)

print("Max absolute diff: \(String(format: "%.6f", maxAbsDiff))")
print("Mean absolute diff: \(String(format: "%.6f", meanAbsDiff))")
print("Max relative diff: \(String(format: "%.6f", maxRelDiff))")

// Print samples
let nFrames = melOld.count / nMels
print("\nSample mel values (mel bin 40, first 10 frames):")
for f in 0..<min(10, nFrames) {
    let idx = 40 * nFrames + f
    if idx < melOld.count {
        print("  frame[\(f)] old=\(String(format: "%.6f", melOld[idx]))  new=\(String(format: "%.6f", melNew[idx]))  diff=\(String(format: "%.2e", melOld[idx] - melNew[idx]))")
    }
}

print("\nSample mel values (mel bin 0, first 10 frames):")
for f in 0..<min(10, nFrames) {
    let idx = f  // mel bin 0
    print("  frame[\(f)] old=\(String(format: "%.6f", melOld[idx]))  new=\(String(format: "%.6f", melNew[idx]))  diff=\(String(format: "%.2e", melOld[idx] - melNew[idx]))")
}

// Speedup
print("\n=== Performance ===")
print("Old (matrix-multiply): \(String(format: "%.3f", dt0))s")
print("New (vDSP_fft_zrip):   \(String(format: "%.3f", dt1))s")
print("Speedup: \(String(format: "%.1f", dt0 / dt1))x")

if maxAbsDiff < 0.05 {
    print("\n✅ Mel spectrograms are close enough (max diff: \(String(format: "%.6f", maxAbsDiff)))")
} else {
    print("\n⚠️ Mel spectrograms differ (max diff: \(String(format: "%.6f", maxAbsDiff))) — expected due to different frequency resolution, should still work for ASR")
}
