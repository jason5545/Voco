// KeyboardAudioRecorder.swift
// AVAudioSession-based recorder for iOS keyboard extension
// Outputs 16kHz mono Float32 samples directly compatible with CoreML Whisper
// [AI-Claude: 2026-03-02]

import AVFoundation
import os

enum KeyboardAudioRecorderError: Error, LocalizedError {
    case microphonePermissionDenied
    case engineStartFailed(Error)

    var errorDescription: String? {
        switch self {
        case .microphonePermissionDenied:
            return "Microphone permission denied. Enable 'Allow Full Access' in Settings."
        case .engineStartFailed(let error):
            return "Failed to start audio engine: \(error.localizedDescription)"
        }
    }
}

/// Records audio using AVAudioEngine, outputting 16kHz mono Float32 samples
class KeyboardAudioRecorder {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "KeyboardAudioRecorder")

    private let audioEngine = AVAudioEngine()
    private var sampleBuffer: [Float] = []
    private let targetSampleRate: Double = 16000
    private var converter: AVAudioConverter?
    private let bufferLock = NSLock()

    /// Current audio level (0.0 - 1.0) for UI metering
    private(set) var currentLevel: Float = 0

    /// Whether the recorder is currently recording
    var isRecording: Bool { audioEngine.isRunning }

    /// Request microphone permission
    /// - Returns: true if permission granted
    static func requestPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    /// Start recording audio
    func startRecording() async throws {
        guard await Self.requestPermission() else {
            throw KeyboardAudioRecorderError.microphonePermissionDenied
        }

        bufferLock.lock()
        sampleBuffer.removeAll()
        bufferLock.unlock()
        currentLevel = 0

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .default)
        try session.setActive(true)

        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        // Target format: 16kHz mono Float32
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw KeyboardAudioRecorderError.engineStartFailed(
                NSError(domain: "KeyboardAudioRecorder", code: -1,
                        userInfo: [NSLocalizedDescriptionKey: "Cannot create target audio format"])
            )
        }

        // Create converter if input format differs from target
        if inputFormat.sampleRate != targetSampleRate || inputFormat.channelCount != 1 {
            converter = AVAudioConverter(from: inputFormat, to: targetFormat)
        } else {
            converter = nil
        }

        let bufferSize: AVAudioFrameCount = 4096
        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) { [weak self] buffer, _ in
            self?.processAudioBuffer(buffer, targetFormat: targetFormat)
        }

        do {
            try audioEngine.start()
            Self.logger.info("Recording started (input: \(inputFormat.sampleRate)Hz \(inputFormat.channelCount)ch)")
        } catch {
            inputNode.removeTap(onBus: 0)
            throw KeyboardAudioRecorderError.engineStartFailed(error)
        }
    }

    /// Stop recording and return accumulated audio samples
    func stopRecording() -> [Float] {
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()

        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)

        bufferLock.lock()
        let samples = sampleBuffer
        sampleBuffer.removeAll()
        bufferLock.unlock()

        currentLevel = 0
        Self.logger.info("Recording stopped, \(samples.count) samples (\(String(format: "%.1f", Double(samples.count) / self.targetSampleRate))s)")
        return samples
    }

    // MARK: - Private

    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer, targetFormat: AVAudioFormat) {
        let samples: [Float]

        if let converter = converter {
            guard let inputBuffer = AVAudioPCMBuffer(
                pcmFormat: buffer.format,
                frameCapacity: buffer.frameLength
            ) else { return }
            inputBuffer.frameLength = buffer.frameLength
            
            if let src = buffer.floatChannelData, let dst = inputBuffer.floatChannelData {
                memcpy(dst[0], src[0], Int(buffer.frameLength) * MemoryLayout<Float>.size)
            }

            let ratio = targetSampleRate / buffer.format.sampleRate
            let outputFrameCount = AVAudioFrameCount(ceil(Double(buffer.frameLength) * ratio))
            guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputFrameCount) else {
                return
            }

            var error: NSError?
            let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                outStatus.pointee = .haveData
                return inputBuffer
            }

            converter.convert(to: outputBuffer, error: &error, withInputFrom: inputBlock)

            if let error = error {
                Self.logger.error("Audio conversion error: \(error)")
                return
            }

            guard let floatData = outputBuffer.floatChannelData else { return }
            let count = Int(outputBuffer.frameLength)
            samples = Array(UnsafeBufferPointer(start: floatData[0], count: count))
        } else {
            guard let floatData = buffer.floatChannelData else { return }
            let count = Int(buffer.frameLength)
            samples = Array(UnsafeBufferPointer(start: floatData[0], count: count))
        }

        if !samples.isEmpty {
            var rms: Float = 0
            for s in samples { rms += s * s }
            rms = sqrt(rms / Float(samples.count))
            currentLevel = min(1.0, rms * 5.0)
        }

        bufferLock.lock()
        sampleBuffer.append(contentsOf: samples)
        bufferLock.unlock()
    }
}
