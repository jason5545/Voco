import Foundation
import AVFoundation
import os

class AudioProcessor {
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "AudioProcessor")

    private enum ProcessingLimits {
        static let chunkDuration: TimeInterval = 10
        static let maxSafeDuration: TimeInterval = 3 * 60 * 60
    }
    
    struct AudioFormat {
        static let targetSampleRate: Double = 16000.0
        static let targetChannels: UInt32 = 1
        static let targetBitDepth: UInt32 = 16
    }
    
    enum AudioProcessingError: LocalizedError {
        case invalidAudioFile
        case conversionFailed
        case exportFailed
        case unsupportedFormat
        case sampleExtractionFailed
        case invalidAudioMetadata
        case audioTooLong(duration: TimeInterval, limit: TimeInterval)
        
        var errorDescription: String? {
            switch self {
            case .invalidAudioFile:
                return "The audio file is invalid or corrupted"
            case .conversionFailed:
                return "Failed to convert the audio format"
            case .exportFailed:
                return "Failed to export the processed audio"
            case .unsupportedFormat:
                return "The audio format is not supported"
            case .sampleExtractionFailed:
                return "Failed to extract audio samples"
            case .invalidAudioMetadata:
                return "The audio file has invalid duration or sample metadata"
            case let .audioTooLong(duration, limit):
                return "Audio is too long to transcribe safely: \(Self.format(duration)) exceeds \(Self.format(limit))"
            }
        }

        private static func format(_ duration: TimeInterval) -> String {
            let minutes = Int(duration / 60)
            let seconds = Int(duration.truncatingRemainder(dividingBy: 60))
            return "\(minutes)m \(seconds)s"
        }
    }

    func transcodeToWhisperWav(_ url: URL, to destinationURL: URL) async throws -> TimeInterval {
        try await Task.detached(priority: .userInitiated) {
            try Self.transcodeToWhisperWavSync(url, to: destinationURL)
        }.value
    }
    
    func processAudioToSamples(_ url: URL) async throws -> [Float] {
        guard let audioFile = try? AVAudioFile(forReading: url) else {
            throw AudioProcessingError.invalidAudioFile
        }
        
        let format = audioFile.processingFormat
        let sampleRate = format.sampleRate
        let channels = format.channelCount
        let totalFrames = audioFile.length
        try Self.validateAudioMetadata(sampleRate: sampleRate, channels: channels, totalFrames: totalFrames)
        
        let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: AudioFormat.targetSampleRate,
            channels: AudioFormat.targetChannels,
            interleaved: false
        )
        
        guard let outputFormat = outputFormat else {
            throw AudioProcessingError.unsupportedFormat
        }
        
        let chunkSize = Self.chunkFrameCount(for: sampleRate)
        var allSamples: [Float] = []
        var currentFrame: AVAudioFramePosition = 0
        
        while currentFrame < totalFrames {
            let remainingFrames = totalFrames - currentFrame
            let framesToRead = min(chunkSize, AVAudioFrameCount(remainingFrames))
            
            guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: framesToRead) else {
                throw AudioProcessingError.conversionFailed
            }
            
            audioFile.framePosition = currentFrame
            try audioFile.read(into: inputBuffer, frameCount: framesToRead)
            
            if sampleRate == AudioFormat.targetSampleRate && channels == AudioFormat.targetChannels {
                let chunkSamples = convertToWhisperFormat(inputBuffer)
                allSamples.append(contentsOf: chunkSamples)
            } else {
                guard let converter = AVAudioConverter(from: format, to: outputFormat) else {
                    throw AudioProcessingError.conversionFailed
                }
                
                let ratio = AudioFormat.targetSampleRate / sampleRate
                let outputFrameCount = AVAudioFrameCount(Double(inputBuffer.frameLength) * ratio)
                
                guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputFrameCount) else {
                    throw AudioProcessingError.conversionFailed
                }
                
                var error: NSError?
                let status = converter.convert(
                    to: outputBuffer,
                    error: &error,
                    withInputFrom: { inNumPackets, outStatus in
                        outStatus.pointee = .haveData
                        return inputBuffer
                    }
                )
                
                if let error = error {
                    throw AudioProcessingError.conversionFailed
                }
                
                if status == .error {
                    throw AudioProcessingError.conversionFailed
                }
                
                let chunkSamples = convertToWhisperFormat(outputBuffer)
                allSamples.append(contentsOf: chunkSamples)
            }
            
            currentFrame += AVAudioFramePosition(framesToRead)
        }
        
        return allSamples
    }

    private static func transcodeToWhisperWavSync(_ sourceURL: URL, to destinationURL: URL) throws -> TimeInterval {
        guard let audioFile = try? AVAudioFile(forReading: sourceURL) else {
            throw AudioProcessingError.invalidAudioFile
        }

        let inputFormat = audioFile.processingFormat
        let sampleRate = inputFormat.sampleRate
        let channels = inputFormat.channelCount
        let totalFrames = audioFile.length
        let duration = try validateAudioMetadata(sampleRate: sampleRate, channels: channels, totalFrames: totalFrames)

        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: AudioFormat.targetSampleRate,
            channels: AudioFormat.targetChannels,
            interleaved: true
        ) else {
            throw AudioProcessingError.unsupportedFormat
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw AudioProcessingError.conversionFailed
        }

        let fileManager = FileManager.default
        try fileManager.createDirectory(
            at: destinationURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if fileManager.fileExists(atPath: destinationURL.path) {
            try fileManager.removeItem(at: destinationURL)
        }

        do {
            let outputFile = try AVAudioFile(
                forWriting: destinationURL,
                settings: outputFormat.settings,
                commonFormat: .pcmFormatInt16,
                interleaved: true
            )

            let chunkSize = chunkFrameCount(for: sampleRate)
            var currentFrame: AVAudioFramePosition = 0

            while currentFrame < totalFrames {
                try Task.checkCancellation()

                try autoreleasepool {
                    let remainingFrames = totalFrames - currentFrame
                    let framesToRead = min(chunkSize, AVAudioFrameCount(remainingFrames))

                    guard let inputBuffer = AVAudioPCMBuffer(
                        pcmFormat: inputFormat,
                        frameCapacity: framesToRead
                    ) else {
                        throw AudioProcessingError.conversionFailed
                    }

                    try audioFile.read(into: inputBuffer, frameCount: framesToRead)
                    guard inputBuffer.frameLength > 0 else {
                        currentFrame = totalFrames
                        return
                    }

                    try convertChunk(
                        inputBuffer,
                        converter: converter,
                        outputFormat: outputFormat,
                        outputFile: outputFile
                    )

                    currentFrame += AVAudioFramePosition(inputBuffer.frameLength)
                }
            }

            try drainConverter(converter, outputFormat: outputFormat, outputFile: outputFile)
            return duration
        } catch {
            try? fileManager.removeItem(at: destinationURL)
            throw error
        }
    }

    private static func convertChunk(
        _ inputBuffer: AVAudioPCMBuffer,
        converter: AVAudioConverter,
        outputFormat: AVAudioFormat,
        outputFile: AVAudioFile
    ) throws {
        let ratio = outputFormat.sampleRate / inputBuffer.format.sampleRate
        let capacity = AVAudioFrameCount(max(1, ceil(Double(inputBuffer.frameLength) * ratio) + 1024))

        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: capacity
        ) else {
            throw AudioProcessingError.conversionFailed
        }

        var didProvideInput = false
        var conversionError: NSError?
        let status = converter.convert(
            to: outputBuffer,
            error: &conversionError,
            withInputFrom: { _, outStatus in
                if didProvideInput {
                    outStatus.pointee = .noDataNow
                    return nil
                }

                didProvideInput = true
                outStatus.pointee = .haveData
                return inputBuffer
            }
        )

        if conversionError != nil || status == .error {
            throw AudioProcessingError.conversionFailed
        }

        if outputBuffer.frameLength > 0 {
            try outputFile.write(from: outputBuffer)
        }
    }

    private static func drainConverter(
        _ converter: AVAudioConverter,
        outputFormat: AVAudioFormat,
        outputFile: AVAudioFile
    ) throws {
        while true {
            guard let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: outputFormat,
                frameCapacity: 4096
            ) else {
                throw AudioProcessingError.conversionFailed
            }

            var conversionError: NSError?
            let status = converter.convert(
                to: outputBuffer,
                error: &conversionError,
                withInputFrom: { _, outStatus in
                    outStatus.pointee = .endOfStream
                    return nil
                }
            )

            if conversionError != nil || status == .error {
                throw AudioProcessingError.conversionFailed
            }

            if outputBuffer.frameLength > 0 {
                try outputFile.write(from: outputBuffer)
            }

            if status == .endOfStream || outputBuffer.frameLength == 0 {
                break
            }
        }
    }

    private static func validateAudioMetadata(
        sampleRate: Double,
        channels: AVAudioChannelCount,
        totalFrames: AVAudioFramePosition
    ) throws -> TimeInterval {
        guard sampleRate.isFinite, sampleRate > 0, channels > 0, totalFrames > 0 else {
            throw AudioProcessingError.invalidAudioMetadata
        }

        let duration = Double(totalFrames) / sampleRate
        guard duration.isFinite, duration > 0 else {
            throw AudioProcessingError.invalidAudioMetadata
        }

        guard duration <= ProcessingLimits.maxSafeDuration else {
            throw AudioProcessingError.audioTooLong(
                duration: duration,
                limit: ProcessingLimits.maxSafeDuration
            )
        }

        return duration
    }

    private static func chunkFrameCount(for sampleRate: Double) -> AVAudioFrameCount {
        let frames = max(1024, Int(sampleRate * ProcessingLimits.chunkDuration))
        return AVAudioFrameCount(min(frames, Int(UInt32.max)))
    }
    
    private func convertToWhisperFormat(_ buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channelData = buffer.floatChannelData else {
            return []
        }
        
        let channelCount = Int(buffer.format.channelCount)
        let frameLength = Int(buffer.frameLength)
        var samples = Array(repeating: Float(0), count: frameLength)
        
        if channelCount == 1 {
            samples = Array(UnsafeBufferPointer(start: channelData[0], count: frameLength))
        } else {
            for frame in 0..<frameLength {
                var sum: Float = 0
                for channel in 0..<channelCount {
                    sum += channelData[channel][frame]
                }
                samples[frame] = sum / Float(channelCount)
            }
        }
        
        let maxSample = samples.map(abs).max() ?? 1
        if maxSample > 0 {
            samples = samples.map { $0 / maxSample }
        }
        
        return samples
    }
    func saveSamplesAsWav(samples: [Float], to url: URL) throws {
        let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: AudioFormat.targetSampleRate,
            channels: AudioFormat.targetChannels,
            interleaved: true
        )

        guard let outputFormat = outputFormat else {
            throw AudioProcessingError.unsupportedFormat
        }

        let buffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: AVAudioFrameCount(samples.count)
        )
        
        guard let buffer = buffer else {
            throw AudioProcessingError.conversionFailed
        }
        
        // Convert float samples to int16
        let int16Samples = samples.map { max(-1.0, min(1.0, $0)) * Float(Int16.max) }.map { Int16($0) }

        // Copy samples to buffer
        int16Samples.withUnsafeBufferPointer { int16Buffer in
            let int16Pointer = int16Buffer.baseAddress!
            buffer.int16ChannelData![0].update(from: int16Pointer, count: int16Samples.count)
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)

        // Create audio file
        let audioFile = try AVAudioFile(
            forWriting: url,
            settings: outputFormat.settings,
            commonFormat: .pcmFormatInt16,
            interleaved: true
        )

        try audioFile.write(from: buffer)
    }
} 
