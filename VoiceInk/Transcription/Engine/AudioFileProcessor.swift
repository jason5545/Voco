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
        let worker = Task.detached(priority: .userInitiated) {
            try Self.transcodeToWhisperWavSync(url, to: destinationURL)
        }
        do {
            let duration = try await withTaskCancellationHandler(operation: {
                try await worker.value
            }, onCancel: {
                worker.cancel()
            })
            try Task.checkCancellation()
            return duration
        } catch {
            worker.cancel()
            if Task.isCancelled {
                try? FileManager.default.removeItem(at: destinationURL)
                throw CancellationError()
            }
            try Task.checkCancellation()
            guard !Self.isPolicyError(error) else { throw error }
            logger.warning("AVAudioFile import failed for \(url.lastPathComponent, privacy: .public): \(error.localizedDescription, privacy: .public). Falling back to AVAssetReader.")
            do {
                return try await transcodeUsingAssetReader(url, to: destinationURL)
            } catch {
                try? FileManager.default.removeItem(at: destinationURL)
                throw error
            }
        }
    }
    
    func processAudioToSamples(_ url: URL) async throws -> [Float] {
        do {
            return try readUsingAudioFile(url)
        } catch {
            try Task.checkCancellation()
            guard !Self.isPolicyError(error) else { throw error }
            logger.warning("AVAudioFile sample extraction failed for \(url.lastPathComponent, privacy: .public): \(error.localizedDescription, privacy: .public). Falling back to AVAssetReader.")
            return try await readUsingAssetReader(url)
        }
    }

    private func readUsingAudioFile(_ url: URL) throws -> [Float] {
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
            try Task.checkCancellation()
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

    private func readUsingAssetReader(_ url: URL) async throws -> [Float] {
        let asset = AVURLAsset(url: url)
        let durationTime = try await asset.load(.duration)
        let duration = CMTimeGetSeconds(durationTime)
        guard duration.isFinite, duration > 0 else {
            throw AudioProcessingError.invalidAudioMetadata
        }
        guard duration <= ProcessingLimits.maxSafeDuration else {
            throw AudioProcessingError.audioTooLong(
                duration: duration,
                limit: ProcessingLimits.maxSafeDuration
            )
        }
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
            throw AudioProcessingError.invalidAudioFile
        }

        let reader = try AVAssetReader(asset: asset)
        let outputSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: AudioFormat.targetSampleRate,
            AVNumberOfChannelsKey: AudioFormat.targetChannels,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw AudioProcessingError.conversionFailed }
        reader.add(output)
        guard reader.startReading() else {
            throw reader.error ?? AudioProcessingError.sampleExtractionFailed
        }

        let maxDecodedSamples = Int(ProcessingLimits.maxSafeDuration * AudioFormat.targetSampleRate)
        var decodedSamples = 0
        var samples: [Float] = []
        do {
            while let sampleBuffer = output.copyNextSampleBuffer() {
                try Task.checkCancellation()
                guard CMSampleBufferDataIsReady(sampleBuffer),
                      let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
                      let streamDescription = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription),
                      let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else {
                    throw AudioProcessingError.sampleExtractionFailed
                }

                let format = streamDescription.pointee
                guard format.mFormatID == kAudioFormatLinearPCM,
                      abs(format.mSampleRate - AudioFormat.targetSampleRate) < 1.0,
                      format.mChannelsPerFrame == AudioFormat.targetChannels,
                      format.mBitsPerChannel == 32,
                      (format.mFormatFlags & kAudioFormatFlagIsFloat) != 0 else {
                    throw AudioProcessingError.conversionFailed
                }

                let byteCount = CMBlockBufferGetDataLength(blockBuffer)
                guard byteCount > 0, byteCount % MemoryLayout<Float>.size == 0 else {
                    throw AudioProcessingError.sampleExtractionFailed
                }
                let chunkSampleCount = byteCount / MemoryLayout<Float>.size
                guard decodedSamples <= maxDecodedSamples - chunkSampleCount else {
                    throw AudioProcessingError.audioTooLong(
                        duration: Double(decodedSamples + chunkSampleCount) / AudioFormat.targetSampleRate,
                        limit: ProcessingLimits.maxSafeDuration
                    )
                }
                var chunk = [Float](repeating: 0, count: byteCount / MemoryLayout<Float>.size)
                let status = chunk.withUnsafeMutableBytes { destination in
                    CMBlockBufferCopyDataBytes(
                        blockBuffer,
                        atOffset: 0,
                        dataLength: byteCount,
                        destination: destination.baseAddress!
                    )
                }
                guard status == kCMBlockBufferNoErr else {
                    throw AudioProcessingError.sampleExtractionFailed
                }
                samples.append(contentsOf: chunk)
                decodedSamples += chunkSampleCount
            }
            if reader.status == .failed {
                throw reader.error ?? AudioProcessingError.sampleExtractionFailed
            }
            if reader.status == .cancelled { throw CancellationError() }
        } catch {
            reader.cancelReading()
            throw error
        }

        guard !samples.isEmpty else { throw AudioProcessingError.sampleExtractionFailed }
        let maxSample = samples.reduce(0) { max($0, abs($1)) }
        if maxSample > 0 {
            samples = samples.map { $0 / maxSample }
        }
        return samples
    }

    private static func isPolicyError(_ error: Error) -> Bool {
        guard let error = error as? AudioProcessingError else { return false }
        switch error {
        case .invalidAudioMetadata, .audioTooLong:
            return true
        default:
            return false
        }
    }

    /// Decode through AVAssetReader when AVAudioFile cannot open a valid media
    /// container. The reader emits bounded PCM sample buffers and each buffer is
    /// written immediately, so a long import never accumulates all samples.
    private func transcodeUsingAssetReader(_ url: URL, to destinationURL: URL) async throws -> TimeInterval {
        let asset = AVURLAsset(url: url)
        let durationTime = try await asset.load(.duration)
        let duration = CMTimeGetSeconds(durationTime)
        guard duration.isFinite, duration > 0 else {
            throw AudioProcessingError.invalidAudioMetadata
        }
        guard duration <= ProcessingLimits.maxSafeDuration else {
            throw AudioProcessingError.audioTooLong(
                duration: duration,
                limit: ProcessingLimits.maxSafeDuration
            )
        }
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
            throw AudioProcessingError.invalidAudioFile
        }

        let reader = try AVAssetReader(asset: asset)
        let outputSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: AudioFormat.targetSampleRate,
            AVNumberOfChannelsKey: AudioFormat.targetChannels,
            AVLinearPCMBitDepthKey: AudioFormat.targetBitDepth,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw AudioProcessingError.conversionFailed }
        reader.add(output)

        let maxDecodedFrames = Int64(ProcessingLimits.maxSafeDuration * AudioFormat.targetSampleRate)
        var decodedFrames: Int64 = 0
        let fileManager = FileManager.default
        try fileManager.createDirectory(
            at: destinationURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if fileManager.fileExists(atPath: destinationURL.path) {
            try fileManager.removeItem(at: destinationURL)
        }

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: AudioFormat.targetSampleRate,
            channels: AudioFormat.targetChannels,
            interleaved: true
        ) else { throw AudioProcessingError.unsupportedFormat }

        do {
            guard reader.startReading() else {
                throw reader.error ?? AudioProcessingError.sampleExtractionFailed
            }
            let audioFile = try AVAudioFile(
                forWriting: destinationURL,
                settings: format.settings,
                commonFormat: .pcmFormatInt16,
                interleaved: true
            )

            while let sampleBuffer = output.copyNextSampleBuffer() {
                try Task.checkCancellation()
                guard CMSampleBufferDataIsReady(sampleBuffer),
                      let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
                      let streamDescription = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription),
                      let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else {
                    throw AudioProcessingError.sampleExtractionFailed
                }
                let streamFormat = streamDescription.pointee
                let formatFlags = streamFormat.mFormatFlags
                guard streamFormat.mFormatID == kAudioFormatLinearPCM,
                      abs(streamFormat.mSampleRate - AudioFormat.targetSampleRate) < 1.0,
                      streamFormat.mChannelsPerFrame == AudioFormat.targetChannels,
                      streamFormat.mBitsPerChannel == AudioFormat.targetBitDepth,
                      (formatFlags & kAudioFormatFlagIsFloat) == 0,
                      (formatFlags & kAudioFormatFlagIsBigEndian) == 0,
                      (formatFlags & kAudioFormatFlagIsNonInterleaved) == 0 else {
                    throw AudioProcessingError.conversionFailed
                }
                let byteCount = CMBlockBufferGetDataLength(blockBuffer)
                guard byteCount > 0,
                      byteCount % MemoryLayout<Int16>.size == 0 else {
                    throw AudioProcessingError.sampleExtractionFailed
                }
                let frameCount = AVAudioFrameCount(byteCount / MemoryLayout<Int16>.size)
                decodedFrames += Int64(frameCount)
                guard decodedFrames <= maxDecodedFrames else {
                    throw AudioProcessingError.audioTooLong(
                        duration: Double(decodedFrames) / AudioFormat.targetSampleRate,
                        limit: ProcessingLimits.maxSafeDuration
                    )
                }
                guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount),
                      let channelData = pcmBuffer.int16ChannelData?.pointee else {
                    throw AudioProcessingError.conversionFailed
                }
                pcmBuffer.frameLength = frameCount
                let status = CMBlockBufferCopyDataBytes(
                    blockBuffer,
                    atOffset: 0,
                    dataLength: byteCount,
                    destination: channelData
                )
                guard status == kCMBlockBufferNoErr else {
                    throw AudioProcessingError.sampleExtractionFailed
                }
                try audioFile.write(from: pcmBuffer)
            }

            if reader.status == .failed {
                throw reader.error ?? AudioProcessingError.sampleExtractionFailed
            }
            if reader.status == .cancelled {
                throw CancellationError()
            }
            guard reader.status == .completed, decodedFrames > 0 else {
                throw AudioProcessingError.sampleExtractionFailed
            }
            return Double(decodedFrames) / AudioFormat.targetSampleRate
        } catch {
            reader.cancelReading()
            try? fileManager.removeItem(at: destinationURL)
            throw error
        }
    }

    /// Internal test hook for exercising the resilient reader with a valid fixture.
    func transcodeUsingAssetReaderForTesting(_ url: URL, to destinationURL: URL) async throws -> TimeInterval {
        do {
            return try await transcodeUsingAssetReader(url, to: destinationURL)
        } catch {
            try? FileManager.default.removeItem(at: destinationURL)
            throw error
        }
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
