import Foundation
import AVFoundation
import CoreAudio
import AudioToolbox
import os

// MARK: - AVAudioEngine-based Recorder with Voice Processing support

final class AVAudioEngineRecorder: RecorderEngine {

    // MARK: - Properties

    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "AVAudioEngineRecorder")

    private var engine: AVAudioEngine?
    private var mixerNode: AVAudioMixerNode?
    private var audioFile: ExtAudioFileRef?

    private var isRecording = false
    private var currentDeviceID: AudioDeviceID = 0
    private var recordingURL: URL?

    // Output format (16kHz mono PCM Int16 for transcription)
    private var outputFormat = AudioStreamBasicDescription(
        mSampleRate: 16000.0,
        mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
        mBytesPerPacket: 2,
        mFramesPerPacket: 1,
        mBytesPerFrame: 2,
        mChannelsPerFrame: 1,
        mBitsPerChannel: 16,
        mReserved: 0
    )

    // Pre-allocated conversion buffer (Float32 → Int16)
    private var conversionBuffer: UnsafeMutablePointer<Int16>?
    private var conversionBufferSize: Int = 0

    // Audio metering (thread-safe)
    private let meterLock = NSLock()
    private var _averagePower: Float = -160.0
    private var _peakPower: Float = -160.0

    var averagePower: Float {
        meterLock.lock()
        defer { meterLock.unlock() }
        return _averagePower
    }

    var peakPower: Float {
        meterLock.lock()
        defer { meterLock.unlock() }
        return _peakPower
    }

    /// Called with raw PCM data (16-bit, 16kHz, mono) for streaming.
    var onAudioChunk: ((_ data: Data) -> Void)?

    /// Whether Voice Processing is enabled for this recorder instance.
    private var voiceProcessingEnabled: Bool

    // MARK: - Initialization

    init(voiceProcessingEnabled: Bool = true) {
        self.voiceProcessingEnabled = voiceProcessingEnabled
    }

    deinit {
        stopRecording()
    }

    // MARK: - Public Interface

    var isCurrentlyRecording: Bool { isRecording }
    var currentRecordingURL: URL? { recordingURL }
    var currentDevice: AudioDeviceID { currentDeviceID }

    func startRecording(toOutputFile url: URL, deviceID: AudioDeviceID) throws {
        // Stop any existing recording
        stopRecording()

        if deviceID == 0 {
            logger.error("Cannot start recording - no valid audio device (deviceID is 0)")
            throw AVAudioEngineRecorderError.deviceNotAvailable
        }

        guard isDeviceAvailable(deviceID) else {
            logger.error("Cannot start recording - device \(deviceID) is no longer available")
            throw AVAudioEngineRecorderError.deviceNotAvailable
        }

        currentDeviceID = deviceID
        recordingURL = url

        logger.notice("🎙️ Starting AVAudioEngine recording from device \(deviceID), VP=\(self.voiceProcessingEnabled)")
        logDeviceDetails(deviceID: deviceID)

        // Step 1: Create engine
        let newEngine = AVAudioEngine()
        engine = newEngine

        // Step 2: Set the input device on the engine's inputNode via AudioUnit property
        let inputNode = newEngine.inputNode
        try setDevice(deviceID, on: inputNode)

        // Step 3: Enable/disable Voice Processing
        let vpActuallyEnabled = enableVoiceProcessingIfRequested(on: inputNode)

        // Step 4: Determine tap format
        // IMPORTANT: installTap format MUST match the inputNode's outputFormat.
        // Requesting a different format (e.g. 16kHz mono) causes AVAudioEngine
        // initialization to fail with -10875 when VP is enabled.
        // We handle sample rate conversion and channel mixing in handleTapBuffer().
        let hwFormat = inputNode.outputFormat(forBus: 0)
        logger.notice("🎙️ InputNode output format: sampleRate=\(hwFormat.sampleRate), channels=\(hwFormat.channelCount)")

        let tapFormat: AVAudioFormat
        let nodeToTap: AVAudioNode

        if !vpActuallyEnabled && hwFormat.channelCount > 1 {
            // Multi-channel without VP → attach a mixer to downmix to mono
            let mixer = AVAudioMixerNode()
            newEngine.attach(mixer)
            newEngine.connect(inputNode, to: mixer, format: hwFormat)
            mixerNode = mixer

            // Mixer output will be mono at the same sample rate
            guard let monoFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: hwFormat.sampleRate,
                channels: 1,
                interleaved: false
            ) else {
                throw AVAudioEngineRecorderError.failedToInstallTap
            }
            newEngine.connect(mixer, to: newEngine.mainMixerNode, format: monoFormat)
            tapFormat = monoFormat
            nodeToTap = mixer
        } else {
            // VP enabled or single-channel device → tap inputNode with its native format
            tapFormat = hwFormat
            nodeToTap = inputNode
        }

        // Step 5: Create output file
        try createOutputFile(at: url)

        // Step 6: Pre-allocate conversion buffer
        let bufferFrames = 4096
        if conversionBuffer == nil || conversionBufferSize < bufferFrames {
            conversionBuffer?.deallocate()
            conversionBuffer = UnsafeMutablePointer<Int16>.allocate(capacity: bufferFrames)
            conversionBufferSize = bufferFrames
        }

        // Step 7: Install tap
        nodeToTap.installTap(onBus: 0, bufferSize: 4096, format: tapFormat) { [weak self] buffer, _ in
            self?.handleTapBuffer(buffer)
        }

        // Step 8: Start engine
        newEngine.prepare()
        do {
            try newEngine.start()
        } catch {
            logger.error("Failed to start AVAudioEngine: \(error.localizedDescription)")
            cleanupEngine()
            throw AVAudioEngineRecorderError.failedToStart(underlying: error)
        }

        isRecording = true
        logger.notice("🎙️ AVAudioEngine recording started successfully")
    }

    func stopRecording() {
        guard isRecording || engine != nil else { return }
        logger.notice("stopRecording: stopping AVAudioEngine recorder")

        cleanupEngine()

        // Close audio file
        if let file = audioFile {
            ExtAudioFileDispose(file)
            audioFile = nil
        }

        // Free conversion buffer
        if let buffer = conversionBuffer {
            buffer.deallocate()
            conversionBuffer = nil
            conversionBufferSize = 0
        }

        isRecording = false
        currentDeviceID = 0
        recordingURL = nil

        // Reset meters
        meterLock.lock()
        _averagePower = -160.0
        _peakPower = -160.0
        meterLock.unlock()
    }

    func switchDevice(to newDeviceID: AudioDeviceID) throws {
        guard isRecording, let currentEngine = engine else {
            throw AVAudioEngineRecorderError.notRecording
        }

        guard newDeviceID != currentDeviceID else { return }

        let oldDeviceID = currentDeviceID
        logger.notice("🎙️ Switching recording device from \(oldDeviceID) to \(newDeviceID)")

        // Remove tap and stop engine (keep file open)
        let inputNode = currentEngine.inputNode
        if let mixer = mixerNode {
            mixer.removeTap(onBus: 0)
        } else {
            inputNode.removeTap(onBus: 0)
        }

        currentEngine.stop()

        // Detach mixer if present
        if let mixer = mixerNode {
            currentEngine.detach(mixer)
            mixerNode = nil
        }

        // Set new device
        do {
            try setDevice(newDeviceID, on: inputNode)
        } catch {
            // Try to recover with old device
            logger.error("Failed to set new device, attempting recovery with old device...")
            do {
                try setDevice(oldDeviceID, on: inputNode)
                try restartEngineWithTap(currentEngine)
            } catch {
                logger.error("Recovery failed: \(error.localizedDescription)")
            }
            throw error
        }

        currentDeviceID = newDeviceID

        // Re-enable VP if needed
        let vpActuallyEnabled = enableVoiceProcessingIfRequested(on: inputNode)

        // Reinstall tap with appropriate format
        let hwFormat = inputNode.outputFormat(forBus: 0)
        logger.notice("🎙️ New device format: sampleRate=\(hwFormat.sampleRate), channels=\(hwFormat.channelCount)")

        let nodeToTap: AVAudioNode
        let tapFormat: AVAudioFormat

        if !vpActuallyEnabled && hwFormat.channelCount > 1 {
            let mixer = AVAudioMixerNode()
            currentEngine.attach(mixer)
            currentEngine.connect(inputNode, to: mixer, format: hwFormat)
            mixerNode = mixer

            guard let monoFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: hwFormat.sampleRate,
                channels: 1,
                interleaved: false
            ) else {
                throw AVAudioEngineRecorderError.failedToInstallTap
            }
            currentEngine.connect(mixer, to: currentEngine.mainMixerNode, format: monoFormat)
            tapFormat = monoFormat
            nodeToTap = mixer
        } else {
            // Use native format — conversion done in handleTapBuffer
            tapFormat = hwFormat
            nodeToTap = inputNode
        }

        nodeToTap.installTap(onBus: 0, bufferSize: 4096, format: tapFormat) { [weak self] buffer, _ in
            self?.handleTapBuffer(buffer)
        }

        // Restart engine
        do {
            try restartEngineWithTap(currentEngine)
        } catch {
            // Try to recover with old device
            logger.error("Failed to restart with new device, recovering...")
            nodeToTap.removeTap(onBus: 0)
            if let mixer = mixerNode {
                currentEngine.detach(mixer)
                mixerNode = nil
            }

            do {
                try setDevice(oldDeviceID, on: inputNode)
                currentDeviceID = oldDeviceID
                _ = enableVoiceProcessingIfRequested(on: inputNode)

                let fallbackFormat = inputNode.outputFormat(forBus: 0)
                inputNode.installTap(onBus: 0, bufferSize: 4096, format: fallbackFormat) { [weak self] buffer, _ in
                    self?.handleTapBuffer(buffer)
                }
                try restartEngineWithTap(currentEngine)
            } catch {
                logger.error("Recovery also failed: \(error.localizedDescription)")
            }
            throw AVAudioEngineRecorderError.failedToStart(underlying: error)
        }

        logger.notice("🎙️ Successfully switched to device \(newDeviceID)")
    }

    // MARK: - Engine Setup Helpers

    private func setDevice(_ deviceID: AudioDeviceID, on inputNode: AVAudioInputNode) throws {
        guard let au = inputNode.audioUnit else {
            throw AVAudioEngineRecorderError.engineNotRunning
        }

        var devID = deviceID
        let status = AudioUnitSetProperty(
            au,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &devID,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )

        if status != noErr {
            logger.error("Failed to set device \(deviceID) on inputNode: \(status)")
            throw AVAudioEngineRecorderError.failedToSetDevice(status: status)
        }
    }

    /// Attempts to enable Voice Processing on the input node.
    /// Returns `true` if VP is actually enabled after this call.
    @discardableResult
    private func enableVoiceProcessingIfRequested(on inputNode: AVAudioInputNode) -> Bool {
        guard voiceProcessingEnabled else {
            // Explicitly disable VP in case it was previously enabled
            do {
                try inputNode.setVoiceProcessingEnabled(false)
            } catch {
                logger.warning("Failed to disable Voice Processing: \(error.localizedDescription)")
            }
            return false
        }

        do {
            try inputNode.setVoiceProcessingEnabled(true)
            logger.notice("🎙️ Voice Processing enabled (beamforming, noise suppression, AEC, AGC)")
            return true
        } catch {
            logger.warning("⚠️ Voice Processing not available for this device, falling back to raw capture: \(error.localizedDescription)")
            // Fallback: continue without VP
            return false
        }
    }

    private func createOutputFile(at url: URL) throws {
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }

        var fileRef: ExtAudioFileRef?
        let status = ExtAudioFileCreateWithURL(
            url as CFURL,
            kAudioFileWAVEType,
            &outputFormat,
            nil,
            AudioFileFlags.eraseFile.rawValue,
            &fileRef
        )

        if status != noErr {
            logger.error("Failed to create audio file at \(url.path): \(status)")
            throw AVAudioEngineRecorderError.failedToCreateFile(status: status)
        }

        audioFile = fileRef

        // Set client format to match what we write (Int16 mono 16kHz)
        let setStatus = ExtAudioFileSetProperty(
            fileRef!,
            kExtAudioFileProperty_ClientDataFormat,
            UInt32(MemoryLayout<AudioStreamBasicDescription>.size),
            &outputFormat
        )

        if setStatus != noErr {
            logger.error("Failed to set file client format: \(setStatus)")
            throw AVAudioEngineRecorderError.failedToCreateFile(status: setStatus)
        }
    }

    private func restartEngineWithTap(_ engine: AVAudioEngine) throws {
        engine.prepare()
        try engine.start()
    }

    private func cleanupEngine() {
        guard let currentEngine = engine else { return }

        let inputNode = currentEngine.inputNode
        if let mixer = mixerNode {
            mixer.removeTap(onBus: 0)
            currentEngine.detach(mixer)
            mixerNode = nil
        } else {
            inputNode.removeTap(onBus: 0)
        }

        currentEngine.stop()
        engine = nil
    }

    // MARK: - Tap Buffer Processing

    private func handleTapBuffer(_ buffer: AVAudioPCMBuffer) {
        guard isRecording else { return }

        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0 else { return }

        // Calculate meters from Float32 data
        calculateMeters(from: buffer)

        // The tap may deliver audio at the hardware sample rate (when using mixer path)
        // or at 16kHz (when tapping inputNode directly with VP).
        // We need to produce 16kHz mono Int16 for the file and streaming callback.
        let tapSampleRate = buffer.format.sampleRate
        let tapChannels = Int(buffer.format.channelCount)

        // Get Float32 samples
        guard let floatData = buffer.floatChannelData else { return }

        // Determine output frame count
        let outputFrameCount: Int
        let needsResample: Bool

        if abs(tapSampleRate - 16000.0) < 1.0 {
            outputFrameCount = frameCount
            needsResample = false
        } else {
            let ratio = 16000.0 / tapSampleRate
            outputFrameCount = Int(Double(frameCount) * ratio)
            needsResample = true
        }

        guard outputFrameCount > 0 else { return }

        // Ensure conversion buffer is large enough
        if outputFrameCount > conversionBufferSize {
            conversionBuffer?.deallocate()
            conversionBuffer = UnsafeMutablePointer<Int16>.allocate(capacity: outputFrameCount)
            conversionBufferSize = outputFrameCount
        }

        guard let outBuf = conversionBuffer else { return }

        if !needsResample {
            // Direct conversion: Float32 → Int16
            // Non-interleaved: channel data in floatData[ch]
            if tapChannels == 1 {
                let samples = floatData[0]
                for i in 0..<frameCount {
                    let scaled = samples[i] * 32767.0
                    let clipped = max(-32768.0, min(32767.0, scaled))
                    outBuf[i] = Int16(clipped)
                }
            } else {
                // Mix channels (shouldn't happen when VP is on, but handle gracefully)
                for i in 0..<frameCount {
                    var sample: Float32 = 0
                    for ch in 0..<tapChannels {
                        sample += floatData[ch][i]
                    }
                    sample /= Float32(tapChannels)
                    let scaled = sample * 32767.0
                    let clipped = max(-32768.0, min(32767.0, scaled))
                    outBuf[i] = Int16(clipped)
                }
            }
        } else {
            // Resample using linear interpolation (same approach as CoreAudioRecorder)
            let ratio = 16000.0 / tapSampleRate
            if tapChannels == 1 {
                let samples = floatData[0]
                for i in 0..<outputFrameCount {
                    let inputIndex = Double(i) / ratio
                    let idx1 = min(Int(inputIndex), frameCount - 1)
                    let idx2 = min(idx1 + 1, frameCount - 1)
                    let frac = Float32(inputIndex - Double(idx1))

                    let sample = samples[idx1] + frac * (samples[idx2] - samples[idx1])
                    let scaled = sample * 32767.0
                    let clipped = max(-32768.0, min(32767.0, scaled))
                    outBuf[i] = Int16(clipped)
                }
            } else {
                for i in 0..<outputFrameCount {
                    let inputIndex = Double(i) / ratio
                    let idx1 = min(Int(inputIndex), frameCount - 1)
                    let idx2 = min(idx1 + 1, frameCount - 1)
                    let frac = Float32(inputIndex - Double(idx1))

                    var sample: Float32 = 0
                    for ch in 0..<tapChannels {
                        let s1 = floatData[ch][idx1]
                        let s2 = floatData[ch][idx2]
                        sample += s1 + frac * (s2 - s1)
                    }
                    sample /= Float32(tapChannels)
                    let scaled = sample * 32767.0
                    let clipped = max(-32768.0, min(32767.0, scaled))
                    outBuf[i] = Int16(clipped)
                }
            }
        }

        // Write to file
        writeToFile(outBuf, frameCount: UInt32(outputFrameCount))

        // Send streaming callback
        if let onAudioChunk = onAudioChunk {
            let byteCount = outputFrameCount * MemoryLayout<Int16>.size
            let data = Data(bytes: outBuf, count: byteCount)
            onAudioChunk(data)
        }
    }

    private func calculateMeters(from buffer: AVAudioPCMBuffer) {
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0, let floatData = buffer.floatChannelData else { return }

        let channelCount = Int(buffer.format.channelCount)
        var sum: Float = 0.0
        var peak: Float = 0.0

        if channelCount == 1 {
            let samples = floatData[0]
            for i in 0..<frameCount {
                let s = abs(samples[i])
                sum += s * s
                if s > peak { peak = s }
            }
        } else {
            for i in 0..<frameCount {
                for ch in 0..<channelCount {
                    let s = abs(floatData[ch][i])
                    sum += s * s
                    if s > peak { peak = s }
                }
            }
        }

        let totalSamples = frameCount * channelCount
        let rms = sqrt(sum / Float(totalSamples))
        let avgDb = 20.0 * log10(max(rms, 0.000001))
        let peakDb = 20.0 * log10(max(peak, 0.000001))

        meterLock.lock()
        _averagePower = avgDb
        _peakPower = peakDb
        meterLock.unlock()
    }

    private func writeToFile(_ buffer: UnsafeMutablePointer<Int16>, frameCount: UInt32) {
        guard let file = audioFile, frameCount > 0 else { return }

        var bufferList = AudioBufferList(
            mNumberBuffers: 1,
            mBuffers: AudioBuffer(
                mNumberChannels: 1,
                mDataByteSize: frameCount * 2,
                mData: buffer
            )
        )

        let status = ExtAudioFileWrite(file, frameCount, &bufferList)
        if status != noErr {
            logger.error("🎙️ ExtAudioFileWrite failed: \(status)")
        }
    }

    // MARK: - Device Info

    private func logDeviceDetails(deviceID: AudioDeviceID) {
        let deviceName = getDeviceStringProperty(deviceID: deviceID, selector: kAudioDevicePropertyDeviceNameCFString) ?? "Unknown"
        let deviceUID = getDeviceStringProperty(deviceID: deviceID, selector: kAudioDevicePropertyDeviceUID) ?? "Unknown"
        logger.notice("🎙️ Device info: name=\(deviceName), uid=\(deviceUID)")
    }

    private func getDeviceStringProperty(deviceID: AudioDeviceID, selector: AudioObjectPropertySelector) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var propertySize = UInt32(MemoryLayout<CFString>.size)
        var property: CFString?

        let status = AudioObjectGetPropertyData(
            deviceID,
            &address,
            0,
            nil,
            &propertySize,
            &property
        )

        if status == noErr, let cfString = property {
            return cfString as String
        }
        return nil
    }

    private func isDeviceAvailable(_ deviceID: AudioDeviceID) -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceIsAlive,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var isAlive: UInt32 = 0
        var propertySize = UInt32(MemoryLayout<UInt32>.size)

        let status = AudioObjectGetPropertyData(
            deviceID,
            &address,
            0,
            nil,
            &propertySize,
            &isAlive
        )

        return status == noErr && isAlive == 1
    }
}

// MARK: - Error Types

enum AVAudioEngineRecorderError: LocalizedError {
    case engineNotRunning
    case notRecording
    case deviceNotAvailable
    case failedToSetDevice(status: OSStatus)
    case failedToEnableVoiceProcessing(underlying: Error)
    case failedToCreateFile(status: OSStatus)
    case failedToStart(underlying: Error)
    case failedToInstallTap

    var errorDescription: String? {
        switch self {
        case .engineNotRunning:
            return "AVAudioEngine is not running"
        case .notRecording:
            return "Not currently recording"
        case .deviceNotAvailable:
            return "Audio device is not available"
        case .failedToSetDevice(let status):
            return "Failed to set input device: \(status)"
        case .failedToEnableVoiceProcessing(let error):
            return "Failed to enable Voice Processing: \(error.localizedDescription)"
        case .failedToCreateFile(let status):
            return "Failed to create audio file: \(status)"
        case .failedToStart(let error):
            return "Failed to start engine: \(error.localizedDescription)"
        case .failedToInstallTap:
            return "Failed to install audio tap"
        }
    }
}
