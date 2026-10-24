import Foundation

final class VocoAudioRenderPipeline: @unchecked Sendable {
    struct Output {
        let samples: UnsafePointer<Float32>
        let frameCount: Int
    }

    private let rnnoiseProcessor = RNNoiseProcessor()
    private var monoBuffer: UnsafeMutablePointer<Float32>?
    private var monoBufferSize: UInt32 = 0

    func allocate(maxFrames: UInt32) {
        guard maxFrames > monoBufferSize else { return }
        monoBuffer?.deallocate()
        monoBuffer = UnsafeMutablePointer<Float32>.allocate(capacity: Int(maxFrames))
        monoBufferSize = maxFrames
    }

    func process(
        inputSamples: UnsafeMutablePointer<Float32>,
        frameCount: UInt32,
        inputChannels: UInt32
    ) -> Output? {
        guard let monoBuffer, frameCount <= monoBufferSize, inputChannels > 0 else { return nil }
        let channels = Int(inputChannels)
        for index in 0..<Int(frameCount) {
            var sample: Float32 = 0
            for channel in 0..<channels {
                sample += inputSamples[index * channels + channel]
            }
            monoBuffer[index] = sample / Float32(inputChannels)
        }

        let (denoised, count) = rnnoiseProcessor.process(
            input: monoBuffer,
            frameCount: Int(frameCount)
        )
        guard count > 0 else { return nil }
        return Output(samples: denoised, frameCount: count)
    }

    func flush() -> (samples: UnsafePointer<Float32>, frameCount: Int) {
        rnnoiseProcessor.flush()
    }

    func reset() {
        rnnoiseProcessor.reset()
    }

    func freeBuffers() {
        monoBuffer?.deallocate()
        monoBuffer = nil
        monoBufferSize = 0
        rnnoiseProcessor.reset()
    }
}
