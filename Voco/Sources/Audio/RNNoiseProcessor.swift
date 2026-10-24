import CRNNoise

/// Wraps the RNNoise C library for real-time noise suppression.
/// All buffers are pre-allocated at init time — zero allocations in the audio callback path.
final class RNNoiseProcessor {

    private var state: OpaquePointer?
    private let frameSize: Int // rnnoise_get_frame_size() — typically 480

    // Pre-allocated buffers
    private let accumulator: UnsafeMutablePointer<Float>   // collects mono input samples
    private var accumCount: Int = 0
    private let rnnoiseIn: UnsafeMutablePointer<Float>     // 480-sample scratch for rnnoise input
    private let rnnoiseOut: UnsafeMutablePointer<Float>    // 480-sample scratch for rnnoise output
    private let outputBuffer: UnsafeMutablePointer<Float>  // holds denoised output to return
    private var outputCount: Int = 0

    // Maximum accumulator capacity (enough for the largest expected callback + leftover)
    // 4096 (max callback) + 479 (max leftover) = 4575
    private let accumulatorCapacity = 4096 + 480

    // Scale factors: RNNoise expects int16 range, our input is [-1, 1]
    private let scaleIn: Float = 32768.0
    private let scaleOut: Float = 1.0 / 32768.0

    init() {
        frameSize = Int(rnnoise_get_frame_size())
        state = rnnoise_create(nil)

        accumulator = .allocate(capacity: accumulatorCapacity)
        rnnoiseIn = .allocate(capacity: frameSize)
        rnnoiseOut = .allocate(capacity: frameSize)
        // Output can produce at most (accumulatorCapacity / frameSize + 1) * frameSize samples
        // but realistically never more than input count + one frame
        outputBuffer = .allocate(capacity: accumulatorCapacity)
    }

    deinit {
        if let st = state {
            rnnoise_destroy(st)
        }
        accumulator.deallocate()
        rnnoiseIn.deallocate()
        rnnoiseOut.deallocate()
        outputBuffer.deallocate()
    }

    /// Process mono Float32 samples (range [-1, 1]).
    /// Returns a pointer to denoised samples and the count of samples produced.
    /// The returned pointer is valid until the next call to process() or flush().
    func process(input: UnsafePointer<Float>, frameCount: Int) -> (UnsafePointer<Float>, Int) {
        guard let st = state else {
            return (UnsafePointer(input), frameCount)
        }

        outputCount = 0

        // Append input to accumulator
        let spaceLeft = accumulatorCapacity - accumCount
        let copyCount = min(frameCount, spaceLeft)
        memcpy(accumulator.advanced(by: accumCount), input, copyCount * MemoryLayout<Float>.size)
        accumCount += copyCount

        // Process as many complete frames as possible
        while accumCount >= frameSize {
            // Scale [-1,1] → [-32768,32767] into rnnoiseIn
            for i in 0..<frameSize {
                rnnoiseIn[i] = accumulator[i] * scaleIn
            }

            // Run RNNoise
            _ = rnnoise_process_frame(st, rnnoiseOut, rnnoiseIn)

            // Scale back and copy to output
            for i in 0..<frameSize {
                outputBuffer[outputCount + i] = rnnoiseOut[i] * scaleOut
            }
            outputCount += frameSize

            // Shift accumulator: move remaining samples to front
            let remaining = accumCount - frameSize
            if remaining > 0 {
                memmove(accumulator, accumulator.advanced(by: frameSize), remaining * MemoryLayout<Float>.size)
            }
            accumCount = remaining
        }

        return (UnsafePointer(outputBuffer), outputCount)
    }

    /// Flush any remaining samples at the end of a recording.
    /// Pads the last partial frame with zeros and processes it.
    func flush() -> (UnsafePointer<Float>, Int) {
        guard let st = state, accumCount > 0 else {
            outputCount = 0
            return (UnsafePointer(outputBuffer), 0)
        }

        outputCount = 0
        let remaining = accumCount

        // Scale existing samples
        for i in 0..<remaining {
            rnnoiseIn[i] = accumulator[i] * scaleIn
        }
        // Zero-pad to fill the frame
        for i in remaining..<frameSize {
            rnnoiseIn[i] = 0
        }

        _ = rnnoise_process_frame(st, rnnoiseOut, rnnoiseIn)

        // Only output the non-padded portion
        for i in 0..<remaining {
            outputBuffer[i] = rnnoiseOut[i] * scaleOut
        }
        outputCount = remaining
        accumCount = 0

        return (UnsafePointer(outputBuffer), outputCount)
    }

    /// Reset state for a new recording session.
    func reset() {
        accumCount = 0
        outputCount = 0
        // Recreate RNNoise state to clear internal buffers
        if let st = state {
            rnnoise_destroy(st)
        }
        state = rnnoise_create(nil)
    }
}
