import Foundation
import os

@MainActor
enum VocoRecordingStartFlow {
    struct Output {
        let recordingURL: URL
        let pendingChunks: OSAllocatedUnfairLock<[Data]>
        let realtimeAudioGate: RealtimeAudioChunkGate
    }

    static func run(
        makeRecordingURL: @MainActor () -> URL,
        setRecordedFile: @MainActor (URL) -> Void,
        setAudioChunkCallback: @MainActor (@escaping (Data) -> Void) -> Void,
        setRecordingStateStarting: @MainActor () -> Void,
        scheduleSystemMute: @MainActor () -> Void,
        startRecording: @MainActor (URL) async throws -> Void,
        checkpoint: @MainActor (String) -> Void = { StartupTracer.checkpoint($0) },
        endTrace: @MainActor (String) -> Void = { StartupTracer.end($0) }
    ) async throws -> Output {
        let recordingURL = makeRecordingURL()
        setRecordedFile(recordingURL)
        checkpoint("toggleRecord_recording_file_prepared")

        let realtimeAudioGate = RealtimeAudioChunkGate()
        let pendingChunks = realtimeAudioGate.pendingChunks
        setAudioChunkCallback { data in
            realtimeAudioGate.receive(data)
        }
        checkpoint("toggleRecord_audio_callback_set")

        setRecordingStateStarting()
        checkpoint("toggleRecord_state_set_starting")
        scheduleSystemMute()

        checkpoint("toggleRecord_before_startRecording")
        try await startRecording(recordingURL)
        endTrace("recorder_startRecording_done")

        return Output(
            recordingURL: recordingURL,
            pendingChunks: pendingChunks,
            realtimeAudioGate: realtimeAudioGate
        )
    }
}

typealias EngineRecordingStartFlow = VocoRecordingStartFlow
