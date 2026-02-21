import Foundation
import CoreAudio

/// Shared interface for recording engines, allowing CoreAudioRecorder and AVAudioEngineRecorder
/// to be used interchangeably by `Recorder`.
protocol RecorderEngine: AnyObject {
    var onAudioChunk: ((_ data: Data) -> Void)? { get set }
    var averagePower: Float { get }
    var peakPower: Float { get }
    var isCurrentlyRecording: Bool { get }
    var currentRecordingURL: URL? { get }
    var currentDevice: AudioDeviceID { get }

    func startRecording(toOutputFile url: URL, deviceID: AudioDeviceID) throws
    func stopRecording()
    func switchDevice(to newDeviceID: AudioDeviceID) throws
}
