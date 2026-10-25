import AppKit
import SwiftUI

struct ModelCardView: View {
    let model: any TranscriptionModel
    let engine: VoiceInkEngine
    let fluidAudioModelManager: FluidAudioModelManager
    let transcriptionModelManager: TranscriptionModelManager
    let isDownloaded: Bool
    let downloadProgress: [String: Double]
    let modelURL: URL?
    let isWarming: Bool

    // Actions
    var deleteAction: () -> Void
    var downloadAction: () -> Void
    var cancelDownloadAction: () -> Void
    var editAction: ((CustomCloudModel) -> Void)? = nil
    var body: some View {
        Group {
            switch model.provider {
            case .whisper:
                if let whisperModel = model as? WhisperModel {
                    WhisperModelCardView(
                        model: whisperModel,
                        isDownloaded: isDownloaded,
                        downloadProgress: downloadProgress,
                        modelURL: modelURL,
                        isWarming: isWarming,
                        deleteAction: deleteAction,
                        downloadAction: downloadAction,
                        cancelDownloadAction: cancelDownloadAction
                    )
                } else if let importedModel = model as? ImportedWhisperModel {
                    ImportedWhisperModelCardView(
                        model: importedModel,
                        isDownloaded: isDownloaded,
                        modelURL: modelURL,
                        deleteAction: deleteAction
                    )
                }
            case .fluidAudio:
                if let fluidAudioModel = model as? FluidAudioModel {
                    FluidAudioModelCardView(model: fluidAudioModel)
                }
            case .transcribeCpp:
                if let transcribeCppModel = model as? TranscribeCppModel {
                    TranscribeCppModelCardView(model: transcribeCppModel)
                }
            case .nativeApple:
                if let nativeAppleModel = model as? NativeAppleModel {
                    NativeAppleModelCardView(
                        model: nativeAppleModel
                    )
                }
            case .groq, .elevenLabs, .deepgram, .mistral, .gemini, .soniox, .speechmatics:
                if let cloudModel = model as? CloudModel {
                    CloudModelCardView(model: cloudModel)
                }
            case .custom:
                if let customModel = model as? CustomCloudModel {
                    CustomModelCardView(
                        model: customModel,
                        deleteAction: deleteAction,
                        editAction: editAction ?? { _ in }
                    )
                }
            default:
                if let cloudModel = model as? CloudModel {
                    CloudModelCardView(model: cloudModel)
                } else {
                    VocoModelCardFactory.view(
                        for: model,
                        engine: engine,
                        transcriptionModelManager: transcriptionModelManager
                    )
                }
            }
        }
    }
}
