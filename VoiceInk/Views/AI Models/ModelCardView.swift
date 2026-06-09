import SwiftUI
import AppKit

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
    var editAction: ((CustomCloudModel) -> Void)?
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
                        downloadAction: downloadAction
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
                    FluidAudioModelCardView(
                        model: fluidAudioModel,
                        fluidAudioModelManager: fluidAudioModelManager
                    )
                }
            case .nativeApple:
                if let nativeAppleModel = model as? NativeAppleModel {
                    NativeAppleModelCardView(
                        model: nativeAppleModel
                    )
                }
            case .qwen3:
                if let qwen3Model = model as? Qwen3Model {
                    Qwen3ModelCardRowView(
                        model: qwen3Model,
                        engine: engine,
                        transcriptionModelManager: transcriptionModelManager
                    )
                }
            case .whisperMLX:
                if let whisperMLXModel = model as? WhisperMLXModel {
                    WhisperMLXModelCardRowView(
                        model: whisperMLXModel,
                        engine: engine,
                        transcriptionModelManager: transcriptionModelManager
                    )
                }
            case .whisperCoreML:
                if let coremlModel = model as? WhisperCoreMLModel {
                    WhisperCoreMLModelCardRowView(
                        model: coremlModel,
                        transcriptionModelManager: transcriptionModelManager
                    )
                }
            case .qwen3CoreML:
                if let qwen3CoreMLModel = model as? Qwen3CoreMLModel {
                    Qwen3CoreMLModelCardRowView(
                        model: qwen3CoreMLModel,
                        engine: engine,
                        transcriptionModelManager: transcriptionModelManager
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
                    CloudModelCardView(
                        model: cloudModel
                    )
                }
            }
        }
    }
}
