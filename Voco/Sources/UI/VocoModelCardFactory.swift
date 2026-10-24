import SwiftUI

@MainActor
enum VocoModelCardFactory {
    @ViewBuilder
    static func view(
        for model: any TranscriptionModel,
        engine: VoiceInkEngine,
        transcriptionModelManager: TranscriptionModelManager
    ) -> some View {
        switch model.provider {
        case .qwen3:
            if let model = model as? Qwen3Model {
                Qwen3ModelCardRowView(
                    model: model,
                    engine: engine,
                    transcriptionModelManager: transcriptionModelManager
                )
            }
        case .whisperMLX:
            if let model = model as? WhisperMLXModel {
                WhisperMLXModelCardRowView(
                    model: model,
                    engine: engine,
                    transcriptionModelManager: transcriptionModelManager
                )
            }
        case .whisperCoreML:
            if let model = model as? WhisperCoreMLModel {
                WhisperCoreMLModelCardRowView(
                    model: model,
                    transcriptionModelManager: transcriptionModelManager
                )
            }
        case .qwen3CoreML:
            if let model = model as? Qwen3CoreMLModel {
                Qwen3CoreMLModelCardRowView(
                    model: model,
                    engine: engine,
                    transcriptionModelManager: transcriptionModelManager
                )
            }
        default:
            EmptyView()
        }
    }
}
