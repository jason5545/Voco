import Foundation

@MainActor
enum VocoModelDownloadStatus {
    static func isDownloaded(_ model: any TranscriptionModel) -> Bool? {
        switch model.provider {
        case .qwen3:
            guard let model = model as? Qwen3Model else { return false }
            return Qwen3ModelManager.isModelDownloaded(modelId: model.modelId)
        case .whisperMLX:
            guard let model = model as? WhisperMLXModel else { return false }
            return WhisperMLXModelManager.isModelDownloaded(modelId: model.huggingFaceRepo)
        case .whisperCoreML:
            guard let model = model as? WhisperCoreMLModel else { return false }
            return WhisperCoreMLModelManager.isModelDownloaded(modelId: model.coremlModelId)
        case .qwen3CoreML:
            guard let model = model as? Qwen3CoreMLModel else { return false }
            return Qwen3CoreMLModelManager.isEncoderDownloaded(modelId: model.coremlModelId)
                && Qwen3ModelManager.isModelDownloaded(modelId: model.mlxModelId)
        default:
            return nil
        }
    }
}
