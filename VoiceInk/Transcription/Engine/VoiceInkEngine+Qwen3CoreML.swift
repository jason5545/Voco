// VoiceInkEngine+Qwen3CoreML.swift
// Download/delete state management for Qwen3-ASR CoreML Hybrid models
// [AI-Claude: 2026-03-13]

import Foundation
import AppKit

extension VoiceInkEngine {
    func isQwen3CoreMLModelDownloaded(_ model: Qwen3CoreMLModel) -> Bool {
        Qwen3CoreMLModelManager.isEncoderDownloaded(modelId: model.coremlModelId)
            && Qwen3ModelManager.isModelDownloaded(modelId: model.mlxModelId)
    }

    func isQwen3CoreMLEncoderDownloaded(_ model: Qwen3CoreMLModel) -> Bool {
        Qwen3CoreMLModelManager.isEncoderDownloaded(modelId: model.coremlModelId)
    }

    func isQwen3CoreMLDecoderDownloaded(_ model: Qwen3CoreMLModel) -> Bool {
        Qwen3ModelManager.isModelDownloaded(modelId: model.mlxModelId)
    }

    func isQwen3CoreMLModelDownloading(_ model: Qwen3CoreMLModel) -> Bool {
        qwen3DownloadStates[model.name] ?? false
    }

    @MainActor
    func downloadQwen3CoreMLModel(_ model: Qwen3CoreMLModel) async {
        if isQwen3CoreMLModelDownloaded(model) {
            return
        }

        let modelName = model.name
        qwen3DownloadStates[modelName] = true
        downloadProgress[modelName] = 0.0

        let timer = Timer.scheduledTimer(withTimeInterval: 1.2, repeats: true) { timer in
            Task { @MainActor in
                if let currentProgress = self.downloadProgress[modelName], currentProgress < 0.9 {
                    self.downloadProgress[modelName] = currentProgress + 0.003
                }
            }
        }

        do {
            // Download CoreML encoder (if not already downloaded)
            if !Qwen3CoreMLModelManager.isEncoderDownloaded(modelId: model.coremlModelId) {
                try await Qwen3CoreMLModelManager.downloadEncoder(
                    modelId: model.coremlModelId,
                    progressHandler: { progress in
                        Task { @MainActor in
                            // Encoder is roughly 50% of total download
                            self.downloadProgress[modelName] = progress * 0.5
                        }
                    }
                )
            }

            // Download MLX decoder (if not already downloaded)
            if !Qwen3ModelManager.isModelDownloaded(modelId: model.mlxModelId) {
                let cacheDir = try Qwen3HuggingFaceDownloader.getCacheDirectory(for: model.mlxModelId)
                try await Qwen3HuggingFaceDownloader.downloadWeights(
                    modelId: model.mlxModelId,
                    to: cacheDir,
                    progressHandler: { progress in
                        Task { @MainActor in
                            // Decoder is the other 50%
                            self.downloadProgress[modelName] = 0.5 + progress * 0.5
                        }
                    }
                )
            }

            downloadProgress[modelName] = 1.0
        } catch {
            // Download failed
        }

        timer.invalidate()
        qwen3DownloadStates[modelName] = false
        downloadProgress[modelName] = nil

        transcriptionModelManager.refreshAllAvailableModels()
    }

    @MainActor
    func deleteQwen3CoreMLModel(_ model: Qwen3CoreMLModel) {
        if let currentModel = transcriptionModelManager.currentTranscriptionModel,
           currentModel.provider == .qwen3CoreML,
           currentModel.name == model.name {
            transcriptionModelManager.clearCurrentTranscriptionModel()
        }

        do {
            try Qwen3CoreMLModelManager.deleteModel(modelId: model.coremlModelId)
        } catch {
            // Silently ignore removal errors
        }

        // Note: We don't delete the MLX decoder model here because it may be shared
        // with the pure MLX Qwen3 model. The user can delete it separately.

        transcriptionModelManager.refreshAllAvailableModels()
    }

    @MainActor
    func showQwen3CoreMLModelInFinder(_ model: Qwen3CoreMLModel) {
        let dir = Qwen3CoreMLModelManager.modelDirectory(for: model.coremlModelId)

        if FileManager.default.fileExists(atPath: dir.path) {
            NSWorkspace.shared.selectFile(dir.path, inFileViewerRootedAtPath: "")
        }
    }
}
