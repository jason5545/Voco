// VoiceInkEngine+Qwen3.swift
// Download/delete state management for Qwen3-ASR models
// [AI-Claude: 2025-02-18]

import Foundation
import AppKit

extension VoiceInkEngine {
    func isQwen3ModelDownloaded(_ model: Qwen3Model) -> Bool {
        Qwen3ModelManager.isModelDownloaded(modelId: model.modelId)
    }

    func isQwen3ModelDownloading(_ model: Qwen3Model) -> Bool {
        qwen3DownloadStates[model.name] ?? false
    }

    @MainActor
    func downloadQwen3Model(_ model: Qwen3Model) async {
        if isQwen3ModelDownloaded(model) {
            return
        }

        let modelName = model.name
        qwen3DownloadStates[modelName] = true
        downloadProgress[modelName] = 0.0

        // Simulated progress timer (actual download progress is coarse per-file)
        let timer = Timer.scheduledTimer(withTimeInterval: 1.2, repeats: true) { timer in
            Task { @MainActor in
                if let currentProgress = self.downloadProgress[modelName], currentProgress < 0.9 {
                    self.downloadProgress[modelName] = currentProgress + 0.005
                }
            }
        }

        do {
            let cacheDir = try Qwen3HuggingFaceDownloader.getCacheDirectory(for: model.modelId)

            try await Qwen3HuggingFaceDownloader.downloadWeights(
                modelId: model.modelId,
                to: cacheDir,
                progressHandler: { progress in
                    Task { @MainActor in
                        self.downloadProgress[modelName] = progress
                    }
                }
            )

            downloadProgress[modelName] = 1.0
        } catch {
            // Download failed, progress will be cleaned up below
        }

        timer.invalidate()
        qwen3DownloadStates[modelName] = false
        downloadProgress[modelName] = nil

        transcriptionModelManager.refreshAllAvailableModels()
    }

    @MainActor
    func deleteQwen3Model(_ model: Qwen3Model) {
        if let currentModel = transcriptionModelManager.currentTranscriptionModel,
           currentModel.provider == .qwen3,
           currentModel.name == model.name {
            transcriptionModelManager.clearCurrentTranscriptionModel()
        }

        do {
            try Qwen3ModelManager.deleteModel(modelId: model.modelId)
        } catch {
            // Silently ignore removal errors
        }

        transcriptionModelManager.refreshAllAvailableModels()
    }

    @MainActor
    func showQwen3ModelInFinder(_ model: Qwen3Model) {
        let dir = Qwen3ModelManager.modelDirectory(for: model.modelId)

        if FileManager.default.fileExists(atPath: dir.path) {
            NSWorkspace.shared.selectFile(dir.path, inFileViewerRootedAtPath: "")
        }
    }
}
