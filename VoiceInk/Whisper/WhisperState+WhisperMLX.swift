// WhisperState+WhisperMLX.swift
// Download/delete state management for Whisper MLX models
// [AI-Claude: 2026-02-27]

import Foundation
import AppKit

extension WhisperState {
    func isWhisperMLXModelDownloaded(_ model: WhisperMLXModel) -> Bool {
        WhisperMLXModelManager.isModelDownloaded(modelId: model.huggingFaceRepo)
    }

    func isWhisperMLXModelDownloading(_ model: WhisperMLXModel) -> Bool {
        whisperMLXDownloadStates[model.name] ?? false
    }

    @MainActor
    func downloadWhisperMLXModel(_ model: WhisperMLXModel) async {
        if isWhisperMLXModelDownloaded(model) {
            return
        }

        let modelName = model.name
        whisperMLXDownloadStates[modelName] = true
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
            try await WhisperMLXModelManager.downloadModel(
                modelId: model.huggingFaceRepo,
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
        whisperMLXDownloadStates[modelName] = false
        downloadProgress[modelName] = nil

        refreshAllAvailableModels()
    }

    @MainActor
    func deleteWhisperMLXModel(_ model: WhisperMLXModel) {
        if let currentModel = currentTranscriptionModel,
           currentModel.provider == .whisperMLX,
           currentModel.name == model.name {
            currentTranscriptionModel = nil
            UserDefaults.standard.removeObject(forKey: "CurrentTranscriptionModel")
        }

        do {
            try WhisperMLXModelManager.deleteModel(modelId: model.huggingFaceRepo)
        } catch {
            // Silently ignore removal errors
        }

        refreshAllAvailableModels()
    }

    @MainActor
    func showWhisperMLXModelInFinder(_ model: WhisperMLXModel) {
        let dir = WhisperMLXModelManager.modelDirectory(for: model.huggingFaceRepo)

        if FileManager.default.fileExists(atPath: dir.path) {
            NSWorkspace.shared.selectFile(dir.path, inFileViewerRootedAtPath: "")
        }
    }
}
