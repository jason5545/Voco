// WhisperState+WhisperKit.swift
// Download/delete state management for WhisperKit models
// [AI-Claude: 2026-02-27]

import Foundation
import AppKit
import WhisperKit

extension WhisperState {
    func isWhisperKitModelDownloaded(_ model: WhisperKitModel) -> Bool {
        WhisperKitModelManager.isModelDownloaded(variant: model.whisperKitVariant)
    }

    func isWhisperKitModelDownloading(_ model: WhisperKitModel) -> Bool {
        whisperKitDownloadStates[model.name] ?? false
    }

    @MainActor
    func downloadWhisperKitModel(_ model: WhisperKitModel) async {
        if isWhisperKitModelDownloaded(model) {
            return
        }

        let modelName = model.name
        whisperKitDownloadStates[modelName] = true
        downloadProgress[modelName] = 0.0

        do {
            let folder = try await WhisperKit.download(
                variant: model.whisperKitVariant,
                from: "argmaxinc/whisperkit-coreml",
                progressCallback: { progress in
                    Task { @MainActor in
                        self.downloadProgress[modelName] = progress.fractionCompleted
                    }
                }
            )

            // Save the actual folder path returned by WhisperKit
            WhisperKitModelManager.saveModelDirectory(folder, for: model.whisperKitVariant)

            downloadProgress[modelName] = 1.0
            logger.notice("WhisperKit model downloaded to: \(folder.path)")
        } catch {
            logger.error("WhisperKit download failed: \(error.localizedDescription)")
        }

        whisperKitDownloadStates[modelName] = false
        downloadProgress[modelName] = nil

        refreshAllAvailableModels()
    }

    @MainActor
    func deleteWhisperKitModel(_ model: WhisperKitModel) {
        if let currentModel = currentTranscriptionModel,
           currentModel.provider == .whisperKit,
           currentModel.name == model.name {
            currentTranscriptionModel = nil
            UserDefaults.standard.removeObject(forKey: "CurrentTranscriptionModel")
        }

        do {
            try WhisperKitModelManager.deleteModel(variant: model.whisperKitVariant)
        } catch {
            logger.error("Failed to delete WhisperKit model: \(error.localizedDescription)")
        }

        // Release loaded model if it matches
        serviceRegistry.whisperKitTranscriptionService.cleanup()

        refreshAllAvailableModels()
    }

    @MainActor
    func showWhisperKitModelInFinder(_ model: WhisperKitModel) {
        let dir = WhisperKitModelManager.modelDirectory(for: model.whisperKitVariant)

        if FileManager.default.fileExists(atPath: dir.path) {
            NSWorkspace.shared.selectFile(dir.path, inFileViewerRootedAtPath: "")
        }
    }
}
