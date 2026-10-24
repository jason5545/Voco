// Qwen3ModelManager.swift
// Manages Qwen3 model files on disk
// [AI-Claude: 2025-02-18]

import Foundation
import os

enum Qwen3ModelManager {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "Qwen3ModelManager")

    /// Base directory for all Qwen3 models
    static var baseDirectory: URL {
        let fm = FileManager.default
        let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport
            .appendingPathComponent(AppIdentifiers.bundleID, isDirectory: true)
            .appendingPathComponent("Qwen3Models", isDirectory: true)
    }

    /// Get the directory for a specific model
    static func modelDirectory(for modelId: String) -> URL {
        let cacheKey = Qwen3HuggingFaceDownloader.sanitizedCacheKey(for: modelId)
        return baseDirectory.appendingPathComponent(cacheKey, isDirectory: true)
    }

    /// Check if a model is downloaded
    static func isModelDownloaded(modelId: String) -> Bool {
        let dir = modelDirectory(for: modelId)
        return Qwen3HuggingFaceDownloader.weightsExist(in: dir)
    }

    /// Delete a downloaded model
    static func deleteModel(modelId: String) throws {
        let dir = modelDirectory(for: modelId)
        if FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.removeItem(at: dir)
            logger.info("Deleted Qwen3 model: \(modelId)")
        }
    }
}
