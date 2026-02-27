// WhisperKitModelManager.swift
// Manages WhisperKit CoreML model files on disk
// [AI-Claude: 2026-02-27]

import Foundation
import os

enum WhisperKitModelManager {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "WhisperKitModelManager")

    private static let defaultsKeyPrefix = "WhisperKitModelFolder_"

    /// Get the stored model folder path for a variant (returned by WhisperKit.download)
    static func modelDirectory(for variant: String) -> URL {
        if let path = UserDefaults.standard.string(forKey: defaultsKeyPrefix + variant) {
            return URL(fileURLWithPath: path)
        }
        // Fallback: default HubApi cache location
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return cacheDir
            .appendingPathComponent("huggingface")
            .appendingPathComponent("models--argmaxinc--whisperkit-coreml")
            .appendingPathComponent(variant, isDirectory: true)
    }

    /// Save the downloaded model folder path
    static func saveModelDirectory(_ url: URL, for variant: String) {
        UserDefaults.standard.set(url.path, forKey: defaultsKeyPrefix + variant)
        logger.info("Saved WhisperKit model path for \(variant): \(url.path)")
    }

    /// Check if a model variant is downloaded (stored path exists and contains .mlmodelc files)
    static func isModelDownloaded(variant: String) -> Bool {
        guard let path = UserDefaults.standard.string(forKey: defaultsKeyPrefix + variant) else {
            return false
        }
        let dir = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: dir.path) else { return false }
        // Check for at least one .mlmodelc bundle inside
        let contents = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        return contents.contains { $0.hasSuffix(".mlmodelc") }
    }

    /// Delete a downloaded model
    static func deleteModel(variant: String) throws {
        if let path = UserDefaults.standard.string(forKey: defaultsKeyPrefix + variant) {
            let dir = URL(fileURLWithPath: path)
            if FileManager.default.fileExists(atPath: dir.path) {
                try FileManager.default.removeItem(at: dir)
                logger.info("Deleted WhisperKit model: \(variant)")
            }
        }
        UserDefaults.standard.removeObject(forKey: defaultsKeyPrefix + variant)
    }
}
