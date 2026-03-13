// Qwen3CoreMLModelManager.swift
// Manages Qwen3 CoreML encoder model files on disk
// [AI-Claude: 2026-03-13]

import Foundation
import os

enum Qwen3CoreMLModelManager {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "Qwen3CoreMLModelManager")

    /// Base directory for all Qwen3 CoreML models
    static var baseDirectory: URL {
        let fm = FileManager.default
        let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport
            .appendingPathComponent(AppIdentifiers.bundleID, isDirectory: true)
            .appendingPathComponent("Qwen3CoreMLModels", isDirectory: true)
    }

    /// Get the directory for a specific CoreML encoder model
    static func modelDirectory(for modelId: String) -> URL {
        let cacheKey = Qwen3HuggingFaceDownloader.sanitizedCacheKey(for: modelId)
        return baseDirectory.appendingPathComponent(cacheKey, isDirectory: true)
    }

    /// Check if a CoreML encoder model is downloaded (has encoder.mlmodelc)
    static func isEncoderDownloaded(modelId: String) -> Bool {
        let dir = modelDirectory(for: modelId)
        let compiledPath = dir.appendingPathComponent("encoder.mlmodelc")
        return FileManager.default.fileExists(atPath: compiledPath.path)
    }

    /// Delete a downloaded CoreML encoder model
    static func deleteModel(modelId: String) throws {
        let dir = modelDirectory(for: modelId)
        if FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.removeItem(at: dir)
            logger.info("Deleted Qwen3 CoreML model: \(modelId)")
        }
    }

    /// Download the CoreML encoder model from HuggingFace
    static func downloadEncoder(
        modelId: String,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws {
        let cacheDir = try getCacheDirectory(for: modelId)

        // Download all files from the HuggingFace repo
        try await Qwen3HuggingFaceDownloader.downloadCoreMLEncoder(
            modelId: modelId,
            to: cacheDir,
            progressHandler: progressHandler
        )
    }

    private static func getCacheDirectory(for modelId: String) throws -> URL {
        let dir = modelDirectory(for: modelId)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}

// MARK: - CoreML-specific download extension

extension Qwen3HuggingFaceDownloader {
    /// Download CoreML encoder files from HuggingFace
    /// Expects the repo to contain an encoder.mlmodelc directory (or encoder.mlpackage)
    static func downloadCoreMLEncoder(
        modelId: String,
        to directory: URL,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws {
        let baseURL = "https://huggingface.co/\(modelId)/resolve/main"
        let session = makeSession()
        defer { session.finishTasksAndInvalidate() }

        // First, try to list files in the repo to find the encoder model
        // CoreML models on HuggingFace are typically stored as directories
        // We need to download the compiled model files

        // Check if there's a file listing we can use
        let encoderDir = directory.appendingPathComponent("encoder.mlmodelc", isDirectory: true)
        if FileManager.default.fileExists(atPath: encoderDir.path) {
            logger.info("CoreML encoder already exists at \(encoderDir.path)")
            progressHandler?(1.0)
            return
        }

        try FileManager.default.createDirectory(at: encoderDir, withIntermediateDirectories: true)

        // Download the model manifest to discover files
        // HuggingFace stores .mlmodelc as a tree of files
        let treeURL = URL(string: "https://huggingface.co/api/models/\(modelId)/tree/main/encoder.mlmodelc")!
        let (treeData, treeResponse) = try await session.data(from: treeURL)

        guard let httpResponse = treeResponse as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            let status = (treeResponse as? HTTPURLResponse)?.statusCode ?? -1
            throw Qwen3DownloadError.failedToDownload("Failed to list encoder files (HTTP \(status))")
        }

        // Parse the tree response to get file paths
        guard let files = try JSONSerialization.jsonObject(with: treeData) as? [[String: Any]] else {
            throw Qwen3DownloadError.failedToDownload("Invalid tree response for encoder.mlmodelc")
        }

        // Collect all file paths (filter to actual files, not directories)
        var filePaths: [String] = []
        collectFiles(from: files, prefix: "", into: &filePaths)

        if filePaths.isEmpty {
            throw Qwen3DownloadError.failedToDownload("No files found in encoder.mlmodelc")
        }

        logger.info("Found \(filePaths.count) files in encoder.mlmodelc")

        // Download each file
        for (index, filePath) in filePaths.enumerated() {
            let remoteURL = URL(string: "\(baseURL)/encoder.mlmodelc/\(filePath)")!
            let localPath = encoderDir.appendingPathComponent(filePath)

            // Create subdirectories if needed
            let parentDir = localPath.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)

            if FileManager.default.fileExists(atPath: localPath.path) {
                progressHandler?(Double(index + 1) / Double(filePaths.count))
                continue
            }

            try await downloadFile(url: remoteURL, to: localPath, session: session, fileName: filePath)
            progressHandler?(Double(index + 1) / Double(filePaths.count))
        }
    }

    /// Recursively collect file paths from HuggingFace tree API response
    private static func collectFiles(from items: [[String: Any]], prefix: String, into paths: inout [String]) {
        for item in items {
            guard let type = item["type"] as? String,
                  let path = item["path"] as? String else { continue }

            // Extract just the filename part after "encoder.mlmodelc/"
            let relativePath: String
            if let range = path.range(of: "encoder.mlmodelc/") {
                relativePath = String(path[range.upperBound...])
            } else {
                relativePath = path
            }

            if type == "file" {
                paths.append(relativePath)
            }
            // Directories in the tree listing are handled by the API returning nested entries
        }
    }

}
