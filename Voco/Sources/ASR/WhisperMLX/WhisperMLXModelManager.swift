// WhisperMLXModelManager.swift
// Manages Whisper MLX model files on disk (download, cache, delete)
// [AI-Claude: 2026-02-27]

import Foundation
import os

enum WhisperMLXDownloadError: Error, LocalizedError {
    case failedToDownload(String)
    case invalidRemoteFileName(String)

    var errorDescription: String? {
        switch self {
        case .failedToDownload(let file):
            return "Failed to download: \(file)"
        case .invalidRemoteFileName(let file):
            return "Refusing to write unsafe remote file name: \(file)"
        }
    }
}

/// Manages Whisper MLX model files on disk
enum WhisperMLXModelManager {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "WhisperMLXModelManager")
    private static let maxRetries = 3

    /// Base directory for all Whisper MLX models
    static var baseDirectory: URL {
        let fm = FileManager.default
        let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport
            .appendingPathComponent(AppIdentifiers.bundleID, isDirectory: true)
            .appendingPathComponent("WhisperMLXModels", isDirectory: true)
    }

    /// Get the directory for a specific model
    static func modelDirectory(for modelId: String) -> URL {
        let cacheKey = sanitizedCacheKey(for: modelId)
        return baseDirectory.appendingPathComponent(cacheKey, isDirectory: true)
    }

    /// Check if a model is downloaded (has weight files)
    static func isModelDownloaded(modelId: String) -> Bool {
        let dir = modelDirectory(for: modelId)
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir.path) else { return false }
        let contents = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return contents.contains {
            $0.pathExtension == "safetensors" || $0.pathExtension == "npz"
        }
    }

    /// Delete a downloaded model
    static func deleteModel(modelId: String) throws {
        let dir = modelDirectory(for: modelId)
        if FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.removeItem(at: dir)
            logger.info("Deleted Whisper MLX model: \(modelId)")
        }
    }

    static func sanitizedCacheKey(for modelId: String) -> String {
        let replaced = modelId.replacingOccurrences(of: "/", with: "_")
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        var scalars: [UnicodeScalar] = []
        scalars.reserveCapacity(replaced.unicodeScalars.count)
        for s in replaced.unicodeScalars {
            scalars.append(allowed.contains(s) ? s : "_")
        }
        var cleaned = String(String.UnicodeScalarView(scalars))
        cleaned = cleaned.trimmingCharacters(in: CharacterSet(charactersIn: "._"))
        if cleaned.isEmpty || cleaned == "." || cleaned == ".." {
            cleaned = "model"
        }
        return cleaned
    }

    // MARK: - Download

    /// Download model files from HuggingFace
    static func downloadModel(
        modelId: String,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws {
        let directory = modelDirectory(for: modelId)
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)

        let baseURL = "https://huggingface.co/\(modelId)/resolve/main"
        let session = makeSession()
        defer { session.finishTasksAndInvalidate() }

        // Start with config.json to determine what other files to download
        var filesToDownload = ["config.json"]

        // Download config first
        let configLocalPath = directory.appendingPathComponent("config.json")
        if !fm.fileExists(atPath: configLocalPath.path) {
            let configURL = URL(string: "\(baseURL)/config.json")!
            try await downloadFile(url: configURL, to: configLocalPath, session: session, fileName: "config.json")
        }

        // Determine weight format and additional files
        let indexPath = directory.appendingPathComponent("model.safetensors.index.json")
        let weightsNPZPath = directory.appendingPathComponent("weights.npz")

        // Check for sharded safetensors index
        if !fm.fileExists(atPath: indexPath.path) {
            let indexURL = URL(string: "\(baseURL)/model.safetensors.index.json")!
            if let (tempURL, resp) = try? await session.download(from: indexURL),
               let httpResp = resp as? HTTPURLResponse, httpResp.statusCode == 200 {
                try? fm.moveItem(at: tempURL, to: indexPath)
            }
        }

        // Determine model weight files
        var modelFiles: [String] = []
        if fm.fileExists(atPath: indexPath.path),
           let indexData = try? Data(contentsOf: indexPath),
           let index = try? JSONSerialization.jsonObject(with: indexData) as? [String: Any],
           let weightMap = index["weight_map"] as? [String: String], !weightMap.isEmpty {
            modelFiles = Array(Set(weightMap.values)).sorted()
        } else {
            try? fm.removeItem(at: indexPath)
            // Try model.safetensors first, fall back to weights.npz
            let safetensorsURL = URL(string: "\(baseURL)/model.safetensors")!
            if let (_, resp) = try? await session.download(from: safetensorsURL),
               let httpResp = resp as? HTTPURLResponse, httpResp.statusCode == 200 {
                modelFiles = ["model.safetensors"]
            } else {
                modelFiles = ["weights.npz"]
            }
        }

        // Tokenizer files to try
        let tokenizerFiles = [
            "multilingual.tiktoken",
            "vocab.json", "merges.txt", "tokenizer_config.json",
        ]

        filesToDownload.append(contentsOf: modelFiles)
        filesToDownload.append(contentsOf: tokenizerFiles)

        // Remove config.json since we already downloaded it
        filesToDownload = filesToDownload.filter { $0 != "config.json" }

        var downloadedCount = 1  // config.json already downloaded
        let totalFiles = filesToDownload.count + 1

        for file in filesToDownload {
            let safeFile = try validatedRemoteFileName(file)
            let localPath = directory.appendingPathComponent(safeFile)

            if fm.fileExists(atPath: localPath.path) {
                downloadedCount += 1
                progressHandler?(Double(downloadedCount) / Double(totalFiles))
                continue
            }

            let fileURL = URL(string: "\(baseURL)/\(safeFile)")!
            do {
                try await downloadFile(url: fileURL, to: localPath, session: session, fileName: safeFile)
            } catch {
                // Tokenizer files are optional (model may use tiktoken OR BPE)
                if tokenizerFiles.contains(safeFile) {
                    logger.info("Optional file not available: \(safeFile)")
                } else {
                    throw error
                }
            }

            downloadedCount += 1
            progressHandler?(Double(downloadedCount) / Double(totalFiles))
        }

        // Fallback: if no tokenizer file was downloaded, fetch multilingual.tiktoken from OpenAI whisper repo
        let hasTokenizer = fm.fileExists(atPath: directory.appendingPathComponent("multilingual.tiktoken").path)
            || fm.fileExists(atPath: directory.appendingPathComponent("vocab.json").path)
        if !hasTokenizer {
            let fallbackURL = URL(string: "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken")!
            let localPath = directory.appendingPathComponent("multilingual.tiktoken")
            logger.notice("No tokenizer found in repo, fetching from OpenAI whisper assets")
            try await downloadFile(url: fallbackURL, to: localPath, session: session, fileName: "multilingual.tiktoken")
        }

        logger.notice("Whisper MLX model downloaded: \(modelId)")
    }

    // MARK: - Network Helpers

    private static func makeSession() -> URLSession {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 600
        config.waitsForConnectivity = true
        return URLSession(configuration: config)
    }

    private static func downloadFile(
        url: URL, to localPath: URL, session: URLSession, fileName: String
    ) async throws {
        var lastError: Error?
        for attempt in 1...maxRetries {
            do {
                let (tempURL, response) = try await session.download(from: url)
                guard let httpResponse = response as? HTTPURLResponse,
                      httpResponse.statusCode == 200 else {
                    let status = (response as? HTTPURLResponse)?.statusCode ?? -1
                    throw WhisperMLXDownloadError.failedToDownload("\(fileName) (HTTP \(status))")
                }
                let fm = FileManager.default
                if fm.fileExists(atPath: localPath.path) {
                    try fm.removeItem(at: localPath)
                }
                try fm.moveItem(at: tempURL, to: localPath)
                return
            } catch {
                lastError = error
                if attempt < maxRetries {
                    let delay = UInt64(pow(2.0, Double(attempt - 1))) * 1_000_000_000
                    logger.info("Retry \(attempt)/\(maxRetries) for \(fileName)")
                    try? await Task.sleep(nanoseconds: delay)
                }
            }
        }
        throw lastError ?? WhisperMLXDownloadError.failedToDownload(fileName)
    }

    private static func validatedRemoteFileName(_ file: String) throws -> String {
        let base = URL(fileURLWithPath: file).lastPathComponent
        guard base == file,
              !base.isEmpty, !base.hasPrefix("."), !base.contains(".."),
              base.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression) != nil else {
            throw WhisperMLXDownloadError.invalidRemoteFileName(file)
        }
        return base
    }
}
