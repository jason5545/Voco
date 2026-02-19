// Qwen3HuggingFaceDownloader.swift
// Adapted from qwen3-asr-swift HuggingFaceDownloader.swift
// Changed: cache dir → ~/Library/Application Support/<BundleID>/Qwen3Models/
// Removed: environment variable overrides
// Fixed: print() → os.Logger
// [AI-Claude: 2025-02-18]

import Foundation
import os

enum Qwen3DownloadError: Error, LocalizedError {
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

/// HuggingFace model downloader for Qwen3-ASR
enum Qwen3HuggingFaceDownloader {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "Qwen3Downloader")
    private static let maxRetries = 3

    /// Get cache directory for a model under Application Support
    static func getCacheDirectory(for modelId: String) throws -> URL {
        let cacheKey = sanitizedCacheKey(for: modelId)
        let fm = FileManager.default

        let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let baseDir = appSupport
            .appendingPathComponent(AppIdentifiers.bundleID, isDirectory: true)
            .appendingPathComponent("Qwen3Models", isDirectory: true)
            .appendingPathComponent(cacheKey, isDirectory: true)

        try fm.createDirectory(at: baseDir, withIntermediateDirectories: true)
        return baseDir
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

    static func weightsExist(in directory: URL) -> Bool {
        let fm = FileManager.default
        let contents = (try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        return contents.contains { $0.pathExtension == "safetensors" }
    }

    static func validatedRemoteFileName(_ file: String) throws -> String {
        let base = URL(fileURLWithPath: file).lastPathComponent
        guard base == file else {
            throw Qwen3DownloadError.invalidRemoteFileName(file)
        }
        guard !base.isEmpty, !base.hasPrefix("."), !base.contains("..") else {
            throw Qwen3DownloadError.invalidRemoteFileName(file)
        }
        guard base.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression) != nil else {
            throw Qwen3DownloadError.invalidRemoteFileName(file)
        }
        return base
    }

    static func validatedLocalPath(directory: URL, fileName: String) throws -> URL {
        let local = directory.appendingPathComponent(fileName, isDirectory: false)
        let dirPath = directory.standardizedFileURL.path
        let localPath = local.standardizedFileURL.path
        let prefix = dirPath.hasSuffix("/") ? dirPath : (dirPath + "/")
        guard localPath.hasPrefix(prefix) else {
            throw Qwen3DownloadError.invalidRemoteFileName(fileName)
        }
        return local
    }

    private static func makeSession() -> URLSession {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 600
        config.waitsForConnectivity = true
        return URLSession(configuration: config)
    }

    private static func downloadFile(
        url: URL,
        to localPath: URL,
        session: URLSession,
        fileName: String
    ) async throws {
        var lastError: Error?

        for attempt in 1...maxRetries {
            do {
                let (tempURL, response) = try await session.download(from: url)

                guard let httpResponse = response as? HTTPURLResponse,
                      httpResponse.statusCode == 200 else {
                    let status = (response as? HTTPURLResponse)?.statusCode ?? -1
                    throw Qwen3DownloadError.failedToDownload("\(fileName) (HTTP \(status))")
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
                    logger.info("Retry \(attempt)/\(maxRetries) for \(fileName): \(error.localizedDescription)")
                    try? await Task.sleep(nanoseconds: delay)
                }
            }
        }

        throw lastError ?? Qwen3DownloadError.failedToDownload(fileName)
    }

    /// Download model files from HuggingFace
    static func downloadWeights(
        modelId: String,
        to directory: URL,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws {
        let baseURL = "https://huggingface.co/\(modelId)/resolve/main"
        let session = makeSession()
        defer { session.finishTasksAndInvalidate() }

        let additionalFiles = ["vocab.json", "merges.txt", "tokenizer_config.json"]

        var filesToDownload = ["config.json"]
        filesToDownload.append(contentsOf: additionalFiles)

        // Determine model file(s) to download
        let indexPath = directory.appendingPathComponent("model.safetensors.index.json")

        if !FileManager.default.fileExists(atPath: indexPath.path) {
            let indexURL = URL(string: "\(baseURL)/model.safetensors.index.json")!
            if let (tempURL, indexResponse) = try? await session.download(from: indexURL),
               let httpResponse = indexResponse as? HTTPURLResponse,
               httpResponse.statusCode == 200 {
                try? FileManager.default.moveItem(at: tempURL, to: indexPath)
            }
        }

        var modelFiles: [String] = []
        if FileManager.default.fileExists(atPath: indexPath.path),
           let indexData = try? Data(contentsOf: indexPath),
           let index = try? JSONSerialization.jsonObject(with: indexData) as? [String: Any],
           let weightMap = index["weight_map"] as? [String: String],
           !weightMap.isEmpty {
            let uniqueFiles = Set(weightMap.values)
            modelFiles = Array(uniqueFiles).sorted()
        } else {
            try? FileManager.default.removeItem(at: indexPath)
            modelFiles = ["model.safetensors"]
        }

        filesToDownload.append(contentsOf: modelFiles)

        for (index, file) in filesToDownload.enumerated() {
            let safeFile = try validatedRemoteFileName(file)
            let localPath = try validatedLocalPath(directory: directory, fileName: safeFile)

            if FileManager.default.fileExists(atPath: localPath.path) {
                progressHandler?(Double(index + 1) / Double(filesToDownload.count))
                continue
            }

            let url = URL(string: "\(baseURL)/\(safeFile)")!
            try await downloadFile(url: url, to: localPath, session: session, fileName: safeFile)

            progressHandler?(Double(index + 1) / Double(filesToDownload.count))
        }
    }
}
