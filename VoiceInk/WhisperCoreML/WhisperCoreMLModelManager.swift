// WhisperCoreMLModelManager.swift
// Manages CoreML Whisper model files on disk (download, cache, delete)
// [AI-Claude: 2026-03-02]

import Foundation
import Compression
import os

enum WhisperCoreMLDownloadError: Error, LocalizedError {
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

/// Manages CoreML Whisper model files on disk
enum WhisperCoreMLModelManager {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "WhisperCoreMLModelManager")
    private static let maxRetries = 3

    /// Base directory for all CoreML Whisper models
    /// On iOS, uses App Group shared container so the keyboard extension can access models
    /// downloaded by the main app. On macOS, uses the standard Application Support path.
    static var baseDirectory: URL {
        #if os(iOS)
        if let groupDir = AppIdentifiers.appGroupDirectory {
            return groupDir.appendingPathComponent("WhisperCoreMLModels", isDirectory: true)
        }
        #endif
        let fm = FileManager.default
        let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport
            .appendingPathComponent(AppIdentifiers.bundleID, isDirectory: true)
            .appendingPathComponent("WhisperCoreMLModels", isDirectory: true)
    }

    /// Get the directory for a specific model
    static func modelDirectory(for modelId: String) -> URL {
        let cacheKey = sanitizedCacheKey(for: modelId)
        return baseDirectory.appendingPathComponent(cacheKey, isDirectory: true)
    }

    /// Check if a model is downloaded (has encoder + decoder + config)
    static func isModelDownloaded(modelId: String) -> Bool {
        let dir = modelDirectory(for: modelId)
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir.path) else { return false }

        let hasEncoder = fm.fileExists(atPath: dir.appendingPathComponent("WhisperEncoder.mlmodelc").path)
            || fm.fileExists(atPath: dir.appendingPathComponent("WhisperEncoder.mlpackage").path)
        let hasDecoder = fm.fileExists(atPath: dir.appendingPathComponent("WhisperDecoder.mlmodelc").path)
            || fm.fileExists(atPath: dir.appendingPathComponent("WhisperDecoder.mlpackage").path)
        let hasConfig = fm.fileExists(atPath: dir.appendingPathComponent("coreml_config.json").path)

        return hasEncoder && hasDecoder && hasConfig
    }

    /// Delete a downloaded model
    static func deleteModel(modelId: String) throws {
        let dir = modelDirectory(for: modelId)
        if FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.removeItem(at: dir)
            logger.info("Deleted CoreML Whisper model: \(modelId)")
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

    /// Download CoreML model files from a release URL
    /// Expected structure at the URL: WhisperEncoder.mlpackage/, WhisperDecoder.mlpackage/,
    /// coreml_config.json, and tokenizer files
    static func downloadModel(
        modelId: String,
        baseURL: String,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws {
        let directory = modelDirectory(for: modelId)
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)

        let session = makeSession()
        defer { session.finishTasksAndInvalidate() }

        // Required files
        let requiredFiles = [
            "coreml_config.json",
        ]

        // MLPackage directories are distributed as zip archives
        let archiveFiles = [
            "WhisperEncoder.mlmodelc.zip",
            "WhisperDecoder.mlmodelc.zip",
        ]

        // Tokenizer files (at least one set required)
        let tokenizerFiles = [
            "multilingual.tiktoken",
            "vocab.json", "merges.txt", "tokenizer_config.json",
        ]

        let allFiles = requiredFiles + archiveFiles + tokenizerFiles
        var downloadedCount = 0
        let totalFiles = allFiles.count

        for file in allFiles {
            let safeFile = try validatedRemoteFileName(file)
            let localPath = directory.appendingPathComponent(safeFile)

            // Skip if already exists (or unpacked version exists)
            let unpackedName = safeFile.replacingOccurrences(of: ".zip", with: "")
            if fm.fileExists(atPath: localPath.path) ||
               (safeFile.hasSuffix(".zip") && fm.fileExists(atPath: directory.appendingPathComponent(unpackedName).path)) {
                downloadedCount += 1
                progressHandler?(Double(downloadedCount) / Double(totalFiles))
                continue
            }

            let fileURL = URL(string: "\(baseURL)/\(safeFile)")!
            do {
                try await downloadFile(url: fileURL, to: localPath, session: session, fileName: safeFile)

                // Unzip .mlmodelc archives
                if safeFile.hasSuffix(".zip") {
                    try unzipFile(at: localPath, to: directory)
                    try? fm.removeItem(at: localPath)
                }
            } catch {
                if tokenizerFiles.contains(safeFile) {
                    logger.info("Optional tokenizer file not available: \(safeFile)")
                } else {
                    throw error
                }
            }

            downloadedCount += 1
            progressHandler?(Double(downloadedCount) / Double(totalFiles))
        }

        // Fallback tokenizer
        let hasTokenizer = fm.fileExists(atPath: directory.appendingPathComponent("multilingual.tiktoken").path)
            || fm.fileExists(atPath: directory.appendingPathComponent("vocab.json").path)
        if !hasTokenizer {
            let fallbackURL = URL(string: "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken")!
            let localPath = directory.appendingPathComponent("multilingual.tiktoken")
            logger.notice("No tokenizer found, fetching from OpenAI whisper assets")
            try await downloadFile(url: fallbackURL, to: localPath, session: session, fileName: "multilingual.tiktoken")
        }

        logger.notice("CoreML Whisper model downloaded: \(modelId)")
    }

    // MARK: - Helpers

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
                    throw WhisperCoreMLDownloadError.failedToDownload("\(fileName) (HTTP \(status))")
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
        throw lastError ?? WhisperCoreMLDownloadError.failedToDownload(fileName)
    }

    private static func unzipFile(at source: URL, to destination: URL) throws {
        #if os(macOS)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        process.arguments = ["-o", source.path, "-d", destination.path]
        process.standardOutput = nil
        process.standardError = nil
        try process.run()
        process.waitUntilExit()

        if process.terminationStatus != 0 {
            throw WhisperCoreMLDownloadError.failedToDownload("Failed to unzip: \(source.lastPathComponent)")
        }
        #else
        // iOS: pure-Swift ZIP extraction (Process is macOS-only)
        let data = try Data(contentsOf: source)
        try unzipData(data, to: destination)
        #endif
    }

    #if !os(macOS)
    /// Minimal ZIP extraction for iOS (handles the simple case of .mlmodelc directories)
    private static func unzipData(_ zipData: Data, to destination: URL) throws {
        let fm = FileManager.default

        // ZIP file structure: Local file headers + data + central directory
        var offset = 0
        let bytes = [UInt8](zipData)
        let count = bytes.count

        while offset + 30 <= count {
            // Local file header signature: 0x04034b50
            guard bytes[offset] == 0x50, bytes[offset+1] == 0x4B,
                  bytes[offset+2] == 0x03, bytes[offset+3] == 0x04 else {
                break  // No more local file headers
            }

            let compressionMethod = UInt16(bytes[offset+8]) | (UInt16(bytes[offset+9]) << 8)
            let compressedSize = Int(UInt32(bytes[offset+18]) | (UInt32(bytes[offset+19]) << 8) |
                                     (UInt32(bytes[offset+20]) << 16) | (UInt32(bytes[offset+21]) << 24))
            let uncompressedSize = Int(UInt32(bytes[offset+22]) | (UInt32(bytes[offset+23]) << 8) |
                                       (UInt32(bytes[offset+24]) << 16) | (UInt32(bytes[offset+25]) << 24))
            let fileNameLen = Int(UInt16(bytes[offset+26]) | (UInt16(bytes[offset+27]) << 8))
            let extraLen = Int(UInt16(bytes[offset+28]) | (UInt16(bytes[offset+29]) << 8))

            let nameStart = offset + 30
            guard nameStart + fileNameLen <= count else { break }
            let fileName = String(bytes: bytes[nameStart..<(nameStart + fileNameLen)], encoding: .utf8) ?? ""

            let dataStart = nameStart + fileNameLen + extraLen
            let dataEnd = dataStart + compressedSize

            guard dataEnd <= count else { break }

            // Security: reject path traversal
            guard !fileName.contains(".."), !fileName.hasPrefix("/") else {
                offset = dataEnd
                continue
            }

            let filePath = destination.appendingPathComponent(fileName)

            if fileName.hasSuffix("/") {
                // Directory entry
                try fm.createDirectory(at: filePath, withIntermediateDirectories: true)
            } else {
                // File entry
                try fm.createDirectory(at: filePath.deletingLastPathComponent(), withIntermediateDirectories: true)

                if compressionMethod == 0 {
                    // Stored (no compression)
                    let fileData = Data(bytes[dataStart..<dataEnd])
                    try fileData.write(to: filePath)
                } else if compressionMethod == 8 {
                    // Deflate — use Apple's Compression framework
                    let compressedData = Data(bytes[dataStart..<dataEnd])
                    let decompressed = try decompressDeflate(compressedData, expectedSize: uncompressedSize)
                    try decompressed.write(to: filePath)
                } else {
                    logger.warning("Unsupported compression method \(compressionMethod) for: \(fileName)")
                }
            }

            offset = dataEnd
        }
    }

    /// Decompress deflate data using Apple's Compression framework
    private static func decompressDeflate(_ data: Data, expectedSize: Int) throws -> Data {
        let bufferSize = max(expectedSize, 65536)
        var decompressed = Data(count: bufferSize)
        let result = data.withUnsafeBytes { srcPtr -> Int in
            decompressed.withUnsafeMutableBytes { dstPtr -> Int in
                let srcBound = srcPtr.bindMemory(to: UInt8.self)
                let dstBound = dstPtr.bindMemory(to: UInt8.self)
                return compression_decode_buffer(
                    dstBound.baseAddress!, bufferSize,
                    srcBound.baseAddress!, data.count,
                    nil,
                    COMPRESSION_ZLIB
                )
            }
        }
        guard result > 0 else {
            throw WhisperCoreMLDownloadError.failedToDownload("Deflate decompression failed")
        }
        decompressed.count = result
        return decompressed
    }
    #endif

    private static func validatedRemoteFileName(_ file: String) throws -> String {
        let base = URL(fileURLWithPath: file).lastPathComponent
        guard base == file,
              !base.isEmpty, !base.hasPrefix("."), !base.contains(".."),
              base.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression) != nil else {
            throw WhisperCoreMLDownloadError.invalidRemoteFileName(file)
        }
        return base
    }
}
