import Combine
import CryptoKit
import Foundation
import OSLog

enum Qwen3ASRContextBiasSourceKind: String, Sendable {
    case downloaded
    case builtin
    case unavailable
}

struct Qwen3ASRContextBiasProfile: Equatable, Sendable {
    let sourceKind: Qwen3ASRContextBiasSourceKind
    let artifactId: String
    let createdAt: String?
    let terms: [String]
    let boost: Float
    let maxTermsPerDecode: Int
    let repeatNgramSize: Int
    let repeatNgramMaxCount: Int
    let sha256: String?
    let fileURL: URL?

    var isUsable: Bool {
        !terms.isEmpty && boost > 0 && maxTermsPerDecode > 0
    }
}

struct Qwen3ASRContextBiasStatus: Equatable, Sendable {
    let sourceKind: Qwen3ASRContextBiasSourceKind
    let artifactId: String?
    let createdAt: String?
    let termCount: Int
    let boost: Float?
    let maxTermsPerDecode: Int?
    let repeatNgramSize: Int?
    let repeatNgramMaxCount: Int?
    let sha256: String?
    let fileURL: URL
    let lastOutcome: String?
    let lastMessage: String?
    let lastDownloadedAt: Date?

    var isAvailable: Bool {
        sourceKind != .unavailable && termCount > 0
    }
}

enum Qwen3ASRContextBiasDownloadOutcome: Equatable, Sendable {
    case installed
    case alreadyCurrent
    case failed(String)

    var rawValue: String {
        switch self {
        case .installed:
            return "INSTALLED"
        case .alreadyCurrent:
            return "ALREADY_CURRENT"
        case .failed:
            return "FAILED"
        }
    }

    var message: String? {
        switch self {
        case .installed:
            return "Installed"
        case .alreadyCurrent:
            return "Already current"
        case .failed(let message):
            return message
        }
    }
}

enum Qwen3ASRContextBiasStoreError: LocalizedError, Equatable {
    case invalidHTTPResponse
    case httpStatus(Int)
    case unsupportedSchema(String)
    case emptyTerms
    case invalidBoost(Float)
    case invalidRemoteURL
    case invalidJSON(String)

    var errorDescription: String? {
        switch self {
        case .invalidHTTPResponse:
            return "Bias download did not return an HTTP response"
        case .httpStatus(let status):
            return "Bias download returned HTTP \(status)"
        case .unsupportedSchema(let schema):
            return "Unsupported ASR context bias schema: \(schema)"
        case .emptyTerms:
            return "ASR context bias terms are empty"
        case .invalidBoost(let boost):
            return "ASR context bias boost is out of range: \(boost)"
        case .invalidRemoteURL:
            return "Invalid ASR context bias URL"
        case .invalidJSON(let reason):
            return "Invalid ASR context bias JSON: \(reason)"
        }
    }
}

@MainActor
final class Qwen3ASRContextBiasStore: ObservableObject {
    static let shared = Qwen3ASRContextBiasStore()

    nonisolated static let enabledKey = "Qwen3ASRContextHotwordBiasEnabled"
    nonisolated static let boostOverrideKey = "Qwen3ASRContextHotwordBiasBoost"
    nonisolated static let supportedSchema = "vocotype.qwen3-asr.context-hotword-bias.v1"
    nonisolated static let remoteProfileURL = URL(
        string: "https://huggingface.co/jason5545/vocotype-qwen3-asr-adapter-int4/resolve/main/runtime/context-hotword-bias-20260705/context-hotword-bias-20260705.json"
    )!
    nonisolated static let profileFileName = "context-hotword-bias.json"

    @Published private(set) var status: Qwen3ASRContextBiasStatus
    @Published private(set) var isDownloading = false

    private let fileURL: URL
    private let defaults: UserDefaults
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "Qwen3ContextBias")
    private var cachedOverride: (modifiedAt: Date, size: Int, profile: Qwen3ASRContextBiasProfile)?

    nonisolated static let builtinProfile = Qwen3ASRContextBiasProfile(
        sourceKind: .builtin,
        artifactId: "builtin-context-hotword-bias-20260705",
        createdAt: "2026-07-05T14:30:00+08:00",
        terms: [
            "repo",
            "GitHub",
            "CLI",
            "MCP",
            "JSON",
            "JSONL",
            "Voco",
            "Qwen",
            "ASR",
            "LoRA",
            "adapter",
            "Markdown",
            "Application Support",
        ],
        boost: 4.0,
        maxTermsPerDecode: 8,
        repeatNgramSize: 4,
        repeatNgramMaxCount: 2,
        sha256: nil,
        fileURL: nil
    )

    init(
        fileURL: URL = Qwen3ASRContextBiasStore.defaultProfileURL,
        defaults: UserDefaults = .standard
    ) {
        self.fileURL = fileURL
        self.defaults = defaults
        self.status = Self.makeStatus(
            from: Self.builtinProfile,
            fileURL: fileURL,
            defaults: defaults
        )
        reload()
    }

    nonisolated static var defaultProfileDirectory: URL {
        AppIdentifiers.appSupportDirectory
            .appendingPathComponent("Qwen3ASRContextBias", isDirectory: true)
    }

    nonisolated static var defaultProfileURL: URL {
        defaultProfileDirectory.appendingPathComponent(profileFileName)
    }

    var isEnabled: Bool {
        if defaults.object(forKey: Self.enabledKey) == nil {
            return true
        }
        return defaults.bool(forKey: Self.enabledKey)
    }

    var boostOverride: Float? {
        let value = defaults.double(forKey: Self.boostOverrideKey)
        return value > 0 ? Float(value) : nil
    }

    func activeProfile() -> Qwen3ASRContextBiasProfile {
        readOverrideProfile() ?? Self.builtinProfile
    }

    func reload() {
        status = Self.makeStatus(
            from: activeProfile(),
            fileURL: fileURL,
            defaults: defaults
        )
    }

    func downloadLatest() async -> Qwen3ASRContextBiasDownloadOutcome {
        guard !isDownloading else { return .failed("Download already in progress") }
        isDownloading = true
        defer { isDownloading = false }

        do {
            let (data, response) = try await URLSession.shared.data(from: Self.remoteProfileURL)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw Qwen3ASRContextBiasStoreError.invalidHTTPResponse
            }
            guard (200...299).contains(httpResponse.statusCode) else {
                throw Qwen3ASRContextBiasStoreError.httpStatus(httpResponse.statusCode)
            }

            let sha = Self.sha256Hex(for: data)
            let downloadedProfile = try Self.decodeProfile(
                from: data,
                sourceKind: .downloaded,
                sha256: sha,
                fileURL: fileURL
            )
            guard downloadedProfile.isUsable else {
                throw Qwen3ASRContextBiasStoreError.emptyTerms
            }

            if readOverrideProfile()?.sha256 == sha {
                recordOutcome(.alreadyCurrent)
                reload()
                return .alreadyCurrent
            }

            try FileManager.default.createDirectory(
                at: fileURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try data.write(to: fileURL, options: .atomic)
            cachedOverride = nil
            recordOutcome(.installed)
            reload()
            logger.info("Installed ASR context bias artifact=\(downloadedProfile.artifactId, privacy: .public) sha=\(sha, privacy: .public)")
            return .installed
        } catch {
            let message = error.localizedDescription
            logger.error("ASR context bias download failed: \(message, privacy: .public)")
            let outcome: Qwen3ASRContextBiasDownloadOutcome = .failed(message)
            recordOutcome(outcome)
            reload()
            return outcome
        }
    }

    private func readOverrideProfile() -> Qwen3ASRContextBiasProfile? {
        let attributes = try? FileManager.default.attributesOfItem(atPath: fileURL.path)
        guard let modifiedAt = attributes?[.modificationDate] as? Date,
              let size = attributes?[.size] as? NSNumber,
              size.intValue > 0 else {
            return nil
        }
        if let cachedOverride,
           cachedOverride.modifiedAt == modifiedAt,
           cachedOverride.size == size.intValue {
            return cachedOverride.profile
        }

        do {
            let data = try Data(contentsOf: fileURL)
            let profile = try Self.decodeProfile(
                from: data,
                sourceKind: .downloaded,
                sha256: Self.sha256Hex(for: data),
                fileURL: fileURL
            )
            cachedOverride = (modifiedAt, size.intValue, profile)
            return profile
        } catch {
            logger.error("Invalid local ASR context bias profile: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }

    private func recordOutcome(_ outcome: Qwen3ASRContextBiasDownloadOutcome) {
        defaults.set(outcome.rawValue, forKey: "Qwen3ASRContextBiasLastOutcome")
        defaults.set(outcome.message, forKey: "Qwen3ASRContextBiasLastMessage")
        defaults.set(Date().timeIntervalSince1970, forKey: "Qwen3ASRContextBiasLastDownloadedAt")
    }

    nonisolated static func decodeProfile(
        from data: Data,
        sourceKind: Qwen3ASRContextBiasSourceKind,
        sha256: String?,
        fileURL: URL?
    ) throws -> Qwen3ASRContextBiasProfile {
        let artifact: RemoteArtifact
        do {
            artifact = try JSONDecoder().decode(RemoteArtifact.self, from: data)
        } catch {
            throw Qwen3ASRContextBiasStoreError.invalidJSON(error.localizedDescription)
        }

        guard artifact.schema == supportedSchema else {
            throw Qwen3ASRContextBiasStoreError.unsupportedSchema(artifact.schema)
        }

        var seenTerms = Set<String>()
        let terms = artifact.decodeBias.terms.compactMap { rawTerm -> String? in
            let term = rawTerm.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !term.isEmpty, seenTerms.insert(term).inserted else { return nil }
            return term
        }
        guard !terms.isEmpty else {
            throw Qwen3ASRContextBiasStoreError.emptyTerms
        }

        let boost = Float(artifact.decodeBias.boost)
        guard boost.isFinite, boost > 0, boost <= 16 else {
            throw Qwen3ASRContextBiasStoreError.invalidBoost(boost)
        }

        return Qwen3ASRContextBiasProfile(
            sourceKind: sourceKind,
            artifactId: artifact.artifactId,
            createdAt: artifact.createdAt,
            terms: terms,
            boost: boost,
            maxTermsPerDecode: max(1, min(artifact.decodeBias.maxTermsPerDecode ?? 8, 32)),
            repeatNgramSize: max(0, min(artifact.decodeBias.repetitionGuard?.repeatNgramSize ?? 4, 16)),
            repeatNgramMaxCount: max(1, min(artifact.decodeBias.repetitionGuard?.repeatNgramMaxCount ?? 2, 16)),
            sha256: sha256,
            fileURL: fileURL
        )
    }

    nonisolated static func sha256Hex(for data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func makeStatus(
        from profile: Qwen3ASRContextBiasProfile,
        fileURL: URL,
        defaults: UserDefaults
    ) -> Qwen3ASRContextBiasStatus {
        let timestamp = defaults.double(forKey: "Qwen3ASRContextBiasLastDownloadedAt")
        return Qwen3ASRContextBiasStatus(
            sourceKind: profile.sourceKind,
            artifactId: profile.artifactId,
            createdAt: profile.createdAt,
            termCount: profile.terms.count,
            boost: profile.boost,
            maxTermsPerDecode: profile.maxTermsPerDecode,
            repeatNgramSize: profile.repeatNgramSize,
            repeatNgramMaxCount: profile.repeatNgramMaxCount,
            sha256: profile.sha256,
            fileURL: profile.fileURL ?? fileURL,
            lastOutcome: defaults.string(forKey: "Qwen3ASRContextBiasLastOutcome"),
            lastMessage: defaults.string(forKey: "Qwen3ASRContextBiasLastMessage"),
            lastDownloadedAt: timestamp > 0 ? Date(timeIntervalSince1970: timestamp) : nil
        )
    }
}

enum Qwen3ContextHotwordBias {
    static func selectedTerms(
        profile: Qwen3ASRContextBiasProfile,
        baselineTranscript: String,
        prompt: String?,
        recentTranscriptions: [String]
    ) -> [String] {
        let context = ([prompt ?? ""] + recentTranscriptions)
            .joined(separator: "\n")
        var selected: [String] = []
        for term in profile.terms where shouldBias(term: term, baseline: baselineTranscript, context: context) {
            selected.append(term)
        }
        return Array(selected.prefix(profile.maxTermsPerDecode))
    }

    private static func shouldBias(term: String, baseline: String, context: String) -> Bool {
        if containsTerm(term, in: context) || containsTerm(term, in: baseline) {
            return true
        }

        let baselineLower = baseline.lowercased()
        let contextHasTerm = containsTerm(term, in: context)
        switch term {
        case "repo":
            return contextHasTerm && (
                containsASCIIWord("report", in: baselineLower) ||
                baselineLower.contains("ripoli") ||
                baselineLower.contains("ripley")
            )
        case "MCP":
            return containsASCIISequence(["m", "c", "p"], in: baselineLower) ||
                containsASCIIWord("mcd", in: baselineLower) ||
                containsASCIIWord("mcb", in: baselineLower) ||
                baselineLower.contains("m c pick")
        case "Qwen":
            return baselineLower.contains("qwen") ||
                baseline.contains("千文") ||
                baseline.contains("鍵蚊")
        case "CLI":
            return containsASCIISequence(["c", "l", "i"], in: baselineLower)
        case "GitHub":
            return baselineLower.contains("git hub") || baselineLower.contains("githup")
        default:
            return false
        }
    }

    private static func containsTerm(_ term: String, in text: String) -> Bool {
        guard !term.isEmpty, !text.isEmpty else { return false }
        if term.range(of: #"^[A-Za-z0-9_.+-]+$"#, options: .regularExpression) != nil {
            return containsASCIIWord(term.lowercased(), in: text.lowercased())
        }
        return text.range(of: term, options: [.caseInsensitive, .diacriticInsensitive]) != nil
    }

    private static func containsASCIIWord(_ word: String, in text: String) -> Bool {
        asciiTokens(in: text).contains(word.lowercased())
    }

    private static func containsASCIISequence(_ sequence: [String], in text: String) -> Bool {
        let tokens = asciiTokens(in: text)
        guard tokens.count >= sequence.count else { return false }
        for index in 0...(tokens.count - sequence.count) {
            if Array(tokens[index..<(index + sequence.count)]) == sequence {
                return true
            }
        }
        return false
    }

    private static func asciiTokens(in text: String) -> [String] {
        var tokens: [String] = []
        var current = ""
        for scalar in text.unicodeScalars {
            if scalar.isASCII, CharacterSet.alphanumerics.contains(scalar) {
                current.unicodeScalars.append(scalar)
            } else if !current.isEmpty {
                tokens.append(current.lowercased())
                current.removeAll(keepingCapacity: true)
            }
        }
        if !current.isEmpty {
            tokens.append(current.lowercased())
        }
        return tokens
    }
}

private struct RemoteArtifact: Decodable {
    let schema: String
    let artifactId: String
    let createdAt: String?
    let decodeBias: DecodeBias
}

private struct DecodeBias: Decodable {
    let boost: Double
    let terms: [String]
    let maxTermsPerDecode: Int?
    let repetitionGuard: RepetitionGuard?
}

private struct RepetitionGuard: Decodable {
    let repeatNgramSize: Int?
    let repeatNgramMaxCount: Int?
}
