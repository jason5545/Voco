import Foundation
import Combine
import CryptoKit
import Darwin
import OSLog

struct VocoAutoApplyPolicyFire: Codable, Equatable {
    let policyId: String
    let policyType: String
    let autoApplyMode: String
    let sourcePattern: String
    let targetText: String
    let sourceSlices: [String]
    let sourceBoundaryMode: String?
    let familyId: String?
    let familyRole: String?

    init(
        policyId: String,
        policyType: String,
        autoApplyMode: String,
        sourcePattern: String,
        targetText: String,
        sourceSlices: [String],
        sourceBoundaryMode: String? = nil,
        familyId: String? = nil,
        familyRole: String? = nil
    ) {
        self.policyId = policyId
        self.policyType = policyType
        self.autoApplyMode = autoApplyMode
        self.sourcePattern = sourcePattern
        self.targetText = targetText
        self.sourceSlices = sourceSlices
        self.sourceBoundaryMode = sourceBoundaryMode
        self.familyId = familyId
        self.familyRole = familyRole
    }
}

struct VocoAutoApplyGuardBlock: Codable, Equatable {
    let guardId: String
    let reason: String
    let term: String
    let blockedText: String
    let allowedPhrases: [String]
}

struct VocoAutoApplyEvaluation: Equatable {
    let inputText: String
    let outputText: String
    let applied: [VocoAutoApplyPolicyFire]
    let suggestions: [VocoAutoApplyPolicyFire]
    let guardBlocks: [VocoAutoApplyGuardBlock]
    let modelVersion: String?
    let modelGeneratedAt: String?

    var changed: Bool { inputText != outputText }
    var requiresReview: Bool { !guardBlocks.isEmpty }
    var policyHitIds: [String] {
        (applied + suggestions).map(\.policyId)
    }

    init(
        inputText: String,
        outputText: String,
        applied: [VocoAutoApplyPolicyFire],
        suggestions: [VocoAutoApplyPolicyFire],
        guardBlocks: [VocoAutoApplyGuardBlock] = [],
        modelVersion: String? = nil,
        modelGeneratedAt: String? = nil
    ) {
        self.inputText = inputText
        self.outputText = outputText
        self.applied = applied
        self.suggestions = suggestions
        self.guardBlocks = guardBlocks
        self.modelVersion = modelVersion
        self.modelGeneratedAt = modelGeneratedAt
    }
}

struct VocoAutoApplyModelStatus: Equatable {
    let isAvailable: Bool
    let message: String
    let modelURL: URL
    let modelVersion: String?
    let modelGeneratedAt: String?
    let schemaVersion: Int?
    let isDegraded: Bool
    let localModelSha256: String?
    let remoteLatestSha256: String?
    let remoteLatestVersion: String?
    let remoteCheckedAt: String?
    let remoteIsInSync: Bool?
    let remoteMessage: String?

    init(
        isAvailable: Bool,
        message: String,
        modelURL: URL,
        modelVersion: String? = nil,
        modelGeneratedAt: String? = nil,
        schemaVersion: Int? = nil,
        isDegraded: Bool = false,
        localModelSha256: String? = nil,
        remoteLatestSha256: String? = nil,
        remoteLatestVersion: String? = nil,
        remoteCheckedAt: String? = nil,
        remoteIsInSync: Bool? = nil,
        remoteMessage: String? = nil
    ) {
        self.isAvailable = isAvailable
        self.message = message
        self.modelURL = modelURL
        self.modelVersion = modelVersion
        self.modelGeneratedAt = modelGeneratedAt
        self.schemaVersion = schemaVersion
        self.isDegraded = isDegraded
        self.localModelSha256 = localModelSha256
        self.remoteLatestSha256 = remoteLatestSha256
        self.remoteLatestVersion = remoteLatestVersion
        self.remoteCheckedAt = remoteCheckedAt
        self.remoteIsInSync = remoteIsInSync
        self.remoteMessage = remoteMessage
    }
}

struct VocoAutoApplyWorkerSyncManifest: Decodable, Equatable {
    struct Privacy: Decodable, Equatable {
        let transcriptUploadAllowed: Bool?
        let evidenceUploadAllowed: Bool?
        let workerDecisionAllowed: Bool?
    }

    struct Readiness: Decodable, Equatable {
        let mergedAutoApplyModelReady: Bool?
        let autoApplyModelReady: Bool?

        var isReady: Bool {
            mergedAutoApplyModelReady == true || autoApplyModelReady == true
        }
    }

    let phase: String
    let version: String?
    let modelSha256: String
    let schemaVersion: Int?
    let runtimeSchemaVersion: Int?
    let autoApplyModelVersion: String?
    let generatedAt: String?
    let policyCounts: [String: Int]
    let policyTypeCounts: [String: Int]
    let source: String?
    let readiness: Readiness?
    let privacy: Privacy
}

enum VocoAutoApplyWorkerSyncError: LocalizedError, Equatable {
    case missingSyncKey
    case invalidHTTPResponse
    case httpStatus(Int)
    case invalidManifest(String)
    case invalidModel(String)
    case sha256Mismatch(expected: String, actual: String)
    case transport(String)

    var errorDescription: String? {
        switch self {
        case .missingSyncKey:
            return "Worker sync key not configured"
        case .invalidHTTPResponse:
            return "Worker did not return an HTTP response"
        case .httpStatus(let status):
            return "Worker returned HTTP \(status)"
        case .invalidManifest(let reason):
            return "Invalid Worker manifest: \(reason)"
        case .invalidModel(let reason):
            return "Invalid downloaded model: \(reason)"
        case .sha256Mismatch(let expected, let actual):
            return "Downloaded model sha mismatch: expected \(expected), got \(actual)"
        case .transport(let message):
            return message
        }
    }

    var preservesLocalModel: Bool {
        true
    }
}

enum VocoAutoApplyWorkerSyncFetchResult: Equatable {
    case upToDate(manifest: VocoAutoApplyWorkerSyncManifest)
    case downloaded(manifest: VocoAutoApplyWorkerSyncManifest, modelData: Data)
}

struct VocoAutoApplyWorkerSyncOutcome: Equatable {
    enum State: Equatable {
        case installed
        case upToDate
        case keptLocal
    }

    let state: State
    let manifest: VocoAutoApplyWorkerSyncManifest?
    let message: String
    let installedModelSha256: String?
    let errorDescription: String?

    static func installed(
        manifest: VocoAutoApplyWorkerSyncManifest,
        message: String
    ) -> VocoAutoApplyWorkerSyncOutcome {
        VocoAutoApplyWorkerSyncOutcome(
            state: .installed,
            manifest: manifest,
            message: message,
            installedModelSha256: manifest.modelSha256,
            errorDescription: nil
        )
    }

    static func upToDate(manifest: VocoAutoApplyWorkerSyncManifest) -> VocoAutoApplyWorkerSyncOutcome {
        VocoAutoApplyWorkerSyncOutcome(
            state: .upToDate,
            manifest: manifest,
            message: String(localized: "Remote model is up to date"),
            installedModelSha256: manifest.modelSha256,
            errorDescription: nil
        )
    }

    static func keptLocal(_ message: String, errorDescription: String?) -> VocoAutoApplyWorkerSyncOutcome {
        VocoAutoApplyWorkerSyncOutcome(
            state: .keptLocal,
            manifest: nil,
            message: message,
            installedModelSha256: nil,
            errorDescription: errorDescription
        )
    }
}

struct VocoAutoApplyWorkerSyncClient {
    typealias Transport = (URLRequest) async throws -> (Data, HTTPURLResponse)

    static let defaultWorkerURL = URL(string: "https://voco-auto-apply-sync.black-hill-f944.workers.dev")!
    static let phase = "phase1-distribution-only"

    let workerURL: URL
    let timeout: TimeInterval
    let transport: Transport

    init(
        workerURL: URL = VocoAutoApplyWorkerSyncClient.defaultWorkerURL,
        timeout: TimeInterval = 20,
        transport: @escaping Transport = VocoAutoApplyWorkerSyncClient.urlSessionTransport
    ) {
        self.workerURL = workerURL
        self.timeout = timeout
        self.transport = transport
    }

    func fetchLatest(
        syncKey: String,
        localModelSha256: String?
    ) async throws -> VocoAutoApplyWorkerSyncFetchResult {
        let manifest = try await fetchManifest(syncKey: syncKey)
        if localModelSha256 == manifest.modelSha256 {
            return .upToDate(manifest: manifest)
        }

        let modelData = try await requestBytes(
            path: "/v1/auto-apply/models/\(manifest.modelSha256)",
            syncKey: syncKey
        )
        let downloadedSha = VocoAutoApplyModelService.sha256Hex(for: modelData)
        guard downloadedSha == manifest.modelSha256 else {
            throw VocoAutoApplyWorkerSyncError.sha256Mismatch(
                expected: manifest.modelSha256,
                actual: downloadedSha
            )
        }
        try VocoAutoApplyModelService.validateDownloadedModelData(modelData, manifest: manifest)
        return .downloaded(manifest: manifest, modelData: modelData)
    }

    func fetchManifest(syncKey: String) async throws -> VocoAutoApplyWorkerSyncManifest {
        let data = try await requestBytes(path: "/v1/auto-apply/manifest", syncKey: syncKey)
        let manifest: VocoAutoApplyWorkerSyncManifest
        do {
            manifest = try JSONDecoder().decode(VocoAutoApplyWorkerSyncManifest.self, from: data)
        } catch {
            throw VocoAutoApplyWorkerSyncError.invalidManifest(error.localizedDescription)
        }
        try validate(manifest)
        return manifest
    }

    private func requestBytes(path: String, syncKey: String) async throws -> Data {
        let trimmedKey = syncKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedKey.isEmpty else {
            throw VocoAutoApplyWorkerSyncError.missingSyncKey
        }

        guard let url = URL(string: "\(workerURL.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/")))/\(path.trimmingCharacters(in: CharacterSet(charactersIn: "/")))") else {
            throw VocoAutoApplyWorkerSyncError.transport("Invalid Worker URL")
        }
        var request = URLRequest(url: url, timeoutInterval: timeout)
        request.httpMethod = "GET"
        request.setValue("Bearer \(trimmedKey)", forHTTPHeaderField: "Authorization")
        request.setValue("Voco-mac-auto-apply-sync/1.0", forHTTPHeaderField: "User-Agent")

        let data: Data
        let response: HTTPURLResponse
        do {
            (data, response) = try await transport(request)
        } catch let error as VocoAutoApplyWorkerSyncError {
            throw error
        } catch {
            throw VocoAutoApplyWorkerSyncError.transport(error.localizedDescription)
        }

        guard (200...299).contains(response.statusCode) else {
            throw VocoAutoApplyWorkerSyncError.httpStatus(response.statusCode)
        }
        return data
    }

    private func validate(_ manifest: VocoAutoApplyWorkerSyncManifest) throws {
        guard manifest.phase == Self.phase else {
            throw VocoAutoApplyWorkerSyncError.invalidManifest("unexpected phase \(manifest.phase)")
        }
        guard Self.isSHA256(manifest.modelSha256) else {
            throw VocoAutoApplyWorkerSyncError.invalidManifest("modelSha256 is missing or invalid")
        }
        if let schemaVersion = manifest.schemaVersion,
           !VocoAutoApplyModelService.supportedSchemaVersions.contains(schemaVersion) {
            throw VocoAutoApplyWorkerSyncError.invalidManifest("unsupported schemaVersion \(schemaVersion)")
        }
        if let runtimeSchemaVersion = manifest.runtimeSchemaVersion,
           !VocoAutoApplyModelService.supportedRuntimeSchemaVersions.contains(runtimeSchemaVersion) {
            throw VocoAutoApplyWorkerSyncError.invalidManifest("unsupported runtimeSchemaVersion \(runtimeSchemaVersion)")
        }
        guard manifest.readiness?.isReady == true else {
            throw VocoAutoApplyWorkerSyncError.invalidManifest("readiness is not true")
        }
        guard manifest.privacy.transcriptUploadAllowed == false else {
            throw VocoAutoApplyWorkerSyncError.invalidManifest("privacy.transcriptUploadAllowed must be false")
        }
        if manifest.privacy.workerDecisionAllowed != nil,
           manifest.privacy.workerDecisionAllowed != false {
            throw VocoAutoApplyWorkerSyncError.invalidManifest("privacy.workerDecisionAllowed must be false")
        }
        if manifest.privacy.evidenceUploadAllowed != nil,
           manifest.privacy.evidenceUploadAllowed != false {
            throw VocoAutoApplyWorkerSyncError.invalidManifest("privacy.evidenceUploadAllowed must be false")
        }
    }

    private static func isSHA256(_ value: String) -> Bool {
        value.count == 64 && value.unicodeScalars.allSatisfy { scalar in
            (48...57).contains(scalar.value) || (97...102).contains(scalar.value)
        }
    }

    private static func urlSessionTransport(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw VocoAutoApplyWorkerSyncError.invalidHTTPResponse
        }
        return (data, httpResponse)
    }
}

final class VocoAutoApplyModelService: ObservableObject {
    static let shared = VocoAutoApplyModelService()
    static let enabledKey = "VocoAutoApplyModelEnabled"
    static let modelFileName = "full-db.auto-apply-model.json"
    static let protectedTermGuardReason = "auto-apply-model-protected-term-guard"
    static let supportedSchemaVersion = 2
    static let supportedSchemaVersions: Set<Int> = [1, 2]
    static let supportedRuntimeSchemaVersion = 3
    static let supportedRuntimeSchemaVersions: Set<Int> = [2, 3]
    static let automaticWorkerSyncInterval: TimeInterval = 60
    static let automaticWorkerSyncInitialDelay: TimeInterval = 5

    static var defaultModelDirectory: URL {
        AppIdentifiers.appSupportDirectory
            .appendingPathComponent("AutoApplyModels", isDirectory: true)
    }

    static var defaultModelURL: URL {
        defaultModelDirectory.appendingPathComponent(modelFileName)
    }

    @Published private(set) var status: VocoAutoApplyModelStatus

    private let modelURL: URL
    private let defaults: UserDefaults
    private let workerSyncClient: VocoAutoApplyWorkerSyncClient
    private let workerSyncKeyProvider: () -> String?
    private let modelBackupRetention: Int
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "AutoApplyModel")
    private let watchQueue = DispatchQueue(label: "com.jasonchien.Voco.autoApplyModelWatcher")
    private var loadedModel: VocoAutoApplyRuntimeModel?
    private var modelFileWatcher: DispatchSourceFileSystemObject?
    private var modelDirectoryWatcher: DispatchSourceFileSystemObject?
    private var pendingModelReload: DispatchWorkItem?
    private let workerSyncCoordinatorLock = NSLock()
    private var activeWorkerSync: (id: UUID, task: Task<VocoAutoApplyWorkerSyncOutcome, Never>)?
    private var automaticWorkerSyncTask: Task<Void, Never>?
    private var remoteLatestSha256: String?
    private var remoteLatestVersion: String?
    private var remoteCheckedAt: String?
    private var remoteMessage: String?

    static let hardCodedActionCommandSurfaces: [String] = ["全部刪除", "全部删除"]
    static let defaultSourceBoundaryMode = "default"
    static let cjkUnsafeContinuationBoundaryMode = "cjk-unsafe-continuation"
    static let currencyNumberNormalizationPolicyId = "runtime.currency-number-normalization"
    static let currencyNumberNormalizationPolicyType = "currencyNumberNormalization"
    private static let unsafeCJKContinuationAfterPairSource: Set<Character> = [
        "分", "性", "化", "度", "感", "型", "式", "區", "市", "縣", "里", "路",
        "街", "段", "號", "款", "項", "章", "篇", "版", "光", "睛"
    ]
    private static let currencyNumberNormalizationSourceSlices = ["runtimeSpecialPolicy"]
    private static let chineseCurrencyAmountCharacters = "零〇一二兩两三四五六七八九壹貳參叁肆伍陸柒捌玖十拾百佰千仟萬万億亿點点"
    private static let currencyApproximationCharacters: Set<Character> = [
        "幾", "几", "多", "來", "余", "餘", "約", "近", "半"
    ]
    private static let chineseCurrencyDigitValues: [Character: Int] = [
        "零": 0, "〇": 0,
        "一": 1, "壹": 1,
        "二": 2, "貳": 2, "兩": 2, "两": 2,
        "三": 3, "參": 3, "叁": 3,
        "四": 4, "肆": 4,
        "五": 5, "伍": 5,
        "六": 6, "陸": 6,
        "七": 7, "柒": 7,
        "八": 8, "捌": 8,
        "九": 9, "玖": 9
    ]
    private static let chineseCurrencySectionUnitValues: [Character: Int] = [
        "十": 10, "拾": 10,
        "百": 100, "佰": 100,
        "千": 1_000, "仟": 1_000
    ]
    private static let chineseCurrencyHighUnitValues: [Character: Int] = [
        "萬": 10_000, "万": 10_000,
        "億": 100_000_000, "亿": 100_000_000
    ]
    private static let currencyPrefixTerms = [
        "新台幣", "新臺幣", "人民幣", "台幣", "臺幣", "美金", "美元", "港幣",
        "日幣", "日圓", "日元", "韓幣", "歐元", "英鎊", "TWD", "NTD", "USD",
        "HKD", "JPY", "RMB", "CNY", "EUR", "GBP", "NT$", "US$"
    ]
    private static let currencySuffixTerms = [
        "塊錢", "新台幣", "新臺幣", "人民幣", "台幣", "臺幣", "美金", "美元",
        "港幣", "日幣", "日圓", "日元", "韓幣", "歐元", "英鎊", "塊", "元", "圓"
    ]
    private static let currencyBoundaryLookahead = "(?=$|[\\s　,，。.!！？?、；;：:）)】\\]\"'」』]|的|了|嗎|呢|吧|啊|喔|呀|耶|整|錢|以上|以下|以內|左右|上下)"
    private static let currencyAmountWithSuffixRegex: NSRegularExpression = {
        let prefix = regexAlternation(currencyPrefixTerms)
        let suffix = regexAlternation(currencySuffixTerms)
        let pattern = "(?:\(prefix)\\s*)?([\(chineseCurrencyAmountCharacters)]+)(?:\(suffix))\(currencyBoundaryLookahead)"
        return try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
    }()
    private static let currencyPrefixAmountRegex: NSRegularExpression = {
        let prefix = regexAlternation(currencyPrefixTerms)
        let pattern = "(?:\(prefix)\\s*)([\(chineseCurrencyAmountCharacters)]+)\(currencyBoundaryLookahead)"
        return try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
    }()

    /// Contract: docs/auto-apply-evaluation-contract.md §2
    static let asciiTokenPattern = "[A-Za-z][A-Za-z0-9_+.#/-]*"
    private static let asciiTokenRegex: NSRegularExpression = {
        try! NSRegularExpression(pattern: asciiTokenPattern, options: [])
    }()

    /// Contract: docs/auto-apply-evaluation-contract.md §5
    /// Matches PUNCT_OR_SPACE_RE from tools/voco_full_db_raw_cleaned_corpus.py:51
    private static let looseKeyPunctuationOrWhitespace = CharacterSet(
        charactersIn: "，,。.!！？?、：:；;「」『』\"'`（）()【】…—-"
    ).union(.whitespacesAndNewlines).union(CharacterSet(charactersIn: "[]"))

    var isUserEnabled: Bool {
        get { defaults.object(forKey: Self.enabledKey) as? Bool ?? true }
        set {
            defaults.set(newValue, forKey: Self.enabledKey)
            objectWillChange.send()
        }
    }

    var isRuntimeEnabled: Bool {
        isUserEnabled && status.isAvailable
    }

    var settingsToggleIsOn: Bool {
        status.isAvailable && isUserEnabled
    }

    var settingsToggleIsEnabled: Bool {
        status.isAvailable
    }

    func protectedTermGuardTerms() -> [String] {
        guard status.isAvailable, let loadedModel else { return [] }
        return Array(Set(
            loadedModel.protectedTermAllowlistGuards
                .map(\.term)
                .filter { !$0.isEmpty }
        )).sorted()
    }

    init(
        modelURL: URL = VocoAutoApplyModelService.defaultModelURL,
        defaults: UserDefaults = .standard,
        workerSyncClient: VocoAutoApplyWorkerSyncClient = VocoAutoApplyWorkerSyncClient(),
        workerSyncKeyProvider: @escaping () -> String? = VocoAutoApplyModelService.defaultWorkerSyncKey,
        modelBackupRetention: Int = 3
    ) {
        self.modelURL = modelURL
        self.defaults = defaults
        self.workerSyncClient = workerSyncClient
        self.workerSyncKeyProvider = workerSyncKeyProvider
        self.modelBackupRetention = modelBackupRetention
        self.status = VocoAutoApplyModelStatus(
            isAvailable: false,
            message: String(localized: "Model not detected"),
            modelURL: modelURL,
            localModelSha256: Self.sha256HexForFileIfExists(modelURL)
        )
        reload()
        startWatchingModelChanges()
    }

    deinit {
        pendingModelReload?.cancel()
        modelFileWatcher?.cancel()
        modelDirectoryWatcher?.cancel()
        automaticWorkerSyncTask?.cancel()
    }

    func reload() {
        let localModelSha256 = Self.sha256HexForFileIfExists(modelURL)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            loadedModel = nil
            status = VocoAutoApplyModelStatus(
                isAvailable: false,
                message: String(localized: "Model not installed"),
                modelURL: modelURL,
                localModelSha256: localModelSha256,
                remoteLatestSha256: remoteLatestSha256,
                remoteLatestVersion: remoteLatestVersion,
                remoteCheckedAt: remoteCheckedAt,
                remoteIsInSync: remoteInSync(localModelSha256: localModelSha256),
                remoteMessage: remoteMessage
            )
            return
        }

        let data: Data
        let decodedModel: VocoDecodedAutoApplyModel
        do {
            data = try Data(contentsOf: modelURL)
            decodedModel = try Self.decodeModel(from: data)
        } catch VocoAutoApplyModelDecodeError.unsupportedSchemaVersion(let schemaVersion) {
            logger.error("Auto-apply model schema version \(schemaVersion) is not supported (expected \(Self.supportedSchemaVersion))")
            if let existing = loadedModel {
                loadedModel = existing
                status = VocoAutoApplyModelStatus(
                    isAvailable: true,
                    message: String(localized: "Model schema unsupported, using previous version"),
                    modelURL: modelURL,
                    modelVersion: existing.modelVersion,
                    modelGeneratedAt: existing.modelGeneratedAt,
                    schemaVersion: existing.schemaVersion,
                    isDegraded: true,
                    localModelSha256: localModelSha256,
                    remoteLatestSha256: remoteLatestSha256,
                    remoteLatestVersion: remoteLatestVersion,
                    remoteCheckedAt: remoteCheckedAt,
                    remoteIsInSync: remoteInSync(localModelSha256: localModelSha256),
                    remoteMessage: remoteMessage
                )
            } else {
                loadedModel = nil
                status = VocoAutoApplyModelStatus(
                    isAvailable: false,
                    message: String(localized: "Model schema unsupported"),
                    modelURL: modelURL,
                    localModelSha256: localModelSha256,
                    remoteLatestSha256: remoteLatestSha256,
                    remoteLatestVersion: remoteLatestVersion,
                    remoteCheckedAt: remoteCheckedAt,
                    remoteIsInSync: remoteInSync(localModelSha256: localModelSha256),
                    remoteMessage: remoteMessage
                )
            }
            return
        } catch VocoAutoApplyModelDecodeError.unsupportedRuntimeSchemaVersion(let runtimeSchemaVersion) {
            logger.error("Auto-apply runtime schema version \(runtimeSchemaVersion) is not supported (expected \(Self.supportedRuntimeSchemaVersion))")
            if let existing = loadedModel {
                loadedModel = existing
                status = VocoAutoApplyModelStatus(
                    isAvailable: true,
                    message: String(localized: "Model schema unsupported, using previous version"),
                    modelURL: modelURL,
                    modelVersion: existing.modelVersion,
                    modelGeneratedAt: existing.modelGeneratedAt,
                    schemaVersion: existing.schemaVersion,
                    isDegraded: true,
                    localModelSha256: localModelSha256,
                    remoteLatestSha256: remoteLatestSha256,
                    remoteLatestVersion: remoteLatestVersion,
                    remoteCheckedAt: remoteCheckedAt,
                    remoteIsInSync: remoteInSync(localModelSha256: localModelSha256),
                    remoteMessage: remoteMessage
                )
            } else {
                loadedModel = nil
                status = VocoAutoApplyModelStatus(
                    isAvailable: false,
                    message: String(localized: "Model schema unsupported"),
                    modelURL: modelURL,
                    localModelSha256: localModelSha256,
                    remoteLatestSha256: remoteLatestSha256,
                    remoteLatestVersion: remoteLatestVersion,
                    remoteCheckedAt: remoteCheckedAt,
                    remoteIsInSync: remoteInSync(localModelSha256: localModelSha256),
                    remoteMessage: remoteMessage
                )
            }
            return
        } catch {
            logger.error("Failed to load auto-apply model: \(error.localizedDescription, privacy: .public)")
            if let existing = loadedModel {
                loadedModel = existing
                status = VocoAutoApplyModelStatus(
                    isAvailable: true,
                    message: String(localized: "Model reload failed, using previous version"),
                    modelURL: modelURL,
                    modelVersion: existing.modelVersion,
                    modelGeneratedAt: existing.modelGeneratedAt,
                    schemaVersion: existing.schemaVersion,
                    isDegraded: true,
                    localModelSha256: localModelSha256,
                    remoteLatestSha256: remoteLatestSha256,
                    remoteLatestVersion: remoteLatestVersion,
                    remoteCheckedAt: remoteCheckedAt,
                    remoteIsInSync: remoteInSync(localModelSha256: localModelSha256),
                    remoteMessage: remoteMessage
                )
            } else {
                loadedModel = nil
                status = VocoAutoApplyModelStatus(
                    isAvailable: false,
                    message: String(localized: "Model unreadable"),
                    modelURL: modelURL,
                    localModelSha256: localModelSha256,
                    remoteLatestSha256: remoteLatestSha256,
                    remoteLatestVersion: remoteLatestVersion,
                    remoteCheckedAt: remoteCheckedAt,
                    remoteIsInSync: remoteInSync(localModelSha256: localModelSha256),
                    remoteMessage: remoteMessage
                )
            }
            return
        }

        guard decodedModel.isReady else {
            loadedModel = nil
            status = VocoAutoApplyModelStatus(
                isAvailable: false,
                message: String(localized: "Model not ready"),
                modelURL: modelURL,
                localModelSha256: localModelSha256,
                remoteLatestSha256: remoteLatestSha256,
                remoteLatestVersion: remoteLatestVersion,
                remoteCheckedAt: remoteCheckedAt,
                remoteIsInSync: remoteInSync(localModelSha256: localModelSha256),
                remoteMessage: remoteMessage
            )
            return
        }

        loadedModel = decodedModel.runtimeModel
        let applyCount = decodedModel.policyCounts["apply"] ?? decodedModel.applyPolicyCount
        let suggestCount = decodedModel.policyCounts["suggest"] ?? decodedModel.suggestPolicyCount
        let replacedCount = decodedModel.policyCounts["replaced"] ?? 0
        let blockedCount = decodedModel.policyCounts["blocked"] ?? 0
        status = VocoAutoApplyModelStatus(
            isAvailable: true,
            message: String(localized: "Model loaded: \(applyCount) apply, \(suggestCount) suggest, \(replacedCount) replaced, \(blockedCount) blocked"),
            modelURL: modelURL,
            modelVersion: decodedModel.runtimeModel.modelVersion,
            modelGeneratedAt: decodedModel.runtimeModel.modelGeneratedAt,
            schemaVersion: decodedModel.runtimeModel.schemaVersion ?? Self.supportedSchemaVersion,
            localModelSha256: localModelSha256,
            remoteLatestSha256: remoteLatestSha256,
            remoteLatestVersion: remoteLatestVersion,
            remoteCheckedAt: remoteCheckedAt,
            remoteIsInSync: remoteInSync(localModelSha256: localModelSha256),
            remoteMessage: remoteMessage
        )
    }

    @discardableResult
    func syncFromWorker() async -> VocoAutoApplyWorkerSyncOutcome {
        let handle = workerSyncHandle {
            Task { [weak self] in
                guard let self else {
                    return .keptLocal(String(localized: "Remote sync unavailable, keeping local model"), errorDescription: nil)
                }
                return await self.performSyncFromWorker()
            }
        }
        let outcome = await handle.task.value
        if handle.ownsTask {
            clearWorkerSyncHandle(id: handle.id)
        }
        return outcome
    }

    func startAutomaticWorkerSync(
        interval: TimeInterval = VocoAutoApplyModelService.automaticWorkerSyncInterval,
        initialDelay: TimeInterval = VocoAutoApplyModelService.automaticWorkerSyncInitialDelay
    ) {
        guard automaticWorkerSyncTask == nil else { return }

        let syncInterval = max(interval, 0.05)
        let startupDelay = max(initialDelay, 0)
        automaticWorkerSyncTask = Task(priority: .utility) { [weak self] in
            if startupDelay > 0 {
                do {
                    try await Task.sleep(nanoseconds: Self.nanoseconds(for: startupDelay))
                } catch {
                    return
                }
            }

            while !Task.isCancelled {
                guard let self else { return }
                if self.hasConfiguredWorkerSyncKey {
                    _ = await self.syncFromWorker()
                }
                do {
                    try await Task.sleep(nanoseconds: Self.nanoseconds(for: syncInterval))
                } catch {
                    return
                }
            }
        }
    }

    func stopAutomaticWorkerSync() {
        automaticWorkerSyncTask?.cancel()
        automaticWorkerSyncTask = nil
    }

    private func performSyncFromWorker() async -> VocoAutoApplyWorkerSyncOutcome {
        let syncKey = currentWorkerSyncKey()
        guard !syncKey.isEmpty else {
            let message = String(localized: "Remote sync key missing, keeping local model")
            await MainActor.run {
                updateRemoteStatus(manifest: nil, message: message)
            }
            return .keptLocal(message, errorDescription: VocoAutoApplyWorkerSyncError.missingSyncKey.localizedDescription)
        }

        do {
            let localSha = Self.sha256HexForFileIfExists(modelURL)
            let result = try await workerSyncClient.fetchLatest(syncKey: syncKey, localModelSha256: localSha)
            switch result {
            case .upToDate(let manifest):
                await MainActor.run {
                    updateRemoteStatus(manifest: manifest, message: String(localized: "Remote model is up to date"))
                }
                return .upToDate(manifest: manifest)
            case .downloaded(let manifest, let modelData):
                try installDownloadedWorkerModel(modelData, expectedSha256: manifest.modelSha256)
                await MainActor.run {
                    updateRemoteStatus(manifest: manifest, message: String(localized: "Remote model installed"))
                }
                return .installed(manifest: manifest, message: String(localized: "Remote model installed"))
            }
        } catch let error as VocoAutoApplyWorkerSyncError {
            logger.error("Auto-apply Worker sync kept local model: \(error.localizedDescription, privacy: .public)")
            let message = String(localized: "Remote sync unavailable, keeping local model")
            await MainActor.run {
                updateRemoteStatus(manifest: nil, message: message)
            }
            return .keptLocal(message, errorDescription: error.localizedDescription)
        } catch {
            logger.error("Auto-apply Worker sync kept local model: \(error.localizedDescription, privacy: .public)")
            let message = String(localized: "Remote sync unavailable, keeping local model")
            await MainActor.run {
                updateRemoteStatus(manifest: nil, message: message)
            }
            return .keptLocal(message, errorDescription: error.localizedDescription)
        }
    }

    private var hasConfiguredWorkerSyncKey: Bool {
        !currentWorkerSyncKey().isEmpty
    }

    private func currentWorkerSyncKey() -> String {
        workerSyncKeyProvider()?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    private func workerSyncHandle(
        starting taskFactory: () -> Task<VocoAutoApplyWorkerSyncOutcome, Never>
    ) -> (id: UUID, task: Task<VocoAutoApplyWorkerSyncOutcome, Never>, ownsTask: Bool) {
        workerSyncCoordinatorLock.lock()
        defer { workerSyncCoordinatorLock.unlock() }

        if let activeWorkerSync {
            return (activeWorkerSync.id, activeWorkerSync.task, false)
        }

        let id = UUID()
        let task = taskFactory()
        activeWorkerSync = (id, task)
        return (id, task, true)
    }

    private func clearWorkerSyncHandle(id: UUID) {
        workerSyncCoordinatorLock.lock()
        defer { workerSyncCoordinatorLock.unlock() }

        if activeWorkerSync?.id == id {
            activeWorkerSync = nil
        }
    }

    private func installDownloadedWorkerModel(_ data: Data, expectedSha256: String) throws {
        try Self.validateDownloadedModelData(data, expectedSha256: expectedSha256)

        let directoryURL = modelURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        _ = try createModelBackupIfNeeded()

        let tempURL = directoryURL.appendingPathComponent(".\(Self.modelFileName).worker-sync-\(UUID().uuidString).tmp")
        do {
            try data.write(to: tempURL, options: .atomic)
            if FileManager.default.fileExists(atPath: modelURL.path) {
                _ = try FileManager.default.replaceItemAt(
                    modelURL,
                    withItemAt: tempURL,
                    backupItemName: nil,
                    options: []
                )
            } else {
                try FileManager.default.moveItem(at: tempURL, to: modelURL)
            }
        } catch {
            try? FileManager.default.removeItem(at: tempURL)
            throw error
        }
    }

    private func createModelBackupIfNeeded() throws -> URL? {
        guard modelBackupRetention > 0,
              FileManager.default.fileExists(atPath: modelURL.path)
        else { return nil }

        let backupDirectory = modelURL.deletingLastPathComponent().appendingPathComponent("Backups", isDirectory: true)
        try FileManager.default.createDirectory(at: backupDirectory, withIntermediateDirectories: true)
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd-HHmmss-SSS"
        let backupURL = backupDirectory.appendingPathComponent(
            "\(Self.modelFileName).bak-\(formatter.string(from: Date()))-worker-sync"
        )
        try FileManager.default.copyItem(at: modelURL, to: backupURL)
        pruneModelBackups(in: backupDirectory)
        return backupURL
    }

    private func pruneModelBackups(in backupDirectory: URL) {
        guard let urls = try? FileManager.default.contentsOfDirectory(
            at: backupDirectory,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else { return }

        let backups = urls
            .filter { $0.lastPathComponent.hasPrefix("\(Self.modelFileName).bak-") }
            .sorted { lhs, rhs in
                let lhsDate = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let rhsDate = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return lhsDate > rhsDate
            }
        for stale in backups.dropFirst(modelBackupRetention) {
            try? FileManager.default.removeItem(at: stale)
        }
    }

    private func updateRemoteStatus(
        manifest: VocoAutoApplyWorkerSyncManifest?,
        message: String
    ) {
        remoteLatestSha256 = manifest?.modelSha256
        remoteLatestVersion = manifest?.version ?? manifest?.autoApplyModelVersion
        remoteCheckedAt = Self.isoString(Date())
        remoteMessage = message
        reload()
    }

    private func remoteInSync(localModelSha256: String?) -> Bool? {
        guard let remoteLatestSha256 else { return nil }
        return localModelSha256 == remoteLatestSha256
    }

    static func validateDownloadedModelData(
        _ data: Data,
        manifest: VocoAutoApplyWorkerSyncManifest
    ) throws {
        try validateDownloadedModelData(data, expectedSha256: manifest.modelSha256)
        if let schemaVersion = manifest.schemaVersion,
           !supportedSchemaVersions.contains(schemaVersion) {
            throw VocoAutoApplyWorkerSyncError.invalidManifest("unsupported schemaVersion \(schemaVersion)")
        }
        if let runtimeSchemaVersion = manifest.runtimeSchemaVersion,
           !supportedRuntimeSchemaVersions.contains(runtimeSchemaVersion) {
            throw VocoAutoApplyWorkerSyncError.invalidManifest("unsupported runtimeSchemaVersion \(runtimeSchemaVersion)")
        }
    }

    static func validateDownloadedModelData(
        _ data: Data,
        expectedSha256: String
    ) throws {
        let actualSha = sha256Hex(for: data)
        guard actualSha == expectedSha256 else {
            throw VocoAutoApplyWorkerSyncError.sha256Mismatch(expected: expectedSha256, actual: actualSha)
        }

        let envelope: VocoAutoApplyModelEnvelope
        do {
            envelope = try JSONDecoder().decode(VocoAutoApplyModelEnvelope.self, from: data)
        } catch {
            throw VocoAutoApplyWorkerSyncError.invalidModel(error.localizedDescription)
        }

        if let schemaVersion = envelope.schemaVersion,
           !supportedSchemaVersions.contains(schemaVersion) {
            throw VocoAutoApplyWorkerSyncError.invalidModel("unsupported schemaVersion \(schemaVersion)")
        }
        if let runtimeSchemaVersion = envelope.runtimeSchemaVersion,
           !supportedRuntimeSchemaVersions.contains(runtimeSchemaVersion) {
            throw VocoAutoApplyWorkerSyncError.invalidModel("unsupported runtimeSchemaVersion \(runtimeSchemaVersion)")
        }
        guard envelope.mergedReplayReadiness?.mergedAutoApplyModelReady == true else {
            throw VocoAutoApplyWorkerSyncError.invalidModel("mergedReplayReadiness.mergedAutoApplyModelReady is not true")
        }

        do {
            let decodedModel = try decodeModel(from: data)
            guard decodedModel.isReady else {
                throw VocoAutoApplyWorkerSyncError.invalidModel("decoded model is not ready")
            }
        } catch let error as VocoAutoApplyWorkerSyncError {
            throw error
        } catch {
            throw VocoAutoApplyWorkerSyncError.invalidModel(error.localizedDescription)
        }
    }

    static func sha256Hex(for data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func sha256HexForFileIfExists(_ url: URL) -> String? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return sha256Hex(for: data)
    }

    static func defaultWorkerSyncKey() -> String? {
        let environmentKey = ProcessInfo.processInfo.environment["VOCO_SYNC_KEY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let environmentKey, !environmentKey.isEmpty {
            return environmentKey
        }

        let secretURL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("GitHub/VocoReplayLab/workers/auto-apply-sync/.secrets/voco_sync_key")
        return try? String(contentsOf: secretURL, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func isoString(_ date: Date) -> String {
        ISO8601DateFormatter().string(from: date)
    }

    private static func nanoseconds(for seconds: TimeInterval) -> UInt64 {
        UInt64(max(seconds, 0) * 1_000_000_000)
    }

    private static func decodeModel(from data: Data) throws -> VocoDecodedAutoApplyModel {
        let decoder = JSONDecoder()
        let indexedDecodeError: Error
        do {
            let indexedModel = try decoder.decode(VocoIndexedRuntimeAutoApplyModel.self, from: data)
            return VocoDecodedAutoApplyModel(indexedModel: indexedModel)
        } catch {
            indexedDecodeError = error
        }

        do {
            let model = try decoder.decode(VocoAutoApplyModel.self, from: data)
            if let schemaVersion = model.schemaVersion,
               !Self.supportedSchemaVersions.contains(schemaVersion) {
                throw VocoAutoApplyModelDecodeError.unsupportedSchemaVersion(schemaVersion)
            }
            return VocoDecodedAutoApplyModel(model: model)
        } catch let decodeError as VocoAutoApplyModelDecodeError {
            switch decodeError {
            case .unsupportedSchemaVersion:
                throw decodeError
            case .missingIndexedRuntimeMarker, .unsupportedRuntimeSchemaVersion:
                if let indexedError = indexedDecodeError as? VocoAutoApplyModelDecodeError {
                    switch indexedError {
                    case .unsupportedRuntimeSchemaVersion:
                        throw indexedError
                    case .missingIndexedRuntimeMarker, .unsupportedSchemaVersion:
                        break
                    }
                }
                throw indexedDecodeError
            }
        } catch {
            if let decodeError = indexedDecodeError as? VocoAutoApplyModelDecodeError {
                switch decodeError {
                case .unsupportedRuntimeSchemaVersion:
                    throw decodeError
                case .missingIndexedRuntimeMarker, .unsupportedSchemaVersion:
                    break
                }
            }
            throw indexedDecodeError
        }
    }

    private func startWatchingModelChanges() {
        watchQueue.async { [weak self] in
            self?.installDirectoryWatcher()
            self?.installFileWatcher()
        }
    }

    private func installDirectoryWatcher() {
        guard modelDirectoryWatcher == nil else { return }

        let directoryURL = modelURL.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        let descriptor = open(directoryURL.path, O_EVTONLY)
        guard descriptor >= 0 else {
            logger.error("Failed to watch auto-apply model directory: \(directoryURL.path, privacy: .public)")
            return
        }

        let source = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: descriptor,
            eventMask: [.write, .delete, .rename, .attrib, .link, .revoke],
            queue: watchQueue
        )
        source.setEventHandler { [weak self] in
            self?.installFileWatcher()
            self?.scheduleModelReload()
        }
        source.setCancelHandler {
            close(descriptor)
        }
        modelDirectoryWatcher = source
        source.resume()
    }

    private func installFileWatcher() {
        modelFileWatcher?.cancel()
        modelFileWatcher = nil

        let descriptor = open(modelURL.path, O_EVTONLY)
        guard descriptor >= 0 else { return }

        let source = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: descriptor,
            eventMask: [.write, .extend, .delete, .rename, .attrib, .link, .revoke],
            queue: watchQueue
        )
        source.setEventHandler { [weak self, weak source] in
            guard let self else { return }
            let events = source?.data ?? []
            scheduleModelReload()
            if events.contains(.delete) || events.contains(.rename) || events.contains(.revoke) {
                installFileWatcher()
            }
        }
        source.setCancelHandler {
            close(descriptor)
        }
        modelFileWatcher = source
        source.resume()
    }

    private func scheduleModelReload() {
        pendingModelReload?.cancel()
        let workItem = DispatchWorkItem { [weak self] in
            self?.reload()
        }
        pendingModelReload = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + .milliseconds(750), execute: workItem)
    }

    func evaluate(_ text: String, context: String = "") -> VocoAutoApplyEvaluation {
        guard isRuntimeEnabled,
              let model = loadedModel,
              !textIsActionCommand(text: text, actionCommandSurfaces: model.actionCommandSurfaces)
        else {
            return VocoAutoApplyEvaluation(inputText: text, outputText: text, applied: [], suggestions: [])
        }

        if let exact = firstExactPolicy(in: model, matching: text),
           exact.isSafeExactApplyPolicy {
            let target = exact.targetText ?? text
            return guardedEvaluation(
                inputText: text,
                proposedOutputText: target,
                applied: [exact.fire],
                suggestions: suggestFires(in: model.suggestPolicies, text: text, context: context),
                protectedTermAllowlistGuards: model.protectedTermAllowlistGuards,
                modelVersion: model.modelVersion,
                modelGeneratedAt: model.modelGeneratedAt
            )
        }

        var output = text
        var applied: [VocoAutoApplyPolicyFire] = []
        for policy in model.scopedApplyPolicies {
            guard policyFires(policy, text: output, context: context),
                  let sourcePattern = policy.sourcePattern,
                  let targetText = policy.targetText
            else { continue }

            let updated = replace(
                sourcePattern,
                with: targetText,
                in: output,
                sourceBoundaryMode: policy.sourceBoundaryMode
            )
            guard updated != output else { continue }
            output = updated
            applied.append(policy.fire)
        }

        let currencyNormalization = normalizeCurrencyNumbers(in: output)
        output = currencyNormalization.outputText
        applied.append(contentsOf: currencyNormalization.applied)

        return guardedEvaluation(
            inputText: text,
            proposedOutputText: output,
            applied: applied,
            suggestions: suggestFires(in: model.suggestPolicies, text: output, context: context),
            protectedTermAllowlistGuards: model.protectedTermAllowlistGuards,
            modelVersion: model.modelVersion,
            modelGeneratedAt: model.modelGeneratedAt
        )
    }

    private func guardedEvaluation(
        inputText: String,
        proposedOutputText: String,
        applied: [VocoAutoApplyPolicyFire],
        suggestions: [VocoAutoApplyPolicyFire],
        protectedTermAllowlistGuards: [VocoProtectedTermAllowlistGuard],
        modelVersion: String?,
        modelGeneratedAt: String?
    ) -> VocoAutoApplyEvaluation {
        let blocks = protectedTermGuardBlocks(
            in: proposedOutputText,
            applied: applied,
            guardRules: protectedTermAllowlistGuards
        )
        guard blocks.isEmpty else {
            return VocoAutoApplyEvaluation(
                inputText: inputText,
                outputText: inputText,
                applied: [],
                suggestions: suggestions,
                guardBlocks: blocks,
                modelVersion: modelVersion,
                modelGeneratedAt: modelGeneratedAt
            )
        }

        return VocoAutoApplyEvaluation(
            inputText: inputText,
            outputText: proposedOutputText,
            applied: applied,
            suggestions: suggestions,
            modelVersion: modelVersion,
            modelGeneratedAt: modelGeneratedAt
        )
    }

    private func firstExactPolicy(
        in model: VocoAutoApplyRuntimeModel,
        matching text: String
    ) -> VocoAutoApplyPolicy? {
        let key = Self.strictTextKey(text)
        return model.exactApplyPolicyByStrictKey[key]
    }

    private func suggestFires(
        in policies: [VocoAutoApplyPolicy],
        text: String,
        context: String
    ) -> [VocoAutoApplyPolicyFire] {
        policies.compactMap { policy in
            guard policyFires(policy, text: text, context: context) else { return nil }
            return policy.fire
        }
    }

    private func policyFires(_ policy: VocoAutoApplyPolicy, text: String, context: String) -> Bool {
        if policy.policyType == .exactTrainablePair {
            guard let inputStrictKey = policy.inputStrictKey else { return false }
            return Self.strictTextKey(text) == inputStrictKey
        }

        guard let sourcePattern = policy.sourcePattern,
              replacementMatches(
                text: text,
                source: sourcePattern,
                sourceBoundaryMode: policy.sourceBoundaryMode
              )
        else { return false }

        let trusted = policy.contextFromContextOnly == true ? context : [text, context].joined(separator: "\n")
        let aliasHits = tokenHits(in: trusted, tokens: policy.contextAliasesAny)
        let tokenHits = tokenHits(in: trusted, tokens: policy.contextTokensAny)
        if policy.requireAlias == true { return !aliasHits.isEmpty }
        if policy.contextRequired == true { return !aliasHits.isEmpty || !tokenHits.isEmpty }
        return true
    }

    private func replacementMatches(
        text: String,
        source: String,
        sourceBoundaryMode: String? = nil
    ) -> Bool {
        guard !source.isEmpty else { return false }
        if containsASCIIToken(source) {
            return rangeForASCIIBoundedSource(source, in: text) != nil
        }
        if sourceBoundaryMode == Self.cjkUnsafeContinuationBoundaryMode {
            return rangeForCJKUnsafeContinuationBoundedSource(source, in: text) != nil
        }
        return text.contains(source)
    }

    private func replace(
        _ source: String,
        with target: String,
        in text: String,
        sourceBoundaryMode: String? = nil
    ) -> String {
        if containsASCIIToken(source) {
            var result = text
            while let range = rangeForASCIIBoundedSource(source, in: result) {
                result.replaceSubrange(range, with: target)
            }
            return result
        }
        if sourceBoundaryMode == Self.cjkUnsafeContinuationBoundaryMode {
            var result = text
            while let range = rangeForCJKUnsafeContinuationBoundedSource(source, in: result) {
                result.replaceSubrange(range, with: target)
            }
            return result
        }
        return text.replacingOccurrences(of: source, with: target)
    }

    private func rangeForASCIIBoundedSource(_ source: String, in text: String) -> Range<String.Index>? {
        var searchStart = text.startIndex
        while searchStart <= text.endIndex,
              let range = text.range(of: source, options: [], range: searchStart..<text.endIndex) {
            let beforeOK = range.lowerBound == text.startIndex || !isASCIIWordCharacter(text[text.index(before: range.lowerBound)])
            let afterOK = range.upperBound == text.endIndex || !isASCIIWordCharacter(text[range.upperBound])
            if beforeOK && afterOK { return range }
            searchStart = range.upperBound
        }
        return nil
    }

    private func rangeForCJKUnsafeContinuationBoundedSource(_ source: String, in text: String) -> Range<String.Index>? {
        var searchStart = text.startIndex
        while searchStart <= text.endIndex,
              let range = text.range(of: source, options: [], range: searchStart..<text.endIndex) {
            if !shouldSkipCJKUnsafeContinuationMatch(source: source, upperBound: range.upperBound, in: text) {
                return range
            }
            searchStart = range.upperBound
        }
        return nil
    }

    private func shouldSkipCJKUnsafeContinuationMatch(
        source: String,
        upperBound: String.Index,
        in text: String
    ) -> Bool {
        guard source.allSatisfy(Self.isCJKCharacter),
              upperBound < text.endIndex
        else { return false }

        return Self.unsafeCJKContinuationAfterPairSource.contains(text[upperBound])
    }

    private func containsASCIIToken(_ text: String) -> Bool {
        let range = NSRange(location: 0, length: text.utf16.count)
        return Self.asciiTokenRegex.firstMatch(in: text, options: [], range: range) != nil
    }

    private func isASCIIWordCharacter(_ character: Character) -> Bool {
        guard character.unicodeScalars.count == 1,
              let scalar = character.unicodeScalars.first
        else { return false }
        return scalar.value == 95 ||
            (48...57).contains(scalar.value) ||
            (65...90).contains(scalar.value) ||
            (97...122).contains(scalar.value)
    }

    private static func isCJKCharacter(_ character: Character) -> Bool {
        character.unicodeScalars.contains { scalar in
            switch scalar.value {
            case 0x4E00...0x9FFF, 0x3400...0x4DBF, 0x20000...0x2A6DF:
                return true
            default:
                return false
            }
        }
    }

    private func tokenHits(in text: String, tokens: [String]) -> [String] {
        tokens.filter { !$0.isEmpty && contextContainsToken(text: text, token: $0) }
    }

    /// Contract: docs/auto-apply-evaluation-contract.md §4
    func contextContainsToken(text: String, token: String) -> Bool {
        guard !text.isEmpty, !token.isEmpty else { return false }
        let normalizedText = text.precomposedStringWithCompatibilityMapping.lowercased()
        let normalizedToken = token
            .precomposedStringWithCompatibilityMapping
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard !normalizedToken.isEmpty else { return false }

        let compactToken = normalizedToken.replacingOccurrences(of: " ", with: "")
        if Self.asciiTokenRegex.firstMatch(in: compactToken, options: [], range: NSRange(location: 0, length: compactToken.utf16.count)) != nil {
            if wordBoundedContains(normalizedText, normalizedToken) {
                return true
            }
            if !compactToken.isEmpty && wordBoundedSpacedContains(normalizedText, compactToken) {
                return true
            }
            return false
        }

        let textKey = looseKey(normalizedText)
        let tokenKey = looseKey(normalizedToken)
        return !tokenKey.isEmpty && textKey.contains(tokenKey)
    }

    /// Contract: docs/auto-apply-evaluation-contract.md §5
    func looseKey(_ value: String) -> String {
        let converted = OpenCCConverter.shared.convert(value)
        let lowercased = converted.lowercased()
        let scalars = lowercased.unicodeScalars.filter { scalar in
            !Self.looseKeyPunctuationOrWhitespace.contains(scalar)
        }
        return String(String.UnicodeScalarView(scalars))
    }

    /// Contract: docs/auto-apply-evaluation-contract.md §6
    func textIsActionCommand(text: String, actionCommandSurfaces: [String]) -> Bool {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return false }
        let surfaces = actionCommandSurfaces.isEmpty ? Self.hardCodedActionCommandSurfaces : actionCommandSurfaces
        let textKey = looseKey(text)
        return surfaces.contains { surface in
            let surfaceKey = looseKey(surface)
            return !surfaceKey.isEmpty && textKey == surfaceKey
        }
    }

    private func wordBoundedContains(_ text: String, _ token: String) -> Bool {
        var searchStart = text.startIndex
        while searchStart <= text.endIndex,
              let range = text.range(of: token, options: [], range: searchStart..<text.endIndex) {
            let beforeOK = range.lowerBound == text.startIndex || !isASCIIWordCharacter(text[text.index(before: range.lowerBound)])
            let afterOK = range.upperBound == text.endIndex || !isASCIIWordCharacter(text[range.upperBound])
            if beforeOK && afterOK { return true }
            searchStart = range.upperBound
        }
        return false
    }

    /// Match `cpp` against text containing `c p p` or `c_p_p` by allowing
    /// `[\s._-]*` between each character of the token.
    private func wordBoundedSpacedContains(_ text: String, _ compactToken: String) -> Bool {
        let chars = Array(compactToken)
        guard !chars.isEmpty else { return false }
        let separatorPattern = "[\\s._-]*"
        let escapedChars = chars.map { NSRegularExpression.escapedPattern(for: String($0)) }
        let pattern = "(?<![a-z0-9_])" + escapedChars.joined(separator: separatorPattern) + "(?![a-z0-9_])"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return false }
        let range = NSRange(location: 0, length: text.utf16.count)
        return regex.firstMatch(in: text, options: [], range: range) != nil
    }

    /// Contract: docs/auto-apply-evaluation-contract.md §3
    func replacementMatchesPublic(
        text: String,
        source: String,
        sourceBoundaryMode: String? = nil
    ) -> Bool {
        replacementMatches(text: text, source: source, sourceBoundaryMode: sourceBoundaryMode)
    }

    func containsAsciiTokenPublic(_ text: String) -> Bool {
        containsASCIIToken(text)
    }

    private func normalizeCurrencyNumbers(in text: String) -> (outputText: String, applied: [VocoAutoApplyPolicyFire]) {
        struct Replacement {
            let nsRange: NSRange
            let source: String
            let target: String
        }

        var replacements: [Replacement] = []

        func collectMatches(from regex: NSRegularExpression) {
            let searchRange = NSRange(location: 0, length: text.utf16.count)
            for match in regex.matches(in: text, options: [], range: searchRange) {
                let amountNSRange = match.range(at: 1)
                guard amountNSRange.location != NSNotFound,
                      !replacements.contains(where: { NSIntersectionRange($0.nsRange, amountNSRange).length > 0 }),
                      let amountRange = Range(amountNSRange, in: text)
                else { continue }

                let source = String(text[amountRange])
                guard let target = normalizedChineseCurrencyAmount(source),
                      target != source
                else { continue }

                replacements.append(Replacement(nsRange: amountNSRange, source: source, target: target))
            }
        }

        collectMatches(from: Self.currencyAmountWithSuffixRegex)
        collectMatches(from: Self.currencyPrefixAmountRegex)

        guard !replacements.isEmpty else {
            return (text, [])
        }

        var output = text
        for replacement in replacements.sorted(by: { $0.nsRange.location > $1.nsRange.location }) {
            guard let outputRange = Range(replacement.nsRange, in: output) else { continue }
            output.replaceSubrange(outputRange, with: replacement.target)
        }

        let fires = replacements
            .sorted(by: { $0.nsRange.location < $1.nsRange.location })
            .map { replacement in
                VocoAutoApplyPolicyFire(
                    policyId: Self.currencyNumberNormalizationPolicyId,
                    policyType: Self.currencyNumberNormalizationPolicyType,
                    autoApplyMode: VocoAutoApplyMode.apply.rawValue,
                    sourcePattern: replacement.source,
                    targetText: replacement.target,
                    sourceSlices: Self.currencyNumberNormalizationSourceSlices
                )
            }
        return (output, fires)
    }

    private func normalizedChineseCurrencyAmount(_ amount: String) -> String? {
        let trimmed = amount.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              !trimmed.contains(where: { Self.currencyApproximationCharacters.contains($0) }),
              !hasApproximateAdjacentCurrencyDigits(trimmed)
        else { return nil }

        let pieces = trimmed.split(separator: "點", omittingEmptySubsequences: false)
        if pieces.count == 1 {
            let pointPieces = trimmed.split(separator: "点", omittingEmptySubsequences: false)
            if pointPieces.count == 2 {
                return normalizedDecimalCurrencyAmount(integerPart: String(pointPieces[0]), fractionPart: String(pointPieces[1]))
            }
            guard pointPieces.count == 1 else { return nil }
            return parseChineseCurrencyInteger(trimmed).map(String.init)
        }
        guard pieces.count == 2 else { return nil }
        return normalizedDecimalCurrencyAmount(integerPart: String(pieces[0]), fractionPart: String(pieces[1]))
    }

    private func normalizedDecimalCurrencyAmount(integerPart: String, fractionPart: String) -> String? {
        guard let integer = parseChineseCurrencyInteger(integerPart),
              !fractionPart.isEmpty
        else { return nil }
        var fractionDigits = ""
        for character in fractionPart {
            guard let digit = Self.chineseCurrencyDigitValues[character] else { return nil }
            fractionDigits.append(String(digit))
        }
        return "\(integer).\(fractionDigits)"
    }

    private func parseChineseCurrencyInteger(_ value: String) -> Int? {
        guard !value.isEmpty else { return nil }
        if value.allSatisfy({ Self.chineseCurrencyDigitValues[$0] != nil }) {
            let digits = value.compactMap { Self.chineseCurrencyDigitValues[$0].map(String.init) }.joined()
            guard !digits.isEmpty else { return nil }
            return Int(digits)
        }

        var total = 0
        var section = ""
        var sawHighUnit = false
        var lastHighUnit = Int.max

        for character in value {
            if let highUnit = Self.chineseCurrencyHighUnitValues[character] {
                guard highUnit < lastHighUnit,
                      let sectionValue = parseChineseCurrencySection(section, allowBareSingleDigit: true)
                else { return nil }
                total += sectionValue * highUnit
                section = ""
                sawHighUnit = true
                lastHighUnit = highUnit
            } else {
                section.append(character)
            }
        }

        guard let trailing = parseChineseCurrencySection(
            section,
            allowBareSingleDigit: !sawHighUnit || section.first == "零" || section.first == "〇"
        ) else { return nil }
        return total + trailing
    }

    private func parseChineseCurrencySection(_ section: String, allowBareSingleDigit: Bool) -> Int? {
        if section.isEmpty { return 0 }
        if section.allSatisfy({ Self.chineseCurrencyDigitValues[$0] != nil }) {
            if section.count == 1 && !allowBareSingleDigit {
                return nil
            }
            let digits = section.compactMap { Self.chineseCurrencyDigitValues[$0].map(String.init) }.joined()
            return Int(digits)
        }

        var total = 0
        var currentDigit: Int?
        var currentDigitFollowsZero = false
        var pendingZero = false
        var sawUnit = false
        var lastUnit = Int.max

        for character in section {
            if let digit = Self.chineseCurrencyDigitValues[character] {
                if digit == 0 {
                    pendingZero = true
                    currentDigit = nil
                    currentDigitFollowsZero = true
                    continue
                }
                guard currentDigit == nil else { return nil }
                currentDigit = digit
                currentDigitFollowsZero = pendingZero
                pendingZero = false
                continue
            }

            guard let unit = Self.chineseCurrencySectionUnitValues[character],
                  unit < lastUnit
            else { return nil }
            let digit = currentDigit ?? (unit == 10 ? 1 : nil)
            guard let digit else { return nil }
            total += digit * unit
            currentDigit = nil
            currentDigitFollowsZero = false
            pendingZero = false
            sawUnit = true
            lastUnit = unit
        }

        if let currentDigit {
            if sawUnit && lastUnit > 10 && !currentDigitFollowsZero {
                return nil
            }
            total += currentDigit
        }
        return total
    }

    private func hasApproximateAdjacentCurrencyDigits(_ value: String) -> Bool {
        guard value.contains(where: {
            Self.chineseCurrencySectionUnitValues[$0] != nil || Self.chineseCurrencyHighUnitValues[$0] != nil
        }) else { return false }

        var previous: Character?
        for character in value {
            defer { previous = character }
            guard let previous,
                  Self.chineseCurrencyDigitValues[previous] != nil,
                  Self.chineseCurrencyDigitValues[character] != nil,
                  Self.chineseCurrencyDigitValues[previous] != 0
            else { continue }
            return true
        }
        return false
    }

    private func protectedTermGuardBlocks(
        in text: String,
        applied: [VocoAutoApplyPolicyFire],
        guardRules: [VocoProtectedTermAllowlistGuard]
    ) -> [VocoAutoApplyGuardBlock] {
        guardRules.compactMap { guardRule in
            guard text.contains(guardRule.term),
                  !allProtectedTermOccurrencesAreAllowed(in: text, guardRule: guardRule),
                  !appliedPolicySupportsProtectedTerm(applied, term: guardRule.term)
            else { return nil }

            return VocoAutoApplyGuardBlock(
                guardId: guardRule.guardId,
                reason: guardRule.reason,
                term: guardRule.term,
                blockedText: text,
                allowedPhrases: guardRule.allowedPhrases
            )
        }
    }

    private func allProtectedTermOccurrencesAreAllowed(in text: String, guardRule: VocoProtectedTermAllowlistGuard) -> Bool {
        var searchStart = text.startIndex
        while searchStart < text.endIndex,
              let termRange = text.range(of: guardRule.term, range: searchStart..<text.endIndex) {
            guard allowlistPhraseContains(termRange, in: text, allowedPhrases: guardRule.allowedPhrases) else {
                return false
            }
            searchStart = termRange.upperBound
        }
        return true
    }

    private func allowlistPhraseContains(
        _ termRange: Range<String.Index>,
        in text: String,
        allowedPhrases: [String]
    ) -> Bool {
        for phrase in allowedPhrases where !phrase.isEmpty {
            var searchStart = text.startIndex
            while searchStart < text.endIndex,
                  let phraseRange = text.range(of: phrase, range: searchStart..<text.endIndex) {
                if phraseRange.lowerBound <= termRange.lowerBound,
                   phraseRange.upperBound >= termRange.upperBound {
                    return true
                }
                searchStart = phraseRange.upperBound
            }
        }
        return false
    }

    private func appliedPolicySupportsProtectedTerm(
        _ applied: [VocoAutoApplyPolicyFire],
        term: String
    ) -> Bool {
        applied.contains { fire in
            fire.sourcePattern.contains(term) || fire.targetText.contains(term)
        }
    }

    static func strictTextKey(_ value: String) -> String {
        let normalized = value
            .precomposedStringWithCompatibilityMapping
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return normalized
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private static func regexAlternation(_ terms: [String]) -> String {
        terms
            .sorted { $0.count > $1.count }
            .map { NSRegularExpression.escapedPattern(for: $0) }
            .joined(separator: "|")
    }
}

private struct VocoAutoApplyModelEnvelope: Decodable {
    let schemaVersion: Int?
    let runtimeSchemaVersion: Int?
    let mergedReplayReadiness: VocoMergedReplayReadiness?
}

private struct VocoAutoApplyModel: Decodable {
    let policyCounts: [String: Int]
    let policyTypeCounts: [String: Int]
    let safetyContract: [String]
    let protectedTermAllowlistGuards: [VocoProtectedTermAllowlistGuard]
    let policies: [VocoAutoApplyPolicy]
    let mergedReplayReadiness: VocoMergedReplayReadiness
    let schemaVersion: Int?
    let actionCommandGuards: [VocoActionCommandGuard]?
    let autoApplyModelVersion: String?
    let generatedAt: String?

    var applyPolicies: [VocoAutoApplyPolicy] {
        policies.filter { $0.autoApplyMode == .apply }
    }

    var suggestPolicies: [VocoAutoApplyPolicy] {
        policies.filter { $0.autoApplyMode == .suggest }
    }

    enum CodingKeys: String, CodingKey {
        case policyCounts
        case policyTypeCounts
        case safetyContract
        case protectedTermAllowlistGuards
        case protectedTermAllowlist
        case policies
        case mergedReplayReadiness
        case schemaVersion
        case actionCommandGuards
        case autoApplyModelVersion
        case generatedAt
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        policyCounts = try container.decodeIfPresent([String: Int].self, forKey: .policyCounts) ?? [:]
        policyTypeCounts = try container.decodeIfPresent([String: Int].self, forKey: .policyTypeCounts) ?? [:]
        safetyContract = try container.decodeIfPresent([String].self, forKey: .safetyContract) ?? []
        protectedTermAllowlistGuards =
            try container.decodeIfPresent([VocoProtectedTermAllowlistGuard].self, forKey: .protectedTermAllowlistGuards) ??
            container.decodeIfPresent([VocoProtectedTermAllowlistGuard].self, forKey: .protectedTermAllowlist) ??
            []
        policies = try container.decode([VocoAutoApplyPolicy].self, forKey: .policies)
        mergedReplayReadiness = try container.decode(VocoMergedReplayReadiness.self, forKey: .mergedReplayReadiness)
        schemaVersion = try container.decodeIfPresent(Int.self, forKey: .schemaVersion)
        actionCommandGuards = try container.decodeIfPresent([VocoActionCommandGuard].self, forKey: .actionCommandGuards)
        autoApplyModelVersion = try container.decodeIfPresent(String.self, forKey: .autoApplyModelVersion)
        generatedAt = try container.decodeIfPresent(String.self, forKey: .generatedAt)
    }
}

private enum VocoAutoApplyModelDecodeError: LocalizedError {
    case missingIndexedRuntimeMarker
    case unsupportedSchemaVersion(Int)
    case unsupportedRuntimeSchemaVersion(Int)

    var errorDescription: String? {
        switch self {
        case .missingIndexedRuntimeMarker:
            return "Missing indexed runtime model marker"
        case .unsupportedSchemaVersion(let schemaVersion):
            return "Unsupported auto-apply schema version \(schemaVersion)"
        case .unsupportedRuntimeSchemaVersion(let runtimeSchemaVersion):
            return "Unsupported auto-apply runtime schema version \(runtimeSchemaVersion)"
        }
    }
}

private struct VocoDecodedAutoApplyModel {
    let runtimeModel: VocoAutoApplyRuntimeModel
    let policyCounts: [String: Int]
    let applyPolicyCount: Int
    let suggestPolicyCount: Int
    let isReady: Bool

    init(model: VocoAutoApplyModel) {
        runtimeModel = VocoAutoApplyRuntimeModel(model: model)
        policyCounts = model.policyCounts
        applyPolicyCount = model.applyPolicies.count
        suggestPolicyCount = model.suggestPolicies.count
        isReady = model.mergedReplayReadiness.mergedAutoApplyModelReady == true
    }

    init(indexedModel: VocoIndexedRuntimeAutoApplyModel) {
        runtimeModel = VocoAutoApplyRuntimeModel(indexedModel: indexedModel)
        policyCounts = indexedModel.policyCounts
        applyPolicyCount = runtimeModel.exactApplyPolicyByStrictKey.count + runtimeModel.scopedApplyPolicies.count
        suggestPolicyCount = runtimeModel.suggestPolicies.count
        isReady = indexedModel.mergedReplayReadiness.mergedAutoApplyModelReady == true
    }
}

private struct VocoIndexedRuntimeAutoApplyModel: Decodable {
    static let modelFormat = "voco-auto-apply-runtime-indexed-v2"

    let policyCounts: [String: Int]
    let policyTypeCounts: [String: Int]
    let safetyContract: [String]
    let protectedTermAllowlistGuards: [VocoProtectedTermAllowlistGuard]
    let exactApplyPolicyByStrictKey: [String: VocoIndexedExactApplyPolicy]
    let scopedApplyPolicies: [VocoAutoApplyPolicy]
    let suggestPolicies: [VocoAutoApplyPolicy]
    let mergedReplayReadiness: VocoMergedReplayReadiness
    let schemaVersion: Int?
    let runtimeSchemaVersion: Int
    let modelFormat: String
    let actionCommandGuards: [VocoActionCommandGuard]?
    let autoApplyModelVersion: String?
    let generatedAt: String?

    enum CodingKeys: String, CodingKey {
        case policyCounts
        case policyTypeCounts
        case safetyContract
        case protectedTermAllowlistGuards
        case protectedTermAllowlist
        case exactApplyPolicyByStrictKey
        case exactApplyPoliciesByStrictKey
        case scopedApplyPolicies
        case scopedReplacementPolicies
        case suggestPolicies
        case mergedReplayReadiness
        case schemaVersion
        case runtimeSchemaVersion
        case modelFormat
        case actionCommandGuards
        case autoApplyModelVersion
        case generatedAt
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let decodedRuntimeSchemaVersion = try container.decodeIfPresent(Int.self, forKey: .runtimeSchemaVersion)
        let decodedModelFormat = try container.decodeIfPresent(String.self, forKey: .modelFormat)
        guard decodedRuntimeSchemaVersion != nil || decodedModelFormat == Self.modelFormat else {
            throw VocoAutoApplyModelDecodeError.missingIndexedRuntimeMarker
        }
        if let decodedRuntimeSchemaVersion,
           !VocoAutoApplyModelService.supportedRuntimeSchemaVersions.contains(decodedRuntimeSchemaVersion) {
            throw VocoAutoApplyModelDecodeError.unsupportedRuntimeSchemaVersion(decodedRuntimeSchemaVersion)
        }

        policyCounts = try container.decodeIfPresent([String: Int].self, forKey: .policyCounts) ?? [:]
        policyTypeCounts = try container.decodeIfPresent([String: Int].self, forKey: .policyTypeCounts) ?? [:]
        safetyContract = try container.decodeIfPresent([String].self, forKey: .safetyContract) ?? []
        protectedTermAllowlistGuards =
            try container.decodeIfPresent([VocoProtectedTermAllowlistGuard].self, forKey: .protectedTermAllowlistGuards) ??
            container.decodeIfPresent([VocoProtectedTermAllowlistGuard].self, forKey: .protectedTermAllowlist) ??
            []
        exactApplyPolicyByStrictKey =
            try container.decodeIfPresent([String: VocoIndexedExactApplyPolicy].self, forKey: .exactApplyPolicyByStrictKey) ??
            container.decode([String: VocoIndexedExactApplyPolicy].self, forKey: .exactApplyPoliciesByStrictKey)
        scopedApplyPolicies =
            try container.decodeIfPresent([VocoAutoApplyPolicy].self, forKey: .scopedApplyPolicies) ??
            container.decodeIfPresent([VocoAutoApplyPolicy].self, forKey: .scopedReplacementPolicies) ??
            []
        suggestPolicies = try container.decodeIfPresent([VocoAutoApplyPolicy].self, forKey: .suggestPolicies) ?? []
        mergedReplayReadiness = try container.decode(VocoMergedReplayReadiness.self, forKey: .mergedReplayReadiness)
        schemaVersion = try container.decodeIfPresent(Int.self, forKey: .schemaVersion)
        runtimeSchemaVersion = decodedRuntimeSchemaVersion ?? VocoAutoApplyModelService.supportedRuntimeSchemaVersion
        modelFormat = decodedModelFormat ?? Self.modelFormat
        actionCommandGuards = try container.decodeIfPresent([VocoActionCommandGuard].self, forKey: .actionCommandGuards)
        autoApplyModelVersion = try container.decodeIfPresent(String.self, forKey: .autoApplyModelVersion)
        generatedAt = try container.decodeIfPresent(String.self, forKey: .generatedAt)
    }
}

private struct VocoIndexedExactApplyPolicy: Decodable {
    let policyId: String
    let sourcePattern: String?
    let targetText: String
    let sourceSlices: [String]

    enum CodingKeys: String, CodingKey {
        case policyId
        case sourcePattern
        case targetText
        case sourceSlices
    }

    func runtimePolicy(inputStrictKey: String) -> VocoAutoApplyPolicy {
        VocoAutoApplyPolicy(
            policyId: policyId,
            autoApplyMode: .apply,
            policyType: .exactTrainablePair,
            sourcePattern: sourcePattern,
            targetText: targetText,
            inputStrictKey: inputStrictKey,
            exactInputRequired: true,
            exactInputResolution: nil,
            contextAliasesAny: [],
            contextTokensAny: [],
            contextFromContextOnly: nil,
            contextRequired: nil,
            requireAlias: nil,
            scopedSourcePhrase: nil,
            sourceSlices: sourceSlices,
            reviewGateConflictRows: []
        )
    }
}

private struct VocoActionCommandGuard: Decodable {
    let surface: String
}

private struct VocoAutoApplyRuntimeModel {
    let protectedTermAllowlistGuards: [VocoProtectedTermAllowlistGuard]
    let exactApplyPolicyByStrictKey: [String: VocoAutoApplyPolicy]
    let scopedApplyPolicies: [VocoAutoApplyPolicy]
    let suggestPolicies: [VocoAutoApplyPolicy]
    let actionCommandSurfaces: [String]
    let modelVersion: String?
    let modelGeneratedAt: String?
    let schemaVersion: Int?

    init(model: VocoAutoApplyModel) {
        protectedTermAllowlistGuards = model.protectedTermAllowlistGuards

        var exactApplyPolicyByStrictKey: [String: VocoAutoApplyPolicy] = [:]
        var scopedApplyPolicies: [VocoAutoApplyPolicy] = []
        var suggestPolicies: [VocoAutoApplyPolicy] = []

        for policy in model.policies {
            switch policy.autoApplyMode {
            case .apply:
                switch policy.policyType {
                case .exactTrainablePair:
                    guard policy.isSafeExactApplyPolicy,
                          let inputStrictKey = policy.inputStrictKey
                    else { break }
                    if exactApplyPolicyByStrictKey[inputStrictKey] == nil {
                        exactApplyPolicyByStrictKey[inputStrictKey] = policy
                    }
                case .scopedReplacement:
                    if policy.isSafeScopedApplyPolicy {
                        scopedApplyPolicies.append(policy)
                    }
                }
            case .suggest:
                suggestPolicies.append(policy)
            case .blocked, .replaced:
                break
            }
        }

        self.exactApplyPolicyByStrictKey = exactApplyPolicyByStrictKey
        self.scopedApplyPolicies = scopedApplyPolicies
        self.suggestPolicies = suggestPolicies
        self.actionCommandSurfaces = (model.actionCommandGuards ?? []).map(\.surface).filter { !$0.isEmpty }
        self.modelVersion = model.autoApplyModelVersion
        self.modelGeneratedAt = model.generatedAt
        self.schemaVersion = model.schemaVersion
    }

    init(indexedModel: VocoIndexedRuntimeAutoApplyModel) {
        protectedTermAllowlistGuards = indexedModel.protectedTermAllowlistGuards

        var exactApplyPolicyByStrictKey: [String: VocoAutoApplyPolicy] = [:]
        for (inputStrictKey, compactPolicy) in indexedModel.exactApplyPolicyByStrictKey {
            let policy = compactPolicy.runtimePolicy(inputStrictKey: inputStrictKey)
            guard policy.isSafeExactApplyPolicy else { continue }
            if exactApplyPolicyByStrictKey[inputStrictKey] == nil {
                exactApplyPolicyByStrictKey[inputStrictKey] = policy
            }
        }

        scopedApplyPolicies = indexedModel.scopedApplyPolicies.filter {
            $0.autoApplyMode == .apply &&
                $0.policyType == .scopedReplacement &&
                $0.isSafeScopedApplyPolicy
        }
        suggestPolicies = indexedModel.suggestPolicies.filter { $0.autoApplyMode == .suggest }
        self.exactApplyPolicyByStrictKey = exactApplyPolicyByStrictKey
        self.actionCommandSurfaces = (indexedModel.actionCommandGuards ?? []).map(\.surface).filter { !$0.isEmpty }
        self.modelVersion = indexedModel.autoApplyModelVersion
        self.modelGeneratedAt = indexedModel.generatedAt
        self.schemaVersion = indexedModel.schemaVersion
    }
}

private struct VocoProtectedTermAllowlistGuard: Decodable, Equatable {
    let guardId: String
    let reason: String
    let term: String
    let allowedPhrases: [String]

    enum CodingKeys: String, CodingKey {
        case guardId
        case reason
        case term
        case allowedPhrases
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        guardId = try container.decode(String.self, forKey: .guardId)
        let decodedReason = try container.decodeIfPresent(String.self, forKey: .reason)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let decodedReason, !decodedReason.isEmpty {
            reason = decodedReason
        } else {
            reason = VocoAutoApplyModelService.protectedTermGuardReason
        }
        term = try container.decode(String.self, forKey: .term)
        allowedPhrases = try container.decodeIfPresent([String].self, forKey: .allowedPhrases) ?? []
    }
}

private struct VocoMergedReplayReadiness: Decodable {
    let mergedAutoApplyModelReady: Bool?
}

private enum VocoAutoApplyMode: String, Decodable {
    case apply
    case suggest
    case blocked
    case replaced

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let rawValue = try container.decode(String.self)
        self = VocoAutoApplyMode(rawValue: rawValue) ?? .blocked
    }
}

private enum VocoAutoApplyPolicyType: String, Decodable {
    case exactTrainablePair
    case scopedReplacement
}

private struct VocoAutoApplyPolicy: Decodable {
    let policyId: String
    let autoApplyMode: VocoAutoApplyMode
    let policyType: VocoAutoApplyPolicyType
    let sourcePattern: String?
    let targetText: String?
    let inputStrictKey: String?
    let exactInputRequired: Bool?
    let exactInputResolution: VocoExactInputResolution?
    let contextAliasesAny: [String]
    let contextTokensAny: [String]
    let contextFromContextOnly: Bool?
    let contextRequired: Bool?
    let requireAlias: Bool?
    let scopedSourcePhrase: String?
    let sourceSlices: [String]
    let reviewGateConflictRows: [Int]
    let manualOverrideRows: [Int]
    let sourceBoundaryMode: String?
    let familyId: String?
    let familyRole: String?
    let migrationSource: String?

    var hasNonEmptyTarget: Bool {
        targetText?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
    }

    var isSafeExactApplyPolicy: Bool {
        autoApplyMode == .apply &&
            policyType == .exactTrainablePair &&
            exactInputRequired == true &&
            inputStrictKey?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false &&
            hasNonEmptyTarget &&
            (reviewGateConflictRows.isEmpty || !manualOverrideRows.isEmpty)
    }

    var isSafeScopedApplyPolicy: Bool {
        autoApplyMode == .apply &&
            policyType == .scopedReplacement &&
            hasNonEmptyTarget &&
            reviewGateConflictRows.isEmpty
    }

    var fire: VocoAutoApplyPolicyFire {
        VocoAutoApplyPolicyFire(
            policyId: policyId,
            policyType: policyType.rawValue,
            autoApplyMode: autoApplyMode.rawValue,
            sourcePattern: sourcePattern ?? "",
            targetText: targetText ?? "",
            sourceSlices: sourceSlices,
            sourceBoundaryMode: sourceBoundaryMode,
            familyId: familyId,
            familyRole: familyRole
        )
    }

    enum CodingKeys: String, CodingKey {
        case policyId
        case autoApplyMode
        case policyType
        case sourcePattern
        case targetText
        case inputStrictKey
        case exactInputRequired
        case exactInputResolution
        case contextAliasesAny
        case contextTokensAny
        case contextFromContextOnly
        case contextRequired
        case requireAlias
        case scopedSourcePhrase
        case sourceSlices
        case reviewGateConflictRows
        case manualOverrideRows
        case sourceBoundaryMode
        case familyId
        case familyRole
        case migrationSource
    }

    init(
        policyId: String,
        autoApplyMode: VocoAutoApplyMode,
        policyType: VocoAutoApplyPolicyType,
        sourcePattern: String?,
        targetText: String?,
        inputStrictKey: String?,
        exactInputRequired: Bool?,
        exactInputResolution: VocoExactInputResolution?,
        contextAliasesAny: [String],
        contextTokensAny: [String],
        contextFromContextOnly: Bool?,
        contextRequired: Bool?,
        requireAlias: Bool?,
        scopedSourcePhrase: String?,
        sourceSlices: [String],
        reviewGateConflictRows: [Int],
        manualOverrideRows: [Int] = [],
        sourceBoundaryMode: String? = nil,
        familyId: String? = nil,
        familyRole: String? = nil,
        migrationSource: String? = nil
    ) {
        self.policyId = policyId
        self.autoApplyMode = autoApplyMode
        self.policyType = policyType
        self.sourcePattern = sourcePattern
        self.targetText = targetText
        self.inputStrictKey = inputStrictKey
        self.exactInputRequired = exactInputRequired
        self.exactInputResolution = exactInputResolution
        self.contextAliasesAny = contextAliasesAny
        self.contextTokensAny = contextTokensAny
        self.contextFromContextOnly = contextFromContextOnly
        self.contextRequired = contextRequired
        self.requireAlias = requireAlias
        self.scopedSourcePhrase = scopedSourcePhrase
        self.sourceSlices = sourceSlices
        self.reviewGateConflictRows = reviewGateConflictRows
        self.manualOverrideRows = manualOverrideRows
        self.sourceBoundaryMode = sourceBoundaryMode
        self.familyId = familyId
        self.familyRole = familyRole
        self.migrationSource = migrationSource
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        policyId = try container.decode(String.self, forKey: .policyId)
        autoApplyMode = try container.decode(VocoAutoApplyMode.self, forKey: .autoApplyMode)
        policyType = try container.decode(VocoAutoApplyPolicyType.self, forKey: .policyType)
        sourcePattern = try container.decodeIfPresent(String.self, forKey: .sourcePattern)
        targetText = try container.decodeIfPresent(String.self, forKey: .targetText)
        inputStrictKey = try container.decodeIfPresent(String.self, forKey: .inputStrictKey)
        exactInputRequired = try container.decodeIfPresent(Bool.self, forKey: .exactInputRequired)
        exactInputResolution = try container.decodeIfPresent(VocoExactInputResolution.self, forKey: .exactInputResolution)
        contextAliasesAny = try container.decodeIfPresent([String].self, forKey: .contextAliasesAny) ?? []
        contextTokensAny = try container.decodeIfPresent([String].self, forKey: .contextTokensAny) ?? []
        contextFromContextOnly = try container.decodeIfPresent(Bool.self, forKey: .contextFromContextOnly)
        contextRequired = try container.decodeIfPresent(Bool.self, forKey: .contextRequired)
        requireAlias = try container.decodeIfPresent(Bool.self, forKey: .requireAlias)
        scopedSourcePhrase = try container.decodeIfPresent(String.self, forKey: .scopedSourcePhrase)
        sourceSlices = try container.decodeIfPresent([String].self, forKey: .sourceSlices) ?? []
        reviewGateConflictRows = try container.decodeIfPresent([Int].self, forKey: .reviewGateConflictRows) ?? []
        manualOverrideRows = try container.decodeIfPresent([Int].self, forKey: .manualOverrideRows) ?? []
        sourceBoundaryMode = try container.decodeIfPresent(String.self, forKey: .sourceBoundaryMode)
        familyId = try container.decodeIfPresent(String.self, forKey: .familyId)
        familyRole = try container.decodeIfPresent(String.self, forKey: .familyRole)
        migrationSource = try container.decodeIfPresent(String.self, forKey: .migrationSource)
    }
}

private struct VocoExactInputResolution: Decodable {
    let targetText: String?
    let targetStrictKey: String?
    let resolutionReason: String?
}
