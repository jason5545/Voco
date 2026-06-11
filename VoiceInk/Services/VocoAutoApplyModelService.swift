import Foundation
import Combine
import Darwin
import OSLog

struct VocoAutoApplyPolicyFire: Codable, Equatable {
    let policyId: String
    let policyType: String
    let autoApplyMode: String
    let sourcePattern: String
    let targetText: String
    let sourceSlices: [String]
}

struct VocoAutoApplyEvaluation: Equatable {
    let inputText: String
    let outputText: String
    let applied: [VocoAutoApplyPolicyFire]
    let suggestions: [VocoAutoApplyPolicyFire]

    var changed: Bool { inputText != outputText }
}

struct VocoAutoApplyModelStatus: Equatable {
    let isAvailable: Bool
    let message: String
    let modelURL: URL
}

final class VocoAutoApplyModelService: ObservableObject {
    static let shared = VocoAutoApplyModelService()
    static let enabledKey = "VocoAutoApplyModelEnabled"
    static let modelFileName = "full-db.auto-apply-model.json"

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
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "AutoApplyModel")
    private let watchQueue = DispatchQueue(label: "com.jasonchien.Voco.autoApplyModelWatcher")
    private var loadedModel: VocoAutoApplyModel?
    private var modelFileWatcher: DispatchSourceFileSystemObject?
    private var modelDirectoryWatcher: DispatchSourceFileSystemObject?
    private var pendingModelReload: DispatchWorkItem?

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

    init(
        modelURL: URL = VocoAutoApplyModelService.defaultModelURL,
        defaults: UserDefaults = .standard
    ) {
        self.modelURL = modelURL
        self.defaults = defaults
        self.status = VocoAutoApplyModelStatus(
            isAvailable: false,
            message: String(localized: "Model not detected"),
            modelURL: modelURL
        )
        reload()
        startWatchingModelChanges()
    }

    deinit {
        pendingModelReload?.cancel()
        modelFileWatcher?.cancel()
        modelDirectoryWatcher?.cancel()
    }

    func reload() {
        do {
            guard FileManager.default.fileExists(atPath: modelURL.path) else {
                loadedModel = nil
                status = VocoAutoApplyModelStatus(
                    isAvailable: false,
                    message: String(localized: "Model not installed"),
                    modelURL: modelURL
                )
                return
            }

            let data = try Data(contentsOf: modelURL)
            let model = try JSONDecoder().decode(VocoAutoApplyModel.self, from: data)
            guard model.mergedReplayReadiness.mergedAutoApplyModelReady == true else {
                loadedModel = nil
                status = VocoAutoApplyModelStatus(
                    isAvailable: false,
                    message: String(localized: "Model not ready"),
                    modelURL: modelURL
                )
                return
            }

            loadedModel = model
            let applyCount = model.policyCounts["apply"] ?? model.applyPolicies.count
            let suggestCount = model.policyCounts["suggest"] ?? model.suggestPolicies.count
            status = VocoAutoApplyModelStatus(
                isAvailable: true,
                message: String(localized: "Model loaded: \(applyCount) apply, \(suggestCount) suggest"),
                modelURL: modelURL
            )
        } catch {
            loadedModel = nil
            status = VocoAutoApplyModelStatus(
                isAvailable: false,
                message: String(localized: "Model unreadable"),
                modelURL: modelURL
            )
            logger.error("Failed to load auto-apply model: \(error.localizedDescription, privacy: .public)")
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
              VoiceCommandService.shared.detectCommand(in: text) == nil
        else {
            return VocoAutoApplyEvaluation(inputText: text, outputText: text, applied: [], suggestions: [])
        }

        let exactApplyPolicies = model.applyPolicies.filter { $0.policyType == .exactTrainablePair }
        if let exact = firstExactPolicy(in: exactApplyPolicies, matching: text),
           exact.isSafeApplyPolicy {
            let target = exact.targetText ?? text
            return VocoAutoApplyEvaluation(
                inputText: text,
                outputText: target,
                applied: [exact.fire],
                suggestions: suggestFires(in: model.suggestPolicies, text: text, context: context)
            )
        }

        var output = text
        var applied: [VocoAutoApplyPolicyFire] = []
        for policy in model.applyPolicies where policy.policyType != .exactTrainablePair {
            guard policy.isSafeApplyPolicy,
                  policyFires(policy, text: output, context: context),
                  let sourcePattern = policy.sourcePattern,
                  let targetText = policy.targetText
            else { continue }

            let updated = replace(sourcePattern, with: targetText, in: output)
            guard updated != output else { continue }
            output = updated
            applied.append(policy.fire)
        }

        return VocoAutoApplyEvaluation(
            inputText: text,
            outputText: output,
            applied: applied,
            suggestions: suggestFires(in: model.suggestPolicies, text: output, context: context)
        )
    }

    private func firstExactPolicy(
        in policies: [VocoAutoApplyPolicy],
        matching text: String
    ) -> VocoAutoApplyPolicy? {
        let key = Self.strictTextKey(text)
        return policies.first {
            $0.inputStrictKey == key && $0.exactInputRequired == true
        }
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
              replacementMatches(text: text, source: sourcePattern)
        else { return false }

        let trusted = policy.contextFromContextOnly == true ? context : [text, context].joined(separator: "\n")
        let aliasHits = tokenHits(in: trusted, tokens: policy.contextAliasesAny)
        let tokenHits = tokenHits(in: trusted, tokens: policy.contextTokensAny)
        if policy.requireAlias == true { return !aliasHits.isEmpty }
        if policy.contextRequired == true { return !aliasHits.isEmpty || !tokenHits.isEmpty }
        return true
    }

    private func replacementMatches(text: String, source: String) -> Bool {
        guard !source.isEmpty else { return false }
        if containsASCIIToken(source) {
            return rangeForASCIIBoundedSource(source, in: text) != nil
        }
        return text.contains(source)
    }

    private func replace(_ source: String, with target: String, in text: String) -> String {
        if containsASCIIToken(source) {
            var result = text
            while let range = rangeForASCIIBoundedSource(source, in: result) {
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

    private func containsASCIIToken(_ text: String) -> Bool {
        text.range(of: #"[A-Za-z][A-Za-z0-9_+.#/-]*"#, options: .regularExpression) != nil
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

    private func tokenHits(in text: String, tokens: [String]) -> [String] {
        tokens.filter { !$0.isEmpty && text.localizedCaseInsensitiveContains($0) }
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
}

private struct VocoAutoApplyModel: Decodable {
    let policyCounts: [String: Int]
    let policyTypeCounts: [String: Int]
    let safetyContract: [String]
    let policies: [VocoAutoApplyPolicy]
    let mergedReplayReadiness: VocoMergedReplayReadiness

    var applyPolicies: [VocoAutoApplyPolicy] {
        policies.filter { $0.autoApplyMode == .apply }
    }

    var suggestPolicies: [VocoAutoApplyPolicy] {
        policies.filter { $0.autoApplyMode == .suggest }
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

    var isSafeApplyPolicy: Bool {
        autoApplyMode == .apply &&
            targetText?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false &&
            reviewGateConflictRows.isEmpty
    }

    var fire: VocoAutoApplyPolicyFire {
        VocoAutoApplyPolicyFire(
            policyId: policyId,
            policyType: policyType.rawValue,
            autoApplyMode: autoApplyMode.rawValue,
            sourcePattern: sourcePattern ?? "",
            targetText: targetText ?? "",
            sourceSlices: sourceSlices
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
    }
}

private struct VocoExactInputResolution: Decodable {
    let targetText: String?
    let targetStrictKey: String?
    let resolutionReason: String?
}
