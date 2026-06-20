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

    init(
        isAvailable: Bool,
        message: String,
        modelURL: URL,
        modelVersion: String? = nil,
        modelGeneratedAt: String? = nil,
        schemaVersion: Int? = nil,
        isDegraded: Bool = false
    ) {
        self.isAvailable = isAvailable
        self.message = message
        self.modelURL = modelURL
        self.modelVersion = modelVersion
        self.modelGeneratedAt = modelGeneratedAt
        self.schemaVersion = schemaVersion
        self.isDegraded = isDegraded
    }
}

final class VocoAutoApplyModelService: ObservableObject {
    static let shared = VocoAutoApplyModelService()
    static let enabledKey = "VocoAutoApplyModelEnabled"
    static let modelFileName = "full-db.auto-apply-model.json"
    static let protectedTermGuardReason = "auto-apply-model-protected-term-guard"
    static let supportedSchemaVersion = 1
    static let supportedRuntimeSchemaVersion = 2

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
    private var loadedModel: VocoAutoApplyRuntimeModel?
    private var modelFileWatcher: DispatchSourceFileSystemObject?
    private var modelDirectoryWatcher: DispatchSourceFileSystemObject?
    private var pendingModelReload: DispatchWorkItem?

    static let hardCodedActionCommandSurfaces: [String] = ["全部刪除", "全部删除"]

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
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            loadedModel = nil
            status = VocoAutoApplyModelStatus(
                isAvailable: false,
                message: String(localized: "Model not installed"),
                modelURL: modelURL
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
                    isDegraded: true
                )
            } else {
                loadedModel = nil
                status = VocoAutoApplyModelStatus(
                    isAvailable: false,
                    message: String(localized: "Model schema unsupported"),
                    modelURL: modelURL
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
                    isDegraded: true
                )
            } else {
                loadedModel = nil
                status = VocoAutoApplyModelStatus(
                    isAvailable: false,
                    message: String(localized: "Model schema unsupported"),
                    modelURL: modelURL
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
                    isDegraded: true
                )
            } else {
                loadedModel = nil
                status = VocoAutoApplyModelStatus(
                    isAvailable: false,
                    message: String(localized: "Model unreadable"),
                    modelURL: modelURL
                )
            }
            return
        }

        guard decodedModel.isReady else {
            loadedModel = nil
            status = VocoAutoApplyModelStatus(
                isAvailable: false,
                message: String(localized: "Model not ready"),
                modelURL: modelURL
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
            schemaVersion: decodedModel.runtimeModel.schemaVersion ?? Self.supportedSchemaVersion
        )
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
               schemaVersion != Self.supportedSchemaVersion {
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

            let updated = replace(sourcePattern, with: targetText, in: output)
            guard updated != output else { continue }
            output = updated
            applied.append(policy.fire)
        }

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
    func replacementMatchesPublic(text: String, source: String) -> Bool {
        replacementMatches(text: text, source: source)
    }

    func containsAsciiTokenPublic(_ text: String) -> Bool {
        containsASCIIToken(text)
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
           decodedRuntimeSchemaVersion != VocoAutoApplyModelService.supportedRuntimeSchemaVersion {
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
        case manualOverrideRows
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
        manualOverrideRows: [Int] = []
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
    }
}

private struct VocoExactInputResolution: Decodable {
    let targetText: String?
    let targetStrictKey: String?
    let resolutionReason: String?
}
