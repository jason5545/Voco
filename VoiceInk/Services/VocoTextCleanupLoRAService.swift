import CryptoKit
import Foundation
import MLX
import os

struct VocoTextCleanupLoRAExpectedIdentity: Equatable {
    let adapterSHA256: String?
    let baseModelSHA256: String?
    let tokenizerConfigSHA256: String?
    let vocabSHA256: String?
    let mergesSHA256: String?
    let auditReportSHA256: String?

    init(
        adapterSHA256: String? = nil,
        baseModelSHA256: String? = nil,
        tokenizerConfigSHA256: String? = nil,
        vocabSHA256: String? = nil,
        mergesSHA256: String? = nil,
        auditReportSHA256: String? = nil
    ) {
        self.adapterSHA256 = adapterSHA256
        self.baseModelSHA256 = baseModelSHA256
        self.tokenizerConfigSHA256 = tokenizerConfigSHA256
        self.vocabSHA256 = vocabSHA256
        self.mergesSHA256 = mergesSHA256
        self.auditReportSHA256 = auditReportSHA256
    }

    static let production = VocoTextCleanupLoRAExpectedIdentity(
        adapterSHA256: VocoTextCleanupLoRAService.expectedAdapterSHA256,
        baseModelSHA256: VocoTextCleanupLoRAService.expectedBaseModelSHA256,
        tokenizerConfigSHA256: VocoTextCleanupLoRAService.expectedTokenizerConfigSHA256,
        vocabSHA256: VocoTextCleanupLoRAService.expectedVocabSHA256,
        mergesSHA256: VocoTextCleanupLoRAService.expectedMergesSHA256,
        auditReportSHA256: VocoTextCleanupLoRAService.expectedAuditReportSHA256
    )
}

struct VocoTextCleanupLoRAIdentity: Codable, Equatable {
    let baseModelPath: String
    let adapterPath: String
    let auditReportPath: String?
    let adapterVersion: String?
    let promotedApplyClass: String?
    let applyAllAllowed: Bool
    let releaseOrDeployAllowed: Bool
    let adapterSHA256: String?
    let baseModelSHA256: String?
    let tokenizerConfigSHA256: String?
    let vocabSHA256: String?
    let mergesSHA256: String?
    let auditReportSHA256: String?
}

struct VocoTextCleanupLoRARequest: Equatable {
    let inputText: String
    let rawTranscript: String?
    let postRuleText: String
    let contextHints: [String]
    let baseModelURL: URL
    let adapterURL: URL
    let maxTokens: Int
    let identity: VocoTextCleanupLoRAIdentity?
}

struct VocoTextCleanupLoRAEvaluation: Codable, Equatable {
    let schema: String
    let eventId: String
    let inputText: String
    let rawTranscript: String?
    let postRuleText: String
    let outputText: String
    let finalText: String
    let candidateText: String?
    let generatedText: String?
    let mode: VocoTextCleanupLoRAService.Mode
    let decision: String
    let chosenAction: String
    let applied: Bool
    let status: String
    let reasonCodes: [String]
    let identity: VocoTextCleanupLoRAIdentity?
    let latencyMilliseconds: Double
    let latencyMs: Double
    let adapterHash: String?
    let adapterPath: String?
    let auditReportSha256: String?
    let applyAllAllowed: Bool?
    let releaseOrDeployAllowed: Bool?

    init(
        schema: String = "voco.text-cleanup-lora-decision.v1",
        eventId: String = UUID().uuidString,
        inputText: String,
        rawTranscript: String? = nil,
        postRuleText: String? = nil,
        outputText: String,
        candidateText: String? = nil,
        mode: VocoTextCleanupLoRAService.Mode,
        chosenAction: String,
        applied: Bool,
        status: String,
        reasonCodes: [String] = [],
        identity: VocoTextCleanupLoRAIdentity? = nil,
        latencyMilliseconds: Double = 0
    ) {
        self.schema = schema
        self.eventId = eventId
        self.inputText = inputText
        self.rawTranscript = rawTranscript
        self.postRuleText = postRuleText ?? inputText
        self.outputText = outputText
        self.finalText = outputText
        self.candidateText = candidateText
        self.generatedText = candidateText
        self.mode = mode
        self.decision = chosenAction
        self.chosenAction = chosenAction
        self.applied = applied
        self.status = status
        self.reasonCodes = reasonCodes
        self.identity = identity
        self.latencyMilliseconds = latencyMilliseconds
        self.latencyMs = latencyMilliseconds
        self.adapterHash = identity?.adapterSHA256
        self.adapterPath = identity?.adapterPath
        self.auditReportSha256 = identity?.auditReportSHA256
        self.applyAllAllowed = identity?.applyAllAllowed
        self.releaseOrDeployAllowed = identity?.releaseOrDeployAllowed
    }
}

enum VocoTextCleanupLoRARuntimeError: Error, LocalizedError, Equatable {
    case timeout
    case loadFailed(String)

    var errorDescription: String? {
        switch self {
        case .timeout:
            return "Text cleanup LoRA timed out"
        case .loadFailed(let message):
            return "Text cleanup LoRA load failed: \(message)"
        }
    }
}

final class VocoTextCleanupLoRAService {
    typealias CandidateGenerator = (VocoTextCleanupLoRARequest) throws -> String?

    static let shared = VocoTextCleanupLoRAService()

    static let modeKey = "VocoTextCleanupLoRAMode"
    static let baseModelPathKey = "VocoTextCleanupLoRABaseModelPath"
    static let adapterPathKey = "VocoTextCleanupLoRAAdapterPath"
    static let maxTokensKey = "VocoTextCleanupLoRAMaxTokens"
    static let auditReportPathKey = "VocoTextCleanupLoRAAuditReportPath"
    static let gateReportPathKey = auditReportPathKey

    static let defaultBaseModelPath = "/Users/jianruicheng/GitHub/VocoReplayLab/local-models/qwen3-asr-1.7b-8bit-text-decoder"
    static let defaultAdapterPath = "/Users/jianruicheng/GitHub/VocoReplayLab/local-adapters/qwen3-asr-cleanup-lora-20260615-v5-focused-promotion/adapters.safetensors"
    static let defaultAuditReportPath = "/Users/jianruicheng/GitHub/VocoReplayLab/artifacts/lora-qwen3-asr-cleanup-20260615-v5-focused-promotion/promotion-audit/report.json"
    static let defaultGateReportPath = defaultAuditReportPath
    static let defaultMaxTokens = 96

    static let expectedAdapterSHA256 = "f253bacd64ab58cbbed73e5b439a49d9e757bc1698be84b402438116cc735932"
    static let expectedBaseModelSHA256 = "163cdc0cddbbe91d0e8e4444bd63cd24288ff9f44427fd7c7955072f8fc1b480"
    static let expectedTokenizerConfigSHA256 = "bfa931e57c356cacf85fc47661fa044684a25c03952cb040d98e7b003550d297"
    static let expectedVocabSHA256 = "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
    static let expectedMergesSHA256 = "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"
    static let expectedAuditReportSHA256 = "4b5233013399f869f3627345df4b03f8c85689aac85527936a1757c3339738e8"
    static let expectedGateReportSHA256 = expectedAuditReportSHA256
    static let v6ComparisonAdapterSHA256 = "09a648760a67705655ab6c7972d9fae06751d58f9644486fdad249964bbeb414"
    static let v6ComparisonAuditReportSHA256 = "32bb24d47affba1a6af5df1b3c7bf0f44ec3944f050fc23751b2f3ed3b61bbae"

    static var defaultEventLogURL: URL {
        AppIdentifiers.appSupportDirectory
            .appendingPathComponent("TextCleanupLoRA", isDirectory: true)
            .appendingPathComponent("text-cleanup-lora-events.jsonl")
    }

    enum Mode: String, Codable, Equatable {
        case off
        case shadow
        case apply
    }

    private struct Runtime {
        let baseModelURL: URL
        let adapterURL: URL
        let model: Qwen3QuantizedTextModel
        let tokenizer: Qwen3Tokenizer
    }

    private enum IdentityError: Error, Equatable {
        case missing(kind: String, url: URL)
        case checksumMismatch(kind: String, url: URL, expected: String, actual: String)
        case auditReportAllowsBroadApplyAll
        case auditReportAllowsRelease
        case auditReportAdapterMismatch(expectedDirectory: String, actualDirectory: String)
        case adapterNotRuntimeApproved(String)
    }

    private let defaults: UserDefaults
    private let eventLogURL: URL?
    private let expectedIdentity: VocoTextCleanupLoRAExpectedIdentity?
    private let candidateGenerator: CandidateGenerator?
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "TextCleanupLoRA")
    private let lock = NSLock()
    private var runtime: Runtime?
    private var identityCache: (baseModelPath: String, adapterPath: String, auditReportPath: String, identity: VocoTextCleanupLoRAIdentity)?

    init(
        defaults: UserDefaults = .standard,
        eventLogURL: URL? = VocoTextCleanupLoRAService.defaultEventLogURL,
        expectedIdentity: VocoTextCleanupLoRAExpectedIdentity? = .production,
        candidateGenerator: CandidateGenerator? = nil
    ) {
        self.defaults = defaults
        self.eventLogURL = eventLogURL
        self.expectedIdentity = expectedIdentity
        self.candidateGenerator = candidateGenerator
    }

    func evaluate(_ text: String, contextHints: [String] = []) -> VocoTextCleanupLoRAEvaluation {
        evaluate(
            text,
            rawTranscript: nil,
            postRuleText: text,
            contextHints: contextHints
        )
    }

    func evaluate(
        _ text: String,
        rawTranscript: String?,
        postRuleText: String,
        contextHints: [String] = []
    ) -> VocoTextCleanupLoRAEvaluation {
        let startedAt = Date()
        let mode = currentMode()
        guard mode != .off else {
            return record(
                noOp(
                    text,
                    rawTranscript: rawTranscript,
                    postRuleText: postRuleText,
                    mode: mode,
                    status: "disabled",
                    reasonCodes: ["disabled"],
                    startedAt: startedAt
                )
            )
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return record(
                noOp(
                    text,
                    rawTranscript: rawTranscript,
                    postRuleText: postRuleText,
                    mode: mode,
                    status: "empty-input",
                    reasonCodes: ["empty-input"],
                    startedAt: startedAt
                )
            )
        }

        do {
            let baseModelURL = baseModelURL()
            let adapterURL = adapterURL()
            let identity = try runtimeIdentity(
                baseModelURL: baseModelURL,
                adapterURL: adapterURL,
                gateReportURL: gateReportURL()
            )
            guard !isActionCommand(text) else {
                return record(
                    noOp(
                        text,
                        rawTranscript: rawTranscript,
                        postRuleText: postRuleText,
                        mode: mode,
                        status: "action-command-bypass",
                        chosenAction: "action-command-bypass",
                        reasonCodes: ["action-command-bypass"],
                        identity: identity,
                        startedAt: startedAt
                    )
                )
            }
            let request = VocoTextCleanupLoRARequest(
                inputText: text,
                rawTranscript: rawTranscript,
                postRuleText: postRuleText,
                contextHints: contextHints,
                baseModelURL: baseModelURL,
                adapterURL: adapterURL,
                maxTokens: maxTokens(),
                identity: identity
            )
            let rawCandidate = try (candidateGenerator ?? generateCandidate)(request)
            let candidate = rawCandidate.flatMap(sanitize)
            guard let candidate, candidate != text else {
                let reasonCodes = candidate == text && shouldPreserveInput(trimmed)
                    ? ["candidate-no-op", "focused-ut-preserve", "apply-all-not-approved"]
                    : ["candidate-no-op"]
                return record(
                    noOp(
                        text,
                        rawTranscript: rawTranscript,
                        postRuleText: postRuleText,
                        mode: mode,
                        candidateText: candidate,
                        status: "no-op",
                        reasonCodes: reasonCodes,
                        identity: identity,
                        startedAt: startedAt
                    )
                )
            }

            let rejectionReasons = safetyRejectionReasons(candidate: candidate, input: text)
            guard rejectionReasons.isEmpty else {
                logger.warning("Text LoRA candidate rejected: \(rejectionReasons.joined(separator: ","), privacy: .public)")
                return record(
                    noOp(
                        text,
                        rawTranscript: rawTranscript,
                        postRuleText: postRuleText,
                        mode: mode,
                        candidateText: candidate,
                        status: "blocked-safety-guard",
                        chosenAction: "block",
                        reasonCodes: rejectionReasons,
                        identity: identity,
                        startedAt: startedAt
                    )
                )
            }

            guard mode == .apply else {
                logger.info("Text LoRA shadow candidate: \(candidate, privacy: .public)")
                return record(
                    noOp(
                        text,
                        rawTranscript: rawTranscript,
                        postRuleText: postRuleText,
                        mode: mode,
                        candidateText: candidate,
                        status: "shadow-candidate",
                        chosenAction: "shadow",
                        reasonCodes: ["shadow-candidate"],
                        identity: identity,
                        startedAt: startedAt
                    )
                )
            }

            guard identity?.adapterSHA256 != Self.v6ComparisonAdapterSHA256 else {
                return record(
                    noOp(
                        text,
                        rawTranscript: rawTranscript,
                        postRuleText: postRuleText,
                        mode: mode,
                        candidateText: candidate,
                        status: "shadow-only-comparison-adapter",
                        chosenAction: "shadow",
                        reasonCodes: ["v6-comparison-only", "apply-all-not-approved"],
                        identity: identity,
                        startedAt: startedAt
                    )
                )
            }

            let focusedReasons = focusedRuntimeReasonCodes(input: text, candidate: candidate)
            guard !focusedReasons.isEmpty else {
                return record(
                    noOp(
                        text,
                        rawTranscript: rawTranscript,
                        postRuleText: postRuleText,
                        mode: mode,
                        candidateText: candidate,
                        status: "blocked-broad-apply-all",
                        chosenAction: "block",
                        reasonCodes: ["broad-apply-all-blocked", "focused-scope-required"],
                        identity: identity,
                        startedAt: startedAt
                    )
                )
            }

            return record(
                VocoTextCleanupLoRAEvaluation(
                    inputText: text,
                    rawTranscript: rawTranscript,
                    postRuleText: postRuleText,
                    outputText: candidate,
                    candidateText: candidate,
                    mode: mode,
                    chosenAction: "apply",
                    applied: true,
                    status: "applied",
                    reasonCodes: ["candidate-safe", "focused-runtime-lora", "apply-all-not-approved"] + focusedReasons,
                    identity: identity,
                    latencyMilliseconds: elapsedMilliseconds(since: startedAt)
                )
            )
        } catch {
            let fallback = fallbackReason(for: error)
            logger.error("Text LoRA fallback: \(fallback.status, privacy: .public) \(String(describing: error), privacy: .public)")
            return record(
                noOp(
                    text,
                    rawTranscript: rawTranscript,
                    postRuleText: postRuleText,
                    mode: mode,
                    status: fallback.status,
                    chosenAction: "fallback",
                    reasonCodes: fallback.reasonCodes,
                    startedAt: startedAt
                )
            )
        }
    }

    private func noOp(
        _ text: String,
        rawTranscript: String?,
        postRuleText: String,
        mode: Mode,
        candidateText: String? = nil,
        status: String,
        chosenAction: String = "noop",
        reasonCodes: [String],
        identity: VocoTextCleanupLoRAIdentity? = nil,
        startedAt: Date
    ) -> VocoTextCleanupLoRAEvaluation {
        VocoTextCleanupLoRAEvaluation(
            inputText: text,
            rawTranscript: rawTranscript,
            postRuleText: postRuleText,
            outputText: text,
            candidateText: candidateText,
            mode: mode,
            chosenAction: chosenAction,
            applied: false,
            status: status,
            reasonCodes: reasonCodes,
            identity: identity,
            latencyMilliseconds: elapsedMilliseconds(since: startedAt)
        )
    }

    private func record(_ evaluation: VocoTextCleanupLoRAEvaluation) -> VocoTextCleanupLoRAEvaluation {
        guard let eventLogURL else { return evaluation }
        do {
            try FileManager.default.createDirectory(
                at: eventLogURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            var data = try encoder.encode(evaluation)
            data.append(0x0A)
            if FileManager.default.fileExists(atPath: eventLogURL.path) {
                let handle = try FileHandle(forWritingTo: eventLogURL)
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: data)
            } else {
                try data.write(to: eventLogURL, options: .atomic)
            }
        } catch {
            logger.error("Failed to write Text LoRA event: \(error.localizedDescription, privacy: .public)")
        }
        return evaluation
    }

    private func currentMode() -> Mode {
        Mode(rawValue: defaults.string(forKey: Self.modeKey) ?? Mode.off.rawValue) ?? .off
    }

    private func baseModelURL() -> URL {
        URL(fileURLWithPath: defaults.string(forKey: Self.baseModelPathKey) ?? Self.defaultBaseModelPath)
    }

    private func adapterURL() -> URL {
        URL(fileURLWithPath: defaults.string(forKey: Self.adapterPathKey) ?? Self.defaultAdapterPath)
    }

    private func gateReportURL() -> URL {
        URL(fileURLWithPath: defaults.string(forKey: Self.auditReportPathKey) ?? Self.defaultAuditReportPath)
    }

    private func maxTokens() -> Int {
        let value = defaults.integer(forKey: Self.maxTokensKey)
        return value > 0 ? value : Self.defaultMaxTokens
    }

    private func runtimeIdentity(
        baseModelURL: URL,
        adapterURL: URL,
        gateReportURL: URL
    ) throws -> VocoTextCleanupLoRAIdentity? {
        guard let expectedIdentity else { return nil }
        let resolvedAdapterURL = resolvedAdapterURL(adapterURL)

        lock.lock()
        defer { lock.unlock() }

        if let identityCache,
           identityCache.baseModelPath == baseModelURL.path,
           identityCache.adapterPath == resolvedAdapterURL.path,
           identityCache.auditReportPath == gateReportURL.path {
            return identityCache.identity
        }

        let adapterSHA = try verifiedSHA256(
            kind: "adapter",
            url: resolvedAdapterURL,
            expected: expectedAdapterSHA(for: resolvedAdapterURL, defaultExpected: expectedIdentity.adapterSHA256),
            required: expectedIdentity.adapterSHA256 != nil
        )
        let baseModelSHA = try verifiedSHA256(
            kind: "base-model",
            url: baseModelURL.appendingPathComponent("model.safetensors"),
            expected: expectedIdentity.baseModelSHA256,
            required: expectedIdentity.baseModelSHA256 != nil
        )
        let tokenizerConfigSHA = try verifiedSHA256(
            kind: "tokenizer-config",
            url: baseModelURL.appendingPathComponent("tokenizer_config.json"),
            expected: expectedIdentity.tokenizerConfigSHA256,
            required: expectedIdentity.tokenizerConfigSHA256 != nil
        )
        let vocabSHA = try verifiedSHA256(
            kind: "vocab",
            url: baseModelURL.appendingPathComponent("vocab.json"),
            expected: expectedIdentity.vocabSHA256,
            required: expectedIdentity.vocabSHA256 != nil
        )
        let mergesSHA = try verifiedSHA256(
            kind: "merges",
            url: baseModelURL.appendingPathComponent("merges.txt"),
            expected: expectedIdentity.mergesSHA256,
            required: expectedIdentity.mergesSHA256 != nil
        )
        let gateSHA = try verifiedSHA256(
            kind: "audit-report",
            url: gateReportURL,
            expected: expectedAuditSHA(for: resolvedAdapterURL, defaultExpected: expectedIdentity.auditReportSHA256),
            required: false
        )
        let audit = try auditReportMetadata(
            from: gateReportURL,
            resolvedAdapterURL: resolvedAdapterURL
        )

        let identity = VocoTextCleanupLoRAIdentity(
            baseModelPath: baseModelURL.path,
            adapterPath: resolvedAdapterURL.path,
            auditReportPath: gateReportURL.path,
            adapterVersion: adapterVersion(for: adapterSHA),
            promotedApplyClass: audit.promotedApplyClass,
            applyAllAllowed: audit.applyAllAllowed,
            releaseOrDeployAllowed: audit.releaseOrDeployAllowed,
            adapterSHA256: adapterSHA,
            baseModelSHA256: baseModelSHA,
            tokenizerConfigSHA256: tokenizerConfigSHA,
            vocabSHA256: vocabSHA,
            mergesSHA256: mergesSHA,
            auditReportSHA256: gateSHA
        )
        identityCache = (
            baseModelPath: baseModelURL.path,
            adapterPath: resolvedAdapterURL.path,
            auditReportPath: gateReportURL.path,
            identity: identity
        )
        return identity
    }

    private func expectedAdapterSHA(for adapterURL: URL, defaultExpected: String?) -> String? {
        if adapterURL.path.contains("qwen3-asr-cleanup-lora-20260615-v6-apply-all-microstep") {
            return Self.v6ComparisonAdapterSHA256
        }
        return defaultExpected
    }

    private func expectedAuditSHA(for adapterURL: URL, defaultExpected: String?) -> String? {
        if adapterURL.path.contains("qwen3-asr-cleanup-lora-20260615-v6-apply-all-microstep") {
            return Self.v6ComparisonAuditReportSHA256
        }
        return defaultExpected
    }

    private func adapterVersion(for adapterSHA: String?) -> String? {
        switch adapterSHA {
        case .some(Self.expectedAdapterSHA256):
            return "v5-focused-promotion"
        case .some(Self.v6ComparisonAdapterSHA256):
            return "v6-apply-all-microstep-comparison"
        default:
            return nil
        }
    }

    private func auditReportMetadata(
        from url: URL,
        resolvedAdapterURL: URL
    ) throws -> VocoTextCleanupLoRAAuditMetadata {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw IdentityError.missing(kind: "audit-report", url: url)
        }

        let data = try Data(contentsOf: url)
        let report = try JSONDecoder().decode(VocoTextCleanupLoRAAuditReport.self, from: data)
        let metadata = report.metadata
        if metadata.applyAllAllowed {
            throw IdentityError.auditReportAllowsBroadApplyAll
        }
        if metadata.releaseOrDeployAllowed {
            throw IdentityError.auditReportAllowsRelease
        }
        if let adapterDir = metadata.adapterDirectory {
            let expectedDirectory = resolvedAdapterURL.deletingLastPathComponent().path
            guard adapterDir == expectedDirectory else {
                throw IdentityError.auditReportAdapterMismatch(
                    expectedDirectory: expectedDirectory,
                    actualDirectory: adapterDir
                )
            }
        }
        return metadata
    }

    private func verifiedSHA256(kind: String, url: URL, expected: String?, required: Bool) throws -> String? {
        guard FileManager.default.fileExists(atPath: url.path) else {
            if required {
                throw IdentityError.missing(kind: kind, url: url)
            }
            return nil
        }

        guard let expected else { return nil }
        let actual = try Self.sha256Hex(of: url)
        guard actual == expected else {
            throw IdentityError.checksumMismatch(kind: kind, url: url, expected: expected, actual: actual)
        }
        return actual
    }

    private func generateCandidate(for request: VocoTextCleanupLoRARequest) throws -> String? {
        lock.lock()
        defer { lock.unlock() }

        let runtime = try loadRuntimeIfNeeded(baseModelURL: request.baseModelURL, adapterURL: request.adapterURL)
        let inputIds = buildChatInputIds(
            text: request.inputText,
            contextHints: request.contextHints,
            tokenizer: runtime.tokenizer
        )
        let generated = try generate(inputIds: inputIds, runtime: runtime, maxTokens: request.maxTokens)
        return generated
    }

    private func loadRuntimeIfNeeded(baseModelURL: URL, adapterURL: URL) throws -> Runtime {
        let resolvedAdapterURL = resolvedAdapterURL(adapterURL)
        if let runtime,
           runtime.baseModelURL.path == baseModelURL.path,
           runtime.adapterURL.path == resolvedAdapterURL.path {
            return runtime
        }

        logger.info("Loading text cleanup LoRA base model: \(baseModelURL.path, privacy: .public)")
        let config = try textDecoderConfig(from: baseModelURL)
        let tokenizer = Qwen3Tokenizer()
        try tokenizer.load(from: baseModelURL.appendingPathComponent("vocab.json"))

        let model = Qwen3QuantizedTextModel(config: config)
        try Qwen3WeightLoader.loadTextDecoderWeights(into: model, from: baseModelURL)
        model.loraStore = try Qwen3TextDecoderLoRAStore.load(from: resolvedAdapterURL, config: config)

        let loaded = Runtime(
            baseModelURL: baseModelURL,
            adapterURL: resolvedAdapterURL,
            model: model,
            tokenizer: tokenizer
        )
        runtime = loaded
        Memory.clearCache()
        return loaded
    }

    private func buildChatInputIds(
        text: String,
        contextHints: [String],
        tokenizer: Qwen3Tokenizer
    ) -> [Int32] {
        let tokens = Qwen3ASRTokens.self
        let systemPrompt = [
            "You are Voco text cleanup.",
            "Return only the corrected final text.",
            "Do not explain.",
            "Preserve the user's language, punctuation, and meaning."
        ].joined(separator: " ")

        var ids: [Int32] = []
        ids.append(contentsOf: [tokens.imStartTokenId, tokens.systemId, tokens.newlineId].map(Int32.init))
        ids.append(contentsOf: tokenizer.encode(systemPrompt).map(Int32.init))
        ids.append(contentsOf: [tokens.imEndTokenId, tokens.newlineId].map(Int32.init))

        ids.append(contentsOf: [tokens.imStartTokenId, tokens.userId, tokens.newlineId].map(Int32.init))
        if !contextHints.isEmpty {
            ids.append(contentsOf: tokenizer.encode(contextHints.joined(separator: "\n")).map(Int32.init))
            ids.append(Int32(tokens.newlineId))
        }
        ids.append(contentsOf: tokenizer.encode(text).map(Int32.init))
        ids.append(contentsOf: [tokens.imEndTokenId, tokens.newlineId].map(Int32.init))

        ids.append(contentsOf: [tokens.imStartTokenId, tokens.assistantId, tokens.newlineId].map(Int32.init))
        return ids
    }

    private func generate(inputIds: [Int32], runtime: Runtime, maxTokens: Int) throws -> String {
        var cache: [(MLXArray, MLXArray)]?
        var generatedTokens: [Int32] = []

        var (hiddenStates, newCache) = try runtime.model(
            inputIds: MLXArray(inputIds).expandedDimensions(axis: 0),
            cache: cache
        )
        cache = newCache

        var token = nextToken(from: hiddenStates, model: runtime.model)
        var tokenIndex = 0
        while tokenIndex < maxTokens {
            if isStopToken(token) {
                break
            }
            generatedTokens.append(token)

            let tokenEmbeds = runtime.model.embedTokens(MLXArray([token]).expandedDimensions(axis: 0))
            (hiddenStates, newCache) = try runtime.model(inputsEmbeds: tokenEmbeds, cache: cache)
            cache = newCache
            token = nextToken(from: hiddenStates, model: runtime.model)
            tokenIndex += 1
        }

        if let cache {
            eval(cache.map { [$0.0, $0.1] }.flatMap { $0 })
        }
        return runtime.tokenizer.decode(tokens: generatedTokens.map(Int.init))
    }

    private func nextToken(from hiddenStates: MLXArray, model: Qwen3QuantizedTextModel) -> Int32 {
        let seqLen = hiddenStates.dim(1)
        let lastHidden = hiddenStates[0..., (seqLen - 1)..<seqLen, 0...]
        let logits = model.embedTokens.asLinear(lastHidden)
        return argMax(logits.reshaped(-1)).item(Int32.self)
    }

    private func isStopToken(_ token: Int32) -> Bool {
        token == Int32(Qwen3ASRTokens.eosTokenId) || token == Int32(Qwen3ASRTokens.padTokenId)
    }

    private func sanitize(_ text: String) -> String? {
        var candidate = text
            .replacingOccurrences(of: "<|im_end|>", with: "")
            .replacingOccurrences(of: "<|endoftext|>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)

        if let range = candidate.range(of: "<|im_start|>") {
            candidate = String(candidate[..<range.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return candidate.isEmpty ? nil : candidate
    }

    private func focusedRuntimeReasonCodes(input: String, candidate: String) -> [String] {
        let trimmedInput = input.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedCandidate = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        var reasons: [String] = []

        if isFocusedVocoBrandContext(input: trimmedInput, candidate: trimmedCandidate) {
            reasons.append("focused-voco-brand-context")
        }
        if isFocusedGoalAcronym(input: trimmedInput, candidate: trimmedCandidate) {
            reasons.append("focused-goal-acronym")
        }
        if isFocusedLiSAJapaneseQuote(input: trimmedInput, candidate: trimmedCandidate) {
            reasons.append("focused-lisa-japanese-quote")
        }
        if isFocusedUTPreserve(input: trimmedInput, candidate: trimmedCandidate) {
            reasons.append("focused-ut-preserve")
        }

        return reasons
    }

    private func isFocusedVocoBrandContext(input: String, candidate: String) -> Bool {
        guard candidate.localizedCaseInsensitiveContains("Voco") else { return false }
        let foldedInput = foldedASCII(input)
        let hasSpelledAlias = foldedInput.contains("boco") || foldedInput.contains("boceo")
        let hasMandarinASRDrift = input.contains("報考") || input.contains("宝考") || input.contains("接回報考")
        let contextLooksLikeVocoRuntime = input.localizedCaseInsensitiveContains("Voco") ||
            input.contains("規則") ||
            input.contains("规则") ||
            input.contains("接回") ||
            input.contains("格式") ||
            input.contains("管線")
        return (hasSpelledAlias || hasMandarinASRDrift) && contextLooksLikeVocoRuntime
    }

    private func isFocusedGoalAcronym(input: String, candidate: String) -> Bool {
        input.contains("Goal. GOAL") && candidate.contains("Goal. G O A L")
    }

    private func isFocusedLiSAJapaneseQuote(input: String, candidate: String) -> Bool {
        let hasLiSAInput = input.localizedCaseInsensitiveContains("lisa") || input.contains("麗莎") || input.contains("理沙")
        let hasLiSATarget = candidate.contains("LiSA")
        let hasJapaneseQuote = input.contains("今日もいい日だ") || candidate.contains("今日もいい日だ")
        return hasLiSAInput && hasLiSATarget && hasJapaneseQuote
    }

    private func isFocusedUTPreserve(input: String, candidate: String) -> Bool {
        shouldPreserveInput(input) && input == candidate
    }

    private func safetyRejectionReasons(candidate: String, input: String) -> [String] {
        var reasons: [String] = []
        let trimmedInput = input.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedCandidate = candidate.trimmingCharacters(in: .whitespacesAndNewlines)

        if trimmedCandidate.isEmpty {
            reasons.append("candidate-empty")
        }
        if trimmedCandidate.count > max(trimmedInput.count * 3, trimmedInput.count + 24) {
            reasons.append("candidate-too-long")
        }
        if isActionCommand(trimmedCandidate) {
            reasons.append("candidate-action-command")
        }
        if shouldPreserveInput(trimmedInput), trimmedCandidate != trimmedInput {
            reasons.append("protected-noop-runtime-input")
        }
        if isLiteralSpellingNegative(trimmedInput), trimmedCandidate != trimmedInput {
            reasons.append("literal-spelling-negative")
        }

        let validation = LLMResponseValidator.shared.validate(
            response: trimmedCandidate,
            original: trimmedInput
        )
        for reason in validation.reasons where isBlockingValidationReason(reason, input: trimmedInput, candidate: trimmedCandidate) {
            reasons.append("validator-\(reason)")
        }

        return deduplicated(reasons)
    }

    private func isBlockingValidationReason(_ reason: String, input: String, candidate: String) -> Bool {
        if reason.hasPrefix("dropped-term") || reason == "short-edit-budget" || reason == "content-drift" {
            return !isAllowedTechnicalRewrite(input: input, candidate: candidate)
        }
        return true
    }

    private func isAllowedTechnicalRewrite(input: String, candidate: String) -> Bool {
        if foldedAlnum(input) == foldedAlnum(candidate) {
            return true
        }

        let foldedInput = foldedAlnum(input)
        let foldedCandidate = foldedAlnum(candidate)
        if isSingleScriptCJK(input), isSingleScriptCJK(candidate), !foldedInput.isEmpty, !foldedCandidate.isEmpty {
            let distance = levenshteinDistance(Array(foldedInput), Array(foldedCandidate))
            let maxLength = max(foldedInput.count, foldedCandidate.count)
            if maxLength >= 5, distance <= 2 {
                return true
            }
        }

        let inputASCII = foldedASCII(input)
        let candidateASCII = foldedASCII(candidate)
        if candidateASCII.contains("voco"),
           inputASCII.contains("boco") || inputASCII.contains("boceo") {
            return true
        }

        guard !inputASCII.isEmpty, !candidateASCII.isEmpty else { return false }
        let distance = levenshteinDistance(Array(inputASCII), Array(candidateASCII))
        let maxLength = max(inputASCII.count, candidateASCII.count)
        return maxLength <= 12 && distance <= max(1, maxLength / 4)
    }

    private func isSingleScriptCJK(_ text: String) -> Bool {
        let content = text.filter { $0.isLetter || $0.isNumber }
        guard !content.isEmpty else { return false }
        return content.allSatisfy { character in
            character.unicodeScalars.contains { scalar in
                (0x4E00...0x9FFF).contains(scalar.value) ||
                    (0x3400...0x4DBF).contains(scalar.value) ||
                    (0x20000...0x2A6DF).contains(scalar.value)
            }
        }
    }

    private func shouldPreserveInput(_ input: String) -> Bool {
        ["UT。", "UT.", "UT"].contains(input)
    }

    private func isLiteralSpellingNegative(_ input: String) -> Bool {
        let markers = ["逐字拼寫", "逐字拼写", "不是產品名稱", "不是产品名称", "不是品牌"]
        guard markers.contains(where: input.contains) else { return false }
        return input.contains { character in
            character.unicodeScalars.contains { scalar in
                scalar.isASCII && CharacterSet.letters.contains(scalar)
            }
        }
    }

    private func isActionCommand(_ text: String) -> Bool {
        let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if normalized.contains("全部刪除") || normalized.contains("全部删除") {
            return true
        }
        return VoiceCommandService.shared.detectCommand(in: text) != nil ||
            VoiceCommandService.shared.detectEditModeCommand(in: text) != nil
    }

    private func textDecoderConfig(from modelURL: URL) throws -> Qwen3TextDecoderConfig {
        let configURL = modelURL.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let config = try? JSONDecoder().decode(HuggingFaceTextDecoderConfig.self, from: data) else {
            return .large
        }

        var decoder = Qwen3TextDecoderConfig()
        decoder.vocabSize = config.vocabSize
        decoder.hiddenSize = config.hiddenSize
        decoder.numLayers = config.numHiddenLayers
        decoder.numHeads = config.numAttentionHeads
        decoder.numKVHeads = config.numKeyValueHeads
        decoder.headDim = config.headDim
        decoder.intermediateSize = config.intermediateSize
        decoder.maxPositionEmbeddings = config.maxPositionEmbeddings
        decoder.rmsNormEps = config.rmsNormEps
        decoder.ropeTheta = config.ropeTheta
        decoder.tieWordEmbeddings = config.tieWordEmbeddings
        decoder.groupSize = config.quantizationConfig?.groupSize ?? config.quantization?.groupSize ?? decoder.groupSize
        decoder.bits = config.quantizationConfig?.bits ?? config.quantization?.bits ?? decoder.bits
        return decoder
    }

    private func fallbackReason(for error: Error) -> (status: String, reasonCodes: [String]) {
        if let identityError = error as? IdentityError {
            switch identityError {
            case .missing(let kind, _):
                return ("fallback-\(kind)-missing", ["\(kind)-missing"])
            case .checksumMismatch(let kind, _, _, _):
                return ("fallback-\(kind)-hash-mismatch", ["\(kind)-hash-mismatch"])
            case .auditReportAllowsBroadApplyAll:
                return ("fallback-audit-broad-apply-all", ["audit-broad-apply-all"])
            case .auditReportAllowsRelease:
                return ("fallback-audit-release-allowed", ["audit-release-allowed"])
            case .auditReportAdapterMismatch:
                return ("fallback-audit-adapter-mismatch", ["audit-adapter-mismatch"])
            case .adapterNotRuntimeApproved:
                return ("fallback-adapter-not-runtime-approved", ["adapter-not-runtime-approved"])
            }
        }

        if let runtimeError = error as? VocoTextCleanupLoRARuntimeError {
            switch runtimeError {
            case .timeout:
                return ("fallback-timeout", ["timeout"])
            case .loadFailed:
                return ("fallback-load-failure", ["load-failure"])
            }
        }

        return ("fallback-runtime-error", ["runtime-error"])
    }

    private func resolvedAdapterURL(_ adapterURL: URL) -> URL {
        adapterURL.pathExtension == "safetensors"
            ? adapterURL
            : adapterURL.appendingPathComponent("adapters.safetensors")
    }

    private func elapsedMilliseconds(since start: Date) -> Double {
        max(0, Date().timeIntervalSince(start) * 1_000)
    }

    private func foldedAlnum(_ text: String) -> String {
        OpenCCConverter.shared.convert(text).lowercased().filter { $0.isLetter || $0.isNumber }
    }

    private func foldedASCII(_ text: String) -> String {
        text.lowercased().filter { character in
            character.unicodeScalars.allSatisfy { scalar in
                scalar.isASCII && (CharacterSet.letters.contains(scalar) || CharacterSet.decimalDigits.contains(scalar))
            }
        }
    }

    private func deduplicated(_ reasons: [String]) -> [String] {
        var seen: Set<String> = []
        return reasons.filter { seen.insert($0).inserted }
    }

    private func levenshteinDistance(_ lhs: [Character], _ rhs: [Character]) -> Int {
        guard !lhs.isEmpty else { return rhs.count }
        guard !rhs.isEmpty else { return lhs.count }

        var previous = Array(0...rhs.count)
        for (i, left) in lhs.enumerated() {
            var current = [i + 1]
            for (j, right) in rhs.enumerated() {
                current.append(
                    left == right
                        ? previous[j]
                        : min(previous[j], previous[j + 1], current[j]) + 1
                )
            }
            previous = current
        }
        return previous[rhs.count]
    }

    private static func sha256Hex(of url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}

private struct HuggingFaceTextDecoderConfig: Decodable {
    struct Quantization: Decodable {
        let bits: Int
        let groupSize: Int

        enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
        }
    }

    let vocabSize: Int
    let hiddenSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let intermediateSize: Int
    let maxPositionEmbeddings: Int
    let rmsNormEps: Float
    let ropeTheta: Float
    let tieWordEmbeddings: Bool
    let quantization: Quantization?
    let quantizationConfig: Quantization?

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case quantization
        case quantizationConfig = "quantization_config"
    }
}

private struct VocoTextCleanupLoRAAuditMetadata: Equatable {
    let adapterDirectory: String?
    let promotedApplyClass: String?
    let applyAllAllowed: Bool
    let releaseOrDeployAllowed: Bool
}

private struct VocoTextCleanupLoRAAuditReport: Decodable {
    struct Decision: Decodable {
        let applyAllAllowed: Bool?
        let applyAllPromoteAllowed: Bool?
        let promotedApplyClass: String?
        let releaseOrDeployAllowed: Bool?
    }

    struct Boundary: Decodable {
        let releaseOrDeployAllowed: Bool?
    }

    let adapterDir: String?
    let decision: Decision?
    let boundary: Boundary?

    var metadata: VocoTextCleanupLoRAAuditMetadata {
        let applyAllAllowed = decision?.applyAllAllowed ??
            decision?.applyAllPromoteAllowed ??
            false
        let releaseOrDeployAllowed = decision?.releaseOrDeployAllowed ??
            boundary?.releaseOrDeployAllowed ??
            false
        return VocoTextCleanupLoRAAuditMetadata(
            adapterDirectory: adapterDir,
            promotedApplyClass: decision?.promotedApplyClass,
            applyAllAllowed: applyAllAllowed,
            releaseOrDeployAllowed: releaseOrDeployAllowed
        )
    }
}
