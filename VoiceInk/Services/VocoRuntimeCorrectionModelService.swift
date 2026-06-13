import Foundation
import CryptoKit
import OSLog

struct VocoRuntimeCorrectionCandidate: Codable, Equatable {
    let source: String
    let target: String
    let score: Double?

    init(source: String, target: String, score: Double? = nil) {
        self.source = source
        self.target = target
        self.score = score
    }
}

struct VocoRuntimeCorrectionFeatures: Equatable {
    let rawTranscript: String
    let canonicalizedText: String
    let postRuleText: String
    let contextHints: [String]
    let appMode: String?
    let deterministicRuleFires: [VocoAutoApplyPolicyFire]
    let actionCommand: Bool
    let protectedTermHits: [String]
    let candidateSpans: [VocoRuntimeCorrectionCandidate]

    init(
        rawTranscript: String,
        canonicalizedText: String,
        postRuleText: String,
        contextHints: [String] = [],
        appMode: String? = nil,
        deterministicRuleFires: [VocoAutoApplyPolicyFire] = [],
        actionCommand: Bool = false,
        protectedTermHits: [String] = [],
        candidateSpans: [VocoRuntimeCorrectionCandidate] = []
    ) {
        self.rawTranscript = rawTranscript
        self.canonicalizedText = canonicalizedText
        self.postRuleText = postRuleText
        self.contextHints = contextHints
        self.appMode = appMode
        self.deterministicRuleFires = deterministicRuleFires
        self.actionCommand = actionCommand
        self.protectedTermHits = protectedTermHits
        self.candidateSpans = candidateSpans
    }
}

struct VocoRuntimeCorrectionDecision: Codable, Equatable {
    let schema: String
    let runtimeMode: String
    let chosenAction: String
    let fallbackReason: String
    let reasonCodes: [String]
    let score: Double?
    let rawTranscript: String
    let postRuleText: String
    let finalText: String
    let modelArtifactId: String
    let modelSha256: String
    let candidateGeneratorSha256: String
    let candidates: [VocoRuntimeCorrectionCandidate]
    let appliedCandidate: VocoRuntimeCorrectionCandidate?
}

struct VocoRuntimeCorrectionEvaluation: Equatable {
    let inputText: String
    let outputText: String
    let decision: VocoRuntimeCorrectionDecision?

    var changed: Bool { inputText != outputText }
}

struct VocoRuntimeCorrectionModelStatus: Equatable {
    let isAvailable: Bool
    let message: String
    let artifactURL: URL
}

final class VocoRuntimeCorrectionModelService {
    static let shared = VocoRuntimeCorrectionModelService()
    static let enabledKey = "VocoRuntimeCorrectionModelEnabled"
    static let artifactFileName = "runtime-correction-artifact.json"
    static let eventLogFileName = "runtime-correction-shadow-events.jsonl"

    static var defaultModelDirectory: URL {
        AppIdentifiers.appSupportDirectory
            .appendingPathComponent("RuntimeCorrectionModels", isDirectory: true)
    }

    static var defaultArtifactURL: URL {
        defaultModelDirectory.appendingPathComponent(artifactFileName)
    }

    static var defaultEventLogURL: URL {
        defaultModelDirectory.appendingPathComponent(eventLogFileName)
    }

    private let artifactURL: URL
    private let eventLogURL: URL?
    private let defaults: UserDefaults
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "RuntimeCorrectionModel")
    private var loadedArtifact: VocoRuntimeCorrectionArtifact?
    private var loadedCandidateSpanModel: VocoRuntimeCandidateSpanModel?

    private(set) var status: VocoRuntimeCorrectionModelStatus

    var isUserEnabled: Bool {
        get { defaults.object(forKey: Self.enabledKey) as? Bool ?? false }
        set { defaults.set(newValue, forKey: Self.enabledKey) }
    }

    var isShadowEnabled: Bool {
        isUserEnabled && status.isAvailable && loadedArtifact?.runtimeMode == "shadow"
    }

    var isGatedApplyEnabled: Bool {
        isUserEnabled && status.isAvailable && loadedArtifact?.runtimeMode == "gatedApply"
    }

    init(
        artifactURL: URL = VocoRuntimeCorrectionModelService.defaultArtifactURL,
        eventLogURL: URL? = VocoRuntimeCorrectionModelService.defaultEventLogURL,
        defaults: UserDefaults = .standard
    ) {
        self.artifactURL = artifactURL
        self.eventLogURL = eventLogURL
        self.defaults = defaults
        self.status = VocoRuntimeCorrectionModelStatus(
            isAvailable: false,
            message: String(localized: "Runtime correction model not detected"),
            artifactURL: artifactURL
        )
        reload()
    }

    func reload() {
        guard artifactURL.pathExtension != "joblib" else {
            loadedArtifact = nil
            loadedCandidateSpanModel = nil
            status = VocoRuntimeCorrectionModelStatus(
                isAvailable: false,
                message: "Runtime correction model requires an explicit artifact manifest, not a joblib ranker",
                artifactURL: artifactURL
            )
            return
        }

        guard FileManager.default.fileExists(atPath: artifactURL.path) else {
            loadedArtifact = nil
            loadedCandidateSpanModel = nil
            status = VocoRuntimeCorrectionModelStatus(
                isAvailable: false,
                message: String(localized: "Runtime correction model not detected"),
                artifactURL: artifactURL
            )
            return
        }

        do {
            let data = try Data(contentsOf: artifactURL)
            let artifact = try JSONDecoder().decode(VocoRuntimeCorrectionArtifact.self, from: data)
            let baseURL = artifactURL.deletingLastPathComponent()
            try artifact.validateRuntimeContract(baseURL: baseURL)
            loadedCandidateSpanModel = try artifact.loadCandidateSpanModel(baseURL: baseURL)
            loadedArtifact = artifact
            let modeLabel = artifact.runtimeMode == "gatedApply" ? "gated apply" : "shadow"
            status = VocoRuntimeCorrectionModelStatus(
                isAvailable: true,
                message: "Runtime correction \(modeLabel) contract loaded",
                artifactURL: artifactURL
            )
        } catch {
            loadedArtifact = nil
            loadedCandidateSpanModel = nil
            status = VocoRuntimeCorrectionModelStatus(
                isAvailable: false,
                message: error.localizedDescription,
                artifactURL: artifactURL
            )
            logger.error("Failed to load runtime correction artifact: \(error.localizedDescription, privacy: .public)")
        }
    }

    func evaluate(_ features: VocoRuntimeCorrectionFeatures) -> VocoRuntimeCorrectionEvaluation {
        guard isUserEnabled, status.isAvailable, let artifact = loadedArtifact else {
            return VocoRuntimeCorrectionEvaluation(
                inputText: features.postRuleText,
                outputText: features.postRuleText,
                decision: nil
            )
        }

        let decision: VocoRuntimeCorrectionDecision
        if features.actionCommand || VoiceCommandService.shared.detectCommand(in: features.postRuleText) != nil {
            decision = makeDecision(
                artifact: artifact,
                features: features,
                chosenAction: "block",
                fallbackReason: "action-command-bypass",
                reasonCodes: ["action-command-bypass"]
            )
        } else if artifact.runtimeMode == "gatedApply" {
            decision = makeGatedApplyDecision(
                artifact: artifact,
                features: featuresWithRuntimeCandidates(features)
            )
        } else {
            decision = makeDecision(
                artifact: artifact,
                features: features,
                chosenAction: "noop",
                fallbackReason: "shadow-contract-fixture-no-runtime-model",
                reasonCodes: ["shadow-only"]
            )
        }

        appendDecisionEvent(decision)
        return VocoRuntimeCorrectionEvaluation(
            inputText: features.postRuleText,
            outputText: decision.finalText,
            decision: decision
        )
    }

    private func featuresWithRuntimeCandidates(
        _ features: VocoRuntimeCorrectionFeatures
    ) -> VocoRuntimeCorrectionFeatures {
        guard features.candidateSpans.isEmpty,
              let loadedCandidateSpanModel
        else {
            return features
        }

        return VocoRuntimeCorrectionFeatures(
            rawTranscript: features.rawTranscript,
            canonicalizedText: features.canonicalizedText,
            postRuleText: features.postRuleText,
            contextHints: features.contextHints,
            appMode: features.appMode,
            deterministicRuleFires: features.deterministicRuleFires,
            actionCommand: features.actionCommand,
            protectedTermHits: features.protectedTermHits,
            candidateSpans: loadedCandidateSpanModel.candidates(for: features)
        )
    }

    private func makeGatedApplyDecision(
        artifact: VocoRuntimeCorrectionArtifact,
        features: VocoRuntimeCorrectionFeatures
    ) -> VocoRuntimeCorrectionDecision {
        if !features.deterministicRuleFires.isEmpty && artifact.safety.jsonExactRulePriority {
            return makeDecision(
                artifact: artifact,
                features: features,
                chosenAction: "block",
                fallbackReason: "deterministic-rule-priority",
                reasonCodes: ["deterministic-rule-priority", "not-worse-than-compiled-json"]
            )
        }

        if !features.protectedTermHits.isEmpty {
            return makeDecision(
                artifact: artifact,
                features: features,
                chosenAction: "block",
                fallbackReason: "protected-term-bypass",
                reasonCodes: ["protected-term-bypass"]
            )
        }

        let threshold = artifact.thresholdConfig?.gatedApply ?? 1.0
        guard let candidate = features.candidateSpans
            .filter({ ($0.score ?? 0) >= threshold })
            .sorted(by: { ($0.score ?? 0) > ($1.score ?? 0) })
            .first
        else {
            return makeDecision(
                artifact: artifact,
                features: features,
                chosenAction: "noop",
                fallbackReason: "no-candidate-above-gated-threshold",
                reasonCodes: ["score-below-threshold"]
            )
        }

        guard !candidate.source.isEmpty,
              !candidate.target.isEmpty,
              features.postRuleText.contains(candidate.source)
        else {
            return makeDecision(
                artifact: artifact,
                features: features,
                chosenAction: "noop",
                fallbackReason: "candidate-source-not-found",
                reasonCodes: ["candidate-source-not-found"],
                score: candidate.score,
                appliedCandidate: candidate
            )
        }

        let finalText = features.postRuleText.replacingOccurrences(of: candidate.source, with: candidate.target)
        guard finalText != features.postRuleText else {
            return makeDecision(
                artifact: artifact,
                features: features,
                chosenAction: "noop",
                fallbackReason: "candidate-does-not-change-output",
                reasonCodes: ["candidate-noop"],
                score: candidate.score,
                appliedCandidate: candidate
            )
        }

        return makeDecision(
            artifact: artifact,
            features: features,
            chosenAction: "apply",
            fallbackReason: "",
            reasonCodes: ["gated-apply", "not-worse-than-compiled-json"],
            finalText: finalText,
            score: candidate.score,
            appliedCandidate: candidate
        )
    }

    private func makeDecision(
        artifact: VocoRuntimeCorrectionArtifact,
        features: VocoRuntimeCorrectionFeatures,
        chosenAction: String,
        fallbackReason: String,
        reasonCodes: [String],
        finalText: String? = nil,
        score: Double? = nil,
        appliedCandidate: VocoRuntimeCorrectionCandidate? = nil
    ) -> VocoRuntimeCorrectionDecision {
        VocoRuntimeCorrectionDecision(
            schema: artifact.decisionSchema.schema,
            runtimeMode: artifact.runtimeMode,
            chosenAction: chosenAction,
            fallbackReason: fallbackReason,
            reasonCodes: reasonCodes,
            score: score,
            rawTranscript: features.rawTranscript,
            postRuleText: features.postRuleText,
            finalText: finalText ?? features.postRuleText,
            modelArtifactId: artifact.artifactId,
            modelSha256: artifact.model.sha256,
            candidateGeneratorSha256: artifact.candidateGenerator.sha256,
            candidates: features.candidateSpans,
            appliedCandidate: appliedCandidate
        )
    }

    private func appendDecisionEvent(_ decision: VocoRuntimeCorrectionDecision) {
        guard let eventLogURL else { return }

        do {
            try FileManager.default.createDirectory(
                at: eventLogURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            var data = try encoder.encode(decision)
            data.append(0x0A)

            if FileManager.default.fileExists(atPath: eventLogURL.path) {
                let handle = try FileHandle(forWritingTo: eventLogURL)
                try handle.seekToEnd()
                try handle.write(contentsOf: data)
                try handle.close()
            } else {
                try data.write(to: eventLogURL, options: .atomic)
            }
        } catch {
            logger.error("Failed to append runtime correction shadow event: \(error.localizedDescription, privacy: .public)")
        }
    }
}

private struct VocoRuntimeCorrectionArtifact: Decodable {
    let schema: String
    let artifactId: String
    let runtimeMode: String
    let intendedUse: String
    let model: Model
    let approval: Approval
    let sourceRanker: SourceRanker
    let safety: Safety
    let decisionSchema: DecisionSchema
    let candidateGenerator: CandidateGenerator
    let thresholdConfig: ThresholdConfig?
    let runtimeReadiness: RuntimeReadiness?

    struct Model: Decodable {
        let format: String
        let modelType: String
        let path: String
        let portableRuntime: Bool
        let sha256: String
    }

    struct Approval: Decodable {
        let allowedModes: [String]
        let runtimeAllowed: Bool
        let requiresJasonApprovalForApply: Bool?
        let approvedBy: String?
        let approvedAt: String?
        let approvalToken: String?
    }

    struct SourceRanker: Decodable {
        let runtimeUsableDirectly: Bool
    }

    struct Safety: Decodable {
        let actionCommandBypass: Bool
        let compiledJsonLoaderMayLoadJoblib: Bool
        let artifactMissingFallback: String
        let timeoutFallback: String
        let jsonExactRulePriority: Bool
        let notWorseThanCompiledJson: Bool?
    }

    struct DecisionSchema: Decodable {
        let schema: String
        let actions: [String]
        let requiresEvidenceEvent: Bool
        let requiresReasonCodes: Bool
        let requiresScore: Bool
    }

    struct CandidateGenerator: Decodable {
        let required: Bool
        let schema: String
        let sha256: String
    }

    struct ThresholdConfig: Decodable {
        let shadow: Double?
        let suggest: Double?
        let gatedApply: Double?
    }

    struct RuntimeReadiness: Decodable {
        let baselineReplayPass: Bool
        let gatedApplyReplayPass: Bool
        let notWorseThanCompiledJson: Bool
        let unsafeApplyFalsePositiveCount: Int
        let finalTextRegressionCount: Int
        let actionCommandBypassVerified: Bool
    }

    func validateRuntimeContract(baseURL: URL) throws {
        guard schema == "voco.runtime-correction-model.v1" else {
            throw VocoRuntimeCorrectionArtifactError.invalidSchema(schema)
        }
        switch runtimeMode {
        case "shadow":
            try validateShadowContract()
        case "gatedApply":
            try validateGatedApplyContract(baseURL: baseURL)
        default:
            throw VocoRuntimeCorrectionArtifactError.invalidRuntimeMode(runtimeMode)
        }
    }

    private func validateShadowContract() throws {
        guard runtimeMode == "shadow" else {
            throw VocoRuntimeCorrectionArtifactError.invalidRuntimeMode(runtimeMode)
        }
        guard approval.allowedModes == ["shadow"], approval.runtimeAllowed == false else {
            throw VocoRuntimeCorrectionArtifactError.invalidApproval
        }
        guard model.portableRuntime == false, model.format == "none" else {
            throw VocoRuntimeCorrectionArtifactError.unexpectedPortableModel
        }
        guard sourceRanker.runtimeUsableDirectly == false else {
            throw VocoRuntimeCorrectionArtifactError.rankRuntimeBoundaryMissing
        }
        guard safety.actionCommandBypass,
              safety.compiledJsonLoaderMayLoadJoblib == false,
              safety.artifactMissingFallback == "return-post-rule-text",
              safety.timeoutFallback == "return-post-rule-text",
              safety.jsonExactRulePriority
        else {
            throw VocoRuntimeCorrectionArtifactError.invalidSafetyContract
        }
        guard decisionSchema.schema == "voco.runtime-correction-decision.v1",
              decisionSchema.actions.contains("noop"),
              decisionSchema.actions.contains("block"),
              decisionSchema.requiresEvidenceEvent,
              decisionSchema.requiresReasonCodes,
              decisionSchema.requiresScore
        else {
            throw VocoRuntimeCorrectionArtifactError.invalidDecisionSchema
        }
        guard candidateGenerator.required,
              candidateGenerator.schema == "voco.runtime-candidate-generator.v1"
        else {
            throw VocoRuntimeCorrectionArtifactError.invalidCandidateGenerator
        }
    }

    private func validateGatedApplyContract(baseURL: URL) throws {
        guard runtimeMode == "gatedApply" else {
            throw VocoRuntimeCorrectionArtifactError.invalidRuntimeMode(runtimeMode)
        }
        guard approval.runtimeAllowed,
              approval.allowedModes.contains("gatedApply"),
              approval.requiresJasonApprovalForApply == true,
              Self.allowedJasonApprovers.contains(approval.approvedBy ?? ""),
              approval.approvalToken?.isEmpty == false,
              approval.approvedAt?.isEmpty == false
        else {
            throw VocoRuntimeCorrectionArtifactError.invalidApproval
        }
        guard model.portableRuntime,
              model.format == "candidate-spans-v1",
              !model.sha256.isEmpty
        else {
            throw VocoRuntimeCorrectionArtifactError.invalidPortableModel
        }

        let modelRelativePath = try Self.safeRelativeModelPath(model.path)
        let modelURL = baseURL.appendingPathComponent(modelRelativePath)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw VocoRuntimeCorrectionArtifactError.invalidPortableModel
        }
        guard (try? Self.sha256Hex(of: modelURL)) == model.sha256 else {
            throw VocoRuntimeCorrectionArtifactError.modelChecksumMismatch
        }

        guard sourceRanker.runtimeUsableDirectly == false else {
            throw VocoRuntimeCorrectionArtifactError.rankRuntimeBoundaryMissing
        }
        guard safety.actionCommandBypass,
              safety.compiledJsonLoaderMayLoadJoblib == false,
              safety.artifactMissingFallback == "return-post-rule-text",
              safety.timeoutFallback == "return-post-rule-text",
              safety.jsonExactRulePriority,
              safety.notWorseThanCompiledJson == true
        else {
            throw VocoRuntimeCorrectionArtifactError.invalidSafetyContract
        }
        guard let threshold = thresholdConfig?.gatedApply,
              threshold >= 0.97
        else {
            throw VocoRuntimeCorrectionArtifactError.invalidThresholdConfig
        }
        guard let runtimeReadiness,
              runtimeReadiness.baselineReplayPass,
              runtimeReadiness.gatedApplyReplayPass,
              runtimeReadiness.notWorseThanCompiledJson,
              runtimeReadiness.unsafeApplyFalsePositiveCount == 0,
              runtimeReadiness.finalTextRegressionCount == 0,
              runtimeReadiness.actionCommandBypassVerified
        else {
            throw VocoRuntimeCorrectionArtifactError.invalidRuntimeReadiness
        }
        try validateDecisionAndCandidateContract()
    }

    func loadCandidateSpanModel(baseURL: URL) throws -> VocoRuntimeCandidateSpanModel? {
        guard runtimeMode == "gatedApply" else { return nil }
        return try VocoRuntimeCandidateSpanModel.load(from: modelURL(baseURL: baseURL))
    }

    private func validateDecisionAndCandidateContract() throws {
        guard decisionSchema.schema == "voco.runtime-correction-decision.v1",
              decisionSchema.actions.contains("noop"),
              decisionSchema.actions.contains("block"),
              decisionSchema.actions.contains("apply"),
              decisionSchema.requiresEvidenceEvent,
              decisionSchema.requiresReasonCodes,
              decisionSchema.requiresScore
        else {
            throw VocoRuntimeCorrectionArtifactError.invalidDecisionSchema
        }
        guard candidateGenerator.required,
              candidateGenerator.schema == "voco.runtime-candidate-generator.v1"
        else {
            throw VocoRuntimeCorrectionArtifactError.invalidCandidateGenerator
        }
    }

    private static let allowedJasonApprovers = [
        "Jason",
        "Jason Chien",
        "Jianrui Cheng"
    ]

    private func modelURL(baseURL: URL) -> URL {
        baseURL.appendingPathComponent(model.path)
    }

    private static func sha256Hex(of url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private static func safeRelativeModelPath(_ path: String) throws -> String {
        guard !path.isEmpty,
              !path.hasPrefix("/"),
              !path.lowercased().hasSuffix(".joblib")
        else {
            throw VocoRuntimeCorrectionArtifactError.invalidPortableModel
        }
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard components.allSatisfy({ component in
            !component.isEmpty && component != "." && component != ".."
        }) else {
            throw VocoRuntimeCorrectionArtifactError.invalidPortableModel
        }
        return path
    }
}

private struct VocoRuntimeCandidateSpanModel: Decodable {
    let schema: String
    let candidates: [Candidate]

    struct Candidate: Decodable {
        let id: String?
        let source: String
        let target: String
        let score: Double
        let rawContains: String?
        let postRuleContains: String?
        let contextHintContains: String?

        func matches(_ features: VocoRuntimeCorrectionFeatures) -> Bool {
            guard features.postRuleText.contains(source) else { return false }

            if let rawContains, !rawContains.isEmpty, !features.rawTranscript.contains(rawContains) {
                return false
            }

            if let postRuleContains, !postRuleContains.isEmpty, !features.postRuleText.contains(postRuleContains) {
                return false
            }

            if let contextHintContains, !contextHintContains.isEmpty {
                let hasContext = features.contextHints.contains { $0.contains(contextHintContains) }
                    || features.appMode?.contains(contextHintContains) == true
                if !hasContext { return false }
            }

            return true
        }
    }

    static func load(from url: URL) throws -> VocoRuntimeCandidateSpanModel {
        let data = try Data(contentsOf: url)
        let model = try JSONDecoder().decode(VocoRuntimeCandidateSpanModel.self, from: data)
        guard model.schema == "voco.runtime-candidate-spans.v1",
              model.candidates.allSatisfy({ candidate in
                  !candidate.source.isEmpty &&
                      !candidate.target.isEmpty &&
                      candidate.source != candidate.target &&
                      candidate.score >= 0 &&
                      candidate.score <= 1
              })
        else {
            throw VocoRuntimeCorrectionArtifactError.invalidCandidateSpanModel
        }
        return model
    }

    func candidates(for features: VocoRuntimeCorrectionFeatures) -> [VocoRuntimeCorrectionCandidate] {
        candidates
            .filter { $0.matches(features) }
            .map {
                VocoRuntimeCorrectionCandidate(
                    source: $0.source,
                    target: $0.target,
                    score: $0.score
                )
            }
    }
}

private enum VocoRuntimeCorrectionArtifactError: LocalizedError {
    case invalidSchema(String)
    case invalidRuntimeMode(String)
    case invalidApproval
    case unexpectedPortableModel
    case invalidPortableModel
    case modelChecksumMismatch
    case rankRuntimeBoundaryMissing
    case invalidSafetyContract
    case invalidDecisionSchema
    case invalidCandidateGenerator
    case invalidCandidateSpanModel
    case invalidThresholdConfig
    case invalidRuntimeReadiness

    var errorDescription: String? {
        switch self {
        case .invalidSchema(let schema):
            "Unsupported runtime correction artifact schema: \(schema)"
        case .invalidRuntimeMode(let mode):
            "Unsupported runtime correction artifact mode: \(mode)"
        case .invalidApproval:
            "Runtime correction artifact approval contract is invalid"
        case .unexpectedPortableModel:
            "Shadow contract must not include a portable runtime model"
        case .invalidPortableModel:
            "Gated apply requires a portable candidate-spans-v1 runtime model artifact"
        case .modelChecksumMismatch:
            "Runtime correction model checksum does not match the artifact manifest"
        case .rankRuntimeBoundaryMissing:
            "Source ranker must not be marked directly runtime usable"
        case .invalidSafetyContract:
            "Runtime correction artifact safety contract is incomplete"
        case .invalidDecisionSchema:
            "Runtime correction decision schema is incomplete"
        case .invalidCandidateGenerator:
            "Runtime correction candidate generator contract is incomplete"
        case .invalidCandidateSpanModel:
            "Runtime correction candidate span model is invalid"
        case .invalidThresholdConfig:
            "Runtime correction gated apply threshold must be at least 0.97"
        case .invalidRuntimeReadiness:
            "Runtime correction gated apply readiness report is not release safe"
        }
    }
}
