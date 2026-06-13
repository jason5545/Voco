import Foundation
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
    let rawTranscript: String
    let postRuleText: String
    let finalText: String
    let modelArtifactId: String
    let modelSha256: String
    let candidateGeneratorSha256: String
    let candidates: [VocoRuntimeCorrectionCandidate]
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

    private(set) var status: VocoRuntimeCorrectionModelStatus

    var isUserEnabled: Bool {
        get { defaults.object(forKey: Self.enabledKey) as? Bool ?? false }
        set { defaults.set(newValue, forKey: Self.enabledKey) }
    }

    var isShadowEnabled: Bool {
        isUserEnabled && status.isAvailable && loadedArtifact?.runtimeMode == "shadow"
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
            status = VocoRuntimeCorrectionModelStatus(
                isAvailable: false,
                message: "Runtime correction model requires an explicit artifact manifest, not a joblib ranker",
                artifactURL: artifactURL
            )
            return
        }

        guard FileManager.default.fileExists(atPath: artifactURL.path) else {
            loadedArtifact = nil
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
            try artifact.validateShadowContract()
            loadedArtifact = artifact
            status = VocoRuntimeCorrectionModelStatus(
                isAvailable: true,
                message: "Runtime correction shadow contract loaded",
                artifactURL: artifactURL
            )
        } catch {
            loadedArtifact = nil
            status = VocoRuntimeCorrectionModelStatus(
                isAvailable: false,
                message: error.localizedDescription,
                artifactURL: artifactURL
            )
            logger.error("Failed to load runtime correction artifact: \(error.localizedDescription, privacy: .public)")
        }
    }

    func evaluate(_ features: VocoRuntimeCorrectionFeatures) -> VocoRuntimeCorrectionEvaluation {
        guard isShadowEnabled, let artifact = loadedArtifact else {
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
                fallbackReason: "action-command-bypass"
            )
        } else {
            decision = makeDecision(
                artifact: artifact,
                features: features,
                chosenAction: "noop",
                fallbackReason: "shadow-contract-fixture-no-runtime-model"
            )
        }

        appendDecisionEvent(decision)
        return VocoRuntimeCorrectionEvaluation(
            inputText: features.postRuleText,
            outputText: features.postRuleText,
            decision: decision
        )
    }

    private func makeDecision(
        artifact: VocoRuntimeCorrectionArtifact,
        features: VocoRuntimeCorrectionFeatures,
        chosenAction: String,
        fallbackReason: String
    ) -> VocoRuntimeCorrectionDecision {
        VocoRuntimeCorrectionDecision(
            schema: artifact.decisionSchema.schema,
            runtimeMode: artifact.runtimeMode,
            chosenAction: chosenAction,
            fallbackReason: fallbackReason,
            rawTranscript: features.rawTranscript,
            postRuleText: features.postRuleText,
            finalText: features.postRuleText,
            modelArtifactId: artifact.artifactId,
            modelSha256: artifact.model.sha256,
            candidateGeneratorSha256: artifact.candidateGenerator.sha256,
            candidates: features.candidateSpans
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

    func validateShadowContract() throws {
        guard schema == "voco.runtime-correction-model.v1" else {
            throw VocoRuntimeCorrectionArtifactError.invalidSchema(schema)
        }
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
}

private enum VocoRuntimeCorrectionArtifactError: LocalizedError {
    case invalidSchema(String)
    case invalidRuntimeMode(String)
    case invalidApproval
    case unexpectedPortableModel
    case rankRuntimeBoundaryMissing
    case invalidSafetyContract
    case invalidDecisionSchema
    case invalidCandidateGenerator

    var errorDescription: String? {
        switch self {
        case .invalidSchema(let schema):
            "Unsupported runtime correction artifact schema: \(schema)"
        case .invalidRuntimeMode(let mode):
            "Runtime correction artifact is not shadow mode: \(mode)"
        case .invalidApproval:
            "Runtime correction artifact approval must allow shadow only"
        case .unexpectedPortableModel:
            "Shadow contract must not include a portable runtime model"
        case .rankRuntimeBoundaryMissing:
            "Source ranker must not be marked directly runtime usable"
        case .invalidSafetyContract:
            "Runtime correction artifact safety contract is incomplete"
        case .invalidDecisionSchema:
            "Runtime correction decision schema is incomplete"
        case .invalidCandidateGenerator:
            "Runtime correction candidate generator contract is incomplete"
        }
    }
}
