import Foundation
import MLX
import os

struct VocoTextCleanupLoRAEvaluation: Equatable {
    let inputText: String
    let outputText: String
    let candidateText: String?
    let mode: VocoTextCleanupLoRAService.Mode
    let applied: Bool
    let status: String
}

final class VocoTextCleanupLoRAService {
    static let shared = VocoTextCleanupLoRAService()

    static let modeKey = "VocoTextCleanupLoRAMode"
    static let baseModelPathKey = "VocoTextCleanupLoRABaseModelPath"
    static let adapterPathKey = "VocoTextCleanupLoRAAdapterPath"
    static let maxTokensKey = "VocoTextCleanupLoRAMaxTokens"

    static let defaultBaseModelPath = "/Users/jianruicheng/GitHub/VocoReplayLab/local-models/qwen3-asr-1.7b-8bit-text-decoder"
    static let defaultAdapterPath = "/Users/jianruicheng/GitHub/VocoReplayLab/local-adapters/qwen3-asr-cleanup-lora-20260615-v2-shadow/adapters.safetensors"
    static let defaultMaxTokens = 96

    enum Mode: String, Equatable {
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

    private let defaults: UserDefaults
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "TextCleanupLoRA")
    private let lock = NSLock()
    private var runtime: Runtime?

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    func evaluate(_ text: String, contextHints: [String] = []) -> VocoTextCleanupLoRAEvaluation {
        let mode = currentMode()
        guard mode != .off else {
            return noOp(text, mode: mode, status: "disabled")
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return noOp(text, mode: mode, status: "empty-input")
        }

        guard VoiceCommandService.shared.detectCommand(in: text) == nil,
              VoiceCommandService.shared.detectEditModeCommand(in: text) == nil else {
            return noOp(text, mode: mode, status: "blocked-action-command")
        }

        do {
            let candidate = try generateCandidate(for: text, contextHints: contextHints)
            guard let candidate, candidate != text else {
                return noOp(text, mode: mode, candidateText: candidate, status: "no-op")
            }
            guard isSafeCandidate(candidate, for: text) else {
                logger.warning("Text LoRA candidate rejected by safety guard")
                return noOp(text, mode: mode, candidateText: candidate, status: "blocked-safety-guard")
            }
            guard mode == .apply else {
                logger.info("Text LoRA shadow candidate: \(candidate, privacy: .public)")
                return noOp(text, mode: mode, candidateText: candidate, status: "shadow-candidate")
            }
            return VocoTextCleanupLoRAEvaluation(
                inputText: text,
                outputText: candidate,
                candidateText: candidate,
                mode: mode,
                applied: true,
                status: "applied"
            )
        } catch {
            logger.error("Text LoRA evaluation failed: \(String(describing: error), privacy: .public)")
            return noOp(text, mode: mode, status: "error")
        }
    }

    private func noOp(
        _ text: String,
        mode: Mode,
        candidateText: String? = nil,
        status: String
    ) -> VocoTextCleanupLoRAEvaluation {
        VocoTextCleanupLoRAEvaluation(
            inputText: text,
            outputText: text,
            candidateText: candidateText,
            mode: mode,
            applied: false,
            status: status
        )
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

    private func maxTokens() -> Int {
        let value = defaults.integer(forKey: Self.maxTokensKey)
        return value > 0 ? value : Self.defaultMaxTokens
    }

    private func generateCandidate(for text: String, contextHints: [String]) throws -> String? {
        lock.lock()
        defer { lock.unlock() }

        let runtime = try loadRuntimeIfNeeded(baseModelURL: baseModelURL(), adapterURL: adapterURL())
        let inputIds = buildChatInputIds(text: text, contextHints: contextHints, tokenizer: runtime.tokenizer)
        let generated = try generate(inputIds: inputIds, runtime: runtime, maxTokens: maxTokens())
        return sanitize(generated)
    }

    private func loadRuntimeIfNeeded(baseModelURL: URL, adapterURL: URL) throws -> Runtime {
        if let runtime,
           runtime.baseModelURL.path == baseModelURL.path,
           runtime.adapterURL.path == adapterURL.path {
            return runtime
        }

        logger.info("Loading text cleanup LoRA base model: \(baseModelURL.path, privacy: .public)")
        let config = try textDecoderConfig(from: baseModelURL)
        let tokenizer = Qwen3Tokenizer()
        try tokenizer.load(from: baseModelURL.appendingPathComponent("vocab.json"))

        let model = Qwen3QuantizedTextModel(config: config)
        try Qwen3WeightLoader.loadTextDecoderWeights(into: model, from: baseModelURL)
        model.loraStore = try Qwen3TextDecoderLoRAStore.load(from: adapterURL, config: config)

        let loaded = Runtime(
            baseModelURL: baseModelURL,
            adapterURL: adapterURL,
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

    private func isSafeCandidate(_ candidate: String, for input: String) -> Bool {
        guard !candidate.isEmpty else { return false }
        guard candidate.count <= max(input.count * 3, input.count + 24) else { return false }
        guard VoiceCommandService.shared.detectCommand(in: candidate) == nil,
              VoiceCommandService.shared.detectEditModeCommand(in: candidate) == nil else {
            return false
        }
        return true
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
