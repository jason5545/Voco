import Foundation
import MLX
import os

enum Qwen3TextDecoderLoRAError: Error, LocalizedError {
    case adapterNotFound(URL)
    case missingPair(String)
    case invalidProjectionKey(String)
    case invalidLayerIndex(Int)
    case shapeMismatch(key: String, reason: String)

    var errorDescription: String? {
        switch self {
        case .adapterNotFound(let url):
            return "Text LoRA adapter not found: \(url.path)"
        case .missingPair(let key):
            return "Text LoRA adapter missing A/B pair for \(key)"
        case .invalidProjectionKey(let key):
            return "Unsupported text LoRA projection key: \(key)"
        case .invalidLayerIndex(let index):
            return "Text LoRA layer index out of range: \(index)"
        case .shapeMismatch(let key, let reason):
            return "Text LoRA shape mismatch for \(key): \(reason)"
        }
    }
}

enum Qwen3TextDecoderLoRATarget: String {
    case qProj
    case kProj
    case vProj
    case oProj
    case gateProj
    case upProj
    case downProj
}

struct Qwen3TextDecoderLoRAProjection {
    let key: String
    let a: MLXArray
    let b: MLXArray
    let scale: Float

    var rank: Int { a.dim(1) }
    var inputDimension: Int { a.dim(0) }
    var outputDimension: Int { b.dim(1) }

    func apply(to input: MLXArray, baseOutput: MLXArray) -> MLXArray {
        let delta = matmul(matmul(input.asType(a.dtype), a), b) * scale
        return baseOutput + delta.asType(baseOutput.dtype)
    }
}

struct Qwen3TextDecoderLoRALayer {
    var qProj: Qwen3TextDecoderLoRAProjection?
    var kProj: Qwen3TextDecoderLoRAProjection?
    var vProj: Qwen3TextDecoderLoRAProjection?
    var oProj: Qwen3TextDecoderLoRAProjection?
    var gateProj: Qwen3TextDecoderLoRAProjection?
    var upProj: Qwen3TextDecoderLoRAProjection?
    var downProj: Qwen3TextDecoderLoRAProjection?

    func projection(_ target: Qwen3TextDecoderLoRATarget) -> Qwen3TextDecoderLoRAProjection? {
        switch target {
        case .qProj: return qProj
        case .kProj: return kProj
        case .vProj: return vProj
        case .oProj: return oProj
        case .gateProj: return gateProj
        case .upProj: return upProj
        case .downProj: return downProj
        }
    }

    mutating func set(_ projection: Qwen3TextDecoderLoRAProjection, target: Qwen3TextDecoderLoRATarget) {
        switch target {
        case .qProj: qProj = projection
        case .kProj: kProj = projection
        case .vProj: vProj = projection
        case .oProj: oProj = projection
        case .gateProj: gateProj = projection
        case .upProj: upProj = projection
        case .downProj: downProj = projection
        }
    }
}

final class Qwen3TextDecoderLoRAStore {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "Qwen3TextLoRA")

    let adapterURL: URL
    let rank: Int
    let scale: Float
    let layers: [Int: Qwen3TextDecoderLoRALayer]
    let appliedProjectionCount: Int

    init(adapterURL: URL, rank: Int, scale: Float, layers: [Int: Qwen3TextDecoderLoRALayer]) {
        self.adapterURL = adapterURL
        self.rank = rank
        self.scale = scale
        self.layers = layers
        self.appliedProjectionCount = layers.values.reduce(0) { total, layer in
            total +
                [layer.qProj, layer.kProj, layer.vProj, layer.oProj, layer.gateProj, layer.upProj, layer.downProj]
                .compactMap { $0 }
                .count
        }
    }

    func layer(at index: Int) -> Qwen3TextDecoderLoRALayer? {
        layers[index]
    }

    static func load(from adapterPath: URL, config: Qwen3TextDecoderConfig) throws -> Qwen3TextDecoderLoRAStore {
        let adapterURL = resolvedSafetensorsURL(from: adapterPath)
        guard FileManager.default.fileExists(atPath: adapterURL.path) else {
            throw Qwen3TextDecoderLoRAError.adapterNotFound(adapterURL)
        }

        let adapterConfig = loadAdapterConfig(near: adapterURL)
        let configuredScale = adapterConfig?.loraParameters?.scale ?? 1.0
        let configuredRank = adapterConfig?.loraParameters?.rank
        let weights = try MLX.loadArrays(url: adapterURL)

        var grouped: [String: (a: MLXArray?, b: MLXArray?)] = [:]
        for (rawKey, value) in weights {
            let key = normalizedAdapterKey(rawKey)
            if key.hasSuffix(".lora_a") {
                let baseKey = String(key.dropLast(".lora_a".count))
                var pair = grouped[baseKey] ?? (nil, nil)
                pair.a = value
                grouped[baseKey] = pair
            } else if key.hasSuffix(".lora_b") {
                let baseKey = String(key.dropLast(".lora_b".count))
                var pair = grouped[baseKey] ?? (nil, nil)
                pair.b = value
                grouped[baseKey] = pair
            }
        }

        var layers: [Int: Qwen3TextDecoderLoRALayer] = [:]
        var inferredRank: Int?
        for key in grouped.keys.sorted() {
            guard let pair = grouped[key], let a = pair.a, let b = pair.b else {
                throw Qwen3TextDecoderLoRAError.missingPair(key)
            }
            let (layerIndex, target) = try parseProjectionKey(key)
            guard layerIndex >= 0, layerIndex < config.numLayers else {
                throw Qwen3TextDecoderLoRAError.invalidLayerIndex(layerIndex)
            }
            try validate(
                a: a,
                b: b,
                key: key,
                target: target,
                config: config,
                configuredRank: configuredRank
            )
            inferredRank = inferredRank ?? a.dim(1)

            let projection = Qwen3TextDecoderLoRAProjection(
                key: key,
                a: a,
                b: b,
                scale: configuredScale
            )
            var layer = layers[layerIndex] ?? Qwen3TextDecoderLoRALayer()
            layer.set(projection, target: target)
            layers[layerIndex] = layer
        }

        let store = Qwen3TextDecoderLoRAStore(
            adapterURL: adapterURL,
            rank: configuredRank ?? inferredRank ?? 0,
            scale: configuredScale,
            layers: layers
        )
        logger.info("Loaded text LoRA adapter with \(store.appliedProjectionCount) projections, rank \(store.rank), scale \(store.scale)")
        return store
    }

    private static func resolvedSafetensorsURL(from url: URL) -> URL {
        url.pathExtension == "safetensors" ? url : url.appendingPathComponent("adapters.safetensors")
    }

    private static func normalizedAdapterKey(_ rawKey: String) -> String {
        var key = rawKey
        for prefix in ["base_model.model.", "model.model."] where key.hasPrefix(prefix) {
            key.removeFirst(prefix.count)
        }
        if key.hasSuffix(".weight") {
            key.removeLast(".weight".count)
        }
        return key
    }

    private static func parseProjectionKey(_ key: String) throws -> (Int, Qwen3TextDecoderLoRATarget) {
        let parts = key.split(separator: ".").map(String.init)
        guard parts.count == 5, parts[0] == "model", parts[1] == "layers", let layerIndex = Int(parts[2]) else {
            throw Qwen3TextDecoderLoRAError.invalidProjectionKey(key)
        }

        switch (parts[3], parts[4]) {
        case ("self_attn", "q_proj"): return (layerIndex, .qProj)
        case ("self_attn", "k_proj"): return (layerIndex, .kProj)
        case ("self_attn", "v_proj"): return (layerIndex, .vProj)
        case ("self_attn", "o_proj"): return (layerIndex, .oProj)
        case ("mlp", "gate_proj"): return (layerIndex, .gateProj)
        case ("mlp", "up_proj"): return (layerIndex, .upProj)
        case ("mlp", "down_proj"): return (layerIndex, .downProj)
        default:
            throw Qwen3TextDecoderLoRAError.invalidProjectionKey(key)
        }
    }

    private static func validate(
        a: MLXArray,
        b: MLXArray,
        key: String,
        target: Qwen3TextDecoderLoRATarget,
        config: Qwen3TextDecoderConfig,
        configuredRank: Int?
    ) throws {
        guard a.ndim == 2, b.ndim == 2 else {
            throw Qwen3TextDecoderLoRAError.shapeMismatch(key: key, reason: "A and B must be rank-2 tensors")
        }
        guard a.dim(1) == b.dim(0) else {
            throw Qwen3TextDecoderLoRAError.shapeMismatch(
                key: key,
                reason: "A rank \(a.dim(1)) does not match B rank \(b.dim(0))"
            )
        }
        if let configuredRank, configuredRank != a.dim(1) {
            throw Qwen3TextDecoderLoRAError.shapeMismatch(
                key: key,
                reason: "configured rank \(configuredRank) does not match tensor rank \(a.dim(1))"
            )
        }

        let expected: (input: Int, output: Int)
        switch target {
        case .qProj:
            expected = (config.hiddenSize, config.numHeads * config.headDim)
        case .kProj, .vProj:
            expected = (config.hiddenSize, config.numKVHeads * config.headDim)
        case .oProj:
            expected = (config.numHeads * config.headDim, config.hiddenSize)
        case .gateProj, .upProj:
            expected = (config.hiddenSize, config.intermediateSize)
        case .downProj:
            expected = (config.intermediateSize, config.hiddenSize)
        }
        guard a.dim(0) == expected.input, b.dim(1) == expected.output else {
            throw Qwen3TextDecoderLoRAError.shapeMismatch(
                key: key,
                reason: "expected A/B dimensions \(expected.input) -> \(expected.output), got \(a.dim(0)) -> \(b.dim(1))"
            )
        }
    }

    private static func loadAdapterConfig(near adapterURL: URL) -> AdapterConfig? {
        let url = adapterURL.deletingLastPathComponent().appendingPathComponent("adapter_config.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(AdapterConfig.self, from: data)
    }
}

private struct AdapterConfig: Decodable {
    struct LoRAParameters: Decodable {
        let rank: Int
        let scale: Float
    }

    let loraParameters: LoRAParameters?

    enum CodingKeys: String, CodingKey {
        case loraParameters = "lora_parameters"
    }
}
