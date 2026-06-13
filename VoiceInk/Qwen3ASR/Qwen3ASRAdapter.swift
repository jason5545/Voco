// Qwen3ASRAdapter.swift
// Audio-side LoRA adapter discovery and application for Qwen3-ASR.

import Foundation
import MLX
import MLXNN
import os

struct Qwen3ASRAdapterMetadata: Equatable {
    let adapterDetected: Bool
    let adapterLoaded: Bool
    let adapterApplied: Bool
    let adapterPath: String?
    let adapterLoadError: String?

    static let unavailable = Qwen3ASRAdapterMetadata(
        adapterDetected: false,
        adapterLoaded: false,
        adapterApplied: false,
        adapterPath: nil,
        adapterLoadError: nil
    )
}

struct Qwen3ASRAdapterDescriptor: Equatable {
    let directory: URL
    let configURL: URL
    let weightsURL: URL
    let config: Qwen3ASRAdapterConfig
}

struct Qwen3ASRAdapterFingerprint: Equatable {
    struct FileSignature: Equatable {
        let exists: Bool
        let size: UInt64?
        let modificationDate: Date?
    }

    let directoryPath: String
    let config: FileSignature
    let weights: FileSignature
}

struct Qwen3ASRAdapterDiscovery: Equatable {
    let descriptor: Qwen3ASRAdapterDescriptor?
    let adapterPath: String?
    let error: String?

    var adapterDetected: Bool {
        descriptor != nil
    }

    static func unavailable(path: String? = nil, error: String? = nil) -> Qwen3ASRAdapterDiscovery {
        Qwen3ASRAdapterDiscovery(descriptor: nil, adapterPath: path, error: error)
    }

    static func available(_ descriptor: Qwen3ASRAdapterDescriptor) -> Qwen3ASRAdapterDiscovery {
        Qwen3ASRAdapterDiscovery(
            descriptor: descriptor,
            adapterPath: descriptor.directory.path,
            error: nil
        )
    }
}

struct Qwen3ASRAdapterCoordinator {
    typealias Discover = (URL) -> Qwen3ASRAdapterDiscovery
    typealias Apply = (Qwen3ASRAdapterDescriptor) throws -> Int

    static func loadIfAvailable(
        modelDirectory: URL,
        discover: Discover,
        apply: Apply
    ) -> Qwen3ASRAdapterMetadata {
        let discovery = discover(modelDirectory)
        guard let descriptor = discovery.descriptor else {
            return Qwen3ASRAdapterMetadata(
                adapterDetected: false,
                adapterLoaded: false,
                adapterApplied: false,
                adapterPath: discovery.adapterPath,
                adapterLoadError: discovery.error
            )
        }

        do {
            let appliedCount = try apply(descriptor)
            return Qwen3ASRAdapterMetadata(
                adapterDetected: true,
                adapterLoaded: true,
                adapterApplied: appliedCount > 0,
                adapterPath: descriptor.directory.path,
                adapterLoadError: nil
            )
        } catch {
            return Qwen3ASRAdapterMetadata(
                adapterDetected: true,
                adapterLoaded: false,
                adapterApplied: false,
                adapterPath: descriptor.directory.path,
                adapterLoadError: error.localizedDescription
            )
        }
    }
}

struct Qwen3ASRAdapterConfig: Decodable, Equatable {
    struct LoRAParameters: Decodable, Equatable {
        let rank: Int
        let scale: Float
        let dropout: Float
        let keys: [String]
    }

    let schema: String
    let baseModel: String
    let boundary: String
    let fineTuneType: String
    let loraParameters: LoRAParameters

    enum CodingKeys: String, CodingKey {
        case schema
        case baseModel = "base_model"
        case boundary
        case fineTuneType = "fine_tune_type"
        case loraParameters = "lora_parameters"
    }
}

enum Qwen3ASRAdapterError: LocalizedError {
    case invalidConfig(String)
    case missingLoRAWeight(String)
    case unsupportedLoRAKey(String)
    case invalidLayerIndex(Int)
    case shapeMismatch(key: String, expected: [Int], actual: [Int])

    var errorDescription: String? {
        switch self {
        case .invalidConfig(let reason):
            return "Invalid Qwen3-ASR adapter config: \(reason)"
        case .missingLoRAWeight(let key):
            return "Missing LoRA weight: \(key)"
        case .unsupportedLoRAKey(let key):
            return "Unsupported LoRA adapter key: \(key)"
        case .invalidLayerIndex(let index):
            return "Invalid Qwen3-ASR audio encoder layer index: \(index)"
        case .shapeMismatch(let key, let expected, let actual):
            return "LoRA shape mismatch for \(key): expected \(expected), got \(actual)"
        }
    }
}

enum Qwen3ASRAudioAdapterLoader {
    static let adapterDirectoryName = "qwen3-asr-speaker-audio-lora-20260612-balanced-64iter"
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "Qwen3ASRAdapter")

    static func adapterDirectory(in modelDirectory: URL) -> URL {
        modelDirectory
            .appendingPathComponent("adapters", isDirectory: true)
            .appendingPathComponent(adapterDirectoryName, isDirectory: true)
    }

    static func fingerprint(in modelDirectory: URL, fileManager: FileManager = .default) -> Qwen3ASRAdapterFingerprint? {
        let adapterDirectory = adapterDirectory(in: modelDirectory)
        guard fileManager.fileExists(atPath: adapterDirectory.path) else {
            return nil
        }

        return Qwen3ASRAdapterFingerprint(
            directoryPath: adapterDirectory.path,
            config: fileSignature(for: adapterDirectory.appendingPathComponent("adapter_config.json"), fileManager: fileManager),
            weights: fileSignature(for: adapterDirectory.appendingPathComponent("adapters.safetensors"), fileManager: fileManager)
        )
    }

    static func discover(in modelDirectory: URL, fileManager: FileManager = .default) -> Qwen3ASRAdapterDiscovery {
        let adapterDirectory = adapterDirectory(in: modelDirectory)

        guard fileManager.fileExists(atPath: adapterDirectory.path) else {
            return .unavailable()
        }

        let configURL = adapterDirectory.appendingPathComponent("adapter_config.json")
        let weightsURL = adapterDirectory.appendingPathComponent("adapters.safetensors")

        guard fileManager.isReadableFile(atPath: configURL.path) else {
            return .unavailable(path: adapterDirectory.path, error: "adapter_config.json not readable")
        }
        guard fileManager.isReadableFile(atPath: weightsURL.path) else {
            return .unavailable(path: adapterDirectory.path, error: "adapters.safetensors not readable")
        }

        do {
            let config = try JSONDecoder().decode(Qwen3ASRAdapterConfig.self, from: Data(contentsOf: configURL))
            try validate(config)
            return .available(Qwen3ASRAdapterDescriptor(
                directory: adapterDirectory,
                configURL: configURL,
                weightsURL: weightsURL,
                config: config
            ))
        } catch {
            return .unavailable(path: adapterDirectory.path, error: error.localizedDescription)
        }
    }

    static func loadAndApplyIfPresent(
        modelDirectory: URL,
        audioEncoder: Qwen3AudioEncoder
    ) -> Qwen3ASRAdapterMetadata {
        let metadata = Qwen3ASRAdapterCoordinator.loadIfAvailable(
            modelDirectory: modelDirectory,
            discover: { discover(in: $0) },
            apply: { descriptor in
                try apply(descriptor: descriptor, to: audioEncoder)
            }
        )

        if metadata.adapterApplied {
            logger.info("Qwen3-ASR audio LoRA adapter applied: \(metadata.adapterPath ?? "unknown")")
        } else if metadata.adapterDetected {
            logger.error("Qwen3-ASR audio LoRA adapter detected but not applied: \(metadata.adapterLoadError ?? "unknown error")")
        } else if let error = metadata.adapterLoadError {
            logger.warning("Qwen3-ASR audio LoRA adapter unavailable: \(error)")
        }

        return metadata
    }

    static func apply(descriptor: Qwen3ASRAdapterDescriptor, to audioEncoder: Qwen3AudioEncoder) throws -> Int {
        let weights = try Qwen3WeightLoader.loadSafetensors(url: descriptor.weightsURL)
        var appliedCount = 0

        for key in descriptor.config.loraParameters.keys {
            let loraAKey = "\(key).lora_a"
            let loraBKey = "\(key).lora_b"
            guard let loraA = weights[loraAKey] else {
                throw Qwen3ASRAdapterError.missingLoRAWeight(loraAKey)
            }
            guard let loraB = weights[loraBKey] else {
                throw Qwen3ASRAdapterError.missingLoRAWeight(loraBKey)
            }
            let linear = try linearLayer(for: key, in: audioEncoder)
            try applyLoRA(
                loraA: loraA,
                loraB: loraB,
                scale: descriptor.config.loraParameters.scale,
                to: linear,
                key: key
            )
            appliedCount += 1
        }

        eval(audioEncoder.convOut.weight)
        logger.info("Applied \(appliedCount) Qwen3-ASR audio LoRA targets")
        return appliedCount
    }

    private static func validate(_ config: Qwen3ASRAdapterConfig) throws {
        guard config.schema == "voco.qwen3-asr-audio-lora-adapter-config.v1" else {
            throw Qwen3ASRAdapterError.invalidConfig("unsupported schema \(config.schema)")
        }
        guard config.fineTuneType == "lora" else {
            throw Qwen3ASRAdapterError.invalidConfig("unsupported fine_tune_type \(config.fineTuneType)")
        }
        guard config.boundary.localizedCaseInsensitiveContains("audio-side") else {
            throw Qwen3ASRAdapterError.invalidConfig("adapter is not marked audio-side")
        }
        guard config.baseModel.contains("qwen3-asr-1.7b-8bit") else {
            throw Qwen3ASRAdapterError.invalidConfig("unexpected base_model \(config.baseModel)")
        }
        guard config.loraParameters.rank > 0, !config.loraParameters.keys.isEmpty else {
            throw Qwen3ASRAdapterError.invalidConfig("empty LoRA targets")
        }
    }

    private static func linearLayer(for key: String, in audioEncoder: Qwen3AudioEncoder) throws -> Linear {
        switch key {
        case "audio_tower.conv_out":
            return audioEncoder.convOut
        case "audio_tower.proj1":
            return audioEncoder.proj1
        case "audio_tower.proj2":
            return audioEncoder.proj2
        default:
            let parts = key.split(separator: ".").map(String.init)
            guard parts.count == 5,
                  parts[0] == "audio_tower",
                  parts[1] == "layers",
                  parts[3] == "self_attn",
                  let layerIndex = Int(parts[2]) else {
                throw Qwen3ASRAdapterError.unsupportedLoRAKey(key)
            }
            guard audioEncoder.layers.indices.contains(layerIndex) else {
                throw Qwen3ASRAdapterError.invalidLayerIndex(layerIndex)
            }
            let attention = audioEncoder.layers[layerIndex].selfAttn
            switch parts[4] {
            case "q_proj":
                return attention.qProj
            case "k_proj":
                return attention.kProj
            case "v_proj":
                return attention.vProj
            case "out_proj":
                return attention.outProj
            default:
                throw Qwen3ASRAdapterError.unsupportedLoRAKey(key)
            }
        }
    }

    private static func applyLoRA(
        loraA: MLXArray,
        loraB: MLXArray,
        scale: Float,
        to linear: Linear,
        key: String
    ) throws {
        let delta = (matmul(loraA, loraB) * scale).transposed()
        let newWeight = linear.weight + delta
        guard newWeight.shape == linear.weight.shape else {
            throw Qwen3ASRAdapterError.shapeMismatch(
                key: key,
                expected: linear.weight.shape,
                actual: newWeight.shape
            )
        }
        linear.update(parameters: ModuleParameters(values: ["weight": .value(newWeight)]))
    }

    private static func fileSignature(for url: URL, fileManager: FileManager) -> Qwen3ASRAdapterFingerprint.FileSignature {
        guard let attributes = try? fileManager.attributesOfItem(atPath: url.path) else {
            return Qwen3ASRAdapterFingerprint.FileSignature(
                exists: false,
                size: nil,
                modificationDate: nil
            )
        }

        return Qwen3ASRAdapterFingerprint.FileSignature(
            exists: true,
            size: (attributes[.size] as? NSNumber)?.uint64Value,
            modificationDate: attributes[.modificationDate] as? Date
        )
    }
}
