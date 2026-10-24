import Foundation

struct Qwen3Model: TranscriptionModel {
    let id = UUID()
    let name: String
    let displayName: String
    let description: String
    let provider: ModelProvider = .qwen3
    let size: String
    let speed: Double
    let accuracy: Double
    let ramUsage: Double
    let modelId: String
    let modelSize: Qwen3ASRModelSize
    var isMultilingualModel: Bool {
        supportedLanguages.count > 1
    }
    let supportedLanguages: [String: String]
}

struct WhisperMLXModel: TranscriptionModel {
    let id = UUID()
    let name: String
    let displayName: String
    let description: String
    let provider: ModelProvider = .whisperMLX
    let size: String
    let speed: Double
    let accuracy: Double
    let ramUsage: Double
    let huggingFaceRepo: String
    var isMultilingualModel: Bool {
        supportedLanguages.count > 1
    }
    let supportedLanguages: [String: String]
}

struct WhisperCoreMLModel: TranscriptionModel {
    let id = UUID()
    let name: String
    let displayName: String
    let description: String
    let provider: ModelProvider = .whisperCoreML
    let size: String
    let speed: Double
    let accuracy: Double
    let ramUsage: Double
    let coremlModelId: String
    var isMultilingualModel: Bool {
        supportedLanguages.count > 1
    }
    let supportedLanguages: [String: String]
}

struct Qwen3CoreMLModel: TranscriptionModel {
    let id = UUID()
    let name: String
    let displayName: String
    let description: String
    let provider: ModelProvider = .qwen3CoreML
    let size: String
    let speed: Double
    let accuracy: Double
    let ramUsage: Double
    let coremlModelId: String
    let mlxModelId: String
    let modelSize: Qwen3ASRModelSize
    var isMultilingualModel: Bool {
        supportedLanguages.count > 1
    }
    let supportedLanguages: [String: String]
}
