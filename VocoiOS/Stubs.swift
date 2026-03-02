// Stubs.swift
// Minimal type stubs so shared TranscriptionModel/PredefinedModels compile on iOS
// without pulling in macOS-only modules
// [AI-Claude: 2026-03-02]

import Foundation

enum Qwen3ASRModelSize {
    case small
    case large
}

class CustomModelManager: ObservableObject {
    static let shared = CustomModelManager()
    @Published var customModels: [CustomCloudModel] = []
}

final class APIKeyManager {
    static let shared = APIKeyManager()
    func getCustomModelAPIKey(forModelId id: UUID) -> String? { nil }
    func saveCustomModelAPIKey(_ key: String, forModelId id: UUID) {}
}

// UncertainWord: original in Qwen3ASR/Qwen3ASRModel.swift (not shared to iOS)
struct UncertainWord {
    let text: String
    let logProb: Double
}
