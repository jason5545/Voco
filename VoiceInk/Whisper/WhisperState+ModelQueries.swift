import Foundation

extension WhisperState {
    /// Returns the best available Whisper model for file transcription.
    /// File transcription accepts both local (ggml) and WhisperMLX models; Qwen3-ASR is for real-time voice input only.
    var bestLocalModelForFileTranscription: (any TranscriptionModel)? {
        let whisperProviders: Set<ModelProvider> = [.local, .whisperMLX]
        // If current model is already Whisper and downloaded, use it
        if let current = currentTranscriptionModel, whisperProviders.contains(current.provider),
           usableModels.contains(where: { $0.name == current.name }) {
            return current
        }
        // Otherwise find the best downloaded Whisper model (highest accuracy)
        let whisperModels = usableModels.filter { whisperProviders.contains($0.provider) }
        return whisperModels.sorted { m1, m2 in
            let a1 = (m1 as? LocalModel)?.accuracy ?? (m1 as? WhisperMLXModel)?.accuracy ?? 0
            let a2 = (m2 as? LocalModel)?.accuracy ?? (m2 as? WhisperMLXModel)?.accuracy ?? 0
            return a1 > a2
        }.first
    }

    var usableModels: [any TranscriptionModel] {
        allAvailableModels.filter { model in
            switch model.provider {
            case .local:
                return availableModels.contains { $0.name == model.name }
            case .parakeet:
                return isParakeetModelDownloaded(named: model.name)
            case .qwen3:
                if let qwen3Model = model as? Qwen3Model {
                    return isQwen3ModelDownloaded(qwen3Model)
                }
                return false
            case .whisperMLX:
                if let whisperMLXModel = model as? WhisperMLXModel {
                    return isWhisperMLXModelDownloaded(whisperMLXModel)
                }
                return false
            case .nativeApple:
                if #available(macOS 26, *) {
                    return true
                } else {
                    return false
                }
            case .groq:
                return APIKeyManager.shared.hasAPIKey(forProvider: "Groq")
            case .elevenLabs:
                return APIKeyManager.shared.hasAPIKey(forProvider: "ElevenLabs")
            case .deepgram:
                return APIKeyManager.shared.hasAPIKey(forProvider: "Deepgram")
            case .mistral:
                return APIKeyManager.shared.hasAPIKey(forProvider: "Mistral")
            case .gemini:
                return APIKeyManager.shared.hasAPIKey(forProvider: "Gemini")
            case .soniox:
                return APIKeyManager.shared.hasAPIKey(forProvider: "Soniox")
            case .custom:
                // Custom models are always usable since they contain their own API keys
                return true
            }
        }
    }
} 
