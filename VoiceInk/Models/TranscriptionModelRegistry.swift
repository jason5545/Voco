import Foundation

enum TranscriptionModelRegistry {

    static var models: [any TranscriptionModel] {
        return predefinedModels + CustomCloudModelManager.shared.customModels
    }

    private static let predefinedModels: [any TranscriptionModel] = {
        let nonCloudModels: [any TranscriptionModel] = [
            // Native Apple Model
            NativeAppleModel(
                name: "apple-speech",
                displayName: "Apple Speech",
                description: "Uses the native Apple Speech framework for transcription. Requires macOS 26",
                isMultilingualModel: true,
                supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .nativeApple)
            ),

            // Parakeet Models
            FluidAudioModel(
                name: "parakeet-tdt-0.6b-v2",
                displayName: "Parakeet V2",
                description: "NVIDIA's Parakeet V2 model optimized for lightning-fast English-only transcription",
                size: "474 MB",
                speed: 0.99,
                accuracy: 0.94,
                ramUsage: 0.8,
                supportsStreaming: true,
                supportedLanguages: LanguageDictionary.forProvider(isMultilingual: false, provider: .fluidAudio)
            ),
            FluidAudioModel(
                name: "parakeet-tdt-0.6b-v3",
                displayName: "Parakeet V3",
                description: "Parakeet V3 with English and 25 European language support",
                size: "494 MB",
                speed: 0.99,
                accuracy: 0.94,
                ramUsage: 0.8,
                supportsStreaming: true,
                supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .fluidAudio)
            ),

            // Qwen3-ASR Models
            Qwen3Model(
                name: "qwen3-asr-0.6b-4bit",
                displayName: "Qwen3-ASR 0.6B",
                description: "Alibaba's Qwen3-ASR model with excellent Chinese/English accuracy, 30+ languages and 22 Chinese dialects",
                size: "~400 MB",
                speed: 0.80,
                accuracy: 0.97,
                ramUsage: 1.2,
                modelId: "mlx-community/Qwen3-ASR-0.6B-4bit",
                modelSize: .small,
                supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .qwen3)
            ),
            Qwen3Model(
                name: "qwen3-asr-1.7b-8bit",
                displayName: "Qwen3-ASR 1.7B",
                description: "Larger Qwen3-ASR model with state-of-the-art accuracy across Chinese (WER 2.71%) and English (WER 2.29%)",
                size: "~1.8 GB",
                speed: 0.60,
                accuracy: 0.99,
                ramUsage: 2.5,
                modelId: "mlx-community/Qwen3-ASR-1.7B-8bit",
                modelSize: .large,
                supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .qwen3)
            ),

            // Qwen3-ASR CoreML Hybrid Models (ANE encoder + GPU decoder)
            Qwen3CoreMLModel(
                name: "qwen3-asr-0.6b-coreml-hybrid",
                displayName: "Qwen3-ASR 0.6B (CoreML Hybrid)",
                description: "CoreML encoder on ANE + MLX decoder on GPU. Frees GPU for other tasks while maintaining accuracy.",
                size: "~500 MB (2 components)",
                speed: 0.85,
                accuracy: 0.97,
                ramUsage: 1.2,
                coremlModelId: "aufklarer/Qwen3-ASR-CoreML",
                mlxModelId: "mlx-community/Qwen3-ASR-0.6B-4bit",
                modelSize: .small,
                supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .qwen3CoreML)
            ),

            // Whisper MLX Models
            WhisperMLXModel(
                name: "whisper-large-v3-turbo-mlx-4bit",
                displayName: "Whisper V3 Turbo (4-bit)",
                description: "MLX GPU accelerated. Fastest Whisper model, compact size.",
                size: "~463 MB",
                speed: 0.85,
                accuracy: 0.95,
                ramUsage: 1.0,
                huggingFaceRepo: "mlx-community/whisper-large-v3-turbo-4bit",
                supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .whisperMLX)
            ),
            WhisperMLXModel(
                name: "whisper-large-v2-mlx-4bit",
                displayName: "Whisper Large v2 (4-bit)",
                description: "MLX GPU accelerated. Whisper v2 with improved training data.",
                size: "~877 MB",
                speed: 0.70,
                accuracy: 0.96,
                ramUsage: 1.5,
                huggingFaceRepo: "mlx-community/whisper-large-v2-asr-4bit",
                supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .whisperMLX)
            ),
            WhisperMLXModel(
                name: "whisper-large-v2-mlx-8bit",
                displayName: "Whisper Large v2 (8-bit)",
                description: "MLX GPU accelerated. Higher precision v2 model.",
                size: "~1.64 GB",
                speed: 0.60,
                accuracy: 0.97,
                ramUsage: 2.5,
                huggingFaceRepo: "mlx-community/whisper-large-v2-asr-8bit",
                supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .whisperMLX)
            ),
            WhisperMLXModel(
                name: "whisper-large-v2-mlx-fp16",
                displayName: "Whisper Large v2 (fp16)",
                description: "MLX GPU accelerated. Full precision v2 model, highest accuracy.",
                size: "~3.08 GB",
                speed: 0.50,
                accuracy: 0.98,
                ramUsage: 4.0,
                huggingFaceRepo: "mlx-community/whisper-large-v2-asr-fp16",
                supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .whisperMLX)
            ),

            // Whisper Models
            WhisperModel(
                name: "ggml-tiny",
                displayName: "Tiny",
                size: "75 MB",
                supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .whisper),
                description: "Tiny model, fastest, least accurate",
                speed: 0.95,
                accuracy: 0.6,
                ramUsage: 0.3
            ),
            WhisperModel(
                name: "ggml-tiny.en",
                displayName: "Tiny (English)",
                size: "75 MB",
                supportedLanguages: LanguageDictionary.forProvider(isMultilingual: false, provider: .whisper),
                description: "Tiny model optimized for English, fastest, least accurate",
                speed: 0.95,
                accuracy: 0.65,
                ramUsage: 0.3
            ),
            WhisperModel(
                name: "ggml-base",
                displayName: "Base",
                size: "142 MB",
                supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .whisper),
                description: "Base model, good balance between speed and accuracy, supports multiple languages",
                speed: 0.85,
                accuracy: 0.72,
                ramUsage: 0.5
            ),
            WhisperModel(
                name: "ggml-base.en",
                displayName: "Base (English)",
                size: "142 MB",
                supportedLanguages: LanguageDictionary.forProvider(isMultilingual: false, provider: .whisper),
                description: "Base model optimized for English, good balance between speed and accuracy",
                speed: 0.85,
                accuracy: 0.75,
                ramUsage: 0.5
            ),
            WhisperModel(
                name: "ggml-large-v2",
                displayName: "Large v2",
                size: "2.9 GB",
                supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .whisper),
                description: "Large model v2, slower than Medium but more accurate",
                speed: 0.3,
                accuracy: 0.96,
                ramUsage: 3.8
            ),
            WhisperModel(
                name: "ggml-large-v3",
                displayName: "Large v3",
                size: "2.9 GB",
                supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .whisper),
                description: "Large model v3, very slow but most accurate",
                speed: 0.3,
                accuracy: 0.98,
                ramUsage: 3.9
            ),
            WhisperModel(
                name: "ggml-large-v3-turbo",
                displayName: "Large v3 Turbo",
                size: "1.5 GB",
                supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .whisper),
                description: "Large model v3 Turbo, faster than v3 with similar accuracy",
                speed: 0.75,
                accuracy: 0.97,
                ramUsage: 1.8
            ),
            WhisperModel(
                name: "ggml-large-v3-turbo-q5_0",
                displayName: "Large v3 Turbo (Quantized)",
                size: "547 MB",
                supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .whisper),
                description: "Quantized version of Large v3 Turbo, faster with slightly lower accuracy",
                speed: 0.75,
                accuracy: 0.95,
                ramUsage: 1.0
            )
        ]

        let cloudModels: [any TranscriptionModel] = CloudProviderRegistry.allProviders.flatMap { $0.models }
        return nonCloudModels + coremlModels + cloudModels
    }()

    #if os(iOS)
    private static let coremlModels: [any TranscriptionModel] = [
        WhisperCoreMLModel(
            name: "whisper-small-coreml-int8",
            displayName: "Whisper Small (CoreML)",
            description: "CoreML ANE-optimized. Best balance of accuracy and efficiency for iOS.",
            size: "~233 MB",
            speed: 0.80,
            accuracy: 0.92,
            ramUsage: 0.5,
            coremlModelId: "whisper-small-int8",
            supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .whisperCoreML)
        ),
        WhisperCoreMLModel(
            name: "whisper-base-coreml-int8",
            displayName: "Whisper Base (CoreML)",
            description: "CoreML ANE-optimized. Lightweight option for basic transcription.",
            size: "~71 MB",
            speed: 0.90,
            accuracy: 0.85,
            ramUsage: 0.2,
            coremlModelId: "whisper-base-int8",
            supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .whisperCoreML)
        ),
        WhisperCoreMLModel(
            name: "whisper-medium-coreml-int8",
            displayName: "Whisper Medium (CoreML)",
            description: "CoreML ANE-optimized. Higher accuracy for demanding use cases.",
            size: "~750 MB",
            speed: 0.60,
            accuracy: 0.95,
            ramUsage: 1.2,
            coremlModelId: "whisper-medium-int8",
            supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .whisperCoreML)
        ),
        WhisperCoreMLModel(
            name: "whisper-large-v2-coreml-int8",
            displayName: "Whisper Large v2 (CoreML)",
            description: "CoreML ANE-optimized. Highest accuracy, requires extended memory entitlement on iOS.",
            size: "~1.5 GB",
            speed: 0.45,
            accuracy: 0.97,
            ramUsage: 3.0,
            coremlModelId: "whisper-large-v2-int8",
            supportedLanguages: LanguageDictionary.forProvider(isMultilingual: true, provider: .whisperCoreML)
        )
    ]
    #else
    private static let coremlModels: [any TranscriptionModel] = []
    #endif
}
