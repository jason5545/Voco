import Foundation

@MainActor
final class VocoTranscriptionServiceRegistry: TranscriptionServiceRegistry {
    private(set) lazy var qwen3TranscriptionService = Qwen3TranscriptionService()
    private(set) lazy var whisperMLXTranscriptionService = WhisperMLXTranscriptionService()
    private(set) lazy var whisperCoreMLTranscriptionService = WhisperCoreMLTranscriptionService()
    private(set) lazy var qwen3CoreMLTranscriptionService = Qwen3CoreMLTranscriptionService()

    override func service(for provider: ModelProvider) -> TranscriptionService {
        switch provider {
        case .qwen3:
            return qwen3TranscriptionService
        case .whisperMLX:
            return whisperMLXTranscriptionService
        case .whisperCoreML:
            return whisperCoreMLTranscriptionService
        case .qwen3CoreML:
            return qwen3CoreMLTranscriptionService
        default:
            return super.service(for: provider)
        }
    }

    override func cleanup() async {
        await super.cleanup()
        await whisperMLXTranscriptionService.cleanup()
        await qwen3TranscriptionService.cleanup()
        await whisperCoreMLTranscriptionService.cleanup()
        await qwen3CoreMLTranscriptionService.cleanup()
    }
}
