import Foundation

struct VocoASRRequestContext {
    let base: TranscriptionRequestContext
    let usesQwen3AudioAdapter: Bool

    var language: String? { base.language }
    var prompt: String? { base.prompt }

    init(
        language: String?,
        prompt: String?,
        usesQwen3AudioAdapter: Bool = true
    ) {
        self.base = TranscriptionRequestContext(language: language, prompt: prompt)
        self.usesQwen3AudioAdapter = usesQwen3AudioAdapter
    }

    init(
        base: TranscriptionRequestContext,
        usesQwen3AudioAdapter: Bool = true
    ) {
        self.base = base
        self.usesQwen3AudioAdapter = usesQwen3AudioAdapter
    }

    func withQwen3AudioAdapter(_ enabled: Bool) -> VocoASRRequestContext {
        VocoASRRequestContext(base: base, usesQwen3AudioAdapter: enabled)
    }
}

extension TranscriptionServiceRegistry {
    func transcribe(
        audioURL: URL,
        model: any TranscriptionModel,
        context: VocoASRRequestContext
    ) async throws -> String {
        if let registry = self as? VocoTranscriptionServiceRegistry, model.provider == .qwen3 {
            return try await registry.qwen3TranscriptionService.transcribe(
                audioURL: audioURL,
                model: model,
                context: context
            )
        }
        return try await transcribe(audioURL: audioURL, model: model, context: context.base)
    }
}
