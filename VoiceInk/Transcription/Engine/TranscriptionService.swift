import Foundation

struct TranscriptionRequestContext {
    let language: String?
    let prompt: String?
    let usesQwen3AudioAdapter: Bool

    static var currentDefaults: TranscriptionRequestContext {
        TranscriptionRequestContext(
            language: UserDefaults.standard.string(forKey: "SelectedLanguage") ?? "auto",
            prompt: UserDefaults.standard.string(forKey: "TranscriptionPrompt"),
            usesQwen3AudioAdapter: true
        )
    }

    init(language: String?, prompt: String?, usesQwen3AudioAdapter: Bool = true) {
        self.language = language
        self.prompt = prompt
        self.usesQwen3AudioAdapter = usesQwen3AudioAdapter
    }

    func withQwen3AudioAdapter(_ enabled: Bool) -> TranscriptionRequestContext {
        TranscriptionRequestContext(
            language: language,
            prompt: prompt,
            usesQwen3AudioAdapter: enabled
        )
    }
}

/// A protocol defining the interface for a transcription service.
/// This allows for a unified way to handle both local and cloud-based transcription models.
protocol TranscriptionService {
    /// Transcribes the audio from a given file URL.
    ///
    /// - Parameters:
    ///   - audioURL: The URL of the audio file to transcribe.
    ///   - model: The `TranscriptionModel` to use for transcription. This provides context about the provider (local, OpenAI, etc.).
    /// - Returns: The transcribed text as a `String`.
    /// - Throws: An error if the transcription fails.
    func transcribe(audioURL: URL, model: any TranscriptionModel, context: TranscriptionRequestContext) async throws -> String
}

extension TranscriptionService {
    func transcribe(audioURL: URL, model: any TranscriptionModel) async throws -> String {
        try await transcribe(audioURL: audioURL, model: model, context: .currentDefaults)
    }
}
