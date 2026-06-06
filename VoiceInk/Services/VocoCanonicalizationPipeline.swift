import Foundation
import SwiftData

@MainActor
enum VocoCanonicalizationPipeline {
    static func normalize(
        _ text: String,
        rawTranscript: String?,
        model: any TranscriptionModel,
        modelContext: ModelContext,
        transcription: Transcription? = nil
    ) -> VocoNormalizationResult {
        let result = VocoCanonicalizationService.shared.normalize(
            text,
            activeContextIDs: activeContextIDs(),
            additionalTerms: VocoCanonicalizationService.vocabularyTerms(from: vocabularyWords(in: modelContext))
        )
        let assessment = confidenceAssessment(for: result, rawTranscript: rawTranscript)

        transcription?.recordASRMetadata(
            rawTranscript: rawTranscript,
            normalizationResult: result,
            confidenceAssessment: assessment,
            asrEngineID: asrEngineID(for: model),
            languageMode: selectedLanguageMode()
        )

        return result
    }

    static func confidenceAssessment(
        for normalizationResult: VocoNormalizationResult,
        rawTranscript: String?
    ) -> VocoConfidenceAssessment {
        VocoConfidenceGateService.shared.assess(
            normalizationResult: normalizationResult,
            rawTranscript: rawTranscript
        )
    }

    static func activeContextIDs() -> [String] {
        var ids = VocoCanonicalizationService.enabledContextPackIDs()
        if let powerMode = PowerModeManager.shared.currentActiveConfiguration,
           powerMode.isEnabled {
            ids.append("power-mode:\(powerMode.id.uuidString)")
        }
        return ids
    }

    static func asrEngineID(for model: any TranscriptionModel) -> String {
        "\(model.provider.rawValue):\(model.name)"
    }

    static func selectedLanguageMode() -> String {
        UserDefaults.standard.string(forKey: "SelectedLanguage") ?? "auto"
    }

    private static func vocabularyWords(in modelContext: ModelContext) -> [String] {
        let descriptor = FetchDescriptor<VocabularyWord>(sortBy: [SortDescriptor(\.word)])
        return ((try? modelContext.fetch(descriptor)) ?? []).map(\.word)
    }
}
