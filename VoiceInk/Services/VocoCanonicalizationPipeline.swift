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
        normalizeWithAssessment(
            text,
            rawTranscript: rawTranscript,
            model: model,
            modelContext: modelContext,
            transcription: transcription
        ).normalizationResult
    }

    static func normalizeWithAssessment(
        _ text: String,
        rawTranscript: String?,
        model: any TranscriptionModel,
        modelContext: ModelContext,
        transcription: Transcription? = nil
    ) -> (normalizationResult: VocoNormalizationResult, confidenceAssessment: VocoConfidenceAssessment) {
        let result = VocoCanonicalizationService.shared.normalize(
            text,
            activeContextIDs: activeContextIDs(),
            additionalTerms: dictionaryTerms(in: modelContext)
        )
        let assessment = confidenceAssessment(
            for: result,
            rawTranscript: rawTranscript,
            modelContext: modelContext,
            excluding: transcription
        )

        transcription?.recordASRMetadata(
            rawTranscript: rawTranscript,
            normalizationResult: result,
            confidenceAssessment: assessment,
            asrEngineID: asrEngineID(for: model),
            languageMode: selectedLanguageMode()
        )

        return (result, assessment)
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

    static func confidenceAssessment(
        for normalizationResult: VocoNormalizationResult,
        rawTranscript: String?,
        modelContext: ModelContext,
        excluding transcription: Transcription? = nil
    ) -> VocoConfidenceAssessment {
        let riskProfile = VocoCorrectionRiskService.profile(
            in: modelContext,
            excluding: transcription?.id
        )
        return VocoConfidenceGateService.shared.assess(
            normalizationResult: normalizationResult,
            rawTranscript: rawTranscript,
            correctionRiskProfile: riskProfile
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

    private static func wordReplacements(in modelContext: ModelContext) -> [WordReplacement] {
        let descriptor = FetchDescriptor<WordReplacement>(
            predicate: #Predicate { $0.isEnabled },
            sortBy: [SortDescriptor(\.originalText)]
        )
        return (try? modelContext.fetch(descriptor)) ?? []
    }

    private static func dictionaryTerms(in modelContext: ModelContext) -> [VocoCanonicalTerm] {
        VocoCanonicalizationService.vocabularyTerms(from: vocabularyWords(in: modelContext)) +
            VocoCanonicalizationService.wordReplacementTerms(from: wordReplacements(in: modelContext))
    }
}
