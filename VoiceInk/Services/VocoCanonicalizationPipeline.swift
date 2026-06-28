import Foundation
import SwiftData

@MainActor
enum VocoCanonicalizationPipeline {
    static func normalize(
        _ text: String,
        rawTranscript: String?,
        model: any TranscriptionModel,
        modelContext: ModelContext,
        transcription: Transcription? = nil,
        appName: String? = nil,
        windowTitle: String? = nil,
        correctionPolicy: VocoCanonicalizationCorrectionPolicy = .full
    ) -> VocoNormalizationResult {
        normalizeWithAssessment(
            text,
            rawTranscript: rawTranscript,
            model: model,
            modelContext: modelContext,
            transcription: transcription,
            appName: appName,
            windowTitle: windowTitle,
            correctionPolicy: correctionPolicy
        ).normalizationResult
    }

    static func normalizeWithAssessment(
        _ text: String,
        rawTranscript: String?,
        model: any TranscriptionModel,
        modelContext: ModelContext,
        transcription: Transcription? = nil,
        appName: String? = nil,
        windowTitle: String? = nil,
        correctionPolicy: VocoCanonicalizationCorrectionPolicy = .full
    ) -> (normalizationResult: VocoNormalizationResult, confidenceAssessment: VocoConfidenceAssessment) {
        let activeMode = ModeManager.shared.currentActiveConfiguration
        let cleanedText = ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: text)
        let rawResult = VocoCanonicalizationService.shared.normalize(
            cleanedText,
            activeContextIDs: activeContextIDs(mode: activeMode),
            additionalTerms: dictionaryTerms(in: modelContext),
            contextHints: VocoCanonicalizationService.contextHints(
                mode: activeMode,
                appName: appName,
                windowTitle: windowTitle
            ),
            correctionPolicy: correctionPolicy
        )
        let result = removingShortTerminalPunctuation(from: rawResult)
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

    private static func removingShortTerminalPunctuation(from result: VocoNormalizationResult) -> VocoNormalizationResult {
        let cleanedText = ShortUtterancePunctuationCleaner.removeTerminalSentencePunctuation(from: result.normalizedText)
        guard cleanedText != result.normalizedText else { return result }
        return VocoNormalizationResult(
            originalText: result.originalText,
            normalizedText: cleanedText,
            activeContextIDs: result.activeContextIDs,
            replacements: result.replacements,
            suggestions: result.suggestions,
            autoApplyModelVersion: result.autoApplyModelVersion,
            autoApplyModelGeneratedAt: result.autoApplyModelGeneratedAt,
            autoApplyPolicyHitIDs: result.autoApplyPolicyHitIDs
        )
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

    static func activeContextIDs(mode: ModeConfig? = ModeManager.shared.currentActiveConfiguration) -> [String] {
        var ids = VocoCanonicalizationService.enabledContextPackIDs()
        if let mode, mode.isEnabled {
            ids.append("power-mode:\(mode.id.uuidString)")
        }
        return ids
    }

    static func asrEngineID(for model: any TranscriptionModel) -> String {
        "\(model.provider.rawValue):\(model.name)"
    }

    static func selectedLanguageMode(defaults: UserDefaults = .standard) -> String {
        TranscriptionLanguageSupport.selectedLanguage(in: defaults)
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
