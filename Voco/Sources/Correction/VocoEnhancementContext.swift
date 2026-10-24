import Foundation
import SwiftData

@MainActor
final class VocoEnhancementContext {
    var cachedAppName: String?
    var cachedWindowTitle: String?
    var cachedSelectedText: String?

    func systemMessage(
        prompt: CustomPrompt,
        configuration: EnhancementRuntimeConfiguration,
        contextSnapshot: RecordingContextSnapshot?,
        customVocabulary: String,
        clipboardText: String?,
        screenText: String?,
        modelContext: ModelContext
    ) -> String {
        let selectedTextContext: String
        if configuration.useSelectedTextContext,
           let selectedText = contextSnapshot?.selectedText ?? cachedSelectedText,
           !selectedText.isEmpty {
            selectedTextContext = "\n\n<CURRENTLY_SELECTED_TEXT>\n\(selectedText)\n</CURRENTLY_SELECTED_TEXT>"
        } else {
            selectedTextContext = ""
        }

        let clipboardContext = if configuration.useClipboardContext,
                              let clipboardText,
                              !clipboardText.isEmpty {
            "\n\n<CLIPBOARD_CONTEXT>\n\(clipboardText)\n</CLIPBOARD_CONTEXT>"
        } else {
            ""
        }

        let screenCaptureContext = if configuration.useScreenCaptureContext,
                                   let screenText,
                                   !screenText.isEmpty {
            "\n\n<CURRENT_WINDOW_CONTEXT>\n\(screenText)\n</CURRENT_WINDOW_CONTEXT>"
        } else {
            ""
        }

        let appContext: String
        if let cachedAppName, !cachedAppName.isEmpty {
            let titlePart = (cachedWindowTitle?.isEmpty == false) ? " - \(cachedWindowTitle!)" : ""
            appContext = "\n\n<ACTIVE_APPLICATION>\n\(cachedAppName)\(titlePart)\n</ACTIVE_APPLICATION>"
        } else {
            appContext = ""
        }

        let allContextSections = selectedTextContext + clipboardContext + screenCaptureContext + appContext
        let customVocabularySection = if !customVocabulary.isEmpty {
            """


            The following are important vocabulary words, proper nouns, and technical terms. When these words or similar-sounding words appear in the <USER_MESSAGE>, ensure they are spelled EXACTLY as shown below:
            <CUSTOM_VOCABULARY>
            \(customVocabulary)
            </CUSTOM_VOCABULARY>
            """
        } else {
            ""
        }

        let postProcessor = ChinesePostProcessingService.shared
        let recentTranscriptionsSection: String
        if postProcessor.isEnabled && postProcessor.isContextMemoryEnabled {
            let recent = postProcessor.contextMemory.getRecent(count: 5)
            recentTranscriptionsSection = recent.isEmpty
                ? ""
                : "\n\n<RECENT_TRANSCRIPTIONS>\n\(recent.enumerated().map { "\($0.offset + 1). \($0.element)" }.joined(separator: "\n"))\n</RECENT_TRANSCRIPTIONS>"
        } else {
            recentTranscriptionsSection = ""
        }

        let uncertainWords = postProcessor.lastUncertainWords
        let uncertainWordsSection = uncertainWords.isEmpty
            ? ""
            : "\n\n<UNCERTAIN_WORDS>\n\(uncertainWords.map { "「\($0.text)」" }.joined(separator: "、"))\n</UNCERTAIN_WORDS>"

        let descriptor = FetchDescriptor<WordReplacement>(
            predicate: #Predicate { $0.isEnabled },
            sortBy: [SortDescriptor(\WordReplacement.dateAdded, order: .reverse)]
        )
        let knownASRErrors = ((try? modelContext.fetch(descriptor)) ?? [])
            .prefix(30)
            .compactMap { replacement -> String? in
                let original = replacement.originalText.split(separator: ",").first?
                    .trimmingCharacters(in: .whitespaces) ?? ""
                guard !original.isEmpty,
                      !replacement.replacementText.isEmpty,
                      original.contains(where: \.isCJK),
                      !CorrectionProtectionList.shared.containsProtectedTerm(in: original) else {
                    return nil
                }
                return "「\(original)」→「\(replacement.replacementText)」"
            }
        let knownASRErrorsSection = knownASRErrors.isEmpty
            ? ""
            : "\n\n<KNOWN_ASR_ERRORS>\n\(knownASRErrors.joined(separator: "\n"))\n</KNOWN_ASR_ERRORS>"

        let finalContextSection = allContextSections
            + customVocabularySection
            + recentTranscriptionsSection
            + uncertainWordsSection
            + knownASRErrorsSection
        return prompt.finalPromptText + finalContextSection
    }

    func clear() {
        cachedAppName = nil
        cachedWindowTitle = nil
        cachedSelectedText = nil
    }
}

extension AIEnhancementService {
    var isEnhancementEnabled: Bool {
        get {
            currentRuntimeConfiguration.isEnabled
        }
        set {
            ModeManager.shared.updateCurrentEffectiveConfiguration { config in
                config.isAIEnhancementEnabled = newValue
            }
            objectWillChange.send()
        }
    }

    var currentRuntimeConfiguration: EnhancementRuntimeConfiguration {
        ModeRuntimeResolver.currentEnhancementConfiguration(
            enhancementService: self,
            aiService: aiService
        )
    }

    var isConfigured: Bool {
        isConfigured(for: currentRuntimeConfiguration)
    }

    var cachedAppName: String? {
        get { vocoContext.cachedAppName }
        set { vocoContext.cachedAppName = newValue }
    }

    var cachedWindowTitle: String? {
        get { vocoContext.cachedWindowTitle }
        set { vocoContext.cachedWindowTitle = newValue }
    }

    var cachedSelectedText: String? {
        get { vocoContext.cachedSelectedText }
        set { vocoContext.cachedSelectedText = newValue }
    }
}
