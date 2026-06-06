import Foundation

enum BackupCategory: String, CaseIterable, Hashable {
    case general
    case prompts
    case powerMode
    case dictionary
    case customModels

    var title: String {
        switch self {
        case .general:
            return "General Settings"
        case .prompts:
            return "Custom Prompts"
        case .powerMode:
            return "Power Mode"
        case .dictionary:
            return "Dictionary"
        case .customModels:
            return "Custom Model Definitions"
        }
    }
}

struct CustomModelBackup: Codable {
    let id: UUID
    let name: String
    let displayName: String
    let description: String
    let apiEndpoint: String
    let modelName: String
    let isMultilingualModel: Bool
    let supportedLanguages: [String: String]
    let apiKey: String?

    init(model: CustomCloudModel) {
        self.id = model.id
        self.name = model.name
        self.displayName = model.displayName
        self.description = model.description
        self.apiEndpoint = model.apiEndpoint
        self.modelName = model.modelName
        self.isMultilingualModel = model.isMultilingualModel
        self.supportedLanguages = model.supportedLanguages
        self.apiKey = nil
    }

    func makeModel() -> CustomCloudModel {
        let model = CustomCloudModel(
            id: id,
            name: name,
            displayName: displayName,
            description: description,
            apiEndpoint: apiEndpoint.trimmingCharacters(in: .whitespacesAndNewlines),
            modelName: modelName.trimmingCharacters(in: .whitespacesAndNewlines),
            isMultilingual: isMultilingualModel,
            supportedLanguages: supportedLanguages
        )

        if let apiKey, !apiKey.isEmpty {
            APIKeyManager.shared.saveCustomModelAPIKey(apiKey, forModelId: id)
        }

        return model
    }
}

struct GeneralBackup: Codable {
    let primaryRecordingShortcut: ShortcutBackup?
    let secondaryRecordingShortcut: ShortcutBackup?
    let pasteLastTranscriptionShortcut: ShortcutBackup?
    let pasteLastEnhancementShortcut: ShortcutBackup?
    let retryLastTranscriptionShortcut: ShortcutBackup?
    let cancelRecorderShortcut: ShortcutBackup?
    let openHistoryWindowShortcut: ShortcutBackup?
    let quickAddToDictionaryShortcut: ShortcutBackup?
    let toggleEnhancementShortcut: ShortcutBackup?
    let primaryRecordingShortcutRawValue: String?
    let secondaryRecordingShortcutRawValue: String?
    let primaryRecordingShortcutModeRawValue: String?
    let secondaryRecordingShortcutModeRawValue: String?
    let isMiddleClickToggleEnabled: Bool?
    let middleClickActivationDelay: Int?
    let launchAtLoginEnabled: Bool?
    let isMenuBarOnly: Bool?
    let recorderType: String?
    let isTranscriptionCleanupEnabled: Bool?
    let transcriptionRetentionMinutes: Int?
    let isAudioCleanupEnabled: Bool?
    let audioRetentionPeriod: Int?

    let isSoundFeedbackEnabled: Bool?
    let isSystemMuteEnabled: Bool?
    let isPauseMediaEnabled: Bool?
    let audioResumptionDelay: Double?
    let isTextFormattingEnabled: Bool?
    let punctuationCleanupMode: PunctuationCleanupMode?
    let removePunctuation: Bool?
    let lowercaseTranscription: Bool?
    let isExperimentalFeaturesEnabled: Bool?
    let restoreClipboardAfterPaste: Bool?
    let clipboardRestoreDelay: Double?
    let pasteMethodRawValue: String?
    let useAppleScriptPaste: Bool?

    // Fork-specific: Chinese post-processing settings
    let correctionProtectedWords: [String]?
    let llmCorrectionExamples: String?
    let llmUserContext: String?

    // Fork-specific: context-aware dictation settings
    let enabledContextPackIDs: [String]?
    let personalStyleGuardEnabled: Bool?
}

struct WordBackup: Codable {
    let word: String

    init(word: String) {
        self.word = word
    }
}

struct WordReplacementBackup: Codable, Equatable {
    let originalText: String
    let replacementText: String
    let isEnabled: Bool
    let source: String
    let hitCount: Int
    let dateAdded: Date
    let lastSeenDate: Date

    init(replacement: WordReplacement) {
        self.originalText = replacement.originalText
        self.replacementText = replacement.replacementText
        self.isEnabled = replacement.isEnabled
        self.source = replacement.source
        self.hitCount = replacement.hitCount
        self.dateAdded = replacement.dateAdded
        self.lastSeenDate = replacement.lastSeenDate
    }

    func makeReplacement(
        originalText: String? = nil,
        replacementText: String? = nil
    ) -> WordReplacement {
        let wordReplacement = WordReplacement(
            originalText: originalText ?? self.originalText,
            replacementText: replacementText ?? self.replacementText,
            dateAdded: dateAdded,
            isEnabled: isEnabled,
            source: source
        )
        wordReplacement.hitCount = max(1, hitCount)
        wordReplacement.lastSeenDate = lastSeenDate
        return wordReplacement
    }
}

struct BackupFile: Codable {
    let version: String
    let customPrompts: [CustomPrompt]
    let powerModeConfigs: [PowerModeConfig]
    let powerModeShortcuts: [String: ShortcutBackup]?
    let vocabularyWords: [WordBackup]?
    let wordReplacements: [String: String]?
    let wordReplacementDetails: [WordReplacementBackup]?
    let generalSettings: GeneralBackup?
    let customEmojis: [String]?
    let customCloudModels: [CustomModelBackup]?

    private enum CodingKeys: String, CodingKey {
        case version, customPrompts, powerModeConfigs, powerModeShortcuts, vocabularyWords, wordReplacements, wordReplacementDetails, generalSettings, customEmojis, customCloudModels
    }

    init(version: String, customPrompts: [CustomPrompt], powerModeConfigs: [PowerModeConfig], powerModeShortcuts: [String: ShortcutBackup]?, vocabularyWords: [WordBackup]?, wordReplacements: [String: String]?, wordReplacementDetails: [WordReplacementBackup]? = nil, generalSettings: GeneralBackup?, customEmojis: [String]?, customCloudModels: [CustomModelBackup]?) {
        self.version = version
        self.customPrompts = customPrompts
        self.powerModeConfigs = powerModeConfigs
        self.powerModeShortcuts = powerModeShortcuts
        self.vocabularyWords = vocabularyWords
        self.wordReplacements = wordReplacements
        self.wordReplacementDetails = wordReplacementDetails
        self.generalSettings = generalSettings
        self.customEmojis = customEmojis
        self.customCloudModels = customCloudModels
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = try container.decodeIfPresent(String.self, forKey: .version) ?? "0.0.0"
        customPrompts = try container.decodeIfPresent([CustomPrompt].self, forKey: .customPrompts) ?? []
        powerModeConfigs = try container.decodeIfPresent([PowerModeConfig].self, forKey: .powerModeConfigs) ?? []
        powerModeShortcuts = try container.decodeIfPresent([String: ShortcutBackup].self, forKey: .powerModeShortcuts)
        vocabularyWords = try container.decodeIfPresent([WordBackup].self, forKey: .vocabularyWords)
        wordReplacements = try container.decodeIfPresent([String: String].self, forKey: .wordReplacements)
        wordReplacementDetails = try container.decodeIfPresent([WordReplacementBackup].self, forKey: .wordReplacementDetails)
        generalSettings = try container.decodeIfPresent(GeneralBackup.self, forKey: .generalSettings)
        customEmojis = try container.decodeIfPresent([String].self, forKey: .customEmojis)
        customCloudModels = try container.decodeIfPresent([CustomModelBackup].self, forKey: .customCloudModels)
    }
}
