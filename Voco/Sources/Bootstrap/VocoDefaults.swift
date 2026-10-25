import Foundation

enum VocoDefaults {
    static let defaultValues: [String: Any] = [
        "hasCompletedOnboarding": false,
        OnboardingSettings.completedV2Key: false,
        OnboardingSettings.preparedV2Key: false,
        "enableAnnouncements": true,
        "restoreClipboardAfterPaste": true,
        "clipboardRestoreDelay": 2.0,
        "useAppleScriptPaste": false,
        "isSystemMuteEnabled": true,
        "audioResumptionDelay": 0.0,
        "isPauseMediaEnabled": false,
        "isSoundFeedbackEnabled": true,
        CustomSoundManager.SoundType.start.builtInSoundKey: CustomSoundManager.SoundType.start.defaultBuiltInSound.rawValue,
        CustomSoundManager.SoundType.stop.builtInSoundKey: CustomSoundManager.SoundType.stop.defaultBuiltInSound.rawValue,
        "IsTextFormattingEnabled": true,
        "IsVADEnabled": true,
        "RemoveFillerWords": true,
        "RemovePunctuation": false,
        "LowercaseTranscription": false,
        TranscriptionLanguageSupport.selectedLanguageKey: TranscriptionLanguageSupport.defaultLanguageCode,
        "AppendTrailingSpace": true,
        "showLiveTextPreview": false,
        "RecorderType": "mini",
        CleanupSettingsKeys.isTranscriptionCleanupEnabled: false,
        CleanupSettingsKeys.transcriptionRetentionMinutes: 1440,
        CleanupSettingsKeys.isAudioCleanupEnabled: false,
        CleanupSettingsKeys.audioRetentionPeriod: 7,
        AutoLearnSettings.isEnabledKey: false,
        "IsMenuBarOnly": false,
        "powerModePersistConfig": false,
        "isMiddleClickToggleEnabled": false,
        "middleClickActivationDelay": 200,
        "SkipShortEnhancement": true,
        "ShortEnhancementWordThreshold": 3,
        EnhancementRequestSettings.timeoutKey: 7.0,
        EnhancementRequestSettings.retryOnTimeoutKey: true,
        PersonalStyleGuardService.enabledKey: true,
        "PrewarmModelOnWake": true,
        "ModelKeepAliveSeconds": 420.0,
        "KeepModelAlive": false,
        "KeepModelAliveOnBattery": false,
        "ContextAwareInsertionEnabled": false,
        "ContextAwareLLMMergeEnabled": false,
        VocoCanonicalizationService.enabledContextPackIDsKey: VocoCanonicalizationService.defaultActiveContextIDs,
        "ChinesePostProcessingEnabled": false,
        "ChinesePostProcessingOpenCC": true,
        "ChinesePostProcessingPinyin": true,
        "ChinesePostProcessingSpokenPunctuation": true,
        "ChinesePostProcessingHalfWidth": true,
        "ChinesePostProcessingRepetition": true,
        "ChinesePostProcessingConfidence": false,
        "ChinesePostProcessingContextMemory": true,
        "ChinesePostProcessingLLMValidation": true,
        "ChinesePostProcessingLogProbThreshold": -0.3,
        PhoneticShadowLogger.shadowLoggingEnabledKey: false,
        PhoneticShadowLogger.candidateApplicationEnabledKey: false,
        VocoAutoApplyModelService.enabledKey: true,
    ]

    static func registerDefaults(defaults: UserDefaults = .standard) {
        defaults.register(defaults: defaultValues)
        if defaults.double(forKey: "clipboardRestoreDelay") < AppDefaults.minimumClipboardRestoreDelay {
            defaults.set(AppDefaults.minimumClipboardRestoreDelay, forKey: "clipboardRestoreDelay")
        }
        PunctuationCleanupMode.migrateLegacyUserDefaultIfNeeded(in: defaults)
        PasteMethod.migrateLegacyUserDefaultIfNeeded(in: defaults)
    }
}

extension AppDefaults {
    static let minimumClipboardRestoreDelay: TimeInterval = 1.0
    static let defaultValues = VocoDefaults.defaultValues
}
