import Foundation

enum AppDefaults {
    static func registerDefaults() {
        UserDefaults.standard.register(defaults: [
            // Onboarding & General
            "hasCompletedOnboarding": false,
            "enableAnnouncements": true,

            // Clipboard
            "restoreClipboardAfterPaste": true,
            "clipboardRestoreDelay": 2.0,
            "useAppleScriptPaste": false,

            // Audio & Media
            "isSystemMuteEnabled": true,
            "audioResumptionDelay": 0.0,
            "isPauseMediaEnabled": false,
            "isSoundFeedbackEnabled": true,
            CustomSoundManager.SoundType.start.builtInSoundKey: CustomSoundManager.SoundType.start.defaultBuiltInSound.rawValue,
            CustomSoundManager.SoundType.stop.builtInSoundKey: CustomSoundManager.SoundType.stop.defaultBuiltInSound.rawValue,

            // Recording & Transcription
            "IsTextFormattingEnabled": true,
            "IsVADEnabled": true,
            "RemoveFillerWords": true,
            "RemovePunctuation": false,
            "LowercaseTranscription": false,
            "SelectedLanguage": "en",
            "AppendTrailingSpace": true,
            "showLiveTextPreview": false,
            "RecorderType": "mini",

            // Cleanup
            "IsTranscriptionCleanupEnabled": false,
            "TranscriptionRetentionMinutes": 1440,
            "IsAudioCleanupEnabled": false,
            "AudioRetentionPeriod": 7,

            // UI & Behavior
            "IsMenuBarOnly": false,
            "powerModePersistConfig": false,
            // Shortcuts
            "isMiddleClickToggleEnabled": false,
            "middleClickActivationDelay": 200,

            // Enhancement
            "SkipShortEnhancement": true,
            "ShortEnhancementWordThreshold": 3,
            "EnhancementTimeoutSeconds": 7,
            "EnhancementRetryOnTimeout": true,
            PersonalStyleGuardService.enabledKey: true,

            // Model
            "PrewarmModelOnWake": true,
            "ModelKeepAliveSeconds": 420.0,
            "KeepModelAlive": false,
            "KeepModelAliveOnBattery": false,

            // Context-Aware Insertion (fork feature)
            "ContextAwareInsertionEnabled": false,
            "ContextAwareLLMMergeEnabled": false,
            VocoCanonicalizationService.enabledContextPackIDsKey: VocoCanonicalizationService.defaultActiveContextIDs,

            // Chinese Post-Processing
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
        ])

        PunctuationCleanupMode.migrateLegacyUserDefaultIfNeeded()
        PasteMethod.migrateLegacyUserDefaultIfNeeded()
    }
}
