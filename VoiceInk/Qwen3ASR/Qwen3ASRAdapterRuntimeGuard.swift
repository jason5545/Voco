import Foundation

enum Qwen3ASRAdapterRuntimeGuard {
    static let longActionCommandAudioThresholdSeconds = 6.0

    static func shouldProbeBaseFallback(
        adapterTranscript: String,
        adapterMetadata: Qwen3ASRAdapterMetadata,
        audioDurationSeconds: Double
    ) -> Bool {
        guard adapterMetadata.adapterApplied,
              audioDurationSeconds >= longActionCommandAudioThresholdSeconds
        else { return false }

        return VoiceCommandService.shared.detectCommand(in: adapterTranscript) != nil
    }

    static func shouldUseBaseFallback(
        adapterTranscript: String,
        baseTranscript: String
    ) -> Bool {
        guard let command = VoiceCommandService.shared.detectCommand(in: adapterTranscript),
              VoiceCommandService.shared.detectCommand(in: baseTranscript) == nil
        else { return false }

        let baseKey = compactTextKey(baseTranscript)
        let commandKeys = commandSurfaceKeys(for: command)
        guard commandKeys.contains(where: { baseKey.contains($0) }) else { return false }

        let shortestCommandLength = commandKeys.map(\.count).min() ?? 0
        return baseKey.count >= shortestCommandLength + 4
    }

    private static func commandSurfaceKeys(for command: VoiceCommand) -> [String] {
        switch command {
        case .deleteAll:
            return ["全部刪除", "全部删除"].map(compactTextKey)
        }
    }

    private static func compactTextKey(_ text: String) -> String {
        let punctuation = CharacterSet(charactersIn: "。，！？、；：.!?,;:「」『』（）()[]【】“”\"'` ")
        let skipped = CharacterSet.whitespacesAndNewlines.union(punctuation)
        let scalars = text
            .precomposedStringWithCompatibilityMapping
            .lowercased()
            .unicodeScalars
            .filter { !skipped.contains($0) }
        return String(String.UnicodeScalarView(scalars))
    }
}
