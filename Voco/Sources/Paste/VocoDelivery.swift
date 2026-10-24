import Foundation

@MainActor
enum VocoDelivery {
    static func paste(
        _ text: String,
        transcription: Transcription,
        output: OutputRuntimeConfiguration,
        dismiss: () async -> Void
    ) async {
        var textToPaste = text
        if let restrictionMessage = LicenseViewModel().usageRestrictionMessage {
            textToPaste = """
                \(restrictionMessage)
                \n\(textToPaste)
                """
        }

        let appendSpace = UserDefaults.standard.bool(forKey: "AppendTrailingSpace")
        let needsTrailingSpace = appendSpace && textToPaste.last?.isWhitespace != true
        let pastedText = textToPaste + (needsTrailingSpace ? " " : "")
        let pasteResult = await CursorPaster.startPasteAtCursor(pastedText).value
        transcription.recordPasteAttempt(
            text: pastedText,
            didPostCommand: pasteResult.didPostPasteCommand
        )
        SoundManager.shared.playStopSound()

        let autoSendKey = output.outputMode == .paste ? output.autoSendKey : .none
        if autoSendKey.isEnabled {
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 500_000_000)
                CursorPaster.performAutoSend(autoSendKey)
            }
        }

        await dismiss()
    }
}
