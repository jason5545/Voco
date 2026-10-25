import Foundation

@MainActor
enum VocoDelivery {
    static func paste(
        _ text: String,
        transcription: Transcription,
        output: OutputRuntimeConfiguration,
        sendKey: FinishAndSendKey? = nil,
        isEphemeral: Bool = false,
        dismiss: () async -> Void
    ) async {
        var textToPaste = text
        if let restrictionMessage = LicenseViewModel.shared.usageRestrictionMessage {
            textToPaste = """
                \(restrictionMessage)
                \n\(textToPaste)
                """
        }

        let appendSpace = UserDefaults.standard.bool(forKey: "AppendTrailingSpace")
        let needsTrailingSpace = appendSpace && textToPaste.last?.isWhitespace != true
        let pastedText = textToPaste + (needsTrailingSpace ? " " : "")
        let pasteOutcome = await CursorPaster.startPasteAtCursor(
            pastedText,
            isEphemeral: isEphemeral
        ).value
        if !isEphemeral {
            transcription.recordPasteAttempt(
                text: pastedText,
                didPostCommand: pasteOutcome.result.didPostPasteCommand
            )
        }
        SoundManager.shared.playStopSound()

        let autoSendKey = sendKey ?? (output.outputMode == .paste ? output.autoSendKey : .none)
        if !isEphemeral, autoSendKey.isEnabled && pasteOutcome.result.didPostPasteCommand {
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 500_000_000)
                if let generation = pasteOutcome.autoLearnGeneration {
                    await AutoLearnService.shared.cancelForAutoSend(generation: generation)
                }
                CursorPaster.performSendKey(autoSendKey)
            }
        }

        await dismiss()
    }
}
