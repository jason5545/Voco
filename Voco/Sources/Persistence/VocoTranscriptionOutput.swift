import Foundation

enum VocoTranscriptionOutput {
    static func isReusable(_ transcription: Transcription) -> Bool {
        let status = transcription.transcriptionStatus.flatMap(TranscriptionStatus.init(rawValue:))
        if let status, status != .completed {
            return false
        }

        return preferredText(for: transcription, preferEnhanced: true) != nil
            || preferredText(for: transcription, preferEnhanced: false) != nil
    }

    static func preferredText(for transcription: Transcription, preferEnhanced: Bool) -> String? {
        let preferred: [String?]
        if preferEnhanced {
            preferred = [
                transcription.enhancedText,
                transcription.finalPastedText,
                transcription.selectedCandidate,
                transcription.normalizedTranscript,
                transcription.text,
            ]
        } else {
            preferred = [
                transcription.finalPastedText,
                transcription.selectedCandidate,
                transcription.normalizedTranscript,
                transcription.text,
            ]
        }

        return preferred.compactMap(reusableText).first
    }

    private static func reusableText(_ text: String?) -> String? {
        guard let trimmed = text?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty,
              trimmed != Transcription.canceledTranscriptionText,
              !trimmed.hasPrefix("Transcription Failed:"),
              !trimmed.hasPrefix("Enhancement failed:")
        else {
            return nil
        }
        return trimmed
    }
}

extension LastTranscriptionService {
    static func isReusableOutput(_ transcription: Transcription) -> Bool {
        VocoTranscriptionOutput.isReusable(transcription)
    }

    static func preferredOutputText(for transcription: Transcription, preferEnhanced: Bool) -> String? {
        VocoTranscriptionOutput.preferredText(for: transcription, preferEnhanced: preferEnhanced)
    }
}
