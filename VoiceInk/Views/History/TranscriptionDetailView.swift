import SwiftUI

struct TranscriptionDetailView: View {
    let transcription: Transcription
    var onInfoTap: (() -> Void)?

    private var hasAudioFile: Bool {
        if let urlString = transcription.audioFileURL,
           let url = URL(string: urlString),
           FileManager.default.fileExists(atPath: url.path) {
            return true
        }
        return false
    }

    private var normalizedDisplayText: String {
        guard let normalizedTranscript = transcription.normalizedTranscript?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !normalizedTranscript.isEmpty
        else {
            return transcription.text
        }
        return normalizedTranscript
    }

    private var selectedDisplayText: String {
        guard let selectedCandidate = transcription.selectedCandidate?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !selectedCandidate.isEmpty
        else {
            return transcription.text
        }
        return selectedCandidate
    }

    private var shouldShowSelectedCandidate: Bool {
        selectedDisplayText != normalizedDisplayText
    }

    var body: some View {
        VStack(spacing: 12) {
            ScrollView {
                VStack(spacing: 16) {
                    if let rawTranscript = transcription.rawTranscript,
                       !rawTranscript.isEmpty,
                       rawTranscript != normalizedDisplayText,
                       rawTranscript != selectedDisplayText {
                        MessageBubble(
                            label: "Raw ASR",
                            text: rawTranscript,
                            isEnhanced: false
                        )
                    }

                    MessageBubble(
                        label: transcription.rawTranscript == nil ? "Original" : "Normalized",
                        text: normalizedDisplayText,
                        isEnhanced: false
                    )

                    if shouldShowSelectedCandidate {
                        MessageBubble(
                            label: "Selected",
                            text: selectedDisplayText,
                            isEnhanced: false
                        )
                    }

                    if let enhancedText = transcription.enhancedText {
                        MessageBubble(
                            label: "Enhanced",
                            text: enhancedText,
                            isEnhanced: true
                        )
                    }

                    if let finalPastedText = transcription.finalPastedText,
                       !finalPastedText.isEmpty,
                       finalPastedText != (transcription.enhancedText ?? ""),
                       finalPastedText != transcription.text {
                        MessageBubble(
                            label: "Pasted",
                            text: finalPastedText,
                            isEnhanced: true
                        )
                    }
                }
                .padding(16)
            }

            if hasAudioFile, let urlString = transcription.audioFileURL,
               let url = URL(string: urlString) {
                VStack(spacing: 0) {
                    Divider()

                    AudioPlayerView(url: url, transcription: transcription, onInfoTap: onInfoTap)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .fill(Color(NSColor.controlBackgroundColor).opacity(0.5))
                        )
                        .padding(.horizontal, 12)
                        .padding(.top, 6)
                }
            }
        }
        .padding(.vertical, 12)
        .background(Color(NSColor.controlBackgroundColor))
    }
}

private struct MessageBubble: View {
    let label: String
    let text: String
    let isEnhanced: Bool

    var body: some View {
        HStack(alignment: .bottom) {
            if isEnhanced { Spacer(minLength: 60) }

            VStack(alignment: isEnhanced ? .leading : .trailing, spacing: 4) {
                Text(label)
                    .font(.system(size: 9, weight: .medium))
                    .foregroundColor(.secondary.opacity(0.7))
                    .padding(.horizontal, 12)

                ScrollView {
                    Text(text)
                        .font(.system(size: 14, weight: .regular))
                        .lineSpacing(2)
                        .textSelection(.enabled)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                }
                .frame(maxHeight: 350)
                .background {
                    if isEnhanced {
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(Color.accentColor.opacity(0.2))
                    } else {
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(.thinMaterial)
                            .overlay(
                                RoundedRectangle(cornerRadius: 18, style: .continuous)
                                    .strokeBorder(Color.primary.opacity(0.06), lineWidth: 0.5)
                            )
                    }
                }
                .overlay(alignment: .bottomTrailing) {
                    CopyIconButton(textToCopy: text)
                        .padding(8)
                }
            }

            if !isEnhanced { Spacer(minLength: 60) }
        }
    }


}
