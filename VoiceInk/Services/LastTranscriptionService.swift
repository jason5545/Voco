import Foundation
import SwiftData

class LastTranscriptionService: ObservableObject {
    
    static func getLastTranscription(from modelContext: ModelContext) -> Transcription? {
        var descriptor = FetchDescriptor<Transcription>(
            sortBy: [SortDescriptor(\.timestamp, order: .reverse)]
        )
        descriptor.fetchLimit = 50
        
        do {
            let transcriptions = try modelContext.fetch(descriptor)
            return transcriptions.first { isReusableOutput($0) }
        } catch {
            print("Error fetching last transcription: \(error)")
            return nil
        }
    }
    
    static func copyLastTranscription(from modelContext: ModelContext) {
        guard let lastTranscription = getLastTranscription(from: modelContext) else {
            Task { @MainActor in
                NotificationManager.shared.showNotification(
                    title: "No transcription available",
                    type: .error
                )
            }
            return
        }
        
        guard let textToCopy = preferredOutputText(for: lastTranscription, preferEnhanced: true) else {
            Task { @MainActor in
                NotificationManager.shared.showNotification(
                    title: "No reusable transcription text",
                    type: .error
                )
            }
            return
        }
        
        let success = ClipboardManager.copyToClipboard(textToCopy)
        
        Task { @MainActor in
            if success {
                NotificationManager.shared.showNotification(
                    title: "Last transcription copied",
                    type: .success
                )
            } else {
                NotificationManager.shared.showNotification(
                    title: "Failed to copy transcription",
                    type: .error
                )
            }
        }
    }

    static func pasteLastTranscription(from modelContext: ModelContext) {
        guard let lastTranscription = getLastTranscription(from: modelContext) else {
            Task { @MainActor in
                NotificationManager.shared.showNotification(
                    title: "No transcription available",
                    type: .error
                )
            }
            return
        }
        
        guard let textToPaste = preferredOutputText(for: lastTranscription, preferEnhanced: false) else {
            Task { @MainActor in
                NotificationManager.shared.showNotification(
                    title: "No reusable transcription text",
                    type: .error
                )
            }
            return
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            CursorPaster.pasteAtCursor(textToPaste)
        }
    }
    
    static func pasteLastEnhancement(from modelContext: ModelContext) {
        guard let lastTranscription = getLastTranscription(from: modelContext) else {
            Task { @MainActor in
                NotificationManager.shared.showNotification(
                    title: "No transcription available",
                    type: .error
                )
            }
            return
        }
        
        guard let textToPaste = preferredOutputText(for: lastTranscription, preferEnhanced: true) else {
            Task { @MainActor in
                NotificationManager.shared.showNotification(
                    title: "No reusable transcription text",
                    type: .error
                )
            }
            return
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            CursorPaster.pasteAtCursor(textToPaste)
        }
    }
    
    static func retryLastTranscription(from modelContext: ModelContext, transcriptionModelManager: TranscriptionModelManager, serviceRegistry: TranscriptionServiceRegistry, enhancementService: AIEnhancementService?) {
        Task { @MainActor in
            guard let lastTranscription = getLastTranscription(from: modelContext),
                  let audioURLString = lastTranscription.audioFileURL,
                  let audioURL = URL(string: audioURLString),
                  FileManager.default.fileExists(atPath: audioURL.path) else {
                NotificationManager.shared.showNotification(
                    title: "Cannot retry: Audio file not found",
                    type: .error
                )
                return
            }

            guard let transcriptionConfiguration = ModeRuntimeResolver.transcriptionConfiguration(
                transcriptionModelManager: transcriptionModelManager
            ) else {
                NotificationManager.shared.showNotification(
                    title: "No transcription model selected",
                    type: .error
                )
                return
            }

            let transcriptionService = AudioTranscriptionService(
                modelContext: modelContext,
                serviceRegistry: serviceRegistry,
                enhancementService: enhancementService
            )
            do {
                let newTranscription = try await transcriptionService.retranscribeAudio(
                    from: audioURL,
                    using: transcriptionConfiguration.model,
                    sourceTranscription: lastTranscription
                )

                guard let textToCopy = preferredOutputText(for: newTranscription, preferEnhanced: true) else {
                    NotificationManager.shared.showNotification(
                        title: "Retry produced no reusable text",
                        type: .error
                    )
                    return
                }

                let success = ClipboardManager.copyToClipboard(textToCopy)
                NotificationManager.shared.showNotification(
                    title: success ? "Copied to clipboard" : "Failed to copy transcription",
                    type: success ? .success : .error
                )
            } catch {
                NotificationManager.shared.showNotification(
                    title: "Retry failed: \(error.localizedDescription)",
                    type: .error
                )
            }
        }
    }

    private static func isReusableOutput(_ transcription: Transcription) -> Bool {
        let status = transcription.transcriptionStatus.flatMap(TranscriptionStatus.init(rawValue:))
        if let status, status != .completed {
            return false
        }

        return preferredOutputText(for: transcription, preferEnhanced: true) != nil ||
            preferredOutputText(for: transcription, preferEnhanced: false) != nil
    }

    private static func preferredOutputText(for transcription: Transcription, preferEnhanced: Bool) -> String? {
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

        return preferred
            .compactMap { reusableText($0) }
            .first
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
