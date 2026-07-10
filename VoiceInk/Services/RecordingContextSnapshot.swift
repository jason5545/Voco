import Foundation
import AppKit

struct RecordingContextSnapshot {
    var capturedAt = Date()
    var selectedText: String?
    var clipboardText: String?
    var screenText: String?
}

@MainActor
final class RecordingContextSnapshotStore {
    private(set) var snapshot = RecordingContextSnapshot()

    func updateSelectedText(_ text: String?) {
        snapshot.selectedText = Self.normalized(text)
    }

    func updateClipboardText(_ text: String?) {
        snapshot.clipboardText = Self.normalized(text)
    }

    func updateScreenText(_ text: String?) {
        snapshot.screenText = Self.normalized(text)
    }

    private static func normalized(_ text: String?) -> String? {
        guard let text else { return nil }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

@MainActor
enum RecordingContextCaptureService {
    static func startCapture(
        into store: RecordingContextSnapshotStore,
        selectedTextProvider: @escaping @MainActor () async -> String?
    ) -> [Task<Void, Never>] {
        [
            Task { @MainActor in
                let clipboardText = await ClipboardManager.getUserClipboardContentForRecordingContext()
                guard !Task.isCancelled else { return }
                store.updateClipboardText(clipboardText)
            },
            Task { @MainActor in
                guard !Task.isCancelled else { return }
                let selectedText = await selectedTextProvider()
                guard !Task.isCancelled else { return }
                store.updateSelectedText(selectedText)
            },
            Task { @MainActor in
                guard CGPreflightScreenCaptureAccess(), !Task.isCancelled else { return }
                let screenCaptureService = ScreenCaptureService()
                let screenText = await screenCaptureService.captureAndExtractText()
                guard !Task.isCancelled else { return }
                store.updateScreenText(screenText)
            }
        ]
    }
}
