import ApplicationServices
import Foundation

@MainActor
final class SelectedTextService {
    static func fetchSelectedText() async -> String? {
        await VocoSelectionService.fetchSelectedText()
    }

    static func fetchSelectedTextForEditModeDetection() async -> String? {
        await VocoSelectionService.fetchSelectedTextForEditModeDetection()
    }

    static func focusedEditableSelectionEvidence(for pid: pid_t) -> EditableTextSelectionEvidence {
        VocoSelectionService.focusedEditableSelectionEvidence(for: pid)
    }

    static func currentEditableSelectionEvidence(
        for pid: pid_t,
        searchFocusedWindow: Bool
    ) async -> EditableTextSelectionEvidence {
        await VocoSelectionService.currentEditableSelectionEvidence(
            for: pid,
            searchFocusedWindow: searchFocusedWindow
        )
    }

    nonisolated static func focusedWindowEditableSelectionEvidence(
        for pid: pid_t,
        traversalLimit: Int = 4_000
    ) -> EditableTextSelectionEvidence {
        VocoSelectionService.focusedWindowEditableSelectionEvidence(
            for: pid,
            traversalLimit: traversalLimit
        )
    }

    static func normalized(_ text: String?) -> String? {
        VocoSelectionService.normalized(text)
    }
}
