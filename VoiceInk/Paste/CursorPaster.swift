import Foundation
import AppKit

class CursorPaster {
    enum PasteResult: Equatable {
        case commandPosted
        case commandNotPosted

        var didPostPasteCommand: Bool {
            self == .commandPosted
        }
    }

    static func pasteAtCursor(_ text: String) {
        VocoPasteService.pasteAtCursor(text)
    }

    @MainActor
    @discardableResult
    static func startPasteAtCursor(_ text: String) -> Task<PasteResult, Never> {
        VocoPasteService.startPasteAtCursor(text)
    }

    @MainActor
    static func pasteAtCursorAndWaitUntilPosted(
        _ text: String,
        targetPID: pid_t? = nil,
        beforePosting: (@MainActor () async -> Bool)? = nil
    ) async -> PasteResult {
        await VocoPasteService.pasteAtCursorAndWaitUntilPosted(
            text,
            targetPID: targetPID,
            beforePosting: beforePosting
        )
    }

    static func deleteSelection() {
        VocoPasteService.deleteSelection()
    }

    static func selectAllAndDelete() {
        VocoPasteService.selectAllAndDelete()
    }

    static func performAutoSend(_ key: AutoSendKey) {
        VocoPasteService.performAutoSend(key)
    }
}
