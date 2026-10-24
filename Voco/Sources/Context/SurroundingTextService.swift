import Foundation
import AppKit
import os

/// Context around the cursor in the target text field.
struct SurroundingTextContext {
    /// Characters immediately before cursor (up to ~100 chars)
    let textBefore: String
    /// Characters immediately after cursor (up to ~100 chars)
    let textAfter: String
    /// Whether context was successfully retrieved
    var isAvailable: Bool { !textBefore.isEmpty || !textAfter.isEmpty }
}

final class SurroundingTextService {
    static let shared = SurroundingTextService()
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "SurroundingText")

    private init() {}

    /// Query surrounding text from the focused element via AX.
    /// Must be called on the main thread. Returns nil if AX unavailable.
    func querySurroundingText(for pid: pid_t) -> SurroundingTextContext? {
        guard AXIsProcessTrusted() else {
            logger.debug("AX not trusted, skipping surrounding text query")
            return nil
        }

        let axApp = AXUIElementCreateApplication(pid)

        // 1. Get focused element
        var focusedObj: AnyObject?
        guard AXUIElementCopyAttributeValue(axApp, kAXFocusedUIElementAttribute as CFString, &focusedObj) == .success else {
            logger.debug("Could not get focused element")
            return nil
        }
        let element = focusedObj as! AXUIElement

        // 2. Verify it's an editable text element
        var roleObj: AnyObject?
        guard AXUIElementCopyAttributeValue(element, kAXRoleAttribute as CFString, &roleObj) == .success,
              let role = roleObj as? String else { return nil }
        let editableRoles: Set<String> = [
            kAXTextFieldRole as String,
            kAXTextAreaRole as String,
            kAXComboBoxRole as String,
        ]
        guard editableRoles.contains(role) else {
            logger.debug("Focused element role \(role) is not editable")
            return nil
        }

        // 3. Get full text value
        var valueObj: AnyObject?
        guard AXUIElementCopyAttributeValue(element, kAXValueAttribute as CFString, &valueObj) == .success,
              let fullText = valueObj as? String else {
            logger.debug("Could not read kAXValueAttribute")
            return nil
        }

        // 4. Get selected text range (cursor position)
        var rangeObj: AnyObject?
        guard AXUIElementCopyAttributeValue(element, kAXSelectedTextRangeAttribute as CFString, &rangeObj) == .success else {
            logger.debug("Could not read kAXSelectedTextRangeAttribute")
            return nil
        }
        var cfRange = CFRange(location: 0, length: 0)
        guard AXValueGetValue(rangeObj as! AXValue, .cfRange, &cfRange) else {
            logger.debug("Could not extract CFRange from AXValue")
            return nil
        }

        // cfRange.location = cursor position (or start of selection)
        // When there's a selection (cfRange.length > 0), use the boundaries of the selection
        let cursorPos = cfRange.location
        let selectionEnd = cfRange.location + cfRange.length
        guard cursorPos >= 0, selectionEnd <= fullText.utf16.count else {
            logger.debug("Cursor position out of bounds: pos=\(cursorPos), selEnd=\(selectionEnd), textLen=\(fullText.utf16.count)")
            return nil
        }

        // 5. Extract before/after context (up to 100 chars each)
        // Use UTF-16 offsets since AX reports positions in UTF-16
        let maxContext = 100

        let utf16 = fullText.utf16
        let beforeStartOffset = max(0, cursorPos - maxContext)
        let beforeStart = String.Index(utf16Offset: beforeStartOffset, in: fullText)
        let beforeEnd = String.Index(utf16Offset: cursorPos, in: fullText)
        let textBefore = String(fullText[beforeStart..<beforeEnd])

        let afterStart = String.Index(utf16Offset: selectionEnd, in: fullText)
        let afterEndOffset = min(fullText.utf16.count, selectionEnd + maxContext)
        let afterEnd = String.Index(utf16Offset: afterEndOffset, in: fullText)
        let textAfter = String(fullText[afterStart..<afterEnd])

        logger.debug("Surrounding text: before=\(textBefore.suffix(20), privacy: .private), after=\(textAfter.prefix(20), privacy: .private)")
        return SurroundingTextContext(textBefore: textBefore, textAfter: textAfter)
    }
}
