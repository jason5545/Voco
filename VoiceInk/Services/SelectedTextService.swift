import ApplicationServices
import AppKit
import Foundation
import os
import SelectedTextKit

@MainActor
final class SelectedTextService {
    struct FocusedEditableTextInfo {
        let role: String
        let fieldValue: String?
        let selectedRangeLength: Int?
    }

    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "SelectedTextService")
    private static let textManager = SelectedTextManager.shared
    private static let editableRoles: Set<String> = [
        kAXTextFieldRole as String,
        kAXTextAreaRole as String,
        kAXComboBoxRole as String,
    ]

    /// Full retrieval strategy used when richer context is acceptable.
    static func fetchSelectedText() async -> String? {
        await fetchSelectedText(using: [.accessibility, .menuAction, .appleScript])
    }

    /// Low-latency retrieval strategy for recorder startup.
    static func fetchSelectedTextForEditModeDetection() async -> String? {
        await fetchSelectedText(using: [.accessibility])
    }

    /// Electron fallback only: discard results that are identical to the
    /// pre-copy pasteboard value. This can drop a real selection that exactly
    /// matches the clipboard, but that is rarer and cheaper than entering edit
    /// mode from stale clipboard echo.
    static func fetchSelectedTextForElectronFallback() async -> String? {
        let baseline = NSPasteboard.general.string(forType: .string)
        let candidate = await fetchSelectedText()
        if EditModeDetectionPolicy.isClipboardEcho(candidate: candidate, clipboardBaseline: baseline) {
            return nil
        }
        return candidate
    }

    private static func fetchSelectedText(using strategies: [TextStrategy]) async -> String? {
        guard AXIsProcessTrusted() else {
            logger.debug("Accessibility is not trusted; selected text capture skipped")
            return nil
        }

        do {
            return normalized(try await textManager.getSelectedText(strategies: strategies))
        } catch {
            logger.debug("SelectedTextKit failed to capture selected text: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }

    /// Check whether the focused UI element in the given process is an editable text field.
    ///
    /// Uses the element's AX role (TextField / TextArea / ComboBox) rather than
    /// `kAXEditableAttribute`, because many apps (including Electron-based ones)
    /// don't implement that attribute while the role is reliably reported.
    ///
    /// Note: AXWebArea is intentionally excluded — most web content is not editable,
    /// and browsers report contenteditable regions as AXTextArea.
    static func isEditableTextFocused(for pid: pid_t) -> Bool {
        guard let focusedElement = focusedElement(for: pid),
              let role = role(for: focusedElement) else {
            return false
        }
        return editableRoles.contains(role)
    }

    static func focusedEditableTextInfo(for pid: pid_t) -> FocusedEditableTextInfo? {
        guard let focusedElement = focusedElement(for: pid),
              let role = role(for: focusedElement),
              editableRoles.contains(role) else {
            return nil
        }

        return FocusedEditableTextInfo(
            role: role,
            fieldValue: stringAttribute(kAXValueAttribute as CFString, for: focusedElement),
            selectedRangeLength: selectedTextRangeLength(for: focusedElement)
        )
    }

    private static func focusedElement(for pid: pid_t) -> AXUIElement? {
        let axApp = AXUIElementCreateApplication(pid)
        var focusedElement: AnyObject?
        guard AXUIElementCopyAttributeValue(axApp, kAXFocusedUIElementAttribute as CFString, &focusedElement) == .success else {
            return nil
        }
        return focusedElement as! AXUIElement
    }

    private static func role(for element: AXUIElement) -> String? {
        var roleValue: AnyObject?
        guard AXUIElementCopyAttributeValue(element, kAXRoleAttribute as CFString, &roleValue) == .success,
              let role = roleValue as? String else {
            return nil
        }
        return role
    }

    private static func stringAttribute(_ attribute: CFString, for element: AXUIElement) -> String? {
        var value: AnyObject?
        guard AXUIElementCopyAttributeValue(element, attribute, &value) == .success else {
            return nil
        }
        return value as? String
    }

    private static func selectedTextRangeLength(for element: AXUIElement) -> Int? {
        var value: AnyObject?
        guard AXUIElementCopyAttributeValue(element, kAXSelectedTextRangeAttribute as CFString, &value) == .success,
              let value,
              CFGetTypeID(value) == AXValueGetTypeID() else {
            return nil
        }

        let axValue = value as! AXValue
        guard AXValueGetType(axValue) == .cfRange else {
            return nil
        }

        var range = CFRange(location: 0, length: 0)
        guard AXValueGetValue(axValue, .cfRange, &range) else {
            return nil
        }
        return range.length
    }

    static func normalized(_ text: String?) -> String? {
        guard let text else { return nil }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : text
    }
}
