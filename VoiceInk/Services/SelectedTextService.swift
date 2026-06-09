import ApplicationServices
import Foundation
import os
import SelectedTextKit

@MainActor
final class SelectedTextService {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "SelectedTextService")
    private static let textManager = SelectedTextManager.shared

    /// Full retrieval strategy used when richer context is acceptable.
    static func fetchSelectedText() async -> String? {
        await fetchSelectedText(using: [.accessibility, .menuAction, .appleScript])
    }

    /// Low-latency retrieval strategy for recorder startup.
    static func fetchSelectedTextForEditModeDetection() async -> String? {
        await fetchSelectedText(using: [.accessibility])
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
        let axApp = AXUIElementCreateApplication(pid)
        var focusedElement: AnyObject?
        guard AXUIElementCopyAttributeValue(axApp, kAXFocusedUIElementAttribute as CFString, &focusedElement) == .success else {
            return false
        }
        let element = focusedElement as! AXUIElement

        var roleValue: AnyObject?
        guard AXUIElementCopyAttributeValue(element, kAXRoleAttribute as CFString, &roleValue) == .success,
              let role = roleValue as? String else {
            return false
        }

        let editableRoles: Set<String> = [
            kAXTextFieldRole as String,
            kAXTextAreaRole as String,
            kAXComboBoxRole as String,
        ]
        return editableRoles.contains(role)
    }

    private static func normalized(_ text: String?) -> String? {
        guard let text else { return nil }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
