import Foundation
import AppKit
import SelectedTextKit

class SelectedTextService {
    static func fetchSelectedText() async -> String? {
        let strategies: [TextStrategy] = [.accessibility, .menuAction]
        do {
            let selectedText = try await SelectedTextManager.shared.getSelectedText(strategies: strategies)
            return selectedText
        } catch {
            print("Failed to get selected text: \(error)")
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
}
