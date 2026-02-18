import AppKit
import os

/// Reads text field content via the macOS Accessibility API.
enum TextFieldReader {
    private static let logger = Logger(subsystem: "com.jasonchien.voco", category: "TextFieldReader")

    /// Reads the value of the currently focused text field.
    /// Returns `nil` if Accessibility is not trusted, the focused element has no value,
    /// or the field is a secure (password) text field.
    static func readFocusedTextFieldValue() -> String? {
        guard AXIsProcessTrusted() else {
            logger.debug("Accessibility not trusted, skipping text field read")
            return nil
        }

        let systemWide = AXUIElementCreateSystemWide()
        var focusedElement: AnyObject?
        let focusResult = AXUIElementCopyAttributeValue(systemWide, kAXFocusedUIElementAttribute as CFString, &focusedElement)
        guard focusResult == .success, let element = focusedElement else {
            logger.debug("No focused UI element found")
            return nil
        }

        // Skip secure text fields (password fields)
        var subrole: AnyObject?
        AXUIElementCopyAttributeValue(element as! AXUIElement, kAXSubroleAttribute as CFString, &subrole)
        if let subroleStr = subrole as? String, subroleStr == kAXSecureTextFieldSubrole as String {
            logger.debug("Focused element is a secure text field, skipping")
            return nil
        }

        var value: AnyObject?
        let valueResult = AXUIElementCopyAttributeValue(element as! AXUIElement, kAXValueAttribute as CFString, &value)
        guard valueResult == .success, let stringValue = value as? String else {
            logger.debug("Could not read value from focused element")
            return nil
        }

        return stringValue
    }

    /// Returns the bundle identifier of the frontmost application.
    static func frontmostAppBundleID() -> String? {
        return NSWorkspace.shared.frontmostApplication?.bundleIdentifier
    }
}
