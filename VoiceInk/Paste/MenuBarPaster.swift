import AppKit
import ApplicationServices
import Carbon

/// Decides when a paste should go through the frontmost app's Edit > Paste
/// menu item instead of a synthetic Cmd+V.
///
/// An input method (RIME/Squirrel, Zhuyin, Pinyin, ...) sees every key event
/// before the app does. In some Chromium text fields — gemini.google.com and
/// Taiwan Mobile's support chat were the reported ones — it can be left
/// holding per-field state that swallows Voco's Cmd+V: the key event arrives,
/// the field keeps focus, and nothing is pasted, while pressing Edit > Paste
/// in the same field still works. The menu item never passes through the input
/// method, so it is used there; every other app keeps the Cmd+V path.
enum MenuPastePolicy {
    static let chromiumBrowserBundleIDs: Set<String> = [
        "net.imput.helium",
        "com.google.Chrome",
        "com.google.Chrome.beta",
        "com.google.Chrome.dev",
        "com.google.Chrome.canary",
        "org.chromium.Chromium",
        "com.brave.Browser",
        "com.microsoft.edgemac",
        "company.thebrowser.Browser",
        "company.thebrowser.dia",
        "com.vivaldi.Vivaldi",
        "com.operasoftware.Opera",
    ]

    /// `inputSourceType` is the current keyboard input source's
    /// `kTISPropertyInputSourceType`. Plain keyboard layouts (ABC, US, Dvorak)
    /// hand Cmd+V straight to the app, so only input methods switch paths.
    static func shouldUseMenuPaste(frontmostBundleID: String?, inputSourceType: String?) -> Bool {
        guard let frontmostBundleID,
              chromiumBrowserBundleIDs.contains(frontmostBundleID),
              let inputSourceType else {
            return false
        }
        return inputSourceType != (kTISTypeKeyboardLayout as String)
    }

    static func currentInputSourceType() -> String? {
        guard let source = TISCopyCurrentKeyboardInputSource()?.takeRetainedValue(),
              let raw = TISGetInputSourceProperty(source, kTISPropertyInputSourceType) else {
            return nil
        }
        return Unmanaged<CFString>.fromOpaque(raw).takeUnretainedValue() as String
    }
}

enum MenuBarPaster {
    /// Per-call AX timeout, so a busy target cannot stall the paste.
    private static let messagingTimeout: Float = 0.25
    /// Give up and fall back to Cmd+V once the menu walk takes this long.
    private static let searchBudget: TimeInterval = 0.5

    /// Presses the enabled menu item bound to plain Cmd+V in the app's menu
    /// bar. Matches the shortcut rather than the title, so localized menus
    /// ("貼上", "Paste") behave the same. Returns false when no such item is
    /// found, it is disabled, or the press fails.
    @MainActor
    static func pressPasteMenuItem(pid: pid_t) -> Bool {
        let app = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(app, messagingTimeout)
        guard let menuBar = element(app, kAXMenuBarAttribute) else {
            return false
        }

        let deadline = Date().addingTimeInterval(searchBudget)
        for barItem in children(of: menuBar) {
            for menu in children(of: barItem) {
                for item in children(of: menu) {
                    guard Date() < deadline else { return false }
                    guard isPlainCommandV(item) else { continue }
                    guard bool(item, kAXEnabledAttribute) == true else { return false }
                    return AXUIElementPerformAction(item, kAXPressAction as CFString) == .success
                }
            }
        }
        return false
    }

    private static func isPlainCommandV(_ item: AXUIElement) -> Bool {
        guard let commandChar = string(item, kAXMenuItemCmdCharAttribute),
              commandChar.caseInsensitiveCompare("V") == .orderedSame else {
            return false
        }
        // kAXMenuItemModifierNone (0) means Command alone; a missing value is
        // treated the same way by AppKit.
        return (int(item, kAXMenuItemCmdModifiersAttribute) ?? 0) == 0
    }

    private static func children(of element: AXUIElement) -> [AXUIElement] {
        AXUIElementSetMessagingTimeout(element, messagingTimeout)
        return (value(element, kAXChildrenAttribute) as? [AXUIElement]) ?? []
    }

    private static func value(_ element: AXUIElement, _ attribute: String) -> CFTypeRef? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attribute as CFString, &value) == .success else {
            return nil
        }
        return value
    }

    private static func element(_ element: AXUIElement, _ attribute: String) -> AXUIElement? {
        guard let value = value(element, attribute),
              CFGetTypeID(value) == AXUIElementGetTypeID() else {
            return nil
        }
        return (value as! AXUIElement)
    }

    private static func string(_ element: AXUIElement, _ attribute: String) -> String? {
        value(element, attribute) as? String
    }

    private static func int(_ element: AXUIElement, _ attribute: String) -> Int? {
        (value(element, attribute) as? NSNumber)?.intValue
    }

    private static func bool(_ element: AXUIElement, _ attribute: String) -> Bool? {
        (value(element, attribute) as? NSNumber)?.boolValue
    }
}
