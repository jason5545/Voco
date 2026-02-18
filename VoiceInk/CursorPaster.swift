import Foundation
import AppKit

class CursorPaster {

    static func pasteAtCursor(_ text: String) {
        let pasteboard = NSPasteboard.general
        let shouldRestoreClipboard = UserDefaults.standard.bool(forKey: "restoreClipboardAfterPaste")

        var savedContents: [(NSPasteboard.PasteboardType, Data)] = []

        if shouldRestoreClipboard {
            let currentItems = pasteboard.pasteboardItems ?? []

            for item in currentItems {
                for type in item.types {
                    if let data = item.data(forType: type) {
                        savedContents.append((type, data))
                    }
                }
            }
        }

        ClipboardManager.setClipboard(text, transient: shouldRestoreClipboard)

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
            if UserDefaults.standard.bool(forKey: "UseAppleScriptPaste") {
                _ = pasteUsingAppleScript()
            } else {
                pasteUsingCommandV()
            }
        }

        if shouldRestoreClipboard {
            let restoreDelay = UserDefaults.standard.double(forKey: "clipboardRestoreDelay")
            let delay = max(restoreDelay, 0.25)

            DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                if !savedContents.isEmpty {
                    pasteboard.clearContents()
                    for (type, data) in savedContents {
                        pasteboard.setData(data, forType: type)
                    }
                }
            }
        }
    }
    
    private static func pasteUsingAppleScript() -> Bool {
        guard AXIsProcessTrusted() else {
            return false
        }
        
        let script = """
        tell application "System Events"
            keystroke "v" using command down
        end tell
        """
        
        var error: NSDictionary?
        if let scriptObject = NSAppleScript(source: script) {
            _ = scriptObject.executeAndReturnError(&error)
            return error == nil
        }
        return false
    }
    
    private static func pasteUsingCommandV() {
        guard AXIsProcessTrusted() else {
            return
        }
        
        let source = CGEventSource(stateID: .hidSystemState)
        
        let cmdDown = CGEvent(keyboardEventSource: source, virtualKey: 0x37, keyDown: true)
        let vDown = CGEvent(keyboardEventSource: source, virtualKey: 0x09, keyDown: true)
        let vUp = CGEvent(keyboardEventSource: source, virtualKey: 0x09, keyDown: false)
        let cmdUp = CGEvent(keyboardEventSource: source, virtualKey: 0x37, keyDown: false)
        
        cmdDown?.flags = .maskCommand
        vDown?.flags = .maskCommand
        vUp?.flags = .maskCommand
        
        cmdDown?.post(tap: .cghidEventTap)
        vDown?.post(tap: .cghidEventTap)
        vUp?.post(tap: .cghidEventTap)
        cmdUp?.post(tap: .cghidEventTap)
    }

    // Simulate pressing Delete key (deletes only selected text range)
    static func deleteSelection() {
        guard AXIsProcessTrusted() else { return }
        let source = CGEventSource(stateID: .hidSystemState)
        let deleteDown = CGEvent(keyboardEventSource: source, virtualKey: 0x33, keyDown: true)
        let deleteUp = CGEvent(keyboardEventSource: source, virtualKey: 0x33, keyDown: false)
        deleteDown?.post(tap: .cghidEventTap)
        deleteUp?.post(tap: .cghidEventTap)
    }

    // Simulate Cmd+A (select all) followed by Delete
    static func selectAllAndDelete() {
        guard AXIsProcessTrusted() else { return }
        let source = CGEventSource(stateID: .hidSystemState)

        // Cmd+A to select all
        let cmdDown = CGEvent(keyboardEventSource: source, virtualKey: 0x37, keyDown: true)
        let aDown = CGEvent(keyboardEventSource: source, virtualKey: 0x00, keyDown: true)
        let aUp = CGEvent(keyboardEventSource: source, virtualKey: 0x00, keyDown: false)
        let cmdUp = CGEvent(keyboardEventSource: source, virtualKey: 0x37, keyDown: false)

        cmdDown?.flags = .maskCommand
        aDown?.flags = .maskCommand
        aUp?.flags = .maskCommand

        cmdDown?.post(tap: .cghidEventTap)
        aDown?.post(tap: .cghidEventTap)
        aUp?.post(tap: .cghidEventTap)
        cmdUp?.post(tap: .cghidEventTap)

        // Wait for selection to complete
        usleep(50_000)

        // Delete key
        let deleteDown = CGEvent(keyboardEventSource: source, virtualKey: 0x33, keyDown: true)
        let deleteUp = CGEvent(keyboardEventSource: source, virtualKey: 0x33, keyDown: false)
        deleteDown?.post(tap: .cghidEventTap)
        deleteUp?.post(tap: .cghidEventTap)
    }

    // Simulate pressing the Return / Enter key
    static func pressEnter() {
        guard AXIsProcessTrusted() else { return }
        let source = CGEventSource(stateID: .hidSystemState)
        let enterDown = CGEvent(keyboardEventSource: source, virtualKey: 0x24, keyDown: true)
        let enterUp = CGEvent(keyboardEventSource: source, virtualKey: 0x24, keyDown: false)
        enterDown?.post(tap: .cghidEventTap)
        enterUp?.post(tap: .cghidEventTap)
    }
}
