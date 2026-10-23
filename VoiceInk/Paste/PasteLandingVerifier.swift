import AppKit
import ApplicationServices

/// Detects a paste that was dispatched but never landed, so CursorPaster can
/// post it again.
///
/// Gemini's prompt box (a Quill 1 editor) handles paste by moving focus to a
/// hidden `.ql-clipboard` element, letting the browser insert there, and
/// copying the result back from a 1 ms timer. When Chromium's insertion is a
/// little late the timer wins, the hidden element is still empty, and the
/// paste disappears: `paste` and `beforeinput insertFromPaste` fire, `input`
/// never does, and nothing is left behind. Posting the same paste again is
/// then safe and usually lands.
///
/// Verification only runs where that failure was observed (Chromium
/// browsers) and only when a retry cannot duplicate text: the focused element
/// must be an editable AX text element with a readable value and no selected
/// range, and it must still be focused when the value is checked.
enum PasteLandingPolicy {
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

    static let editableRoles: Set<String> = [
        kAXTextFieldRole as String,
        kAXTextAreaRole as String,
        kAXComboBoxRole as String,
    ]

    static let maximumRetries = 2
    /// How long a dispatched paste may take to show up in the AX value
    /// before it is treated as lost.
    static let landingTimeout: TimeInterval = 0.7
    static let pollInterval: TimeInterval = 0.03

    static func shouldVerify(
        bundleID: String?,
        role: String?,
        valueIsReadable: Bool,
        selectedRangeLength: Int?
    ) -> Bool {
        guard let bundleID, chromiumBrowserBundleIDs.contains(bundleID),
              let role, editableRoles.contains(role),
              valueIsReadable,
              selectedRangeLength == 0 else {
            return false
        }
        return true
    }

    enum Observation: Equatable {
        case value(String)
        /// Another element has focus, e.g. Quill's hidden `.ql-clipboard`
        /// for a few milliseconds while it handles the paste.
        case focusElsewhere
        case unreadable
    }

    enum Verdict: Equatable {
        case landed
        case notLanded
        case keepWaiting
    }

    /// A paste landed when the field's value changed. Anything that cannot be
    /// confirmed as "unchanged" ends without a retry, so an unknown state
    /// never produces a second paste.
    static func verdict(before: String, observation: Observation, deadlinePassed: Bool) -> Verdict {
        switch observation {
        case .value(let after) where after != before:
            return .landed
        case .value:
            return deadlinePassed ? .notLanded : .keepWaiting
        case .focusElsewhere:
            return deadlinePassed ? .landed : .keepWaiting
        case .unreadable:
            return .landed
        }
    }
}

/// AX snapshot of the focused text element taken right before a paste.
struct PasteTargetSnapshot {
    let element: AXUIElement
    let value: String

    private static let messagingTimeout: Float = 0.25

    /// Returns a snapshot only when `PasteLandingPolicy` allows verifying
    /// this target.
    static func capture(for app: NSRunningApplication?) -> PasteTargetSnapshot? {
        guard let app, AXIsProcessTrusted() else { return nil }
        let axApp = AXUIElementCreateApplication(app.processIdentifier)
        AXUIElementSetMessagingTimeout(axApp, messagingTimeout)
        guard let focused = copyElement(axApp, kAXFocusedUIElementAttribute) else { return nil }
        AXUIElementSetMessagingTimeout(focused, messagingTimeout)

        let role = copyValue(focused, kAXRoleAttribute) as? String
        let value = copyValue(focused, kAXValueAttribute) as? String
        guard PasteLandingPolicy.shouldVerify(
            bundleID: app.bundleIdentifier,
            role: role,
            valueIsReadable: value != nil,
            selectedRangeLength: selectedRangeLength(of: focused)
        ), let value else {
            return nil
        }
        return PasteTargetSnapshot(element: focused, value: value)
    }

    func observe(pid: pid_t) -> PasteLandingPolicy.Observation {
        let axApp = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(axApp, Self.messagingTimeout)
        guard let focused = Self.copyElement(axApp, kAXFocusedUIElementAttribute) else {
            return .unreadable
        }
        guard CFEqual(focused, element) else { return .focusElsewhere }
        guard let value = Self.copyValue(element, kAXValueAttribute) as? String else {
            return .unreadable
        }
        return .value(value)
    }

    private static func selectedRangeLength(of element: AXUIElement) -> Int? {
        guard let raw = copyValue(element, kAXSelectedTextRangeAttribute),
              CFGetTypeID(raw) == AXValueGetTypeID() else {
            return nil
        }
        var range = CFRange()
        guard AXValueGetValue(raw as! AXValue, .cfRange, &range) else { return nil }
        return range.length
    }

    private static func copyValue(_ element: AXUIElement, _ attribute: String) -> CFTypeRef? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attribute as CFString, &value) == .success else {
            return nil
        }
        return value
    }

    private static func copyElement(_ element: AXUIElement, _ attribute: String) -> AXUIElement? {
        guard let value = copyValue(element, attribute),
              CFGetTypeID(value) == AXUIElementGetTypeID() else {
            return nil
        }
        return (value as! AXUIElement)
    }
}
