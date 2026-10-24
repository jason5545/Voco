#!/usr/bin/env swift
// test_chrome_ax.swift
// Diagnostic script: Query Chrome's AX tree to understand URL bar behavior.
// Usage: swift scripts/test_chrome_ax.swift
// Requirement: Chrome must be frontmost with text selected in URL bar (or a web text field).

import AppKit
import ApplicationServices

// MARK: - Helpers

func getAttributeString(_ element: AXUIElement, _ attr: String) -> String? {
    var value: AnyObject?
    let err = AXUIElementCopyAttributeValue(element, attr as CFString, &value)
    guard err == .success else { return nil }
    return value as? String
}

func getAttributeBool(_ element: AXUIElement, _ attr: String) -> Bool? {
    var value: AnyObject?
    let err = AXUIElementCopyAttributeValue(element, attr as CFString, &value)
    guard err == .success else { return nil }
    return (value as? NSNumber)?.boolValue
}

func getAttributeNames(_ element: AXUIElement) -> [String] {
    var names: CFArray?
    let err = AXUIElementCopyAttributeNames(element, &names)
    guard err == .success, let cfNames = names else { return [] }
    return cfNames as! [String]
}

func describeAXError(_ err: AXError) -> String {
    switch err {
    case .success: return "success"
    case .failure: return "failure"
    case .illegalArgument: return "illegalArgument"
    case .invalidUIElement: return "invalidUIElement"
    case .invalidUIElementObserver: return "invalidUIElementObserver"
    case .cannotComplete: return "cannotComplete"
    case .attributeUnsupported: return "attributeUnsupported"
    case .actionUnsupported: return "actionUnsupported"
    case .notificationUnsupported: return "notificationUnsupported"
    case .notImplemented: return "notImplemented"
    case .notificationAlreadyRegistered: return "notificationAlreadyRegistered"
    case .notificationNotRegistered: return "notificationNotRegistered"
    case .apiDisabled: return "apiDisabled"
    case .noValue: return "noValue"
    case .parameterizedAttributeUnsupported: return "parameterizedAttributeUnsupported"
    case .notEnoughPrecision: return "notEnoughPrecision"
    @unknown default: return "unknown(\(err.rawValue))"
    }
}

// MARK: - Main

guard AXIsProcessTrusted() else {
    print("ERROR: Accessibility permission not granted. Open System Settings > Privacy & Security > Accessibility and add Terminal/cmux.")
    exit(1)
}

// Find Chrome
guard let chromeApp = NSWorkspace.shared.runningApplications.first(where: {
    $0.bundleIdentifier == "com.google.Chrome"
}) else {
    print("ERROR: Chrome is not running.")
    exit(1)
}

let chromePid = chromeApp.processIdentifier
let isFrontmost = chromeApp.isActive
print("=== Chrome AX Diagnostic ===")
print("Chrome PID: \(chromePid)")
print("Chrome is frontmost: \(isFrontmost)")
if !isFrontmost {
    print("WARNING: Chrome is NOT the frontmost app. Results may not reflect actual usage.")
}
print()

let axApp = AXUIElementCreateApplication(chromePid)

// 1. Focused UI Element
print("--- Focused UI Element ---")
var focusedObj: AnyObject?
let focusedErr = AXUIElementCopyAttributeValue(axApp, kAXFocusedUIElementAttribute as CFString, &focusedObj)
print("AXFocusedUIElement result: \(describeAXError(focusedErr))")

guard focusedErr == .success else {
    print("Cannot get focused element. focusedElementUnavailable would be TRUE in cache.")
    print("This means the deferred path (isEditable || focusedElementUnavailable) would trigger.")
    exit(0)
}

let focused = focusedObj as! AXUIElement

// 2. Role and Subrole
let role = getAttributeString(focused, kAXRoleAttribute as String)
let subrole = getAttributeString(focused, kAXSubroleAttribute as String)
let roleDescription = getAttributeString(focused, kAXRoleDescriptionAttribute as String)
let identifier = getAttributeString(focused, kAXIdentifierAttribute as String)
let description_attr = getAttributeString(focused, kAXDescriptionAttribute as String)

print("Role: \(role ?? "(nil)")")
print("Subrole: \(subrole ?? "(nil)")")
print("RoleDescription: \(roleDescription ?? "(nil)")")
print("Identifier: \(identifier ?? "(nil)")")
print("Description: \(description_attr ?? "(nil)")")

// 3. Editable role check (same logic as EditModeCacheService.performAXPoll)
let editableRoles: Set<String> = ["AXTextField", "AXTextArea", "AXComboBox"]
let isEditable = role.map { editableRoles.contains($0) } ?? false
print()
print("--- Editability Analysis ---")
print("Role is in editableRoles set: \(isEditable)")
print("  (editableRoles = \(editableRoles.sorted()))")

// 4. All attribute names
let attrNames = getAttributeNames(focused)
print()
print("--- All Attributes (\(attrNames.count)) ---")
for name in attrNames.sorted() {
    print("  \(name)")
}

// 5. Selected text via AX
print()
print("--- Selected Text Queries ---")

// Direct kAXSelectedTextAttribute
var selectedTextObj: AnyObject?
let selTextErr = AXUIElementCopyAttributeValue(focused, kAXSelectedTextAttribute as CFString, &selectedTextObj)
print("kAXSelectedTextAttribute result: \(describeAXError(selTextErr))")
if selTextErr == .success, let txt = selectedTextObj as? String {
    print("  Selected text: \"\(txt)\"")
} else {
    print("  Selected text: (not available)")
}

// kAXValueAttribute (full field content)
var valueObj: AnyObject?
let valueErr = AXUIElementCopyAttributeValue(focused, kAXValueAttribute as CFString, &valueObj)
print("kAXValueAttribute result: \(describeAXError(valueErr))")
if valueErr == .success, let txt = valueObj as? String {
    print("  Value: \"\(txt)\"")
}

// kAXSelectedTextRangeAttribute
var rangeObj: AnyObject?
let rangeErr = AXUIElementCopyAttributeValue(focused, kAXSelectedTextRangeAttribute as CFString, &rangeObj)
print("kAXSelectedTextRangeAttribute result: \(describeAXError(rangeErr))")
if rangeErr == .success {
    let rangeValue = rangeObj as! AXValue
    var cfRange = CFRange(location: 0, length: 0)
    AXValueGetValue(rangeValue, .cfRange, &cfRange)
    print("  Selected range: location=\(cfRange.location), length=\(cfRange.length)")
}

// 6. System-wide focused element (what SelectedTextKit uses)
print()
print("--- System-wide Element ---")
let systemWide = AXUIElementCreateSystemWide()
var sysFocusedObj: AnyObject?
let sysFocusedErr = AXUIElementCopyAttributeValue(systemWide, kAXFocusedUIElementAttribute as CFString, &sysFocusedObj)
print("System-wide AXFocusedUIElement result: \(describeAXError(sysFocusedErr))")
if sysFocusedErr == .success {
    let sysFocused = sysFocusedObj as! AXUIElement
    let sysRole = getAttributeString(sysFocused, kAXRoleAttribute as String)
    print("System-wide focused role: \(sysRole ?? "(nil)")")

    var sysSelTextObj: AnyObject?
    let sysSelErr = AXUIElementCopyAttributeValue(sysFocused, kAXSelectedTextAttribute as CFString, &sysSelTextObj)
    print("System-wide selected text result: \(describeAXError(sysSelErr))")
    if sysSelErr == .success, let txt = sysSelTextObj as? String {
        print("  System-wide selected text: \"\(txt)\"")
    }
}

// 7. Walk parent chain to understand element hierarchy
print()
print("--- Parent Chain ---")
var current = focused
for depth in 0..<10 {
    let r = getAttributeString(current, kAXRoleAttribute as String) ?? "(nil)"
    let sr = getAttributeString(current, kAXSubroleAttribute as String) ?? "(nil)"
    let t = getAttributeString(current, kAXTitleAttribute as String) ?? "(nil)"
    print("  [\(depth)] Role=\(r), Subrole=\(sr), Title=\(t)")

    var parentObj: AnyObject?
    let parentErr = AXUIElementCopyAttributeValue(current, kAXParentAttribute as CFString, &parentObj)
    guard parentErr == .success else { break }
    current = parentObj as! AXUIElement
}

// 8. Simulate what EditModeCacheService.performAXPoll would conclude
print()
print("=== Simulated EditModeCacheService.performAXPoll Result ===")
print("isEditable: \(isEditable)")
print("focusedElementUnavailable: false (focused element was obtainable)")
let hasSelectedText: Bool
if selTextErr == .success, let txt = selectedTextObj as? String, !txt.isEmpty {
    hasSelectedText = true
    print("selectedText: \"\(txt)\"")
} else {
    hasSelectedText = false
    print("selectedText: nil (AX query returned \(describeAXError(selTextErr)))")
}

print()
print("=== Predicted detectEditMode() Behavior ===")
if isEditable && hasSelectedText {
    print("PATH: Happy path (line 255)")
    print("  -> isEditMode = true, editModeSelectedText set directly from cache")
} else if isEditable || false /* focusedElementUnavailable */ {
    print("PATH: Deferred menuAction fallback (line 258)")
    print("  -> isEditMode initially false, Task spawns fetchSelectedText([.accessibility, .menuAction])")
    print("  -> .accessibility will likely fail again (same AX issue)")
    print("  -> .menuAction will try Edit > Copy menu action (simulated Cmd+C)")
    if isEditable {
        print("  -> TIMING ISSUE: deferred Task sets isEditMode=true AFTER toggleRecord already started")
        print("     Line 136: engine.toggleRecord() runs synchronously after detectEditMode()")
        print("     The deferred Task hasn't completed yet when recording starts.")
    }
} else {
    print("PATH: Not editable, Edit Mode disabled (line 272)")
    print("  -> isEditMode = false")
}

print()
print("=== Summary ===")
print("Chrome URL bar AX role: \(role ?? "(nil)")")
print("Is in editable roles: \(isEditable)")
print("AX selected text available: \(hasSelectedText)")
if isEditable && !hasSelectedText {
    print("")
    print("ROOT CAUSE ANALYSIS:")
    print("The Chrome URL bar reports role=\(role ?? "?") which IS in the editable set,")
    print("but kAXSelectedTextAttribute returns \(describeAXError(selTextErr)).")
    print("This means the cache has isEditable=true, selectedText=nil.")
    print("detectEditMode() enters the deferred path (line 258-270),")
    print("which spawns a background Task to call fetchSelectedText().")
    print("")
    print("HOWEVER: the deferred Task runs CONCURRENTLY with engine.toggleRecord().")
    print("At line 131: detectEditMode(engine) runs (sets isEditMode=false, spawns Task)")
    print("At line 136: engine.toggleRecord() runs IMMEDIATELY after")
    print("The engine captures forkState.isEditMode at recording start (line 22 of ForkFeatures),")
    print("which is still false because the deferred Task hasn't completed yet.")
    print("")
    print("Even if fetchSelectedText() eventually succeeds (via menuAction/Cmd+C),")
    print("the engine already captured isEditMode=false and won't use edit mode.")
}
