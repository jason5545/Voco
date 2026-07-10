import ApplicationServices
import AppKit
import Foundation
import os
import SelectedTextKit

struct EditableTextSelectionObservation: Equatable, Sendable {
    let role: String
    let selectedText: String?
    let selectedRangeLength: Int?
    let isFocused: Bool?

    init(
        role: String,
        selectedText: String?,
        selectedRangeLength: Int?,
        isFocused: Bool? = nil
    ) {
        self.role = role
        self.selectedText = selectedText
        self.selectedRangeLength = selectedRangeLength
        self.isFocused = isFocused
    }
}

enum EditableTextSelectionEvidence: Equatable, Sendable {
    case selected(String)
    case noSelection
    case unavailable
}

enum EditableTextSelectionPolicy {
    static let editableRoles: Set<String> = [
        kAXTextFieldRole as String,
        kAXTextAreaRole as String,
        kAXComboBoxRole as String,
    ]

    /// Edit Mode needs positive selection provenance from one editable AX
    /// element. A non-empty string without a positive range is unverified: it
    /// can be stale text supplied by an Electron accessibility bridge.
    static func resolve(
        observations: [EditableTextSelectionObservation],
        requireFocusedElement: Bool = false
    ) -> EditableTextSelectionEvidence {
        let editable = observations.filter {
            editableRoles.contains($0.role) &&
                (!requireFocusedElement || $0.isFocused == true)
        }

        for observation in editable {
            guard let rangeLength = observation.selectedRangeLength,
                  rangeLength > 0,
                  let selectedText = meaningfulSelectionText(observation.selectedText) else {
                continue
            }
            return .selected(selectedText)
        }

        if editable.contains(where: { $0.selectedRangeLength == 0 }) {
            return .noSelection
        }

        return .unavailable
    }

    static func meaningfulSelectionText(_ text: String?) -> String? {
        guard let text else { return nil }

        var ignored = CharacterSet.whitespacesAndNewlines
        ignored.formUnion(CharacterSet(charactersIn: "\u{FFFC}\u{200B}\u{200C}\u{200D}\u{FEFF}"))
        let hasMeaningfulScalar = text.unicodeScalars.contains { !ignored.contains($0) }
        return hasMeaningfulScalar ? text : nil
    }
}

enum EditableMarkerSelectionPolicy {
    static func resolve(
        selectedText: String?,
        rangeLength: Int,
        endpointsShareEditableAncestor: Bool,
        editableAncestorIsFocused: Bool
    ) -> EditableTextSelectionEvidence {
        guard rangeLength > 0 else { return .noSelection }
        guard endpointsShareEditableAncestor,
              editableAncestorIsFocused,
              let selectedText = EditableTextSelectionPolicy.meaningfulSelectionText(selectedText) else {
            return .noSelection
        }
        return .selected(selectedText)
    }
}

@MainActor
final class SelectedTextService {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "SelectedTextService")
    private static let textManager = SelectedTextManager.shared
    nonisolated private static let focusedWindowTraversalLimit = 4_000

    /// Full retrieval strategy used when richer context is acceptable.
    static func fetchSelectedText() async -> String? {
        await ClipboardTransactionCoordinator.shared.withExclusiveAccessUnlessCancelled {
            await fetchSelectedText(using: [.accessibility, .menuAction, .appleScript])
        } ?? nil
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

    /// Reads selection text and range from the same focused AX element so stale
    /// text from a separate system-wide lookup cannot be mixed into the result.
    static func focusedEditableSelectionEvidence(for pid: pid_t) -> EditableTextSelectionEvidence {
        guard let focusedElement = focusedElement(for: pid),
              let observation = editableSelectionObservation(for: focusedElement) else {
            return .unavailable
        }
        return EditableTextSelectionPolicy.resolve(observations: [observation])
    }

    /// Resolves the current editable selection without using the clipboard.
    /// Known Electron apps get a focused-window marker scan when their app-level
    /// focused element is incomplete or reports a collapsed/stale range.
    static func currentEditableSelectionEvidence(
        for pid: pid_t,
        searchFocusedWindow: Bool
    ) async -> EditableTextSelectionEvidence {
        let directEvidence = focusedEditableSelectionEvidence(for: pid)
        if case .selected = directEvidence {
            return directEvidence
        }
        guard searchFocusedWindow else {
            return directEvidence
        }

        let scanTask = Task.detached(priority: .userInitiated) {
            focusedWindowEditableSelectionEvidence(for: pid)
        }
        return await withTaskCancellationHandler(
            operation: { await scanTask.value },
            onCancel: { scanTask.cancel() }
        )
    }

    /// Electron apps can temporarily expose AXGroup/AXWebArea (or no focused
    /// element) at the application level while their real contenteditable host
    /// remains an AXTextArea deeper in the focused window. Search that window
    /// and only arm Edit Mode from an explicit non-zero selection range.
    nonisolated static func focusedWindowEditableSelectionEvidence(
        for pid: pid_t,
        traversalLimit: Int = focusedWindowTraversalLimit
    ) -> EditableTextSelectionEvidence {
        let axApp = AXUIElementCreateApplication(pid)
        guard let focusedWindow = elementAttribute(
            kAXFocusedWindowAttribute as CFString,
            for: axApp
        ) else {
            return .unavailable
        }

        var queue = [focusedWindow]
        var nextIndex = 0
        var visited = 0
        var observations: [EditableTextSelectionObservation] = []

        while nextIndex < queue.count, visited < traversalLimit, !Task.isCancelled {
            let element = queue[nextIndex]
            nextIndex += 1
            visited += 1

            if role(for: element) == "AXWebArea",
               let markerEvidence = editableMarkerSelectionEvidence(for: element) {
                switch markerEvidence {
                case .selected:
                    return markerEvidence
                case .noSelection:
                    // Chromium's document marker is the authoritative current
                    // selection. A collapsed marker must not be overridden by
                    // a hidden editor that retained an old range.
                    return .noSelection
                case .unavailable:
                    break
                }
            }

            if let observation = editableSelectionObservation(for: element) {
                observations.append(observation)
                let evidence = EditableTextSelectionPolicy.resolve(
                    observations: [observation],
                    requireFocusedElement: true
                )
                if case .selected = evidence {
                    return evidence
                }
            }

            queue.append(contentsOf: childElements(for: element))
        }

        return EditableTextSelectionPolicy.resolve(
            observations: observations,
            requireFocusedElement: true
        )
    }

    private static func focusedElement(for pid: pid_t) -> AXUIElement? {
        let axApp = AXUIElementCreateApplication(pid)
        return elementAttribute(kAXFocusedUIElementAttribute as CFString, for: axApp)
    }

    nonisolated private static func role(for element: AXUIElement) -> String? {
        var roleValue: AnyObject?
        guard AXUIElementCopyAttributeValue(element, kAXRoleAttribute as CFString, &roleValue) == .success,
              let role = roleValue as? String else {
            return nil
        }
        return role
    }

    nonisolated private static func editableSelectionObservation(
        for element: AXUIElement
    ) -> EditableTextSelectionObservation? {
        guard let role = role(for: element),
              EditableTextSelectionPolicy.editableRoles.contains(role) else {
            return nil
        }

        let selectionRange = selectedTextRange(for: element)
        var selectedText = stringAttribute(kAXSelectedTextAttribute as CFString, for: element)
        if EditableTextSelectionPolicy.meaningfulSelectionText(selectedText) == nil,
           let selectionRange,
           selectionRange.length > 0 {
            selectedText = parameterizedAttribute(
                kAXStringForRangeParameterizedAttribute as CFString,
                parameter: selectionRange.value,
                for: element
            ) as? String
        }

        return EditableTextSelectionObservation(
            role: role,
            selectedText: selectedText,
            selectedRangeLength: selectionRange?.length,
            isFocused: boolAttribute(kAXFocusedAttribute as CFString, for: element)
        )
    }

    nonisolated private static func boolAttribute(
        _ attribute: CFString,
        for element: AXUIElement
    ) -> Bool? {
        var value: AnyObject?
        guard AXUIElementCopyAttributeValue(element, attribute, &value) == .success else {
            return nil
        }
        return value as? Bool
    }

    nonisolated private static func stringAttribute(_ attribute: CFString, for element: AXUIElement) -> String? {
        var value: AnyObject?
        guard AXUIElementCopyAttributeValue(element, attribute, &value) == .success else {
            return nil
        }
        return value as? String
    }

    nonisolated private static func selectedTextRange(
        for element: AXUIElement
    ) -> (value: AXValue, length: Int)? {
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
        return (axValue, range.length)
    }

    /// Chromium exposes its document selection as a text-marker range on the
    /// web area. Accept it only when both endpoints map back to editable AX
    /// ancestors; selecting static transcript/status text must not arm Edit Mode.
    nonisolated private static func editableMarkerSelectionEvidence(
        for webArea: AXUIElement
    ) -> EditableTextSelectionEvidence? {
        guard let markerRangeValue = rawAttribute(
            kAXSelectedTextMarkerRangeAttribute as CFString,
            for: webArea
        ),
        CFGetTypeID(markerRangeValue) == AXTextMarkerRangeGetTypeID(),
        let length = parameterizedAttribute(
            kAXLengthForTextMarkerRangeParameterizedAttribute as CFString,
            parameter: markerRangeValue,
            for: webArea
        ) as? NSNumber else {
            return nil
        }

        let selectedText = parameterizedAttribute(
            kAXStringForTextMarkerRangeParameterizedAttribute as CFString,
            parameter: markerRangeValue,
            for: webArea
        ) as? String

        let markerRange = markerRangeValue as! AXTextMarkerRange
        let startMarker = AXTextMarkerRangeCopyStartMarker(markerRange)
        let endMarker = AXTextMarkerRangeCopyEndMarker(markerRange)
        let endpointsShareEditableAncestor: Bool
        let editableAncestorIsFocused: Bool
        if let startEditable = editableAncestor(for: startMarker, in: webArea),
           let endEditable = editableAncestor(for: endMarker, in: webArea),
           CFEqual(startEditable, endEditable) {
            endpointsShareEditableAncestor = CFEqual(startEditable, endEditable)
            editableAncestorIsFocused = boolAttribute(
                kAXFocusedAttribute as CFString,
                for: startEditable
            ) == true
        } else {
            endpointsShareEditableAncestor = false
            editableAncestorIsFocused = false
        }

        return EditableMarkerSelectionPolicy.resolve(
            selectedText: selectedText,
            rangeLength: length.intValue,
            endpointsShareEditableAncestor: endpointsShareEditableAncestor,
            editableAncestorIsFocused: editableAncestorIsFocused
        )
    }

    nonisolated private static func editableAncestor(
        for marker: AXTextMarker,
        in webArea: AXUIElement
    ) -> AXUIElement? {
        guard let markerElement = parameterizedAttribute(
            kAXUIElementForTextMarkerParameterizedAttribute as CFString,
            parameter: marker,
            for: webArea
        ),
        CFGetTypeID(markerElement) == AXUIElementGetTypeID() else {
            return nil
        }

        let element = markerElement as! AXUIElement
        if let role = role(for: element),
           EditableTextSelectionPolicy.editableRoles.contains(role) {
            return element
        }
        return elementAttribute(kAXEditableAncestorAttribute as CFString, for: element)
    }

    nonisolated private static func rawAttribute(
        _ attribute: CFString,
        for element: AXUIElement
    ) -> AnyObject? {
        var value: AnyObject?
        guard AXUIElementCopyAttributeValue(element, attribute, &value) == .success else {
            return nil
        }
        return value
    }

    nonisolated private static func parameterizedAttribute(
        _ attribute: CFString,
        parameter: AnyObject,
        for element: AXUIElement
    ) -> AnyObject? {
        var value: AnyObject?
        guard AXUIElementCopyParameterizedAttributeValue(
            element,
            attribute,
            parameter,
            &value
        ) == .success else {
            return nil
        }
        return value
    }

    nonisolated private static func elementAttribute(
        _ attribute: CFString,
        for element: AXUIElement
    ) -> AXUIElement? {
        var value: AnyObject?
        guard AXUIElementCopyAttributeValue(element, attribute, &value) == .success,
              let value,
              CFGetTypeID(value) == AXUIElementGetTypeID() else {
            return nil
        }
        return unsafeBitCast(value, to: AXUIElement.self)
    }

    nonisolated private static func childElements(for element: AXUIElement) -> [AXUIElement] {
        var value: AnyObject?
        guard AXUIElementCopyAttributeValue(
            element,
            kAXChildrenAttribute as CFString,
            &value
        ) == .success,
        let children = value as? [AXUIElement] else {
            return []
        }
        return children
    }

    static func normalized(_ text: String?) -> String? {
        guard let text else { return nil }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : text
    }
}
