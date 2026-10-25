import AppKit
import ApplicationServices
import Foundation

struct VocoEphemeralAXContent: Equatable {
    var windowText: [String]
    var webAreaText: [String]

    init(windowText: [String] = [], webAreaText: [String] = []) {
        self.windowText = windowText
        self.webAreaText = webAreaText
    }

    static let empty = VocoEphemeralAXContent()
}

enum VocoEphemeralSignalMatcher {
    static let marker = "無痕對話"

    static func isEphemeral(
        bundleID: String?,
        installedT3BundleIDs: Set<String>,
        content: VocoEphemeralAXContent
    ) -> Bool {
        guard let bundleID, installedT3BundleIDs.contains(bundleID) else { return false }
        return (content.windowText + content.webAreaText).contains { $0.contains(marker) }
    }
}

@MainActor
protocol VocoEphemeralSignalDetecting {
    func isEphemeral() -> Bool
}

@MainActor
final class VocoEphemeralSignalDetector: VocoEphemeralSignalDetecting {
    typealias FrontmostApplication = (pid: pid_t, bundleID: String?)
    typealias FrontmostApplicationProvider = () -> FrontmostApplication?
    typealias BundleIDProvider = () -> Set<String>
    typealias AXReader = (pid_t) -> VocoEphemeralAXContent

    private let frontmostApplication: FrontmostApplicationProvider
    private let installedT3BundleIDs: BundleIDProvider
    private let readAX: AXReader

    init(
        frontmostApplication: @escaping FrontmostApplicationProvider,
        installedT3BundleIDs: @escaping BundleIDProvider,
        readAX: @escaping AXReader
    ) {
        self.frontmostApplication = frontmostApplication
        self.installedT3BundleIDs = installedT3BundleIDs
        self.readAX = readAX
    }

    convenience init() {
        self.init(
            frontmostApplication: {
                guard let app = NSWorkspace.shared.frontmostApplication else { return nil }
                return (app.processIdentifier, app.bundleIdentifier)
            },
            installedT3BundleIDs: Self.discoverT3BundleIDs,
            readAX: VocoT3AXReader.read
        )
    }

    func isEphemeral() -> Bool {
        guard let app = frontmostApplication() else { return false }
        let bundleIDs = installedT3BundleIDs()
        guard bundleIDs.contains(app.bundleID ?? "") else { return false }
        return VocoEphemeralSignalMatcher.isEphemeral(
            bundleID: app.bundleID,
            installedT3BundleIDs: bundleIDs,
            content: readAX(app.pid)
        )
    }

    static func discoverT3BundleIDs() -> Set<String> {
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: "/Applications") else {
            return []
        }

        return Set(entries.compactMap { entry -> String? in
            guard entry.hasPrefix("T3"), entry.hasSuffix(".app") else { return nil }
            return Bundle(path: "/Applications/\(entry)")?.bundleIdentifier
        })
    }
}

enum VocoT3AXReader {
    private static let manualAccessibilityAttribute = "AXManualAccessibility"

    static func read(pid: pid_t) -> VocoEphemeralAXContent {
        guard AXIsProcessTrusted() else { return .empty }
        let app = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(app, 0.5)

        let previousManualAccessibility = boolAttribute(manualAccessibilityAttribute, from: app)
        if previousManualAccessibility == false {
            _ = AXUIElementSetAttributeValue(app, manualAccessibilityAttribute as CFString, kCFBooleanTrue)
        }
        defer {
            if previousManualAccessibility == false {
                _ = AXUIElementSetAttributeValue(app, manualAccessibilityAttribute as CFString, kCFBooleanFalse)
            }
        }

        var windowText: [String] = []
        if let window = elementAttribute(kAXFocusedWindowAttribute, from: app) {
            appendText(stringAttribute(kAXTitleAttribute, from: window), to: &windowText)
            appendText(stringAttribute(kAXDescriptionAttribute, from: window), to: &windowText)
        }

        var webAreaText: [String] = []
        if let window = elementAttribute(kAXFocusedWindowAttribute, from: app) {
            collectWebAreaText(from: window, into: &webAreaText)
        }
        if let focused = elementAttribute(kAXFocusedUIElementAttribute, from: app) {
            var current: AXUIElement? = focused
            var depth = 0
            while let element = current, depth < 30 {
                if stringAttribute(kAXRoleAttribute, from: element) == "AXWebArea" {
                    appendText(stringAttribute(kAXTitleAttribute, from: element), to: &webAreaText)
                    appendText(stringAttribute(kAXDescriptionAttribute, from: element), to: &webAreaText)
                }
                current = elementAttribute(kAXParentAttribute, from: element)
                depth += 1
            }
        }

        return VocoEphemeralAXContent(windowText: windowText, webAreaText: webAreaText)
    }

    private static func collectWebAreaText(from root: AXUIElement, into output: inout [String]) {
        var queue = [root]
        var seen = Set<Int>()
        var visited = 0

        while !queue.isEmpty, visited < 1_000 {
            let element = queue.removeFirst()
            let identity = Unmanaged.passUnretained(element).toOpaque().hashValue
            guard seen.insert(identity).inserted else { continue }
            visited += 1

            let role = stringAttribute(kAXRoleAttribute, from: element)
            if role == "AXWebArea" {
                appendText(stringAttribute(kAXTitleAttribute, from: element), to: &output)
                appendText(stringAttribute(kAXDescriptionAttribute, from: element), to: &output)
            }
            guard role?.contains("Menu") != true else { continue }
            queue.append(contentsOf: childElements(of: element))
        }
    }

    private static func appendText(_ text: String?, to output: inout [String]) {
        guard let text, !text.isEmpty else { return }
        output.append(text)
    }

    private static func boolAttribute(_ attribute: String, from element: AXUIElement) -> Bool? {
        var value: AnyObject?
        guard AXUIElementCopyAttributeValue(element, attribute as CFString, &value) == .success else {
            return nil
        }
        return (value as? NSNumber)?.boolValue
    }

    private static func stringAttribute(_ attribute: String, from element: AXUIElement) -> String? {
        var value: AnyObject?
        guard AXUIElementCopyAttributeValue(element, attribute as CFString, &value) == .success else {
            return nil
        }
        return value as? String
    }

    private static func elementAttribute(_ attribute: String, from element: AXUIElement) -> AXUIElement? {
        var value: AnyObject?
        guard AXUIElementCopyAttributeValue(element, attribute as CFString, &value) == .success,
              let value,
              CFGetTypeID(value) == AXUIElementGetTypeID()
        else {
            return nil
        }
        return unsafeBitCast(value, to: AXUIElement.self)
    }

    private static func childElements(of element: AXUIElement) -> [AXUIElement] {
        var value: AnyObject?
        guard AXUIElementCopyAttributeValue(element, kAXChildrenAttribute as CFString, &value) == .success else {
            return []
        }
        return value as? [AXUIElement] ?? []
    }
}
