import Foundation
import AppKit
import os

/// Background polling service that caches the frontmost app's edit state every ~1 second.
/// When the user presses the recording shortcut, `toggleMiniRecorder()` reads the cache
/// instead of performing a live AX query — eliminating the 0.5s Chrome/Electron delay.
final class EditModeCacheService: @unchecked Sendable {
    static let shared = EditModeCacheService()

    /// Terminal apps where Edit Mode should be skipped.
    /// Shared with `WhisperState+UI.swift` to avoid duplication.
    static let terminalBundleIDs: Set<String> = [
        "com.apple.Terminal",
        "com.googlecode.iterm2",
        "net.kovidgoyal.kitty",
        "com.mitchellh.ghostty",
        "io.alacritty",
        "dev.warp.Warp-Stable",
        "com.github.wez.wezterm",
        "co.zeit.hyper",
        "org.tabby",
    ]

    // MARK: - Cached State

    private let lock = NSLock()

    private var _cachedIsEditable: Bool = false
    private var _cachedSelectedText: String?
    private var _cachedAppName: String?
    private var _cachedBundleID: String?
    private var _cachedPid: pid_t?
    private var _cachedWindowTitle: String?

    var cachedIsEditable: Bool { lock.withLock { _cachedIsEditable } }
    var cachedSelectedText: String? { lock.withLock { _cachedSelectedText } }
    var cachedAppName: String? { lock.withLock { _cachedAppName } }
    var cachedBundleID: String? { lock.withLock { _cachedBundleID } }
    var cachedPid: pid_t? { lock.withLock { _cachedPid } }
    var cachedWindowTitle: String? { lock.withLock { _cachedWindowTitle } }

    // MARK: - Polling

    private var pollingTask: Task<Void, Never>?
    private var activationObserver: NSObjectProtocol?
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "EditModeCache")

    private init() {}

    func startPolling() {
        guard pollingTask == nil else { return }
        logger.debug("Edit mode cache polling started")
        pollingTask = Task.detached(priority: .utility) { [weak self] in
            while !Task.isCancelled {
                await self?.pollOnce()
                try? await Task.sleep(for: .seconds(1))
            }
        }

        // Eagerly refresh cache when the user switches apps
        if activationObserver == nil {
            activationObserver = NSWorkspace.shared.notificationCenter.addObserver(
                forName: NSWorkspace.didActivateApplicationNotification,
                object: nil,
                queue: nil
            ) { [weak self] _ in
                guard let self else { return }
                self.invalidate()
                Task.detached(priority: .utility) {
                    await self.pollOnce()
                }
            }
        }
    }

    func stopPolling() {
        pollingTask?.cancel()
        pollingTask = nil
        if let observer = activationObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(observer)
            activationObserver = nil
        }
        logger.debug("Edit mode cache polling stopped")
    }

    func invalidate() {
        lock.withLock {
            _cachedIsEditable = false
            _cachedSelectedText = nil
            _cachedAppName = nil
            _cachedBundleID = nil
            _cachedPid = nil
            _cachedWindowTitle = nil
        }
    }

    // MARK: - Single Poll Cycle

    private func pollOnce() async {
        // Step 1: Get frontmost app info (fast, non-AX)
        let frontApp = NSWorkspace.shared.frontmostApplication
        let bundleID = frontApp?.bundleIdentifier
        let pid = frontApp?.processIdentifier
        let appName = frontApp?.localizedName
        let isTerminal = bundleID.map { Self.terminalBundleIDs.contains($0) } ?? false
        let axTrusted = AXIsProcessTrusted()

        // If terminal or no AX trust, just cache the basic info with isEditable = false
        guard axTrusted, !isTerminal, let pid = pid else {
            lock.withLock {
                _cachedIsEditable = false
                _cachedSelectedText = nil
                _cachedAppName = appName
                _cachedBundleID = bundleID
                _cachedPid = pid
                _cachedWindowTitle = nil
            }
            return
        }

        // Step 2: AX queries with 800ms timeout
        let axResult: AXPollResult? = await withTaskGroup(of: AXPollResult?.self) { group in
            group.addTask {
                // Actual AX work
                let isEditable = SelectedTextService.isEditableTextFocused(for: pid)

                var selectedText: String?
                if isEditable {
                    selectedText = await SelectedTextService.fetchSelectedTextForEditModeDetection()
                    if let text = selectedText, text.isEmpty {
                        selectedText = nil
                    }
                }

                // Window title via AX
                var windowTitle: String?
                let axApp = AXUIElementCreateApplication(pid)
                var focusedWindow: AnyObject?
                if AXUIElementCopyAttributeValue(axApp, kAXFocusedWindowAttribute as CFString, &focusedWindow) == .success {
                    var titleValue: AnyObject?
                    if AXUIElementCopyAttributeValue(focusedWindow as! AXUIElement, kAXTitleAttribute as CFString, &titleValue) == .success {
                        windowTitle = titleValue as? String
                    }
                }

                return AXPollResult(
                    isEditable: isEditable,
                    selectedText: selectedText,
                    windowTitle: windowTitle
                )
            }

            group.addTask {
                // 800ms timeout sentinel
                try? await Task.sleep(for: .milliseconds(800))
                return nil
            }

            // First non-nil result wins
            for await value in group {
                if let v = value {
                    group.cancelAll()
                    return v
                }
            }
            group.cancelAll()
            return nil // timeout
        }

        // Step 3: Write cache
        lock.withLock {
            if let result = axResult {
                _cachedIsEditable = result.isEditable
                _cachedSelectedText = result.selectedText
                _cachedWindowTitle = result.windowTitle
            } else {
                // Timeout — keep basic info, mark not editable (conservative)
                _cachedIsEditable = false
                _cachedSelectedText = nil
                _cachedWindowTitle = nil
            }
            _cachedAppName = appName
            _cachedBundleID = bundleID
            _cachedPid = pid
        }
    }
}

/// Internal result type for a single AX poll cycle.
private struct AXPollResult {
    let isEditable: Bool
    let selectedText: String?
    let windowTitle: String?
}
