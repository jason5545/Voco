import Foundation
import AppKit
import os

/// Background polling service that caches the frontmost app's edit state every ~1 second.
/// When the user presses the recording shortcut, `toggleMiniRecorder()` reads the cache
/// instead of performing a live AX query — eliminating the 0.5s Chrome/Electron delay.
final class EditModeCacheService: @unchecked Sendable {
    static let shared = EditModeCacheService()

    /// Terminal apps where Edit Mode should be skipped.
    /// Shared with `RecorderUIManager` to avoid duplication.
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
    private var _cachedFocusedElementUnavailable: Bool = false
    private var _cachedSelectedText: String?
    private var _cachedAppName: String?
    private var _cachedBundleID: String?
    private var _cachedPid: pid_t?
    private var _cachedWindowTitle: String?

    var cachedIsEditable: Bool { lock.withLock { _cachedIsEditable } }
    var cachedFocusedElementUnavailable: Bool { lock.withLock { _cachedFocusedElementUnavailable } }
    var cachedSelectedText: String? { lock.withLock { _cachedSelectedText } }
    var cachedAppName: String? { lock.withLock { _cachedAppName } }
    var cachedBundleID: String? { lock.withLock { _cachedBundleID } }
    var cachedPid: pid_t? { lock.withLock { _cachedPid } }
    var cachedWindowTitle: String? { lock.withLock { _cachedWindowTitle } }

    /// Atomic snapshot of edit mode state — avoids race with activation observer invalidate().
    struct EditModeSnapshot {
        let isEditable: Bool
        let focusedElementUnavailable: Bool
        let selectedText: String?
    }

    func snapshotEditModeState() -> EditModeSnapshot {
        lock.withLock {
            EditModeSnapshot(
                isEditable: _cachedIsEditable,
                focusedElementUnavailable: _cachedFocusedElementUnavailable,
                selectedText: _cachedSelectedText
            )
        }
    }

    // MARK: - Polling

    private var pollingTask: Task<Void, Never>?
	private var refreshCoordinatorTask: Task<Void, Never>?
    private var activationObserver: NSObjectProtocol?
	private var pollingState = EditModePollingState()
    private let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "EditModeCache")

    private init() {}

    func startPolling() {
		let generation: UInt64? = lock.withLock {
			guard let generation = pollingState.startPolling() else { return nil }
			ensureRefreshCoordinatorLocked()
			return generation
		}
		guard let generation else { return }

        logger.debug("Edit mode cache polling started")
		pollingTask = Task(priority: .utility) { [weak self] in
			await self?.runPollingLoop(expectedGeneration: generation)
		}

        // Eagerly refresh cache when the user switches apps
        if activationObserver == nil {
            activationObserver = NSWorkspace.shared.notificationCenter.addObserver(
                forName: NSWorkspace.didActivateApplicationNotification,
                object: nil,
                queue: nil
            ) { [weak self] _ in
                guard let self else { return }
				self.requestRefresh(reason: "activation", invalidateCache: true)
            }
        }
    }

    func stopPolling() {
		let didStop = lock.withLock {
			pollingState.stopPolling()
			let hadActivePolling = pollingTask != nil || activationObserver != nil
			pollingTask?.cancel()
			pollingTask = nil
			return hadActivePolling
		}
        if let observer = activationObserver {
            NSWorkspace.shared.notificationCenter.removeObserver(observer)
            activationObserver = nil
        }
		if didStop {
			logger.debug("Edit mode cache polling stopped")
		}
    }

    func invalidate() {
        lock.withLock {
			invalidateLocked()
        }
    }

	private func ensureRefreshCoordinatorLocked() {
		guard refreshCoordinatorTask == nil else { return }
		refreshCoordinatorTask = Task(priority: .utility) { [weak self] in
			await self?.drainRefreshQueue()
		}
	}

	private func requestRefresh(reason: String, invalidateCache: Bool = false) {
		let disposition = lock.withLock { () -> EditModeRefreshDisposition in
			if invalidateCache {
				invalidateLocked()
			}
			let disposition = pollingState.enqueueRefresh()
			if disposition != .ignoredStopped {
				ensureRefreshCoordinatorLocked()
			}
			return disposition
		}

		switch disposition {
		case .scheduled:
			logger.debug("Scheduled edit mode refresh (reason: \(reason, privacy: .public))")
		case .queuedBehindInFlight:
			logger.debug("Queued edit mode refresh behind active poll (reason: \(reason, privacy: .public))")
		case .coalesced:
			logger.debug("Coalesced edit mode refresh request (reason: \(reason, privacy: .public))")
		case .ignoredStopped:
			logger.debug("Skipped edit mode refresh because polling is stopped (reason: \(reason, privacy: .public))")
		}
	}

	private func runPollingLoop(expectedGeneration: UInt64) async {
		while !Task.isCancelled {
			do {
				try await Task.sleep(for: .seconds(1))
			} catch {
				break
			}

			let shouldContinue = lock.withLock {
				pollingState.shouldContinuePolling(expectedGeneration: expectedGeneration)
			}
			guard shouldContinue else { break }

			requestRefresh(reason: "interval")
		}
	}

	private func drainRefreshQueue() async {
		while true {
			let expectedGeneration = lock.withLock { () -> UInt64? in
				guard let generation = pollingState.beginNextRefresh() else {
					refreshCoordinatorTask = nil
					return nil
				}
				return generation
			}

			guard let expectedGeneration else { return }
			await pollOnce(expectedGeneration: expectedGeneration)

			let shouldContinue = lock.withLock { () -> Bool in
				pollingState.finishRefresh()
				let hasPendingRefresh = pollingState.hasPendingRefresh
				if !hasPendingRefresh {
					refreshCoordinatorTask = nil
				}
				return hasPendingRefresh
			}

			if !shouldContinue {
				return
			}
		}
	}

	private func invalidateLocked() {
		_cachedIsEditable = false
		_cachedFocusedElementUnavailable = false
		_cachedSelectedText = nil
		_cachedAppName = nil
		_cachedBundleID = nil
		_cachedPid = nil
		_cachedWindowTitle = nil
	}

    // MARK: - Single Poll Cycle

	private func pollOnce(expectedGeneration: UInt64) async {
        // Step 1: Get frontmost app info (fast, non-AX)
        let frontApp = NSWorkspace.shared.frontmostApplication
        let bundleID = frontApp?.bundleIdentifier
        let pid = frontApp?.processIdentifier
        let appName = frontApp?.localizedName
        let isTerminal = bundleID.map { Self.terminalBundleIDs.contains($0) } ?? false
        let axTrusted = AXIsProcessTrusted()

        // If terminal or no AX trust, just cache the basic info with isEditable = false
        guard axTrusted, !isTerminal, let pid = pid else {
			guard shouldApplyPollResult(expectedGeneration: expectedGeneration) else {
				logger.debug("Suppressing stale basic edit mode refresh result")
				return
			}
            lock.withLock {
                _cachedIsEditable = false
                _cachedSelectedText = nil
				_cachedFocusedElementUnavailable = false
                _cachedAppName = appName
                _cachedBundleID = bundleID
                _cachedPid = pid
                _cachedWindowTitle = nil
            }
            return
        }

		let axResult = await Self.performAXPoll(pid: pid)
		guard shouldApplyPollResult(expectedGeneration: expectedGeneration) else {
			logger.debug("Suppressing stale AX edit mode refresh result")
			return
		}

        // Step 3: Write cache
        lock.withLock {
			_cachedIsEditable = axResult.isEditable
			_cachedFocusedElementUnavailable = axResult.focusedElementUnavailable
			_cachedSelectedText = axResult.selectedText
			_cachedWindowTitle = axResult.windowTitle
            _cachedAppName = appName
            _cachedBundleID = bundleID
            _cachedPid = pid
        }
    }

	private func shouldApplyPollResult(expectedGeneration: UInt64) -> Bool {
		lock.withLock {
			pollingState.shouldApplyResult(for: expectedGeneration)
		}
	}

	private static func performAXPoll(pid: pid_t) async -> AXPollResult {
		// Actual AX work — inline focused element check to distinguish
		// "element unavailable" (e.g. Claude desktop) from "not editable"
		let axApp = AXUIElementCreateApplication(pid)

		var isEditable = false
		var focusedElementUnavailable = false

		var focusedElementObj: AnyObject?
		let focusedResult = AXUIElementCopyAttributeValue(axApp, kAXFocusedUIElementAttribute as CFString, &focusedElementObj)

		if focusedResult == .success {
			let element = focusedElementObj as! AXUIElement
			var roleValue: AnyObject?
			if AXUIElementCopyAttributeValue(element, kAXRoleAttribute as CFString, &roleValue) == .success,
			   let role = roleValue as? String {
				let editableRoles: Set<String> = [
					kAXTextFieldRole as String,
					kAXTextAreaRole as String,
					kAXComboBoxRole as String,
				]
				isEditable = editableRoles.contains(role)
			}
		} else {
			focusedElementUnavailable = true
		}

		var selectedText: String?
		if isEditable {
			selectedText = await SelectedTextService.fetchSelectedTextForEditModeDetection()
			if let text = selectedText, text.isEmpty {
				selectedText = nil
			}
		}

		// Window title via AX
		var windowTitle: String?
		var focusedWindow: AnyObject?
		if AXUIElementCopyAttributeValue(axApp, kAXFocusedWindowAttribute as CFString, &focusedWindow) == .success {
			var titleValue: AnyObject?
			if AXUIElementCopyAttributeValue(focusedWindow as! AXUIElement, kAXTitleAttribute as CFString, &titleValue) == .success {
				windowTitle = titleValue as? String
			}
		}

		return AXPollResult(
			isEditable: isEditable,
			focusedElementUnavailable: focusedElementUnavailable,
			selectedText: selectedText,
			windowTitle: windowTitle
		)
	}
}

/// Internal result type for a single AX poll cycle.
private struct AXPollResult {
    let isEditable: Bool
    let focusedElementUnavailable: Bool
    let selectedText: String?
    let windowTitle: String?
}

enum EditModeRefreshDisposition: Equatable {
	case scheduled
	case queuedBehindInFlight
	case coalesced
	case ignoredStopped
}

struct EditModePollingState {
	private(set) var generation: UInt64 = 0
	private(set) var isPollingEnabled = false
	private(set) var hasPendingRefresh = false
	private(set) var isPollInFlight = false

	mutating func startPolling() -> UInt64? {
		guard !isPollingEnabled else { return nil }
		generation &+= 1
		isPollingEnabled = true
		hasPendingRefresh = true
		return generation
	}

	mutating func stopPolling() {
		guard isPollingEnabled || hasPendingRefresh || isPollInFlight else { return }
		generation &+= 1
		isPollingEnabled = false
		hasPendingRefresh = false
	}

	mutating func enqueueRefresh() -> EditModeRefreshDisposition {
		guard isPollingEnabled else { return .ignoredStopped }
		if hasPendingRefresh {
			return .coalesced
		}
		hasPendingRefresh = true
		return isPollInFlight ? .queuedBehindInFlight : .scheduled
	}

	mutating func beginNextRefresh() -> UInt64? {
		guard hasPendingRefresh, !isPollInFlight else { return nil }
		hasPendingRefresh = false
		isPollInFlight = true
		return generation
	}

	mutating func finishRefresh() {
		isPollInFlight = false
	}

	func shouldContinuePolling(expectedGeneration: UInt64) -> Bool {
		isPollingEnabled && generation == expectedGeneration
	}

	func shouldApplyResult(for expectedGeneration: UInt64) -> Bool {
		isPollingEnabled && generation == expectedGeneration
	}
}
