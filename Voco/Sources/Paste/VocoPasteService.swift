import Foundation
import AppKit
import Carbon
import os

enum VocoPasteService {
    private typealias ClipboardItemSnapshot = [(NSPasteboard.PasteboardType, Data)]
    private typealias ClipboardSnapshot = [ClipboardItemSnapshot]
    private struct ClipboardPasteOwnership {
        let session: ClipboardPasteSessionIdentity?
        let text: String
        let changeCount: Int
    }

    private static let logger = Logger(subsystem: "com.prakashjoshipax.voiceink", category: "CursorPaster")
    private static let prePasteDelay: TimeInterval = 0.10
    private static let pasteShortcutEventDelay: TimeInterval = 0.01
    private static let minimumClipboardRestoreDelay: TimeInterval = AppDefaults.minimumClipboardRestoreDelay
    @MainActor private static var restoreChain = ClipboardRestoreChain<ClipboardSnapshot>()
    @MainActor private static var pendingRestoreTask: Task<Void, Never>?

    static func pasteAtCursor(_ text: String) {
        Task {
            let pasteTask = await MainActor.run {
                startPasteAtCursor(text)
            }
            _ = await pasteTask.value
        }
    }

    @MainActor
    @discardableResult
    static func startPasteAtCursor(_ text: String) -> Task<CursorPaster.PasteResult, Never> {
        Task { @MainActor in
            await performPasteSession(text, targetPID: nil)
        }
    }

    @MainActor
    static func pasteAtCursorAndWaitUntilPosted(
        _ text: String,
        targetPID: pid_t? = nil,
        beforePosting: (@MainActor () async -> Bool)? = nil
    ) async -> CursorPaster.PasteResult {
        await performPasteSession(
            text,
            targetPID: targetPID,
            beforePosting: beforePosting
        )
    }

    @MainActor
    private static func performPasteSession(
        _ text: String,
        targetPID: pid_t?,
        beforePosting: (@MainActor () async -> Bool)? = nil
    ) async -> CursorPaster.PasteResult {
        await ClipboardTransactionCoordinator.shared.withExclusiveAccess {
            await performExclusivePasteSession(
                text,
                targetPID: targetPID,
                beforePosting: beforePosting
            )
        }
    }

    @MainActor
    private static func performExclusivePasteSession(
        _ text: String,
        targetPID: pid_t?,
        beforePosting: (@MainActor () async -> Bool)?
    ) async -> CursorPaster.PasteResult {
        let pasteboard = NSPasteboard.general
        let shouldRestoreClipboard = UserDefaults.standard.bool(forKey: "restoreClipboardAfterPaste")
        let needsAbortSnapshot = targetPID != nil
        let sessionID = UUID().uuidString
        let session = ClipboardPasteSessionIdentity(id: sessionID, text: text)
        let savedContents: ClipboardSnapshot

        if shouldRestoreClipboard || needsAbortSnapshot {
            savedContents = restoreChain.originalSnapshotForNextPaste(
                currentSession: pasteSessionIdentity(on: pasteboard),
                makeSnapshot: { snapshotClipboard(from: pasteboard) }
            )
        } else {
            savedContents = []
        }
        if !shouldRestoreClipboard {
            restoreChain.clear(ifSessionMatches: nil)
        }
        pendingRestoreTask?.cancel()
        pendingRestoreTask = nil

        guard ClipboardManager.setClipboard(
            text,
            transient: shouldRestoreClipboard,
            sessionID: shouldRestoreClipboard ? sessionID : nil
        ) else {
            logger.error("Failed to prepare clipboard for paste")
            if shouldRestoreClipboard || needsAbortSnapshot {
                restoreClipboard(savedContents, on: pasteboard)
            }
            restoreChain.clear(ifSessionMatches: nil)
            return .commandNotPosted
        }
        let ownership = ClipboardPasteOwnership(
            session: shouldRestoreClipboard ? session : nil,
            text: text,
            changeCount: pasteboard.changeCount
        )

        await wait(prePasteDelay)

        let dispatchValidationPassed = await beforePosting?() ?? true
        let targetIsFrontmost = targetPID.map {
            NSWorkspace.shared.frontmostApplication?.processIdentifier == $0
        } ?? true
        guard !Task.isCancelled,
              dispatchValidationPassed,
              targetIsFrontmost,
              pasteboardStillOwned(pasteboard, ownership: ownership) else {
            logger.warning("Paste target, selection, or clipboard changed before command dispatch; aborting")
            if pasteboardStillOwned(pasteboard, ownership: ownership) {
                restoreClipboard(savedContents, on: pasteboard)
            }
            restoreChain.clear(ifSessionMatches: nil)
            return .commandNotPosted
        }

        let pasteTargetApp = NSWorkspace.shared.frontmostApplication
        let pasteTarget = PasteTargetSnapshot.capture(for: pasteTargetApp)
        let pasteResult = await postPasteCommand()
        if pasteResult.didPostPasteCommand, let pasteTarget, let pasteTargetApp {
            await repostPasteUntilItLands(
                target: pasteTarget,
                app: pasteTargetApp,
                pasteboard: pasteboard,
                ownership: ownership
            )
        }
        if !pasteResult.didPostPasteCommand, needsAbortSnapshot {
            if pasteboardStillOwned(pasteboard, ownership: ownership) {
                restoreClipboard(savedContents, on: pasteboard)
            }
            restoreChain.clear(ifSessionMatches: nil)
            return pasteResult
        }
        if shouldRestoreClipboard {
            restoreChain.begin(session: session, originalSnapshot: savedContents)
            scheduleClipboardRestore(
                savedContents,
                session: session,
                ownership: ownership,
                on: pasteboard
            )
        }

        return pasteResult
    }

    private static func snapshotClipboard(from pasteboard: NSPasteboard) -> ClipboardSnapshot {
        (pasteboard.pasteboardItems ?? []).map { item in
            item.types.compactMap { type in
                if let data = item.data(forType: type) {
                    return (type, data)
                }
                return nil
            }
        }
    }

    @MainActor
    private static func repostPasteUntilItLands(
        target: PasteTargetSnapshot,
        app: NSRunningApplication,
        pasteboard: NSPasteboard,
        ownership: ClipboardPasteOwnership
    ) async {
        let bundleID = app.bundleIdentifier ?? "unknown"
        for attempt in 0...PasteLandingPolicy.maximumRetries {
            if await pasteLanded(target: target, pid: app.processIdentifier) {
                if attempt > 0 {
                    logger.notice("Paste landed in \(bundleID, privacy: .public) after \(attempt) retry(s)")
                }
                return
            }
            guard attempt < PasteLandingPolicy.maximumRetries,
                  !Task.isCancelled,
                  NSWorkspace.shared.frontmostApplication?.processIdentifier == app.processIdentifier,
                  pasteboardStillOwned(pasteboard, ownership: ownership) else {
                break
            }
            logger.warning("Paste did not land in \(bundleID, privacy: .public); posting it again (retry \(attempt + 1))")
            guard await postPasteCommand().didPostPasteCommand else { break }
        }
        logger.warning("Paste still not visible in \(bundleID, privacy: .public) after retries")
    }

    @MainActor
    private static func pasteLanded(target: PasteTargetSnapshot, pid: pid_t) async -> Bool {
        let deadline = Date().addingTimeInterval(PasteLandingPolicy.landingTimeout)
        while true {
            await wait(PasteLandingPolicy.pollInterval)
            switch PasteLandingPolicy.verdict(
                before: target.value,
                observation: target.observe(pid: pid),
                deadlinePassed: Date() >= deadline
            ) {
            case .landed: return true
            case .notLanded: return false
            case .keepWaiting: continue
            }
        }
    }

    @MainActor
    private static func postPasteCommand() async -> CursorPaster.PasteResult {
        if PasteMethod.current() == .appleScript {
            return pasteUsingAppleScript() ? .commandPosted : .commandNotPosted
        }
        return await pasteFromClipboard()
    }

    @MainActor
    private static func scheduleClipboardRestore(
        _ savedContents: ClipboardSnapshot,
        session: ClipboardPasteSessionIdentity,
        ownership: ClipboardPasteOwnership,
        on pasteboard: NSPasteboard
    ) {
        let delay = max(
            UserDefaults.standard.double(forKey: "clipboardRestoreDelay"),
            minimumClipboardRestoreDelay
        )

        pendingRestoreTask = Task { @MainActor in
            do {
                try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            } catch {
                return
            }

            await ClipboardTransactionCoordinator.shared.withExclusiveAccess {
                guard restoreChain.activeSession == session else {
                    return
                }
                defer {
                    restoreChain.clear(ifSessionMatches: session)
                    pendingRestoreTask = nil
                }
                guard pasteboardStillOwned(pasteboard, ownership: ownership) else {
                    return
                }
                restoreClipboard(savedContents, on: pasteboard)
            }
        }
    }

    private static func pasteboardStillOwned(
        _ pasteboard: NSPasteboard,
        ownership: ClipboardPasteOwnership
    ) -> Bool {
        guard pasteboard.changeCount == ownership.changeCount,
              pasteboard.string(forType: .string) == ownership.text else {
            return false
        }
        if let session = ownership.session {
            return pasteSessionIdentity(on: pasteboard) == session
        }
        return pasteboard.string(forType: ClipboardManager.pasteSessionType) == nil
    }

    private static func pasteSessionIdentity(
        on pasteboard: NSPasteboard
    ) -> ClipboardPasteSessionIdentity? {
        guard let id = pasteboard.string(forType: ClipboardManager.pasteSessionType),
              let text = pasteboard.string(forType: .string) else {
            return nil
        }
        return ClipboardPasteSessionIdentity(id: id, text: text)
    }

    private static func restoreClipboard(
        _ savedContents: ClipboardSnapshot,
        on pasteboard: NSPasteboard
    ) {
        pasteboard.clearContents()
        if !savedContents.isEmpty {
            pasteboard.writeObjects(pasteboardItems(from: savedContents))
        }
    }

    private static func pasteboardItems(from snapshot: ClipboardSnapshot) -> [NSPasteboardItem] {
        snapshot.map { itemSnapshot in
            let item = NSPasteboardItem()
            for (type, data) in itemSnapshot {
                item.setData(data, forType: type)
            }
            return item
        }
    }

    private static func makeScript(_ source: String) -> NSAppleScript? {
        let script = NSAppleScript(source: source)
        var error: NSDictionary?
        script?.compileAndReturnError(&error)
        return script
    }

    private static let pasteScriptKeystroke = makeScript("tell application \"System Events\" to keystroke \"v\" using command down")
    private static let pasteScriptKeyCode = makeScript("tell application \"System Events\" to key code 9 using command down")

    @MainActor
    private static var layoutSwitchesToQWERTYOnCommand: Bool {
        let source = TISCopyCurrentKeyboardInputSource().takeRetainedValue()
        guard let nameRef = TISGetInputSourceProperty(source, kTISPropertyLocalizedName) else { return false }
        return (Unmanaged<CFString>.fromOpaque(nameRef).takeUnretainedValue() as String).hasSuffix("⌘")
    }

    @MainActor
    private static func pasteUsingAppleScript() -> Bool {
        guard let script = layoutSwitchesToQWERTYOnCommand ? pasteScriptKeyCode : pasteScriptKeystroke else {
            logger.error("AppleScript paste script is unavailable")
            return false
        }

        var error: NSDictionary?
        script.executeAndReturnError(&error)
        if let error {
            logger.error("AppleScript paste failed: \(String(describing: error), privacy: .public)")
        }
        return error == nil
    }

    @MainActor
    private static func pasteFromClipboard() async -> CursorPaster.PasteResult {
        guard AXIsProcessTrusted() else {
            logger.error("Accessibility permission is required to paste with simulated key events")
            return .commandNotPosted
        }

        let source = CGEventSource(stateID: .privateState)
        guard let cmdDown = CGEvent(keyboardEventSource: source, virtualKey: 0x37, keyDown: true),
              let vDown = CGEvent(keyboardEventSource: source, virtualKey: 0x09, keyDown: true),
              let vUp = CGEvent(keyboardEventSource: source, virtualKey: 0x09, keyDown: false),
              let cmdUp = CGEvent(keyboardEventSource: source, virtualKey: 0x37, keyDown: false) else {
            logger.error("Failed to create Cmd+V keyboard events")
            return .commandNotPosted
        }

        cmdDown.flags = .maskCommand
        vDown.flags = .maskCommand
        vUp.flags = .maskCommand
        cmdUp.flags = []

        cmdDown.post(tap: .cghidEventTap)
        await wait(pasteShortcutEventDelay)
        vDown.post(tap: .cghidEventTap)
        await wait(pasteShortcutEventDelay)
        vUp.post(tap: .cghidEventTap)
        await wait(pasteShortcutEventDelay)
        cmdUp.post(tap: .cghidEventTap)

        return .commandPosted
    }

    private static func wait(_ seconds: TimeInterval) async {
        guard seconds > 0 else { return }
        let nanoseconds = UInt64(seconds * 1_000_000_000)
        try? await Task.sleep(nanoseconds: nanoseconds)
    }

    static func deleteSelection() {
        guard AXIsProcessTrusted() else { return }
        let source = CGEventSource(stateID: .hidSystemState)
        let deleteDown = CGEvent(keyboardEventSource: source, virtualKey: 0x33, keyDown: true)
        let deleteUp = CGEvent(keyboardEventSource: source, virtualKey: 0x33, keyDown: false)
        deleteDown?.post(tap: .cghidEventTap)
        deleteUp?.post(tap: .cghidEventTap)
    }

    static func selectAllAndDelete() {
        guard AXIsProcessTrusted() else { return }
        let source = CGEventSource(stateID: .hidSystemState)

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

        usleep(50_000)

        let deleteDown = CGEvent(keyboardEventSource: source, virtualKey: 0x33, keyDown: true)
        let deleteUp = CGEvent(keyboardEventSource: source, virtualKey: 0x33, keyDown: false)
        deleteDown?.post(tap: .cghidEventTap)
        deleteUp?.post(tap: .cghidEventTap)
    }

    static func performAutoSend(_ key: FinishAndSendKey) {
        guard key.isEnabled else { return }
        guard AXIsProcessTrusted() else { return }

        let source = CGEventSource(stateID: .privateState)
        let enterDown = CGEvent(keyboardEventSource: source, virtualKey: 0x24, keyDown: true)
        let enterUp = CGEvent(keyboardEventSource: source, virtualKey: 0x24, keyDown: false)

        switch key {
        case .none: return
        case .enter: break
        case .shiftEnter:
            enterDown?.flags = .maskShift
            enterUp?.flags = .maskShift
        case .commandEnter:
            enterDown?.flags = .maskCommand
            enterUp?.flags = .maskCommand
        }

        enterDown?.post(tap: .cghidEventTap)
        enterUp?.post(tap: .cghidEventTap)
    }
}
