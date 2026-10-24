import AppKit

@MainActor
final class ClipboardTransactionCoordinator {
    static let shared = ClipboardTransactionCoordinator()

    private var isLocked = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    init() {}

    func withExclusiveAccess<T>(
        _ operation: @MainActor () async -> T
    ) async -> T {
        await acquire()
        defer { release() }
        return await operation()
    }

    func withExclusiveAccessUnlessCancelled<T>(
        _ operation: @MainActor () async -> T
    ) async -> T? {
        await acquire()
        defer { release() }
        guard !Task.isCancelled else { return nil }
        return await operation()
    }

    private func acquire() async {
        if !isLocked {
            isLocked = true
            return
        }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    private func release() {
        guard !waiters.isEmpty else {
            isLocked = false
            return
        }
        let next = waiters.removeFirst()
        next.resume()
    }
}

struct ClipboardPasteSessionIdentity: Equatable {
    let id: String
    let text: String
}

struct ClipboardRestoreChain<Snapshot> {
    private(set) var activeSession: ClipboardPasteSessionIdentity?
    private var originalSnapshot: Snapshot?

    mutating func originalSnapshotForNextPaste(
        currentSession: ClipboardPasteSessionIdentity?,
        makeSnapshot: () -> Snapshot
    ) -> Snapshot {
        if currentSession == activeSession, let originalSnapshot {
            return originalSnapshot
        }
        clear()
        return makeSnapshot()
    }

    mutating func begin(
        session: ClipboardPasteSessionIdentity,
        originalSnapshot: Snapshot
    ) {
        activeSession = session
        self.originalSnapshot = originalSnapshot
    }

    mutating func clear(ifSessionMatches session: ClipboardPasteSessionIdentity? = nil) {
        if let session, activeSession != session {
            return
        }
        clear()
    }

    private mutating func clear() {
        activeSession = nil
        originalSnapshot = nil
    }
}

enum ClipboardContextPolicy {
    static func userClipboardText(_ text: String?, pasteSessionID: String?) -> String? {
        pasteSessionID == nil ? text : nil
    }
}

extension ClipboardManager {
    @MainActor
    static func getUserClipboardContentForRecordingContext() async -> String? {
        await ClipboardTransactionCoordinator.shared.withExclusiveAccessUnlessCancelled {
            let pasteboard = NSPasteboard.general
            return ClipboardContextPolicy.userClipboardText(
                pasteboard.string(forType: .string),
                pasteSessionID: pasteboard.string(forType: pasteSessionType)
            )
        } ?? nil
    }
}
