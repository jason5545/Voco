import Foundation

final class VocoEvidenceBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var actions: [() -> Void] = []

    func append(_ action: @escaping () -> Void) {
        lock.lock()
        actions.append(action)
        lock.unlock()
    }

    func commit() {
        let pending = takeActions()
        pending.forEach { $0() }
    }

    func discard() {
        _ = takeActions()
    }

    private func takeActions() -> [() -> Void] {
        lock.lock()
        let pending = actions
        actions.removeAll()
        lock.unlock()
        return pending
    }
}

enum VocoEvidenceScope {
    @TaskLocal static var current: VocoEvidenceBuffer?

    static func record(_ action: @escaping () -> Void) {
        if let current {
            current.append(action)
        } else {
            action()
        }
    }
}
