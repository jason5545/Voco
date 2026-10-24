import Foundation
import os

final class RealtimeAudioChunkGate: @unchecked Sendable {
    private struct State {
        var callback: ((Data) -> Void)?
        var isActive = false
        var droppedChunks = 0
    }

    let pendingChunks: OSAllocatedUnfairLock<[Data]>
    private let maxBufferedChunks = 2_048
    private let state = OSAllocatedUnfairLock(initialState: State())

    init() {
        pendingChunks = OSAllocatedUnfairLock(initialState: [])
    }

    func receive(_ data: Data) {
        let callback = state.withLock { state -> ((Data) -> Void)? in
            guard state.isActive else {
                pendingChunks.withLock { chunks in
                    if chunks.count < maxBufferedChunks {
                        chunks.append(data)
                    } else {
                        state.droppedChunks += 1
                    }
                }
                return nil
            }
            return state.callback
        }
        callback?(data)
    }

    func activate(_ callback: @escaping (Data) -> Void) {
        state.withLock { state in
            state.callback = callback
            state.isActive = false
        }

        while true {
            let chunks = pendingChunks.withLock { chunks -> [Data] in
                let result = chunks
                chunks.removeAll(keepingCapacity: true)
                return result
            }
            for chunk in chunks {
                callback(chunk)
            }

            let shouldFinish = state.withLock { state -> Bool in
                let hasPending = pendingChunks.withLock { !$0.isEmpty }
                if !hasPending {
                    state.isActive = true
                }
                return !hasPending
            }
            if shouldFinish { return }
        }
    }

    func reset() {
        state.withLock { state in
            state.callback = nil
            state.isActive = false
            pendingChunks.withLock { $0.removeAll(keepingCapacity: false) }
        }
    }

    func takeDroppedChunkCount() -> Int {
        state.withLock { state in
            let count = state.droppedChunks
            state.droppedChunks = 0
            return count
        }
    }
}
