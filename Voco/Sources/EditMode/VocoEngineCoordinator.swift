import Combine
import Foundation

@MainActor
final class VocoEngineCoordinator: ObservableObject {
    @Published var downloadProgress: [String: Double] = [:]
    @Published var whisperMLXDownloadStates: [String: Bool] = [:]
    @Published var qwen3DownloadStates: [String: Bool] = [:]
    @Published private(set) var editModeSelection: EditModeSelectionSnapshot?
    @Published var pendingDictionaryEntry: WordSubstitution?

    var editModeDetectionTask: Task<Void, Never>?
    var capturedFrontAppPID: pid_t?
    var dictionaryDismissTimer: DispatchWorkItem?
    var deferredModelCleanupTask: Task<Void, Never>?

    private var forwardingCancellable: AnyCancellable?

    var isEditMode: Bool { editModeSelection != nil }
    var editModeSelectedText: String? { editModeSelection?.text }

    func armEditMode(selectedText: String, pid: pid_t) {
        editModeSelection = EditModeSelectionSnapshot(text: selectedText, pid: pid)
    }

    func clearEditMode() {
        editModeSelection = nil
    }

    func forwardChanges(to engine: VoiceInkEngine) {
        forwardingCancellable = objectWillChange.sink { [weak engine] _ in
            engine?.objectWillChange.send()
        }
    }
}

typealias ForkEngineState = VocoEngineCoordinator
