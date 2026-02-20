import SwiftUI
import AppKit

@MainActor
class MiniWindowManager: ObservableObject {
    @Published var isVisible = false
    private var windowController: NSWindowController?
    private var miniPanel: MiniRecorderPanel?
    private var hostingController: NSViewController?
    private weak var whisperState: WhisperState?
    private weak var recorder: Recorder?
    
    init(whisperState: WhisperState, recorder: Recorder) {
        self.whisperState = whisperState
        self.recorder = recorder
        setupNotifications()
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
        if let panel = miniPanel {
            panel.contentViewController = nil
            panel.contentView = nil
            panel.orderOut(nil)
        }
        hostingController = nil
        windowController?.close()
    }
    
    private func setupNotifications() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleHideNotification),
            name: NSNotification.Name("HideMiniRecorder"),
            object: nil
        )
    }
    
    @objc private func handleHideNotification() {
        hide()
    }
    func show() {
        if isVisible { return }

        let activeScreen = NSApp.keyWindow?.screen ?? NSScreen.main ?? NSScreen.screens[0]

        initializeWindow(screen: activeScreen)
        self.isVisible = true 
        miniPanel?.show()
    }

    func hide() {
        guard isVisible else { return }

        self.isVisible = false
        self.miniPanel?.hide { [weak self] in
            guard let self = self else { return }
            self.deinitializeWindow()
        }
    }
    
    private func initializeWindow(screen: NSScreen) {
        deinitializeWindow()

        guard let whisperState = whisperState, let recorder = recorder else { return }

        let metrics = MiniRecorderPanel.calculateWindowMetrics()
        let panel = MiniRecorderPanel(contentRect: metrics)

        let miniRecorderView = MiniRecorderView(whisperState: whisperState, recorder: recorder)
            .environmentObject(self)
            .environmentObject(whisperState.enhancementService!)

        let hostingController = NSHostingController(rootView: miniRecorderView)
        panel.contentViewController = hostingController

        self.miniPanel = panel
        self.windowController = NSWindowController(window: panel)
        self.hostingController = hostingController

        panel.orderFrontRegardless()
    }
    
    private func deinitializeWindow() {
        if let panel = miniPanel {
            panel.contentViewController = nil
            panel.contentView = nil
            panel.orderOut(nil)
        }
        hostingController = nil
        windowController?.close()
        windowController = nil
        miniPanel = nil
    }
    
    func toggle() {
        if isVisible {
            hide()
        } else {
            show()
        }
    }
} 
