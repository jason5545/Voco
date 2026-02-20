import SwiftUI
import AppKit

@MainActor
class NotchWindowManager: ObservableObject {
    @Published var isVisible = false
    private var windowController: NSWindowController?
    var notchPanel: NotchRecorderPanel?
    private var hostingController: NSViewController?
    private weak var whisperState: WhisperState?
    private weak var recorder: Recorder?

    init(whisperState: WhisperState, recorder: Recorder) {
        self.whisperState = whisperState
        self.recorder = recorder

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleHideNotification),
            name: NSNotification.Name("HideNotchRecorder"),
            object: nil
        )
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
        if let panel = notchPanel {
            panel.contentViewController = nil
            panel.contentView = nil
            panel.orderOut(nil)
        }
        hostingController = nil
        windowController?.close()
    }
    
    @objc private func handleHideNotification() {
        hide()
    }
    
    func show() {
        if isVisible { return }

        let activeScreen = NSApp.keyWindow?.screen ?? NSScreen.main ?? NSScreen.screens[0]

        initializeWindow(screen: activeScreen)
        self.isVisible = true
        notchPanel?.show()
    }
    
    func hide() {
        guard isVisible else { return }

        self.isVisible = false

        self.notchPanel?.hide { [weak self] in
            guard let self = self else { return }
            self.deinitializeWindow()
        }
    }
    
    private func initializeWindow(screen: NSScreen) {
        deinitializeWindow()

        guard let whisperState = whisperState, let recorder = recorder else { return }

        let metrics = NotchRecorderPanel.calculateWindowMetrics()
        let panel = NotchRecorderPanel(contentRect: metrics.frame)

        let notchRecorderView = NotchRecorderView(whisperState: whisperState, recorder: recorder)
            .environmentObject(whisperState.enhancementService!)

        let hostingController = NotchRecorderHostingController(rootView: notchRecorderView)
        panel.contentViewController = hostingController

        self.notchPanel = panel
        self.windowController = NSWindowController(window: panel)
        self.hostingController = hostingController

        panel.orderFrontRegardless()
    }
    
    private func deinitializeWindow() {
        if let panel = notchPanel {
            panel.contentViewController = nil
            panel.contentView = nil
            panel.orderOut(nil)
        }
        hostingController = nil
        windowController?.close()
        windowController = nil
        notchPanel = nil
    }
    
    func toggle() {
        if isVisible {
            hide()
        } else {
            show()
        }
    }
} 
