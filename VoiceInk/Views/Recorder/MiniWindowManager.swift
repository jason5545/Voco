import SwiftUI
import AppKit

@MainActor
class MiniWindowManager: ObservableObject {
    @Published var isVisible = false
    private var windowController: NSWindowController?
    private var miniPanel: MiniRecorderPanel?

    // Type-erased view factory stored as closure
    private let makeView: (MiniWindowManager) -> AnyView

    init(engine: VoiceInkEngine, recorder: Recorder) {
        guard let enhancementService = engine.enhancementService else {
            preconditionFailure("VoiceInkEngine.enhancementService must be non-nil when creating MiniWindowManager")
        }
        self.makeView = { manager in
            AnyView(
                MiniRecorderView(stateProvider: engine, recorder: recorder)
                    .environmentObject(manager)
                    .environmentObject(enhancementService)
            )
        }
        setupNotifications()
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
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
        StartupTracer.checkpoint("MiniWindowManager.show_enter")

        if miniPanel == nil {
            let activeScreen = NSApp.keyWindow?.screen ?? NSScreen.main ?? NSScreen.screens[0]
            initializeWindow(screen: activeScreen)
        }

        self.isVisible = true
        miniPanel?.show()
        StartupTracer.checkpoint("MiniWindowManager.show_panel_visible")
    }

    func hide() {
        guard isVisible else { return }
        self.isVisible = false
        miniPanel?.orderOut(nil)
    }

    func destroyWindow() {
        isVisible = false
        deinitializeWindow()
    }

    private func initializeWindow(screen: NSScreen) {
        StartupTracer.checkpoint("initializeWindow_enter")
        deinitializeWindow()

        let metrics = MiniRecorderPanel.calculateWindowMetrics()
        StartupTracer.checkpoint("initializeWindow_metrics_calculated")
        let panel = MiniRecorderPanel(contentRect: metrics)
        StartupTracer.checkpoint("initializeWindow_panel_created")

        let miniRecorderView = makeView(self)
        StartupTracer.checkpoint("initializeWindow_swiftui_view_created")
        let hostingController = NSHostingController(rootView: miniRecorderView)
        StartupTracer.checkpoint("initializeWindow_hosting_controller_created")
        panel.contentView = hostingController.view

        self.miniPanel = panel
        self.windowController = NSWindowController(window: panel)

        panel.orderFrontRegardless()
        StartupTracer.checkpoint("initializeWindow_orderFrontRegardless_done")
    }

    private func deinitializeWindow() {
        miniPanel?.orderOut(nil)
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
