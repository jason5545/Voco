import SwiftUI
import AppKit

@MainActor
class MiniWindowManager: ObservableObject {
    @Published var isVisible = false
    private var windowController: NSWindowController?
    private var panel: MiniRecorderPanel?

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
        if panel == nil { initializeWindow() }
        isVisible = true
        panel?.show()
        StartupTracer.checkpoint("MiniWindowManager.show_panel_visible")
    }

    func hide() {
        guard isVisible else { return }
        isVisible = false
        panel?.orderOut(nil)
    }

    func destroyWindow() {
        isVisible = false
        deinitializeWindow()
    }

    private func initializeWindow() {
        StartupTracer.checkpoint("initializeWindow_enter")
        deinitializeWindow()
        let metrics = MiniRecorderPanel.calculateWindowMetrics()
        StartupTracer.checkpoint("initializeWindow_metrics_calculated")
        let newPanel = MiniRecorderPanel(contentRect: metrics)
        StartupTracer.checkpoint("initializeWindow_panel_created")
        let view = makeView(self)
        StartupTracer.checkpoint("initializeWindow_swiftui_view_created")
        let hostingController = NSHostingController(rootView: view)
        StartupTracer.checkpoint("initializeWindow_hosting_controller_created")
        newPanel.contentView = hostingController.view
        panel = newPanel
        windowController = NSWindowController(window: newPanel)
        newPanel.orderFrontRegardless()
        StartupTracer.checkpoint("initializeWindow_orderFrontRegardless_done")
    }

    private func deinitializeWindow() {
        panel?.orderOut(nil)
        windowController?.close()
        windowController = nil
        panel = nil
    }

    func toggle() {
        isVisible ? hide() : show()
    }
}
