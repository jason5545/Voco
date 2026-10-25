import SwiftUI

struct VocoMenuBarView: View {
    @Environment(\.openWindow) private var openWindow
    @EnvironmentObject var engine: VoiceInkEngine
    @EnvironmentObject var recorderUIManager: RecorderUIManager
    @EnvironmentObject var transcriptionModelManager: TranscriptionModelManager
    @EnvironmentObject var whisperModelManager: WhisperModelManager
    @EnvironmentObject var recordingShortcutManager: RecordingShortcutManager
    @EnvironmentObject var menuBarManager: MenuBarManager
    @EnvironmentObject var mainWindowNavigation: MainWindowNavigation
    @EnvironmentObject var updaterViewModel: UpdaterViewModel
    @EnvironmentObject var enhancementService: AIEnhancementService
    @EnvironmentObject var aiService: AIService
    @ObservedObject private var launchAtLoginManager = LaunchAtLoginManager.shared
    @ObservedObject var audioDeviceManager = AudioDeviceManager.shared

    var body: some View {
        VStack {
            Button("Toggle Recorder") {
                recorderUIManager.handleToggleRecorderPanelNotification()
            }

            Divider()

            transcriptionModelMenu

            Divider()

            Toggle("AI Enhancement", isOn: $enhancementService.isEnhancementEnabled)

            aiProviderMenu

            LanguageSelectionView(
                transcriptionModelManager: transcriptionModelManager,
                displayMode: .menuItem,
                whisperPrompt: whisperModelManager.whisperPrompt
            )

            audioInputMenu

            Divider()

            Button("Retry Last Transcription") {
                LastTranscriptionService.retryLastTranscription(
                    from: engine.modelContext,
                    transcriptionModelManager: transcriptionModelManager,
                    serviceRegistry: engine.serviceRegistry,
                    enhancementService: enhancementService
                )
            }

            Button("Copy Last Transcription") {
                LastTranscriptionService.copyLastTranscription(from: engine.modelContext)
            }
            .keyboardShortcut("c", modifiers: [.command, .shift])

            Button("Quick History") {
                menuBarManager.openQuickHistory()
            }
            .keyboardShortcut("h", modifiers: [.command, .shift])

            Button("Settings") {
                showMainWindowAndNavigate(to: "Settings")
            }
            .keyboardShortcut(",", modifiers: .command)

            Button(menuBarManager.isMenuBarOnly ? "Show Dock Icon" : "Hide Dock Icon") {
                let shouldShowMainWindow = menuBarManager.isMenuBarOnly
                menuBarManager.toggleMenuBarOnly()

                if shouldShowMainWindow {
                    showMainWindow()
                }
            }
            .keyboardShortcut("d", modifiers: [.command, .shift])

            Toggle(
                "Launch at Login",
                isOn: Binding(
                    get: { launchAtLoginManager.isEnabled },
                    set: { launchAtLoginManager.setEnabled($0) }
                )
            )
            .disabled(launchAtLoginManager.isUpdating)

            Divider()

            Button("Check for Updates") {
                updaterViewModel.checkForUpdates()
            }
            .disabled(!updaterViewModel.canCheckForUpdates)

            Button("Help and Support") {
                EmailSupport.openSupportEmail()
            }

            Divider()

            Button("Quit Voco") {
                NSApplication.shared.terminate(nil)
            }
        }
    }

    private var transcriptionModelMenu: some View {
        Menu {
            ForEach(transcriptionModelManager.usableModels, id: \.id) { model in
                Button {
                    Task {
                        transcriptionModelManager.setDefaultTranscriptionModel(model)
                    }
                } label: {
                    HStack {
                        Text(model.displayName)
                        if transcriptionModelManager.currentTranscriptionModel?.id == model.id {
                            Image(systemName: "checkmark")
                        }
                    }
                }
            }

            Divider()

            Button("Manage Models") {
                showMainWindowAndNavigate(to: "AI Models")
            }
        } label: {
            HStack {
                Text("Transcription Model: \(transcriptionModelManager.currentTranscriptionModel?.displayName ?? "None")")
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 10))
            }
        }
    }

    private var aiProviderMenu: some View {
        Menu {
            ForEach(aiService.connectedProviders, id: \.self) { provider in
                Button {
                    aiService.selectedProvider = provider
                } label: {
                    HStack {
                        Text(provider.rawValue)
                        if aiService.selectedProvider == provider {
                            Image(systemName: "checkmark")
                        }
                    }
                }
            }

            if aiService.connectedProviders.isEmpty {
                Text("No providers connected")
                    .foregroundColor(.secondary)
            }
        } label: {
            HStack {
                Text("AI Provider: \(aiService.selectedProvider.rawValue)")
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 10))
            }
        }
    }

    private var audioInputMenu: some View {
        Menu {
            ForEach(audioDeviceManager.availableDevices, id: \.id) { device in
                Button {
                    audioDeviceManager.selectDeviceAndSwitchToCustomMode(id: device.id)
                } label: {
                    HStack {
                        Text(device.name)
                        if audioDeviceManager.getCurrentDevice() == device.id {
                            Image(systemName: "checkmark")
                        }
                    }
                }
            }

            if audioDeviceManager.availableDevices.isEmpty {
                Text("No devices available")
                    .foregroundColor(.secondary)
            }
        } label: {
            HStack {
                Text("Audio Input")
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 10))
            }
        }
    }

    private func showMainWindow() {
        let existingWindow = WindowManager.shared.currentMainWindow()
        menuBarManager.activateForPresentedWindow()

        if existingWindow == nil {
            WindowManager.shared.prepareForUserRequestedMainWindow()
            openWindow(id: AppWindowID.main)
        } else {
            openWindow(id: AppWindowID.main)
            WindowManager.shared.showMainWindow()
        }
    }

    private func showMainWindowAndNavigate(to destination: String) {
        mainWindowNavigation.navigate(to: destination)
        showMainWindow()
    }
}
