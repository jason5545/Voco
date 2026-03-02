// KeyboardViewController.swift
// iOS Custom Keyboard Extension for Voco voice input
// [AI-Claude: 2026-03-02]

import UIKit
import SwiftUI
import os

class KeyboardViewController: UIInputViewController {
    private static let logger = Logger(subsystem: AppIdentifiers.subsystem, category: "KeyboardViewController")

    private let viewModel = KeyboardViewModel()
    private let recorder = KeyboardAudioRecorder()
    private let engine = WhisperCoreMLEngine()
    private let enhancementService = KeyboardEnhancementService.shared

    private var levelTimer: Timer?
    private var hostingController: UIHostingController<KeyboardView>?

    // MARK: - Lifecycle

    override func viewDidLoad() {
        super.viewDidLoad()

        setupSwiftUIView()
        setupCallbacks()
        loadLanguagePreference()
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        checkFullAccess()
        checkModelAvailability()
    }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        stopLevelTimer()
        if recorder.isRecording {
            _ = recorder.stopRecording()
            viewModel.isRecording = false
        }
    }

    // MARK: - Setup

    private func setupSwiftUIView() {
        let keyboardView = KeyboardView(viewModel: viewModel)
        let hostingVC = UIHostingController(rootView: keyboardView)
        hostingVC.view.translatesAutoresizingMaskIntoConstraints = false
        hostingVC.view.backgroundColor = .clear

        addChild(hostingVC)
        view.addSubview(hostingVC.view)
        hostingVC.didMove(toParent: self)

        NSLayoutConstraint.activate([
            hostingVC.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            hostingVC.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            hostingVC.view.topAnchor.constraint(equalTo: view.topAnchor),
            hostingVC.view.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            hostingVC.view.heightAnchor.constraint(equalToConstant: 160),
        ])

        self.hostingController = hostingVC
    }

    private func setupCallbacks() {
        viewModel.onStartRecording = { [weak self] in
            self?.startVoiceInput()
        }
        viewModel.onStopRecording = { [weak self] in
            self?.stopAndTranscribe()
        }
        viewModel.onDeleteBackward = { [weak self] in
            self?.textDocumentProxy.deleteBackward()
        }
        viewModel.onInsertNewline = { [weak self] in
            self?.textDocumentProxy.insertText("\n")
        }
        viewModel.onNextKeyboard = { [weak self] in
            self?.advanceToNextInputMode()
        }
        viewModel.onLanguageChanged = { [weak self] code in
            self?.saveLanguagePreference(code)
        }
    }

    // MARK: - Full Access Check

    private func checkFullAccess() {
        let hasAccess = hasFullAccess
        Task { @MainActor in
            viewModel.hasFullAccess = hasAccess
            if !hasAccess {
                Self.logger.warning("Full access not granted — microphone unavailable")
            }
        }
    }

    // MARK: - Model Availability

    private func checkModelAvailability() {
        // Check if any CoreML model is downloaded in the App Group container
        let modelId = preferredModelId()
        let available = WhisperCoreMLModelManager.isModelDownloaded(modelId: modelId)

        Task { @MainActor in
            viewModel.isModelLoaded = available
            if !available {
                Self.logger.info("No CoreML model available for keyboard extension")
            }
        }
    }

    private func preferredModelId() -> String {
        let defaults = UserDefaults(suiteName: AppIdentifiers.appGroupID)
        return defaults?.string(forKey: "KeyboardModelId") ?? "whisper-small-int8"
    }

    // MARK: - Voice Input

    private func startVoiceInput() {
        Task { @MainActor in
            viewModel.isRecording = true
            viewModel.statusText = "Listening..."

            do {
                try await recorder.startRecording()
                startLevelTimer()
            } catch {
                viewModel.isRecording = false
                viewModel.statusText = "Error: \(error.localizedDescription)"
                Self.logger.error("Failed to start recording: \(error)")
            }
        }
    }

    private func stopAndTranscribe() {
        stopLevelTimer()
        let samples = recorder.stopRecording()

        Task { @MainActor in
            viewModel.isRecording = false

            guard !samples.isEmpty else {
                viewModel.statusText = "No audio recorded"
                return
            }

            viewModel.isTranscribing = true
            viewModel.statusText = "Transcribing..."

            do {
                let modelId = preferredModelId()
                let modelDir = WhisperCoreMLModelManager.modelDirectory(for: modelId)
                try await engine.loadModel(from: modelDir)

                let language = viewModel.selectedLanguage == "auto" ? nil : viewModel.selectedLanguage
                let result = try await engine.transcribe(samples: samples, language: language)

                if !result.text.isEmpty {
                    // Set model provider for confidence routing
                    ChinesePostProcessingService.shared.lastModelProvider = .whisperCoreML

                    // Run enhancement pipeline (Chinese post-processing + optional LLM)
                    let enhancedText = await enhancementService.enhance(result.text, language: language)
                    textDocumentProxy.insertText(enhancedText)
                    viewModel.statusText = "Done (\(result.tokenCount) tokens)"
                } else {
                    viewModel.statusText = "No speech detected"
                }
            } catch {
                viewModel.statusText = "Error: \(error.localizedDescription)"
                Self.logger.error("Transcription failed: \(error)")
            }

            viewModel.isTranscribing = false
        }
    }

    // MARK: - Audio Level Timer

    private func startLevelTimer() {
        levelTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            Task { @MainActor in
                self.viewModel.audioLevel = self.recorder.currentLevel
            }
        }
    }

    private func stopLevelTimer() {
        levelTimer?.invalidate()
        levelTimer = nil
    }

    // MARK: - Language Preference (App Group shared)

    private func loadLanguagePreference() {
        let defaults = UserDefaults(suiteName: AppIdentifiers.appGroupID)
        if let lang = defaults?.string(forKey: "KeyboardLanguage") {
            viewModel.selectedLanguage = lang
        }
    }

    private func saveLanguagePreference(_ code: String) {
        let defaults = UserDefaults(suiteName: AppIdentifiers.appGroupID)
        defaults?.set(code, forKey: "KeyboardLanguage")
    }
}
