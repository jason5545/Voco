// KeyboardView.swift
// SwiftUI keyboard UI for Voco voice input
// [AI-Claude: 2026-03-02]

import SwiftUI

/// View model shared between KeyboardView and KeyboardViewController
@MainActor
class KeyboardViewModel: ObservableObject {
    @Published var isRecording = false
    @Published var isTranscribing = false
    @Published var statusText = "Tap to start"
    @Published var audioLevel: Float = 0
    @Published var selectedLanguage: String = "auto"
    @Published var hasFullAccess = false
    @Published var isModelLoaded = false

    /// Callbacks wired by KeyboardViewController
    var onStartRecording: (() -> Void)?
    var onStopRecording: (() -> Void)?
    var onDeleteBackward: (() -> Void)?
    var onInsertNewline: (() -> Void)?
    var onNextKeyboard: (() -> Void)?
    var onLanguageChanged: ((String) -> Void)?

    let availableLanguages: [(code: String, name: String)] = [
        ("auto", "Auto"),
        ("en", "English"),
        ("zh", "中文"),
        ("ja", "日本語"),
        ("ko", "한국어"),
        ("de", "Deutsch"),
        ("fr", "Français"),
        ("es", "Español"),
    ]

    func toggleRecording() {
        if isRecording {
            onStopRecording?()
        } else {
            onStartRecording?()
        }
    }
}

struct KeyboardView: View {
    @ObservedObject var viewModel: KeyboardViewModel

    var body: some View {
        VStack(spacing: 8) {
            // Status bar
            statusBar

            // Main controls
            HStack(spacing: 12) {
                // Next keyboard button
                Button(action: { viewModel.onNextKeyboard?() }) {
                    Image(systemName: "globe")
                        .font(.title2)
                        .frame(width: 44, height: 44)
                }
                .foregroundColor(.primary)

                // Record button (main action)
                recordButton

                // Delete button
                Button(action: { viewModel.onDeleteBackward?() }) {
                    Image(systemName: "delete.left")
                        .font(.title2)
                        .frame(width: 44, height: 44)
                }
                .foregroundColor(.primary)

                // Newline button
                Button(action: { viewModel.onInsertNewline?() }) {
                    Image(systemName: "return")
                        .font(.title2)
                        .frame(width: 44, height: 44)
                }
                .foregroundColor(.primary)
            }
            .padding(.horizontal, 8)

            // Language picker
            languagePicker
        }
        .padding(.vertical, 8)
        .background(Color(.systemBackground))
    }

    // MARK: - Subviews

    private var statusBar: some View {
        Group {
            if !viewModel.hasFullAccess {
                Text("Enable \"Allow Full Access\" in Settings > Voco")
                    .font(.caption)
                    .foregroundColor(.red)
                    .padding(.horizontal)
            } else if !viewModel.isModelLoaded {
                Text("Download a model in the Voco app first")
                    .font(.caption)
                    .foregroundColor(.orange)
                    .padding(.horizontal)
            } else {
                HStack {
                    if viewModel.isRecording {
                        audioLevelIndicator
                    }
                    Text(viewModel.statusText)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal)
            }
        }
    }

    private var recordButton: some View {
        Button(action: { viewModel.toggleRecording() }) {
            ZStack {
                Circle()
                    .fill(viewModel.isRecording ? Color.red : Color.accentColor)
                    .frame(width: 60, height: 60)

                if viewModel.isTranscribing {
                    ProgressView()
                        .tint(.white)
                } else {
                    Image(systemName: viewModel.isRecording ? "stop.fill" : "mic.fill")
                        .font(.title)
                        .foregroundColor(.white)
                }
            }
        }
        .disabled(viewModel.isTranscribing || !viewModel.hasFullAccess || !viewModel.isModelLoaded)
    }

    private var audioLevelIndicator: some View {
        HStack(spacing: 2) {
            ForEach(0..<5, id: \.self) { i in
                RoundedRectangle(cornerRadius: 1)
                    .fill(Float(i) / 5.0 < viewModel.audioLevel ? Color.green : Color.gray.opacity(0.3))
                    .frame(width: 3, height: 8 + CGFloat(i) * 2)
            }
        }
    }

    private var languagePicker: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(viewModel.availableLanguages, id: \.code) { lang in
                    Button(action: {
                        viewModel.selectedLanguage = lang.code
                        viewModel.onLanguageChanged?(lang.code)
                    }) {
                        Text(lang.name)
                            .font(.caption2)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(
                                viewModel.selectedLanguage == lang.code
                                    ? Color.accentColor.opacity(0.2)
                                    : Color.gray.opacity(0.1)
                            )
                            .cornerRadius(8)
                    }
                    .foregroundColor(viewModel.selectedLanguage == lang.code ? .accentColor : .primary)
                }
            }
            .padding(.horizontal, 8)
        }
    }
}
