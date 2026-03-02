// SetupGuideView.swift
// Keyboard installation guide for iOS
// [AI-Claude: 2026-03-02]

import SwiftUI

struct SetupGuideView: View {
    @State private var hasModel = false

    var body: some View {
        NavigationStack {
            List {
                Section {
                    modelStatusRow
                } header: {
                    Text("Model Status")
                }

                Section {
                    stepRow(number: 1, title: "Open Settings", detail: "Go to Settings > General > Keyboard")
                    stepRow(number: 2, title: "Add Keyboard", detail: "Tap \"Keyboards\" > \"Add New Keyboard...\"")
                    stepRow(number: 3, title: "Select Voco", detail: "Find and tap \"Voco\" in the third-party keyboards list")
                    stepRow(number: 4, title: "Allow Full Access", detail: "Tap \"Voco\" > enable \"Allow Full Access\". This is required for microphone access.")
                } header: {
                    Text("Enable Voco Keyboard")
                }

                Section {
                    stepRow(number: 5, title: "Switch Keyboard", detail: "In any text field, tap the globe icon to switch to Voco")
                    stepRow(number: 6, title: "Start Dictating", detail: "Tap the microphone button on the Voco keyboard")
                } header: {
                    Text("Usage")
                }
            }
            .navigationTitle("Setup Guide")
            .onAppear { checkModelStatus() }
        }
    }

    private var modelStatusRow: some View {
        HStack {
            Image(systemName: hasModel ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                .foregroundStyle(hasModel ? .green : .orange)
                .font(.title2)

            VStack(alignment: .leading, spacing: 2) {
                Text(hasModel ? "Model Ready" : "No Model Downloaded")
                    .font(.headline)
                Text(hasModel
                     ? "You can use the Voco keyboard for voice-to-text."
                     : "Download a model in the Models tab before using the keyboard.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }

    private func stepRow(number: Int, title: String, detail: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text("\(number)")
                .font(.callout.bold())
                .foregroundStyle(.white)
                .frame(width: 26, height: 26)
                .background(Circle().fill(.blue))

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.body.bold())
                Text(detail)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }

    private func checkModelStatus() {
        let models = PredefinedModels.models.compactMap { $0 as? WhisperCoreMLModel }
        hasModel = models.contains { WhisperCoreMLModelManager.isModelDownloaded(modelId: $0.coremlModelId) }
    }
}
