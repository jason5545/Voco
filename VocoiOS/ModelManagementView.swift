// ModelManagementView.swift
// CoreML model management for iOS
// [AI-Claude: 2026-03-02]

import SwiftUI

struct ModelManagementView: View {
    @State private var downloadProgress: [String: Double] = [:]
    @State private var downloadingModels: Set<String> = []
    @State private var errorMessage: String?
    @State private var refreshTrigger = false

    private let coremlModels: [WhisperCoreMLModel] = {
        PredefinedModels.models.compactMap { $0 as? WhisperCoreMLModel }
    }()

    private var defaultModelId: String {
        let defaults = UserDefaults(suiteName: AppIdentifiers.appGroupID)
        return defaults?.string(forKey: "KeyboardModelId") ?? "whisper-small-int8"
    }

    var body: some View {
        NavigationStack {
            List {
                if let error = errorMessage {
                    Section {
                        Label(error, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.red)
                    }
                }

                Section {
                    ForEach(coremlModels, id: \.name) { model in
                        modelRow(model)
                    }
                } header: {
                    Text("Whisper CoreML Models")
                } footer: {
                    Text("Models are stored in a shared container accessible by the keyboard extension. Download at least one model to use voice-to-text.")
                }
            }
            .navigationTitle("Models")
            .refreshable { refreshTrigger.toggle() }
        }
    }

    private func modelRow(_ model: WhisperCoreMLModel) -> some View {
        let modelId = model.coremlModelId
        let isDownloaded = WhisperCoreMLModelManager.isModelDownloaded(modelId: modelId)
        let isDefault = defaultModelId == modelId
        let isDownloading = downloadingModels.contains(modelId)

        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(model.displayName)
                            .font(.headline)
                        if isDefault {
                            Text("Default")
                                .font(.caption2.bold())
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Capsule().fill(.blue))
                                .foregroundStyle(.white)
                        }
                    }

                    HStack(spacing: 12) {
                        Label(model.size, systemImage: "internaldrive")
                        Label("RAM: \(model.ramUsage, specifier: "%.1f") GB", systemImage: "memorychip")
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)

                    Text(model.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }

                Spacer()

                if isDownloading {
                    ProgressView(value: downloadProgress[modelId] ?? 0)
                        .progressViewStyle(.circular)
                        .frame(width: 32, height: 32)
                } else if isDownloaded {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .font(.title2)
                }
            }

            if isDownloading {
                ProgressView(value: downloadProgress[modelId] ?? 0)
                    .progressViewStyle(.linear)
                Text("Downloading... \(Int((downloadProgress[modelId] ?? 0) * 100))%")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else if isDownloaded {
                HStack(spacing: 12) {
                    if !isDefault {
                        Button("Set as Default") {
                            setDefaultModel(modelId)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }

                    Button("Delete", role: .destructive) {
                        deleteModel(modelId)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            } else {
                Button {
                    Task { await downloadModel(model) }
                } label: {
                    Label("Download", systemImage: "arrow.down.circle")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            }
        }
        .padding(.vertical, 4)
    }

    private func downloadModel(_ model: WhisperCoreMLModel) async {
        let modelId = model.coremlModelId
        let baseURL = "https://github.com/jason5545/Voco/releases/download/coreml-models"

        downloadingModels.insert(modelId)
        downloadProgress[modelId] = 0
        errorMessage = nil

        do {
            try await WhisperCoreMLModelManager.downloadModel(
                modelId: modelId,
                baseURL: baseURL
            ) { progress in
                Task { @MainActor in
                    downloadProgress[modelId] = progress
                }
            }

            // If no default model set yet, auto-set this one
            let defaults = UserDefaults(suiteName: AppIdentifiers.appGroupID)
            if defaults?.string(forKey: "KeyboardModelId") == nil {
                defaults?.set(modelId, forKey: "KeyboardModelId")
            }
        } catch {
            errorMessage = "Download failed: \(error.localizedDescription)"
        }

        downloadingModels.remove(modelId)
        downloadProgress.removeValue(forKey: modelId)
        refreshTrigger.toggle()
    }

    private func setDefaultModel(_ modelId: String) {
        let defaults = UserDefaults(suiteName: AppIdentifiers.appGroupID)
        defaults?.set(modelId, forKey: "KeyboardModelId")
        refreshTrigger.toggle()
    }

    private func deleteModel(_ modelId: String) {
        do {
            try WhisperCoreMLModelManager.deleteModel(modelId: modelId)
            // If deleting the default, clear the preference
            if defaultModelId == modelId {
                let defaults = UserDefaults(suiteName: AppIdentifiers.appGroupID)
                defaults?.removeObject(forKey: "KeyboardModelId")
            }
        } catch {
            errorMessage = "Delete failed: \(error.localizedDescription)"
        }
        refreshTrigger.toggle()
    }
}
