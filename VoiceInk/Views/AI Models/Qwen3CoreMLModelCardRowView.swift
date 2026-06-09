// Qwen3CoreMLModelCardRowView.swift
// UI card for Qwen3-ASR CoreML Hybrid model display and management
// [AI-Claude: 2026-03-13]

import SwiftUI
import Combine
import AppKit

struct Qwen3CoreMLModelCardRowView: View {
    let model: Qwen3CoreMLModel
    @ObservedObject var engine: VoiceInkEngine
    @ObservedObject var transcriptionModelManager: TranscriptionModelManager

    var isCurrent: Bool {
        transcriptionModelManager.currentTranscriptionModel?.name == model.name
    }

    var isDownloaded: Bool {
        engine.isQwen3CoreMLModelDownloaded(model)
    }

    var isDownloading: Bool {
        engine.isQwen3CoreMLModelDownloading(model)
    }

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 6) {
                headerSection
                metadataSection
                descriptionSection
                componentStatus
                progressSection
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            actionSection
        }
        .padding(16)
        .background(AppMaterialCardBackground(isSelected: isCurrent, cornerRadius: AppTheme.Radius.card))
    }

    private var headerSection: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(model.displayName)
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(Color(.labelColor))

            Text("ANE + GPU")
                .font(.system(size: 11, weight: .medium))
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Capsule().fill(Color.orange.opacity(0.8)))
                .foregroundColor(.white)

            statusBadge
            Spacer()
        }
    }

    private var statusBadge: some View {
        Group {
            if isCurrent {
                Text("Default")
                    .font(.system(size: 11, weight: .medium))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color.accentColor))
                    .foregroundColor(.white)
            } else if isDownloaded {
                Text("Downloaded")
                    .font(.system(size: 11, weight: .medium))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color(.quaternaryLabelColor)))
                    .foregroundColor(Color(.labelColor))
            }
        }
    }

    private var metadataSection: some View {
        HStack(spacing: 12) {
            Label(model.language, systemImage: "globe")
            Label(model.size, systemImage: "internaldrive")
            HStack(spacing: 3) {
                Text("Speed")
                progressDotsWithNumber(value: model.speed * 10)
            }
            .fixedSize(horizontal: true, vertical: false)
            HStack(spacing: 3) {
                Text("Accuracy")
                progressDotsWithNumber(value: model.accuracy * 10)
            }
            .fixedSize(horizontal: true, vertical: false)
        }
        .font(.system(size: 11))
        .foregroundColor(Color(.secondaryLabelColor))
        .lineLimit(1)
    }

    private var descriptionSection: some View {
        Text(model.description)
            .font(.system(size: 11))
            .foregroundColor(Color(.secondaryLabelColor))
            .lineLimit(2)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.top, 4)
    }

    private var componentStatus: some View {
        Group {
            if !isDownloaded && !isDownloading {
                HStack(spacing: 12) {
                    componentBadge(
                        label: "CoreML Encoder",
                        isReady: engine.isQwen3CoreMLEncoderDownloaded(model)
                    )
                    componentBadge(
                        label: "MLX Decoder",
                        isReady: engine.isQwen3CoreMLDecoderDownloaded(model)
                    )
                }
                .padding(.top, 4)
            }
        }
    }

    private func componentBadge(label: String, isReady: Bool) -> some View {
        HStack(spacing: 4) {
            Image(systemName: isReady ? "checkmark.circle.fill" : "circle")
                .font(.system(size: 10))
                .foregroundColor(isReady ? .green : Color(.tertiaryLabelColor))
            Text(label)
                .font(.system(size: 10))
                .foregroundColor(Color(.secondaryLabelColor))
        }
    }

    private var progressSection: some View {
        Group {
            if isDownloading {
                let progress = engine.downloadProgress[model.name] ?? 0.0
                ProgressView(value: progress)
                    .progressViewStyle(LinearProgressViewStyle())
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 8)
            }
        }
    }

    private var actionSection: some View {
        HStack(spacing: 8) {
            if isCurrent {
                Text("Default Model")
                    .font(.system(size: 12))
                    .foregroundColor(Color(.secondaryLabelColor))
            } else if isDownloaded {
                Button(action: {
                    transcriptionModelManager.setDefaultTranscriptionModel(model)
                }) {
                    Text("Set as Default")
                        .font(.system(size: 12))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            } else {
                Button(action: {
                    Task {
                        await engine.downloadQwen3CoreMLModel(model)
                    }
                }) {
                    HStack(spacing: 4) {
                        Text(isDownloading ? "Downloading..." : "Download")
                        Image(systemName: "arrow.down.circle")
                    }
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(Capsule().fill(Color.accentColor))
                }
                .buttonStyle(.plain)
                .disabled(isDownloading)
            }

            if isDownloaded {
                Menu {
                    Button(action: {
                        engine.deleteQwen3CoreMLModel(model)
                    }) {
                        Label("Delete Model", systemImage: "trash")
                    }

                    Button {
                        engine.showQwen3CoreMLModelInFinder(model)
                    } label: {
                        Label("Show in Finder", systemImage: "folder")
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .font(.system(size: 14))
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .frame(width: 20, height: 20)
            }
        }
    }
}
