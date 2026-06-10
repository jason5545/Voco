import SwiftUI
import SwiftData

enum VocabularySortMode: String {
    case wordAsc = "wordAsc"
    case wordDesc = "wordDesc"
}

struct VocabularyView: View {
    @Query private var vocabularyWords: [VocabularyWord]
    @Environment(\.modelContext) private var modelContext
    @State private var newWord = ""
    @State private var showAlert = false
    @State private var alertMessage = ""
    @State private var sortMode: VocabularySortMode = .wordAsc
    @State private var isShowingRimeImport = false

    init() {
        if let savedSort = UserDefaults.standard.string(forKey: "vocabularySortMode"),
           let mode = VocabularySortMode(rawValue: savedSort) {
            _sortMode = State(initialValue: mode)
        }
    }

    private var sortedItems: [VocabularyWord] {
        switch sortMode {
        case .wordAsc:
            return vocabularyWords.sorted { $0.word.localizedCaseInsensitiveCompare($1.word) == .orderedAscending }
        case .wordDesc:
            return vocabularyWords.sorted { $0.word.localizedCaseInsensitiveCompare($1.word) == .orderedDescending }
        }
    }

    private func toggleSort() {
        sortMode = (sortMode == .wordAsc) ? .wordDesc : .wordAsc
        UserDefaults.standard.set(sortMode.rawValue, forKey: "vocabularySortMode")
    }

    private var shouldShowAddButton: Bool {
        !newWord.isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                TextField("", text: $newWord, prompt: Text("Add word to vocabulary"))
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 13))
                    .onSubmit { addWords() }
                    .labelsHidden()

                if shouldShowAddButton {
                    AddIconButton(
                        helpText: "Add word",
                        isDisabled: newWord.isEmpty,
                        action: addWords
                    )
                }

                AppIconButton(
                    systemName: "tray.and.arrow.down",
                    help: "Import RIME vocabulary",
                    size: 28,
                    iconSize: 13,
                    cornerRadius: 6
                ) {
                    isShowingRimeImport = true
                }
            }
            .animation(.easeInOut(duration: 0.2), value: shouldShowAddButton)

            if !vocabularyWords.isEmpty {
                VStack(alignment: .leading, spacing: 12) {
                    Button(action: toggleSort) {
                        HStack(spacing: 4) {
                            Text("Vocabulary Words (\(vocabularyWords.count))")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundColor(.secondary)

                            Image(systemName: sortMode == .wordAsc ? "chevron.up" : "chevron.down")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .buttonStyle(.plain)
                    .help("Sort alphabetically")

                    FlowLayout(spacing: 8) {
                        ForEach(sortedItems) { item in
                            VocabularyWordView(item: item) {
                                removeWord(item)
                            }
                        }
                    }
                    .padding(.vertical, 4)
                }
                .padding(.top, 4)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .alert("Vocabulary", isPresented: $showAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(alertMessage)
        }
        .sheet(isPresented: $isShowingRimeImport) {
            RimeVocabularyImportSheet()
        }
    }
    
    private func addWords() {
        let input = newWord.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty else { return }
        if let error = DictionaryService.addVocabularyWords(input, existing: Array(vocabularyWords), context: modelContext) {
            alertMessage = error
            showAlert = true
            return
        }
        newWord = ""
    }

    private func removeWord(_ word: VocabularyWord) {
        modelContext.delete(word)

        do {
            try modelContext.save()
        } catch {
            // Rollback the delete to restore UI consistency
            modelContext.rollback()
            alertMessage = "Failed to remove word: \(error.localizedDescription)"
            showAlert = true
        }
    }
}

private struct RimeVocabularyImportSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var modelContext

    @State private var preview: RimeVocabularyPreview?
    @State private var selectedItemIDs: Set<String> = []
    @State private var isLoading = false
    @State private var alertTitle = ""
    @State private var alertMessage = ""
    @State private var showAlert = false

    private let service = RimeVocabularyImportService.shared

    var body: some View {
        VStack(spacing: 0) {
            AppPanelHeader(title: "RIME Vocabulary Import") {
                dismiss()
            }

            content

            Divider()

            footer
        }
        .frame(width: 760, height: 620)
        .onAppear(perform: loadPreview)
        .alert(alertTitle, isPresented: $showAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(alertMessage)
        }
    }

    @ViewBuilder
    private var content: some View {
        if isLoading {
            ProgressView()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if let preview {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    summaryView(preview.summary)

                    if !preview.sourceWarnings.isEmpty {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(preview.sourceWarnings, id: \.self) { warning in
                                Label(warning, systemImage: "exclamationmark.triangle")
                                    .font(.system(size: 12))
                                    .foregroundStyle(AppTheme.Status.warning)
                            }
                        }
                    }

                    LazyVStack(spacing: 8) {
                        ForEach(preview.items) { item in
                            RimeVocabularyPreviewRow(
                                item: item,
                                isSelected: Binding(
                                    get: { selectedItemIDs.contains(item.id) },
                                    set: { isSelected in
                                        if isSelected {
                                            selectedItemIDs.insert(item.id)
                                        } else {
                                            selectedItemIDs.remove(item.id)
                                        }
                                    }
                                )
                            )
                        }
                    }

                    RimeUserDBPolicyView(policy: preview.learnedUserDBPolicy)
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .topLeading)
            }
        } else {
            Text("No preview available")
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var footer: some View {
        HStack(spacing: 10) {
            Button("Refresh Preview") {
                loadPreview()
            }

            Spacer()

            Button("Cancel", role: .cancel) {
                dismiss()
            }

            Button("Import Selected (\(selectedItemIDs.count))") {
                importSelected()
            }
            .keyboardShortcut(.defaultAction)
            .disabled(selectedItemIDs.isEmpty)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private func summaryView(_ summary: RimeVocabularyPreviewSummary) -> some View {
        HStack(spacing: 10) {
            RimePreviewMetric(title: "Total", value: summary.totalCount)
            RimePreviewMetric(title: "New", value: summary.newCount)
            RimePreviewMetric(title: "Existing", value: summary.existingCount)
            RimePreviewMetric(title: "Skipped", value: summary.skippedCount)
            RimePreviewMetric(title: "Review", value: summary.reviewOnlyCount)
        }
    }

    private func loadPreview() {
        isLoading = true
        let loadedPreview = service.makePreview(context: modelContext)
        preview = loadedPreview
        selectedItemIDs = Set(loadedPreview.items.filter(\.isImportable).map(\.id))
        isLoading = false
    }

    private func importSelected() {
        guard let preview else { return }

        let selectedItems = preview.items.filter { selectedItemIDs.contains($0.id) }

        do {
            let result = try service.importSelectedItems(selectedItems, context: modelContext)
            loadPreview()
            alertTitle = "RIME Import Complete"
            alertMessage = "Added \(result.insertedVocabularyCount) vocabulary words and \(result.insertedProtectedTermCount) protected terms."
            showAlert = true
        } catch {
            alertTitle = "RIME Import Failed"
            alertMessage = error.localizedDescription
            showAlert = true
        }
    }
}

private struct RimePreviewMetric: View {
    let title: String
    let value: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
            Text("\(value)")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(.primary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background {
            RoundedRectangle(cornerRadius: 8)
                .fill(AppTheme.Surface.window.opacity(0.5))
        }
    }
}

private struct RimeVocabularyPreviewRow: View {
    let item: RimeVocabularyPreviewItem
    @Binding var isSelected: Bool

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            if item.isImportable {
                Toggle("", isOn: $isSelected)
                    .toggleStyle(.checkbox)
                    .labelsHidden()
                    .padding(.top, 2)
            } else {
                Image(systemName: item.isSkipped ? "minus.circle" : "info.circle")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(item.isSkipped ? AppTheme.Status.warning : .secondary)
                    .frame(width: 16, height: 16)
                    .padding(.top, 3)
            }

            VStack(alignment: .leading, spacing: 5) {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(item.candidate.term)
                        .font(.system(size: 14, weight: .semibold))
                        .lineLimit(2)

                    Text(item.candidate.categoryGuess.label)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background {
                            RoundedRectangle(cornerRadius: 4)
                                .fill(AppTheme.Surface.window.opacity(0.6))
                        }
                }

                Text(metadataText)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)

                HStack(spacing: 8) {
                    Text(item.destinationLabel)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.primary)

                    Text(item.statusLabel)
                        .font(.system(size: 12))
                        .foregroundStyle(item.isSkipped ? AppTheme.Status.warning : .secondary)
                }
            }

            Spacer(minLength: 0)
        }
        .padding(10)
        .background {
            RoundedRectangle(cornerRadius: 8)
                .fill(AppTheme.Surface.card)
        }
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(AppTheme.Border.subtle, lineWidth: 1)
        }
    }

    private var metadataText: String {
        let section = item.candidate.section ?? "No section"
        let weight = item.candidate.weight.map { " · weight \($0)" } ?? ""
        return "\(item.candidate.sourceFile) · \(section) · \(item.candidate.code)\(weight)"
    }
}

private struct RimeUserDBPolicyView: View {
    let policy: RimeLearnedUserDBImportPolicy

    var body: some View {
        DisclosureGroup("userdb learned phrase policy") {
            VStack(alignment: .leading, spacing: 6) {
                Text(policy.frequencyThresholdRule)
                Text(policy.uncommonTermDetectionRule)
                Text(policy.technicalTermPriorityRule)
                Text(policy.personNameRule)
                Text(policy.previewOnlyRule)
                Text(policy.lowConfidenceRule)
            }
            .font(.system(size: 12))
            .foregroundStyle(.secondary)
            .padding(.top, 6)
        }
        .font(.system(size: 12, weight: .medium))
    }
}

struct VocabularyWordView: View {
    let item: VocabularyWord
    let onDelete: () -> Void
    @State private var isDeleteHovered = false

    var body: some View {
        HStack(spacing: 6) {
            Text(item.word)
                .font(.system(size: 13))
                .lineLimit(1)
                .foregroundColor(.primary)

            Button(action: onDelete) {
                Image(systemName: "xmark.circle.fill")
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(isDeleteHovered ? AppTheme.Status.error : .secondary)
                    .contentTransition(.symbolEffect(.replace))
            }
            .buttonStyle(.borderless)
            .help("Remove word")
            .onHover { hover in
                withAnimation(.easeInOut(duration: 0.2)) {
                    isDeleteHovered = hover
                }
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background {
            RoundedRectangle(cornerRadius: 6)
                .fill(AppTheme.Surface.window.opacity(0.4))
        }
        .overlay {
            RoundedRectangle(cornerRadius: 6)
                .stroke(AppTheme.Border.subtle, lineWidth: 1)
        }
        .shadow(color: Color.black.opacity(0.05), radius: 2, y: 1)
    }
} 
