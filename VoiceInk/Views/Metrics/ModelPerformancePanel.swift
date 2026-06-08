import SwiftUI
import SwiftData

// MARK: - Time filter

enum TimeFilter: String, CaseIterable, Identifiable {
    case last7Days  = "Last 7 Days"
    case last30Days = "Last 30 Days"
    case thisYear   = "This Year"
    case allTime    = "All Time"

    var id: String { rawValue }

    var localizedTitle: String {
        String(localized: String.LocalizationValue(rawValue))
    }

    var predicate: Predicate<SessionMetric>? {
        let now = Date()
        switch self {
        case .allTime:
            return nil
        case .last7Days:
            let start = now.addingTimeInterval(-7 * 24 * 3600)
            return #Predicate<SessionMetric> { $0.timestamp >= start }
        case .last30Days:
            let start = now.addingTimeInterval(-30 * 24 * 3600)
            return #Predicate<SessionMetric> { $0.timestamp >= start }
        case .thisYear:
            guard let start = Calendar.current.dateInterval(of: .year, for: now)?.start else { return nil }
            return #Predicate<SessionMetric> { $0.timestamp >= start }
        }
    }
}

// MARK: - Panel shell (owns filter state)

struct ModelPerformancePanel: View {
    @AppStorage("modelPerfPanelFilter") private var filterRaw: String = TimeFilter.last7Days.rawValue
    let onClose: () -> Void

    private var filter: TimeFilter { TimeFilter(rawValue: filterRaw) ?? .last7Days }

    var body: some View {
        VStack(spacing: 0) {
            header
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .background(Color(NSColor.windowBackgroundColor))
                .overlay(Divider().opacity(0.5), alignment: .bottom)
                .zIndex(1)

            ModelPerformancePanelContent(filter: filter)
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Text("Model Performance")
                .font(.headline.weight(.semibold))
            Spacer()
            Picker("", selection: Binding(get: { filter }, set: { filterRaw = $0.rawValue })) {
                ForEach(TimeFilter.allCases) { f in
                    Text(f.localizedTitle).tag(f)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .fixedSize()
            Button(action: onClose) {
                Image(systemName: "xmark")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(.secondary)
                    .padding(6)
                    .background(Color.secondary.opacity(0.1))
                    .clipShape(Circle())
            }
            .buttonStyle(.plain)
        }
    }
}

// MARK: - Content (owns @Query, reacts to filter)

private struct ModelPerformancePanelContent: View {
    @AppStorage("modelPerfPanelShowsDiagnostics") private var showsDiagnostics = false
    @Query private var metrics: [SessionMetric]

    init(filter: TimeFilter) {
        if let predicate = filter.predicate {
            _metrics = Query(filter: predicate)
        } else {
            _metrics = Query()
        }
    }

    private var modelStats: [ModelPerformanceStat] {
        var accumulators: [String: ModelPerformanceAccumulator] = [:]
        for metric in metrics {
            guard let name = metric.transcriptionModelName,
                  let processingDuration = metric.transcriptionDuration,
                  processingDuration > 0 else { continue }
            accumulators[name, default: ModelPerformanceAccumulator()].add(
                audioDuration: metric.audioDuration,
                processingDuration: processingDuration
            )
        }
        return accumulators.map { name, acc in acc.stat(named: name) }
            .sorted { $0.avgProcessingTime < $1.avgProcessingTime }
    }

    private var enhancementStats: [EnhancementStat] {
        var accumulators: [String: EnhancementAccumulator] = [:]
        for metric in metrics {
            guard let name = metric.aiEnhancementModelName,
                  let duration = metric.enhancementDuration,
                  duration > 0 else { continue }
            accumulators[name, default: EnhancementAccumulator()].add(duration: duration)
        }
        return accumulators.map { name, acc in acc.stat(named: name) }
            .sorted { $0.avgDuration < $1.avgDuration }
    }

    private var assistiveSummary: AssistiveSignalSummary {
        AssistiveSignalSummary(metrics: metrics)
    }

    private let gridColumns = [
        GridItem(.flexible(), spacing: 12),
        GridItem(.flexible(), spacing: 12)
    ]

    var body: some View {
        if modelStats.isEmpty && enhancementStats.isEmpty && !assistiveSummary.hasData {
            emptyState
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    if !modelStats.isEmpty {
                        modelsSection
                    }
                    if !enhancementStats.isEmpty {
                        enhancementSection
                    }
                    if assistiveSummary.hasData {
                        assistiveSection
                    }
                }
                .padding(16)
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "chart.bar.xaxis")
                .font(.system(size: 32, weight: .light))
                .foregroundColor(.secondary)
            Text("No data for this period")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Assistive signals

    private var assistiveSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            if assistiveSummary.hasPrimaryData {
                primaryAssistiveSection
            }
            if assistiveSummary.hasDiagnosticData {
                diagnosticAssistiveSection
            }
        }
    }

    private var primaryAssistiveSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader("Key Signals")
            LazyVGrid(columns: gridColumns, spacing: 12) {
                if assistiveSummary.confidenceRouteSampleCount > 0 {
                    assistiveTile(
                        icon: "checkmark.shield",
                        title: "Direct Insertions",
                        value: formatPercent(assistiveSummary.directInsertionRate),
                        detail: String(localized: "\(assistiveSummary.directInsertionCount) of \(assistiveSummary.confidenceRouteSampleCount) routed sessions"),
                        color: .mint
                    )

                    assistiveTile(
                        icon: "exclamationmark.bubble",
                        title: "Review Suggested",
                        value: formatPercent(assistiveSummary.reviewSuggestedRate),
                        detail: String(localized: "\(assistiveSummary.reviewSuggestedCount) of \(assistiveSummary.confidenceRouteSampleCount) routed sessions"),
                        color: .orange
                    )
                }

                if assistiveSummary.confidenceScoreSampleCount > 0 {
                    assistiveTile(
                        icon: "slider.horizontal.3",
                        title: "Average Confidence",
                        value: formatPercent(assistiveSummary.averageConfidenceScore),
                        detail: String(localized: "\(assistiveSummary.confidenceScoreSampleCount) scored sessions"),
                        color: .teal
                    )
                }

                if assistiveSummary.retranscriptionSampleCount > 0 {
                    assistiveTile(
                        icon: "arrow.triangle.2.circlepath",
                        title: "Retranscriptions",
                        value: formatPercent(assistiveSummary.meaningfulRetranscriptionRate),
                        detail: assistiveSummary.retranscriptionDetail,
                        color: .pink
                    )
                }
            }
        }
    }

    private var diagnosticAssistiveSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Button {
                withAnimation(.easeInOut(duration: 0.18)) {
                    showsDiagnostics.toggle()
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: showsDiagnostics ? "chevron.down" : "chevron.right")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.secondary)
                        .frame(width: 14, height: 14)
                    sectionHeader("Advanced Diagnostics")
                    Spacer()
                    Text("\(assistiveSummary.diagnosticTileCount)")
                        .font(.system(size: 10, weight: .semibold, design: .rounded))
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 3)
                        .background(Color.secondary.opacity(0.12))
                        .clipShape(Capsule())
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if showsDiagnostics {
                diagnosticTiles
            }
        }
    }

    private var diagnosticTiles: some View {
        LazyVGrid(columns: gridColumns, spacing: 12) {
            if assistiveSummary.reviewTriggerCount > 0 {
                assistiveTile(
                    icon: "list.bullet.clipboard",
                    title: "Review Triggers",
                    value: "\(assistiveSummary.reviewTriggerCount)",
                    detail: assistiveSummary.reviewTriggerDetail,
                    color: .red
                )
            }

            if assistiveSummary.candidateSelectionCount > 0 {
                assistiveTile(
                    icon: "cursorarrow.click.2",
                    title: "Candidate Selections",
                    value: "\(assistiveSummary.candidateSelectionCount)",
                    detail: String(localized: "\(assistiveSummary.userSelectionCount) user / \(assistiveSummary.fallbackSelectionCount) fallback"),
                    color: .indigo
                )
            }

            if assistiveSummary.correctionFeedbackSessionCount > 0 {
                assistiveTile(
                    icon: "text.badge.checkmark",
                    title: "Feedback",
                    value: "\(assistiveSummary.correctionFeedbackCount)",
                    detail: assistiveSummary.correctionFeedbackDetail,
                    color: .brown
                )
            }

            if assistiveSummary.styleGuardRejectionSessionCount > 0 {
                assistiveTile(
                    icon: "shield.lefthalf.filled",
                    title: "Style Guard",
                    value: "\(assistiveSummary.styleGuardRejectionSessionCount)",
                    detail: assistiveSummary.styleGuardDetail,
                    color: .green
                )
            }

            if assistiveSummary.candidateSourceSampleCount > 0 {
                assistiveTile(
                    icon: "rectangle.stack",
                    title: "Candidate Sources",
                    value: "\(assistiveSummary.candidateSourceCandidateCount)",
                    detail: assistiveSummary.candidateSourceDetail,
                    color: .cyan
                )
            }

            if assistiveSummary.candidateDivergenceRatioSampleCount > 0 {
                assistiveTile(
                    icon: "arrow.left.and.right",
                    title: "Candidate Divergence",
                    value: formatPercent(assistiveSummary.averageCandidateDivergenceRatio),
                    detail: String(localized: "\(assistiveSummary.candidateDivergenceRatioSampleCount) compared sessions"),
                    color: .blue
                )
            }

            if assistiveSummary.canonicalizedSessionCount > 0 || assistiveSummary.suggestedSessionCount > 0 {
                assistiveTile(
                    icon: "text.magnifyingglass",
                    title: "Canonicalization",
                    value: "\(assistiveSummary.canonicalizedSessionCount)",
                    detail: String(localized: "\(assistiveSummary.totalCanonicalizationReplacementCount) replacements / \(assistiveSummary.totalCanonicalizationSuggestionCount) suggestions"),
                    color: .purple
                )
            }

            if assistiveSummary.pasteCommandSampleCount > 0 {
                assistiveTile(
                    icon: "doc.on.clipboard",
                    title: "Paste Commands",
                    value: formatPercent(assistiveSummary.pasteCommandPostedRate),
                    detail: String(localized: "\(assistiveSummary.pasteCommandPostedCount) of \(assistiveSummary.pasteCommandSampleCount) recorded"),
                    color: .blue
                )
            }
        }
    }

    private func assistiveTile(
        icon: String,
        title: LocalizedStringKey,
        value: String,
        detail: String,
        color: Color
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(color)
                    .frame(width: 18, height: 18)

                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
            }

            Text(value)
                .font(.system(size: 24, weight: .bold, design: .rounded))
                .foregroundColor(color)
                .lineLimit(1)
                .minimumScaleFactor(0.65)

            Text(detail)
                .font(.system(size: 10))
                .foregroundColor(.secondary)
                .lineLimit(2)
                .minimumScaleFactor(0.75)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(MetricCardBackground(color: color))
        .cornerRadius(12)
    }

    // MARK: - Models grid

    private var modelsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader("Transcription Models")
            LazyVGrid(columns: gridColumns, spacing: 12) {
                ForEach(modelStats) { stat in
                    modelTile(stat)
                }
            }
        }
    }

    private func modelTile(_ stat: ModelPerformanceStat) -> some View {
        VStack(spacing: 10) {
            VStack(spacing: 2) {
                Text(stat.name)
                    .font(.system(size: 12, weight: .semibold))
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                Text(String(localized: "\(stat.sessionCount) sessions"))
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity)

            VStack(spacing: 3) {
                Text(String(format: "%.1fx", stat.speedFactor))
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                    .foregroundColor(.mint)
                Text(String(localized: String.LocalizationValue(stat.speedFactor >= 1.0 ? "Faster than Real-time" : "Slower than Real-time")))
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
            }

            Divider().padding(.horizontal, 8)

            HStack(spacing: 0) {
                VStack(spacing: 2) {
                    Text(formatDuration(stat.avgAudioDuration))
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .foregroundColor(.indigo)
                    Text("Avg. Audio")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity)

                Rectangle()
                    .fill(Color(NSColor.separatorColor))
                    .frame(width: 1, height: 24)

                VStack(spacing: 2) {
                    Text(String(format: "%.2fs", stat.avgProcessingTime))
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .foregroundColor(.teal)
                    Text("Avg. Processing")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity)
            }
        }
        .padding(14)
        .background(MetricCardBackground(color: .mint))
        .cornerRadius(12)
    }

    // MARK: - Enhancement Models

    private var enhancementSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader("Enhancement Models")
            LazyVGrid(columns: gridColumns, spacing: 12) {
                ForEach(enhancementStats) { stat in
                    enhancementTile(stat)
                }
            }
        }
    }

    private func enhancementTile(_ stat: EnhancementStat) -> some View {
        VStack(spacing: 10) {
            VStack(spacing: 2) {
                Text(stat.name)
                    .font(.system(size: 12, weight: .semibold))
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                Text(String(localized: "\(stat.sessionCount) sessions"))
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity)

            VStack(spacing: 3) {
                Text(String(format: "%.2fs", stat.avgDuration))
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                    .foregroundColor(.indigo)
                Text("Avg. Enhancement Time")
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
            }
        }
        .padding(14)
        .background(MetricCardBackground(color: .indigo))
        .cornerRadius(12)
    }

    // MARK: - Helpers

    private func sectionHeader(_ title: LocalizedStringKey) -> some View {
        Text(title)
            .font(.system(size: 12, weight: .semibold))
            .foregroundColor(.secondary)
            .textCase(.uppercase)
            .tracking(0.5)
    }

    private func formatDuration(_ duration: TimeInterval) -> String {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.minute, .second]
        formatter.unitsStyle = .abbreviated
        return formatter.string(from: duration) ?? "0s"
    }

    private func formatPercent(_ value: Double?) -> String {
        guard let value else { return "-" }
        return "\(Int((value * 100).rounded()))%"
    }
}

// MARK: - Data models

struct AssistiveSignalSummary: Equatable {
    let sessionCount: Int
    let confidenceRouteSampleCount: Int
    let directInsertionCount: Int
    let reviewSuggestedCount: Int
    let reviewTriggerSessionCount: Int
    let reviewTriggerCount: Int
    let reviewTriggerCounts: [String: Int]
    let reviewTriggerSummaryCounts: [String: Int]
    let confidenceScoreSampleCount: Int
    let averageConfidenceScore: Double?
    let candidateSelectionCount: Int
    let userSelectionCount: Int
    let dismissedFallbackCount: Int
    let timeoutFallbackCount: Int
    let automaticFallbackCount: Int
    let correctionFeedbackSessionCount: Int
    let correctionFeedbackCount: Int
    let correctiveFeedbackCount: Int
    let correctionFeedbackReasonCounts: [String: Int]
    let styleGuardRejectionSessionCount: Int
    let styleGuardReasonCount: Int
    let styleGuardReasonCounts: [String: Int]
    let styleGuardRejectedCharacterCount: Int
    let candidateSourceSampleCount: Int
    let candidateSourceCandidateCount: Int
    let candidateSourceCounts: [String: Int]
    let reviewRequiredCandidateCount: Int
    let candidateDivergenceRatioSampleCount: Int
    let averageCandidateDivergenceRatio: Double?
    let selectedCandidateSourceCounts: [String: Int]
    let canonicalizedSessionCount: Int
    let suggestedSessionCount: Int
    let totalCanonicalizationReplacementCount: Int
    let totalCanonicalizationSuggestionCount: Int
    let retranscriptionSampleCount: Int
    let unchangedRetranscriptionCount: Int
    let minorRetranscriptionCount: Int
    let meaningfulRetranscriptionCount: Int
    let retranscriptionChangeRatioSampleCount: Int
    let averageRetranscriptionChangeRatio: Double?
    let retranscriptionConfidenceDeltaSampleCount: Int
    let averageRetranscriptionConfidenceDelta: Double?
    let pasteCommandSampleCount: Int
    let pasteCommandPostedCount: Int

    init(metrics: [SessionMetric]) {
        sessionCount = metrics.count
        confidenceRouteSampleCount = metrics.filter { $0.confidenceRoute != nil }.count
        directInsertionCount = metrics.filter { $0.confidenceRoute == VocoConfidenceRoute.directInsertion.rawValue }.count
        reviewSuggestedCount = metrics.filter { $0.confidenceRoute == VocoConfidenceRoute.reviewSuggested.rawValue }.count

        let reviewTriggerBreakdowns = metrics.map(\.reviewTriggerIDs).filter { !$0.isEmpty }
        reviewTriggerSessionCount = metrics.filter {
            $0.reviewTriggerCount > 0 ||
            !$0.reviewTriggerIDs.isEmpty ||
            !$0.reviewTriggerSummaries.isEmpty
        }.count
        reviewTriggerCounts = Self.mergedCounts(
            reviewTriggerBreakdowns.map { ids in
                ids.reduce(into: [:]) { counts, id in
                    counts[id, default: 0] += 1
                }
            }
        )
        reviewTriggerSummaryCounts = Self.mergedCounts(
            metrics
                .map(Self.reviewTriggerDisplayItems)
                .filter { !$0.isEmpty }
                .map { summaries in
                    summaries.reduce(into: [:]) { counts, summary in
                        counts[summary, default: 0] += 1
                    }
                }
        )
        let storedReviewTriggerCount = metrics.reduce(0) { $0 + $1.reviewTriggerCount }
        let idReviewTriggerCount = reviewTriggerCounts.values.reduce(0, +)
        let summaryReviewTriggerCount = reviewTriggerSummaryCounts.values.reduce(0, +)
        reviewTriggerCount = max(storedReviewTriggerCount, max(idReviewTriggerCount, summaryReviewTriggerCount))

        let confidenceScores = metrics.compactMap(\.confidenceScore)
        confidenceScoreSampleCount = confidenceScores.count
        averageConfidenceScore = confidenceScores.isEmpty
            ? nil
            : confidenceScores.reduce(0, +) / Double(confidenceScores.count)

        userSelectionCount = metrics.filter { $0.candidateSelectionSource == VocoCandidateSelectionSource.userSelection.rawValue }.count
        dismissedFallbackCount = metrics.filter { $0.candidateSelectionSource == VocoCandidateSelectionSource.dismissedFallback.rawValue }.count
        timeoutFallbackCount = metrics.filter { $0.candidateSelectionSource == VocoCandidateSelectionSource.timeoutFallback.rawValue }.count
        automaticFallbackCount = metrics.filter { $0.candidateSelectionSource == VocoCandidateSelectionSource.automaticFallback.rawValue }.count
        candidateSelectionCount = userSelectionCount + dismissedFallbackCount + timeoutFallbackCount + automaticFallbackCount

        correctionFeedbackSessionCount = metrics.filter {
            $0.correctionFeedbackCount > 0 || !$0.correctionFeedbackReasons.isEmpty
        }.count
        correctionFeedbackCount = metrics.reduce(0) { $0 + $1.correctionFeedbackCount }
        correctiveFeedbackCount = metrics.reduce(0) { $0 + $1.correctiveFeedbackCount }
        correctionFeedbackReasonCounts = Self.mergedCounts(
            metrics
                .map(\.correctionFeedbackReasons)
                .filter { !$0.isEmpty }
                .map(Self.countsByValue)
        )

        styleGuardRejectionSessionCount = metrics.filter {
            $0.styleGuardReasonCount > 0 ||
                !$0.styleGuardReasons.isEmpty ||
                $0.styleGuardRejectedCharacterCount > 0
        }.count
        styleGuardReasonCount = metrics.reduce(0) { $0 + $1.styleGuardReasonCount }
        styleGuardReasonCounts = Self.mergedCounts(
            metrics
                .map(\.styleGuardReasons)
                .filter { !$0.isEmpty }
                .map(Self.countsByValue)
        )
        styleGuardRejectedCharacterCount = metrics.reduce(0) { $0 + $1.styleGuardRejectedCharacterCount }

        let sourceBreakdowns = metrics.map(\.candidateSourceCounts).filter { !$0.isEmpty }
        candidateSourceSampleCount = sourceBreakdowns.count
        candidateSourceCounts = Self.mergedCounts(sourceBreakdowns)
        candidateSourceCandidateCount = candidateSourceCounts.values.reduce(0, +)
        reviewRequiredCandidateCount = metrics.reduce(0) { $0 + $1.reviewRequiredCandidateCount }
        let candidateDivergenceRatios = metrics.compactMap(\.candidateDivergenceRatio)
        candidateDivergenceRatioSampleCount = candidateDivergenceRatios.count
        averageCandidateDivergenceRatio = candidateDivergenceRatios.isEmpty
            ? nil
            : candidateDivergenceRatios.reduce(0, +) / Double(candidateDivergenceRatios.count)
        selectedCandidateSourceCounts = Self.mergedCounts(
            metrics.compactMap(\.selectedCandidateHypothesisSource).map { [$0: 1] }
        )

        canonicalizedSessionCount = metrics.filter { $0.canonicalizationReplacementCount > 0 }.count
        suggestedSessionCount = metrics.filter { $0.canonicalizationSuggestionCount > 0 }.count
        totalCanonicalizationReplacementCount = metrics.reduce(0) { $0 + $1.canonicalizationReplacementCount }
        totalCanonicalizationSuggestionCount = metrics.reduce(0) { $0 + $1.canonicalizationSuggestionCount }

        retranscriptionSampleCount = metrics.filter { ($0.retranscriptionChangeCategory ?? "").isEmpty == false }.count
        unchangedRetranscriptionCount = metrics.filter { $0.retranscriptionChangeCategory == RetranscriptionChangeCategory.unchanged.rawValue }.count
        minorRetranscriptionCount = metrics.filter { $0.retranscriptionChangeCategory == RetranscriptionChangeCategory.minorChange.rawValue }.count
        meaningfulRetranscriptionCount = metrics.filter { $0.retranscriptionChangeCategory == RetranscriptionChangeCategory.meaningfulChange.rawValue }.count

        let retranscriptionChangeRatios = metrics.compactMap(\.retranscriptionChangeRatio)
        retranscriptionChangeRatioSampleCount = retranscriptionChangeRatios.count
        averageRetranscriptionChangeRatio = retranscriptionChangeRatios.isEmpty
            ? nil
            : retranscriptionChangeRatios.reduce(0, +) / Double(retranscriptionChangeRatios.count)

        let retranscriptionConfidenceDeltas = metrics.compactMap(\.retranscriptionConfidenceDelta)
        retranscriptionConfidenceDeltaSampleCount = retranscriptionConfidenceDeltas.count
        averageRetranscriptionConfidenceDelta = retranscriptionConfidenceDeltas.isEmpty
            ? nil
            : retranscriptionConfidenceDeltas.reduce(0, +) / Double(retranscriptionConfidenceDeltas.count)

        pasteCommandSampleCount = metrics.filter { $0.pasteCommandPosted != nil }.count
        pasteCommandPostedCount = metrics.filter { $0.pasteCommandPosted == true }.count
    }

    var hasData: Bool {
        hasPrimaryData || hasDiagnosticData
    }

    var hasPrimaryData: Bool {
        confidenceRouteSampleCount > 0 ||
            confidenceScoreSampleCount > 0 ||
            retranscriptionSampleCount > 0
    }

    var hasDiagnosticData: Bool {
        diagnosticTileCount > 0
    }

    var diagnosticTileCount: Int {
        var count = 0
        if reviewTriggerCount > 0 { count += 1 }
        if candidateSelectionCount > 0 { count += 1 }
        if correctionFeedbackSessionCount > 0 { count += 1 }
        if styleGuardRejectionSessionCount > 0 { count += 1 }
        if candidateSourceSampleCount > 0 { count += 1 }
        if candidateDivergenceRatioSampleCount > 0 { count += 1 }
        if canonicalizedSessionCount > 0 || suggestedSessionCount > 0 { count += 1 }
        if pasteCommandSampleCount > 0 { count += 1 }
        return count
    }

    var fallbackSelectionCount: Int {
        dismissedFallbackCount + timeoutFallbackCount + automaticFallbackCount
    }

    var correctionFeedbackDetail: String {
        guard correctionFeedbackSessionCount > 0 || correctionFeedbackCount > 0 else {
            return String(localized: "No feedback recorded")
        }

        var parts = [
            String(localized: "\(correctiveFeedbackCount) corrective / \(Self.sessionCount(correctionFeedbackSessionCount))"),
        ]
        let reasonText = Self.signalReasonSummary(correctionFeedbackReasonCounts, limit: 2)
        if !reasonText.isEmpty {
            parts.append(reasonText)
        }
        return parts.joined(separator: " / ")
    }

    var styleGuardDetail: String {
        guard styleGuardRejectionSessionCount > 0 else {
            return String(localized: "No style rejections")
        }

        var parts = [
            Self.sessionCount(styleGuardRejectionSessionCount),
        ]
        if styleGuardRejectedCharacterCount > 0 {
            parts.append(Self.rejectedCharacterCount(styleGuardRejectedCharacterCount))
        }

        let reasonText = Self.styleGuardReasonSummary(styleGuardReasonCounts, limit: 2)
        if !reasonText.isEmpty {
            parts.append(reasonText)
        }
        return parts.joined(separator: " / ")
    }

    var reviewTriggerDetail: String {
        guard reviewTriggerCount > 0 else {
            return String(localized: "No review triggers")
        }

        let sessionText = Self.sessionCount(reviewTriggerSessionCount)
        let triggerText = Self.reviewTriggerSummary(
            reviewTriggerSummaryCounts,
            fallbackCounts: reviewTriggerCounts,
            limit: 3
        )
        guard !triggerText.isEmpty else {
            return String(localized: "\(reviewTriggerCount) recorded")
        }
        return "\(sessionText) / \(triggerText)"
    }

    var candidateSourceDetail: String {
        guard candidateSourceCandidateCount > 0 else {
            return String(localized: "No source breakdown")
        }

        let reviewText = String(localized: "\(reviewRequiredCandidateCount) review")
        let sourceText = Self.sourceSummary(candidateSourceCounts, limit: 3)
        var parts = [reviewText]
        if !sourceText.isEmpty {
            parts.append(sourceText)
        }
        if let averageCandidateDivergenceRatio {
            parts.append(String(localized: "avg delta \(Self.percent(averageCandidateDivergenceRatio))"))
        }
        return parts.joined(separator: " / ")
    }

    var directInsertionRate: Double? {
        rate(directInsertionCount, confidenceRouteSampleCount)
    }

    var reviewSuggestedRate: Double? {
        rate(reviewSuggestedCount, confidenceRouteSampleCount)
    }

    var meaningfulRetranscriptionRate: Double? {
        rate(meaningfulRetranscriptionCount, retranscriptionSampleCount)
    }

    var retranscriptionDetail: String {
        var parts = [
            String(localized: "\(meaningfulRetranscriptionCount) meaningful / \(retranscriptionSampleCount) analyzed")
        ]

        if let averageRetranscriptionChangeRatio {
            parts.append(String(localized: "avg change \(Self.percent(averageRetranscriptionChangeRatio))"))
        }

        if let averageRetranscriptionConfidenceDelta {
            parts.append(String(localized: "avg confidence \(Self.signedPercent(averageRetranscriptionConfidenceDelta))"))
        }

        return parts.joined(separator: ", ")
    }

    var pasteCommandPostedRate: Double? {
        rate(pasteCommandPostedCount, pasteCommandSampleCount)
    }

    private func rate(_ numerator: Int, _ denominator: Int) -> Double? {
        guard denominator > 0 else { return nil }
        return Double(numerator) / Double(denominator)
    }

    private static func percent(_ value: Double) -> String {
        "\(Int((value * 100).rounded()))%"
    }

    private static func signedPercent(_ value: Double) -> String {
        let sign = value >= 0 ? "+" : ""
        return "\(sign)\(percent(value))"
    }

    private static func sessionCount(_ count: Int) -> String {
        if count == 1 {
            return String(localized: "\(count) session")
        }
        return String(localized: "\(count) sessions")
    }

    private static func rejectedCharacterCount(_ count: Int) -> String {
        if count == 1 {
            return String(localized: "\(count) char rejected")
        }
        return String(localized: "\(count) chars rejected")
    }

    private static func mergedCounts(_ counts: [[String: Int]]) -> [String: Int] {
        counts.reduce(into: [:]) { merged, next in
            for (key, value) in next {
                merged[key, default: 0] += value
            }
        }
    }

    private static func countsByValue(_ values: [String]) -> [String: Int] {
        values.reduce(into: [:]) { counts, value in
            counts[value, default: 0] += 1
        }
    }

    private static func signalReasonSummary(_ counts: [String: Int], limit: Int) -> String {
        sortedReasonCounts(counts, displayName: VocoSignalDisplayFormatter.displayReason(for:), limit: limit)
            .map { "\(VocoSignalDisplayFormatter.displayReason(for: $0.key)) \($0.value)" }
            .joined(separator: ", ")
    }

    private static func styleGuardReasonSummary(_ counts: [String: Int], limit: Int) -> String {
        let categoryCounts = counts.reduce(into: [String: Int]()) { totals, entry in
            let category = VocoSignalDisplayFormatter.displayStyleGuardReasonCategory(for: entry.key)
            totals[category, default: 0] += entry.value
        }

        return sortedReasonCounts(categoryCounts, displayName: { $0 }, limit: limit)
            .map { "\($0.key) \($0.value)" }
            .joined(separator: ", ")
    }

    private static func sortedReasonCounts(
        _ counts: [String: Int],
        displayName: (String) -> String,
        limit: Int
    ) -> [(key: String, value: Int)] {
        counts
            .filter { $0.value > 0 }
            .sorted { lhs, rhs in
                if lhs.value != rhs.value {
                    return lhs.value > rhs.value
                }
                return displayName(lhs.key) < displayName(rhs.key)
            }
            .prefix(limit)
            .map { $0 }
    }

    private static func sourceSummary(_ counts: [String: Int], limit: Int) -> String {
        counts
            .filter { $0.value > 0 }
            .sorted { lhs, rhs in
                if lhs.value != rhs.value {
                    return lhs.value > rhs.value
                }
                let lhsOrder = sourceSortOrder(lhs.key)
                let rhsOrder = sourceSortOrder(rhs.key)
                if lhsOrder != rhsOrder {
                    return lhsOrder < rhsOrder
                }
                return sourceDisplayName(lhs.key) < sourceDisplayName(rhs.key)
            }
            .prefix(limit)
            .map { "\(sourceDisplayName($0.key)) \($0.value)" }
            .joined(separator: ", ")
    }

    private static func reviewTriggerSummary(
        _ summaryCounts: [String: Int],
        fallbackCounts: [String: Int],
        limit: Int
    ) -> String {
        if !summaryCounts.isEmpty {
            return sortedReviewTriggerSummaryCounts(summaryCounts, limit: limit)
                .map { "\($0.key) \($0.value)" }
                .joined(separator: ", ")
        }

        return sortedReviewTriggerIDCounts(fallbackCounts, limit: limit)
            .map { "\(reviewTriggerDisplayName($0.key)) \($0.value)" }
            .joined(separator: ", ")
    }

    private static func sortedReviewTriggerSummaryCounts(_ counts: [String: Int], limit: Int) -> [(key: String, value: Int)] {
        counts
            .filter { $0.value > 0 }
            .sorted { lhs, rhs in
                if lhs.value != rhs.value {
                    return lhs.value > rhs.value
                }
                return lhs.key < rhs.key
            }
            .prefix(limit)
            .map { $0 }
    }

    private static func sortedReviewTriggerIDCounts(_ counts: [String: Int], limit: Int) -> [(key: String, value: Int)] {
        counts
            .filter { $0.value > 0 }
            .sorted { lhs, rhs in
                if lhs.value != rhs.value {
                    return lhs.value > rhs.value
                }
                return reviewTriggerDisplayName(lhs.key) < reviewTriggerDisplayName(rhs.key)
            }
            .prefix(limit)
            .map { $0 }
    }

    private static func reviewTriggerDisplayItems(for metric: SessionMetric) -> [String] {
        if !metric.reviewTriggerSummaries.isEmpty {
            return metric.reviewTriggerSummaries
        }

        return metric.reviewTriggerIDs.map(reviewTriggerDisplayName)
    }

    private static func reviewTriggerDisplayName(_ id: String) -> String {
        VocoSignalDisplayFormatter.displayReason(for: id)
    }

    private static func sourceDisplayName(_ source: String) -> String {
        VocoHypothesisSource(rawValue: source)?.displayName ?? source
    }

    private static func sourceSortOrder(_ source: String) -> Int {
        VocoHypothesisSource(rawValue: source)?.analyticsSortPriority ?? 99
    }
}

struct ModelPerformanceStat: Identifiable {
    var id: String { name }
    let name: String
    let sessionCount: Int
    let totalProcessingTime: TimeInterval
    let avgProcessingTime: TimeInterval
    let avgAudioDuration: TimeInterval
    let speedFactor: Double
}

struct ModelPerformanceAccumulator {
    var sessionCount = 0
    var totalProcessingTime: TimeInterval = 0
    var totalAudioDuration: TimeInterval = 0

    mutating func add(audioDuration: TimeInterval, processingDuration: TimeInterval) {
        sessionCount += 1
        totalProcessingTime += processingDuration
        totalAudioDuration += audioDuration
    }

    func stat(named name: String) -> ModelPerformanceStat {
        let safeCount = max(sessionCount, 1)
        let speedFactor = totalProcessingTime > 0 ? totalAudioDuration / totalProcessingTime : 0
        return ModelPerformanceStat(
            name: name,
            sessionCount: sessionCount,
            totalProcessingTime: totalProcessingTime,
            avgProcessingTime: totalProcessingTime / Double(safeCount),
            avgAudioDuration: totalAudioDuration / Double(safeCount),
            speedFactor: speedFactor
        )
    }
}

struct EnhancementStat: Identifiable {
    var id: String { name }
    let name: String
    let sessionCount: Int
    let avgDuration: TimeInterval
}

struct EnhancementAccumulator {
    var sessionCount = 0
    var totalDuration: TimeInterval = 0

    mutating func add(duration: TimeInterval) {
        sessionCount += 1
        totalDuration += duration
    }

    func stat(named name: String) -> EnhancementStat {
        let safeCount = max(sessionCount, 1)
        return EnhancementStat(
            name: name,
            sessionCount: sessionCount,
            avgDuration: totalDuration / Double(safeCount)
        )
    }
}
