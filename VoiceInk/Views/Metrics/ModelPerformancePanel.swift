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
                    if assistiveSummary.hasData {
                        assistiveSection
                    }
                    if !modelStats.isEmpty {
                        modelsSection
                    }
                    if !enhancementStats.isEmpty {
                        enhancementSection
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
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader("Assistive Signals")
            LazyVGrid(columns: gridColumns, spacing: 12) {
                assistiveTile(
                    icon: "checkmark.shield",
                    title: "Direct Insertions",
                    value: formatPercent(assistiveSummary.directInsertionRate),
                    detail: "\(assistiveSummary.directInsertionCount) of \(assistiveSummary.confidenceRouteSampleCount) routed sessions",
                    color: .mint
                )

                assistiveTile(
                    icon: "exclamationmark.bubble",
                    title: "Review Suggested",
                    value: formatPercent(assistiveSummary.reviewSuggestedRate),
                    detail: "\(assistiveSummary.reviewSuggestedCount) of \(assistiveSummary.confidenceRouteSampleCount) routed sessions",
                    color: .orange
                )

                assistiveTile(
                    icon: "slider.horizontal.3",
                    title: "Average Confidence",
                    value: formatPercent(assistiveSummary.averageConfidenceScore),
                    detail: "\(assistiveSummary.confidenceScoreSampleCount) scored sessions",
                    color: .teal
                )

                assistiveTile(
                    icon: "cursorarrow.click.2",
                    title: "Candidate Selections",
                    value: "\(assistiveSummary.candidateSelectionCount)",
                    detail: "\(assistiveSummary.userSelectionCount) user / \(assistiveSummary.fallbackSelectionCount) fallback",
                    color: .indigo
                )

                assistiveTile(
                    icon: "rectangle.stack",
                    title: "Candidate Sources",
                    value: "\(assistiveSummary.candidateSourceCandidateCount)",
                    detail: assistiveSummary.candidateSourceDetail,
                    color: .cyan
                )

                assistiveTile(
                    icon: "text.magnifyingglass",
                    title: "Canonicalization",
                    value: "\(assistiveSummary.canonicalizedSessionCount)",
                    detail: "\(assistiveSummary.totalCanonicalizationReplacementCount) replacements / \(assistiveSummary.totalCanonicalizationSuggestionCount) suggestions",
                    color: .purple
                )

                assistiveTile(
                    icon: "arrow.triangle.2.circlepath",
                    title: "Retranscriptions",
                    value: formatPercent(assistiveSummary.meaningfulRetranscriptionRate),
                    detail: assistiveSummary.retranscriptionDetail,
                    color: .pink
                )

                assistiveTile(
                    icon: "doc.on.clipboard",
                    title: "Paste Commands",
                    value: formatPercent(assistiveSummary.pasteCommandPostedRate),
                    detail: "\(assistiveSummary.pasteCommandPostedCount) of \(assistiveSummary.pasteCommandSampleCount) recorded",
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
    let confidenceScoreSampleCount: Int
    let averageConfidenceScore: Double?
    let candidateSelectionCount: Int
    let userSelectionCount: Int
    let dismissedFallbackCount: Int
    let timeoutFallbackCount: Int
    let automaticFallbackCount: Int
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

        let confidenceScores = metrics.compactMap(\.confidenceScore)
        confidenceScoreSampleCount = confidenceScores.count
        averageConfidenceScore = confidenceScores.isEmpty
            ? nil
            : confidenceScores.reduce(0, +) / Double(confidenceScores.count)

        candidateSelectionCount = metrics.filter { ($0.candidateSelectionSource ?? "").isEmpty == false }.count
        userSelectionCount = metrics.filter { $0.candidateSelectionSource == VocoCandidateSelectionSource.userSelection.rawValue }.count
        dismissedFallbackCount = metrics.filter { $0.candidateSelectionSource == VocoCandidateSelectionSource.dismissedFallback.rawValue }.count
        timeoutFallbackCount = metrics.filter { $0.candidateSelectionSource == VocoCandidateSelectionSource.timeoutFallback.rawValue }.count
        automaticFallbackCount = metrics.filter { $0.candidateSelectionSource == VocoCandidateSelectionSource.automaticFallback.rawValue }.count

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
        confidenceRouteSampleCount > 0 ||
        confidenceScoreSampleCount > 0 ||
        candidateSelectionCount > 0 ||
        candidateSourceSampleCount > 0 ||
        candidateDivergenceRatioSampleCount > 0 ||
        canonicalizedSessionCount > 0 ||
        suggestedSessionCount > 0 ||
        retranscriptionSampleCount > 0 ||
        pasteCommandSampleCount > 0
    }

    var fallbackSelectionCount: Int {
        dismissedFallbackCount + timeoutFallbackCount + automaticFallbackCount
    }

    var candidateSourceDetail: String {
        guard candidateSourceCandidateCount > 0 else { return "No source breakdown" }

        let reviewText = "\(reviewRequiredCandidateCount) review"
        let sourceText = Self.sourceSummary(candidateSourceCounts, limit: 3)
        var parts = [reviewText]
        if !sourceText.isEmpty {
            parts.append(sourceText)
        }
        if let averageCandidateDivergenceRatio {
            parts.append("avg delta \(Self.percent(averageCandidateDivergenceRatio))")
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
            "\(meaningfulRetranscriptionCount) meaningful / \(retranscriptionSampleCount) analyzed"
        ]

        if let averageRetranscriptionChangeRatio {
            parts.append("avg change \(Self.percent(averageRetranscriptionChangeRatio))")
        }

        if let averageRetranscriptionConfidenceDelta {
            parts.append("avg confidence \(Self.signedPercent(averageRetranscriptionConfidenceDelta))")
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

    private static func mergedCounts(_ counts: [[String: Int]]) -> [String: Int] {
        counts.reduce(into: [:]) { merged, next in
            for (key, value) in next {
                merged[key, default: 0] += value
            }
        }
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
