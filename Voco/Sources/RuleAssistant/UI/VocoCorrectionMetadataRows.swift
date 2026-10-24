import SwiftUI

struct VocoCorrectionMetadataRows: View {
    let transcription: Transcription
    @ObservedObject private var coverageStore = RowCorrectionCoverageStore.shared

    var body: some View {
        if let coverage = transcription.correctionCoverage {
            metadataRow(
                icon: coverage.badgeIcon,
                label: String(localized: "Rule coverage"),
                value: coverage.label.text
            )
            metadataRow(
                icon: "list.bullet",
                label: String(localized: "Covered policies"),
                value: coverage.policyIds.prefix(3).joined(separator: ", ")
                    + (coverage.policyIds.count > 3 ? " +\(coverage.policyIds.count - 3)" : "")
            )
        }

        if let marking = transcription.correctionMarkingLabel {
            metadataRow(
                icon: "checkmark.seal",
                label: "Correction",
                value: marking.text
            )

            let events = transcription.corrections
            if events.count > 1 {
                let latestId = events.compactMap(\.eventId).last.map { String($0.prefix(12)) } ?? "-"
                metadataRow(
                    icon: "list.bullet.rectangle",
                    label: "Correction events",
                    value: "\(events.count) · \(latestId)"
                )
            }
        }
    }

    private func metadataRow(icon: String, label: String, value: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.secondary)
                .frame(width: 20, height: 20)

            Text(label)
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.secondary)

            Spacer(minLength: 0)

            Text(value)
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(.primary)
                .lineLimit(1)
        }
    }
}
