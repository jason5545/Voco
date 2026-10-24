import SwiftUI

/// Voco-owned display strings for queued audio imports.
///
/// The upstream `QueueItemStatus.ProcessingPhase` stays untouched; this extension only adds
/// the localized labels the Voco import UI shows.
extension QueueItemStatus.ProcessingPhase {
    var localizedName: LocalizedStringKey {
        switch self {
        case .loading: "Loading model..."
        case .processingAudio: "Processing audio..."
        case .transcribing: "Transcribing..."
        case .enhancing: "Enhancing..."
        }
    }
}
