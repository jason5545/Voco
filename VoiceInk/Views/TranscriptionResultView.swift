import SwiftUI

enum TranscriptionTab: CaseIterable, Hashable {
    case original
    case enhanced

    var localizedName: LocalizedStringKey {
        switch self {
        case .original: "Original"
        case .enhanced: "Enhanced"
        }
    }
}
