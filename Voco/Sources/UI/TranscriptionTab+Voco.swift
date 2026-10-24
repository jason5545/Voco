import SwiftUI

extension TranscriptionTab {
    var localizedName: LocalizedStringKey {
        switch self {
        case .original: "Original"
        case .enhanced: "Enhanced"
        }
    }
}
