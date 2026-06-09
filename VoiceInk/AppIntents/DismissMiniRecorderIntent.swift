import AppIntents
import Foundation
import AppKit

struct DismissMiniRecorderIntent: AppIntent {
    static var title: LocalizedStringResource = "Dismiss Voco Recorder"
    static var description = IntentDescription("Dismiss the Voco recorder and cancel any active recording.")
    
    static var openAppWhenRun: Bool = false
    
    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        NotificationCenter.default.post(name: .dismissRecorderPanel, object: nil)
        
        let dialog = IntentDialog(stringLiteral: "Voco recorder dismissed")
        return .result(dialog: dialog)
    }
}
