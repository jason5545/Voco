import AppKit
import Foundation

@MainActor
struct EmailSupport {
    static func generateSupportEmailBody() -> String {
        VocoSupport.generateSupportEmailBody()
    }

    static func generateSupportEmailURL() -> URL? {
        VocoSupport.generateSupportEmailURL()
    }

    static func openSupportEmail() {
        VocoSupport.openSupportEmail()
    }
}
