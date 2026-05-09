import Foundation
import SwiftUI
import AppKit

struct EmailSupport {
    private static let supportEmailAddress = "prakashjoshipax@gmail.com"
    private static let supportEmailSubject = "Voco Support Request"

    static func generateSupportEmailBody() -> String {
        let systemInfo = SystemInfoService.shared.getSystemInfoString()

        let body = """

        ------------------------
        ✨ **SCREEN RECORDING HIGHLY RECOMMENDED** ✨
        ▶️ Create a quick screen recording showing the issue!
        ▶️ It helps me understand and fix the problem much faster.

        📝 ISSUE DETAILS:
        - What steps did you take before the issue occurred?
        - What did you expect to happen?
        - What actually happened instead?


        ## 📋 COMMON ISSUES:
        Check out our Common Issues page before sending an email: https://tryvoiceink.com/common-issues
        ------------------------

        System Information:
        \(systemInfo)


        """
        return body
    }
    
    static func generateSupportEmailURL() -> URL? {
        let body = generateSupportEmailBody()
        let encodedSubject = supportEmailSubject.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
        let encodedBody = body.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
        
        return URL(string: "mailto:\(supportEmailAddress)?subject=\(encodedSubject)&body=\(encodedBody)")
    }
    
    static func openSupportEmail() {
        if let emailURL = generateSupportEmailURL() {
            NSWorkspace.shared.open(emailURL)
        }
    }
    
    
}
