import Foundation

enum VocoLicensePolicy {
    static let isPersonalNonCommercialBuild = true

    static var hasFullAccess: Bool {
        isPersonalNonCommercialBuild
    }

    static var showsSalesUI: Bool {
        !isPersonalNonCommercialBuild
    }

    static var showsLicenseDestination: Bool {
        showsSalesUI
    }
}
