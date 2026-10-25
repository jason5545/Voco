import Foundation

enum VocoNavigationPolicy {
    static var secondaryItems: [ViewType] {
        VocoLicensePolicy.showsLicenseDestination ? [.settings, .license] : [.settings]
    }

    static var visibleViewTypes: Set<ViewType> {
        Set(ViewType.allCases.filter(isVisible(_:)))
    }

    static func isVisible(_ viewType: ViewType) -> Bool {
        viewType != .license || VocoLicensePolicy.showsLicenseDestination
    }
}
