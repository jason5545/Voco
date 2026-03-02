import Foundation

/// Central identifiers for the app, derived from the bundle ID at runtime.
///
/// **Forking Voco?** Just change `PRODUCT_BUNDLE_IDENTIFIER` in Xcode project
/// settings — all identifiers below adapt automatically.
enum AppIdentifiers {
    /// The main app's bundle identifier (always the host app, not the extension).
    /// For keyboard extension `com.jasonchien.Voco.VocoKeyboard`, this returns `com.jasonchien.Voco`.
    static let bundleID: String = {
        let raw = Bundle.main.bundleIdentifier ?? "com.voco.app"
        // Strip extension suffix (e.g. ".VocoKeyboard") to get the host app bundle ID
        if raw.contains(".") {
            let components = raw.components(separatedBy: ".")
            // A standard app bundle ID has 3 parts (com.company.App)
            // An extension has 4+ (com.company.App.Extension)
            if components.count > 3 {
                return components.dropLast().joined(separator: ".")
            }
        }
        return raw
    }()

    /// Lowercase variant used for Logger subsystem and queue labels.
    static let subsystem = bundleID.lowercased()

    /// App Group identifier for sharing data between main app and extensions.
    static let appGroupID = "group.com.jasonchien.Voco"

    /// Base Application Support directory for this app.
    static var appSupportDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(bundleID, isDirectory: true)
    }

    /// App Group shared container directory (available to main app + extensions).
    static var appGroupDirectory: URL? {
        FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: appGroupID)
    }

    /// iCloud CloudKit container identifier (non-local builds only).
    static let cloudKitContainer = "iCloud.\(bundleID)"
}
