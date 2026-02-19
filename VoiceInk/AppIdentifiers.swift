import Foundation

/// Central identifiers for the app, derived from the bundle ID at runtime.
///
/// **Forking Voco?** Just change `PRODUCT_BUNDLE_IDENTIFIER` in Xcode project
/// settings — all identifiers below adapt automatically.
enum AppIdentifiers {
    /// The app's bundle identifier (e.g. `com.jasonchien.Voco`).
    static let bundleID = Bundle.main.bundleIdentifier ?? "com.voco.app"

    /// Lowercase variant used for Logger subsystem and queue labels.
    static let subsystem = bundleID.lowercased()

    /// Base Application Support directory for this app.
    static var appSupportDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(bundleID, isDirectory: true)
    }

    /// iCloud CloudKit container identifier (non-local builds only).
    static let cloudKitContainer = "iCloud.\(bundleID)"
}
