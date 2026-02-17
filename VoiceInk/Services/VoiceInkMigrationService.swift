import Foundation
import os
import Security

/// Migrates data from original VoiceInk to Voco on first launch
class VoiceInkMigrationService {
    static let shared = VoiceInkMigrationService()

    private let logger = Logger(subsystem: "com.jasonchien.voco", category: "VoiceInkMigration")

    private let migrationCompletedKey = "VocoMigrationFromVoiceInkCompleted"
    private let voiceInkBundleId = "com.prakashjoshipax.VoiceInk"

    /// Result of migration attempt
    struct MigrationResult {
        var userDefaultsMigrated = false
        var swiftDataMigrated = false
        var whisperModelsMigrated = false
        var recordingsMigrated = false
        var keychainMigrated = false
        var errors: [String] = []

        var isFullySuccessful: Bool {
            return errors.isEmpty
        }

        var summary: String {
            var parts: [String] = []
            if userDefaultsMigrated { parts.append("UserDefaults") }
            if swiftDataMigrated { parts.append("SwiftData") }
            if whisperModelsMigrated { parts.append("Whisper Models") }
            if recordingsMigrated { parts.append("Recordings") }
            if keychainMigrated { parts.append("Keychain") }
            let migrated = parts.isEmpty ? "None" : parts.joined(separator: ", ")
            let errorSummary = errors.isEmpty ? "" : " | Errors: \(errors.joined(separator: "; "))"
            return "Migrated: \(migrated)\(errorSummary)"
        }
    }

    // MARK: - Public API

    /// Whether migration is needed (VoiceInk data exists and hasn't been migrated)
    var needsMigration: Bool {
        return !UserDefaults.standard.bool(forKey: migrationCompletedKey)
            && voiceInkDataExists()
    }

    /// Perform migration if needed
    func migrateIfNeeded() async -> MigrationResult {
        guard needsMigration else {
            return MigrationResult()
        }

        logger.notice("Starting VoiceInk → Voco migration...")
        var result = MigrationResult()

        let fm = FileManager.default
        let voiceInkAppSupport = voiceInkAppSupportURL()
        let vocoAppSupport = vocoAppSupportURL()

        // Ensure Voco app support directory exists
        do {
            try fm.createDirectory(at: vocoAppSupport, withIntermediateDirectories: true)
        } catch {
            result.errors.append("Failed to create Voco app support directory: \(error.localizedDescription)")
            logger.error("Migration failed: cannot create Voco directory")
            return result
        }

        // 1. Migrate UserDefaults
        result.userDefaultsMigrated = migrateUserDefaults()

        // 2. Migrate SwiftData stores
        result.swiftDataMigrated = migrateSwiftData(from: voiceInkAppSupport, to: vocoAppSupport, result: &result)

        // 3. Migrate Whisper Models (symlink)
        result.whisperModelsMigrated = migrateWhisperModels(from: voiceInkAppSupport, to: vocoAppSupport, result: &result)

        // 4. Migrate Recordings (symlink)
        result.recordingsMigrated = migrateRecordings(from: voiceInkAppSupport, to: vocoAppSupport, result: &result)

        // 5. Migrate Keychain items
        result.keychainMigrated = migrateKeychain(result: &result)

        // Mark as completed only if fully successful
        if result.isFullySuccessful {
            UserDefaults.standard.set(true, forKey: migrationCompletedKey)
            logger.notice("Migration completed successfully: \(result.summary)")
        } else {
            logger.warning("Migration partially completed: \(result.summary)")
        }

        return result
    }

    // MARK: - Detection

    private func voiceInkDataExists() -> Bool {
        let appSupport = voiceInkAppSupportURL()
        return FileManager.default.fileExists(atPath: appSupport.path)
    }

    private func voiceInkAppSupportURL() -> URL {
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(voiceInkBundleId)
    }

    private func vocoAppSupportURL() -> URL {
        let bundleId = Bundle.main.bundleIdentifier ?? voiceInkBundleId
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(bundleId)
    }

    // MARK: - UserDefaults Migration

    private func migrateUserDefaults() -> Bool {
        let plistPath = NSHomeDirectory() + "/Library/Preferences/\(voiceInkBundleId).plist"
        guard FileManager.default.fileExists(atPath: plistPath),
              let plistData = NSDictionary(contentsOfFile: plistPath) as? [String: Any] else {
            logger.info("No VoiceInk UserDefaults plist found, skipping")
            return true // Not an error, just nothing to migrate
        }

        let currentDefaults = UserDefaults.standard
        var count = 0
        for (key, value) in plistData {
            // Don't overwrite existing Voco values
            if currentDefaults.object(forKey: key) == nil {
                currentDefaults.set(value, forKey: key)
                count += 1
            }
        }

        logger.info("Migrated \(count) UserDefaults entries")
        return true
    }

    // MARK: - SwiftData Migration

    private func migrateSwiftData(from source: URL, to dest: URL, result: inout MigrationResult) -> Bool {
        let fm = FileManager.default
        let storeFiles = ["default.store", "dictionary.store"]
        var success = true

        for store in storeFiles {
            let sourceFile = source.appendingPathComponent(store)
            let destFile = dest.appendingPathComponent(store)

            guard fm.fileExists(atPath: sourceFile.path) else { continue }

            if fm.fileExists(atPath: destFile.path) {
                logger.info("SwiftData store \(store) already exists in Voco, skipping")
                continue
            }

            do {
                // Copy the main store file
                try fm.copyItem(at: sourceFile, to: destFile)

                // Also copy WAL and SHM files if they exist
                for suffix in ["-wal", "-shm"] {
                    let walSource = source.appendingPathComponent(store + suffix)
                    let walDest = dest.appendingPathComponent(store + suffix)
                    if fm.fileExists(atPath: walSource.path) && !fm.fileExists(atPath: walDest.path) {
                        try fm.copyItem(at: walSource, to: walDest)
                    }
                }

                logger.info("Copied SwiftData store: \(store)")
            } catch {
                result.errors.append("Failed to copy \(store): \(error.localizedDescription)")
                success = false
            }
        }

        return success
    }

    // MARK: - Whisper Models Migration (Symlink)

    private func migrateWhisperModels(from source: URL, to dest: URL, result: inout MigrationResult) -> Bool {
        let fm = FileManager.default
        let sourceModels = source.appendingPathComponent("WhisperModels")
        let destModels = dest.appendingPathComponent("WhisperModels")

        guard fm.fileExists(atPath: sourceModels.path) else {
            logger.info("No VoiceInk WhisperModels directory found, skipping")
            return true
        }

        if fm.fileExists(atPath: destModels.path) {
            logger.info("WhisperModels directory already exists in Voco, skipping")
            return true
        }

        do {
            // Create symlink to avoid duplicating large model files
            try fm.createSymbolicLink(at: destModels, withDestinationURL: sourceModels)
            logger.info("Created symlink for WhisperModels")
            return true
        } catch {
            result.errors.append("Failed to symlink WhisperModels: \(error.localizedDescription)")
            return false
        }
    }

    // MARK: - Recordings Migration (Symlink)

    private func migrateRecordings(from source: URL, to dest: URL, result: inout MigrationResult) -> Bool {
        let fm = FileManager.default
        let sourceRecordings = source.appendingPathComponent("Recordings")
        let destRecordings = dest.appendingPathComponent("Recordings")

        guard fm.fileExists(atPath: sourceRecordings.path) else {
            logger.info("No VoiceInk Recordings directory found, skipping")
            return true
        }

        if fm.fileExists(atPath: destRecordings.path) {
            logger.info("Recordings directory already exists in Voco, skipping")
            return true
        }

        do {
            try fm.createSymbolicLink(at: destRecordings, withDestinationURL: sourceRecordings)
            logger.info("Created symlink for Recordings")
            return true
        } catch {
            result.errors.append("Failed to symlink Recordings: \(error.localizedDescription)")
            return false
        }
    }

    // MARK: - Keychain Migration

    private func migrateKeychain(result: inout MigrationResult) -> Bool {
        // Query for all keychain items with VoiceInk service name
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: voiceInkBundleId,
            kSecReturnAttributes as String: true,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitAll,
        ]

        var queryResult: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &queryResult)

        guard status == errSecSuccess,
              let items = queryResult as? [[String: Any]] else {
            if status == errSecItemNotFound {
                logger.info("No VoiceInk keychain items found, skipping")
                return true
            }
            result.errors.append("Keychain query failed: \(status)")
            return false
        }

        let vocoBundleId = Bundle.main.bundleIdentifier ?? voiceInkBundleId
        var success = true

        for item in items {
            guard let account = item[kSecAttrAccount as String] as? String,
                  let data = item[kSecValueData as String] as? Data else { continue }

            // Check if already exists in Voco keychain
            let checkQuery: [String: Any] = [
                kSecClass as String: kSecClassGenericPassword,
                kSecAttrService as String: vocoBundleId,
                kSecAttrAccount as String: account,
            ]

            let checkStatus = SecItemCopyMatching(checkQuery as CFDictionary, nil)
            if checkStatus == errSecSuccess {
                logger.info("Keychain item '\(account)' already exists in Voco, skipping")
                continue
            }

            // Add to Voco keychain
            let addQuery: [String: Any] = [
                kSecClass as String: kSecClassGenericPassword,
                kSecAttrService as String: vocoBundleId,
                kSecAttrAccount as String: account,
                kSecValueData as String: data,
            ]

            let addStatus = SecItemAdd(addQuery as CFDictionary, nil)
            if addStatus == errSecSuccess {
                logger.info("Migrated keychain item: \(account)")
            } else {
                result.errors.append("Failed to migrate keychain item '\(account)': \(addStatus)")
                success = false
            }
        }

        return success
    }
}
