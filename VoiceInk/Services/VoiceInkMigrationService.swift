import Foundation
import os
import Security

/// Migrates data from original VoiceInk to Voco on first launch
class VoiceInkMigrationService {
    static let shared = VoiceInkMigrationService()

    private let logger = Logger(subsystem: "com.jasonchien.voco", category: "VoiceInkMigration")

    private let migrationCompletedKey = "VocoMigrationFromVoiceInkCompleted"
    private let xvoiceMigrationCompletedKey = "VocoMigrationFromXVoiceCompleted"
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

    /// Whether VoiceInk migration is needed
    var needsMigration: Bool {
        // Already marked as completed
        if UserDefaults.standard.bool(forKey: migrationCompletedKey) {
            return false
        }
        // No VoiceInk data to migrate from
        if !voiceInkDataExists() {
            return false
        }
        // Voco already has its own SwiftData stores — migration was already done
        // (manually or partially) so no need to run again
        let vocoAppSupport = vocoAppSupportURL()
        let fm = FileManager.default
        if fm.fileExists(atPath: vocoAppSupport.appendingPathComponent("default.store").path) {
            logger.info("Voco already has SwiftData stores, marking migration as completed")
            UserDefaults.standard.set(true, forKey: migrationCompletedKey)
            return false
        }
        return true
    }

    /// Whether XVoice API key migration is needed
    var needsXVoiceMigration: Bool {
        return !UserDefaults.standard.bool(forKey: xvoiceMigrationCompletedKey)
    }

    /// Migrate XVoice OpenRouter API key from .env file
    func migrateXVoiceAPIKeyIfNeeded() {
        guard needsXVoiceMigration else { return }

        let xvoiceEnvPaths = [
            NSHomeDirectory() + "/GitHub/xvoice/.env",
            NSHomeDirectory() + "/xvoice/.env",
        ]

        for envPath in xvoiceEnvPaths {
            guard FileManager.default.fileExists(atPath: envPath),
                  let content = try? String(contentsOfFile: envPath, encoding: .utf8) else { continue }

            for line in content.components(separatedBy: .newlines) {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard trimmed.hasPrefix("OPENROUTER_API_KEY=") else { continue }
                let key = String(trimmed.dropFirst("OPENROUTER_API_KEY=".count))
                guard !key.isEmpty else { continue }

                // Only set if Voco doesn't already have an OpenRouter key
                if !APIKeyManager.shared.hasAPIKey(forProvider: "openrouter") {
                    APIKeyManager.shared.saveAPIKey(key, forProvider: "openrouter")
                    logger.notice("Migrated OpenRouter API key from XVoice")
                } else {
                    logger.info("OpenRouter API key already exists in Voco, skipping XVoice migration")
                }

                UserDefaults.standard.set(true, forKey: xvoiceMigrationCompletedKey)
                return
            }
        }

        // No .env found or no key in it, mark as done
        UserDefaults.standard.set(true, forKey: xvoiceMigrationCompletedKey)
        logger.info("No XVoice .env found, skipping API key migration")
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

        // Always mark as completed after first run — don't retry on every launch
        UserDefaults.standard.set(true, forKey: migrationCompletedKey)
        if result.isFullySuccessful {
            logger.notice("Migration completed successfully: \(result.summary)")
        } else {
            logger.warning("Migration completed with issues: \(result.summary)")
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
        // Use kSecAttrSynchronizableAny to match both syncable and non-syncable items
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: voiceInkBundleId,
            kSecAttrSynchronizable as String: kSecAttrSynchronizableAny,
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

        let keychainService = KeychainService.shared
        var success = true

        for item in items {
            guard let account = item[kSecAttrAccount as String] as? String,
                  let data = item[kSecValueData as String] as? Data else { continue }

            if keychainService.exists(forKey: account) {
                logger.info("Keychain item '\(account)' already exists in Voco, skipping")
                continue
            }

            // Use KeychainService to write with correct attributes (syncable + data protection)
            if keychainService.save(data: data, forKey: account) {
                logger.info("Migrated keychain item: \(account)")
            } else {
                result.errors.append("Failed to migrate keychain item '\(account)'")
                success = false
            }
        }

        return success
    }
}
