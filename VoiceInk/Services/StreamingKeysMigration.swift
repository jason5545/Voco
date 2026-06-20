import Foundation

enum StreamingKeysMigration {
    static func run() {
        let defaults = UserDefaults.standard
        if !defaults.bool(forKey: "streaming-keys-migrated") {
            let legacyStreamingMappings: [(old: String, new: [String])] = [
                ("parakeet-streaming-enabled", [
                    "streaming-enabled-parakeet-tdt-0.6b-v2",
                    "streaming-enabled-parakeet-tdt-0.6b-v3",
                ]),
            ]

            for mapping in legacyStreamingMappings {
                if let value = defaults.object(forKey: mapping.old) as? Bool {
                    for newKey in mapping.new {
                        defaults.set(value, forKey: newKey)
                    }
                    defaults.removeObject(forKey: mapping.old)
                }
            }

            // Remap CurrentTranscriptionModel if it points to a removed streaming-only model name.
            let removedModelMappings: [String: String] = [
                "stt-rt-v4": "stt-async-v5",
                "voxtral-mini-transcribe-realtime-2602": "voxtral-mini-latest",
            ]

            if let savedModel = defaults.string(forKey: "CurrentTranscriptionModel"),
               let replacement = removedModelMappings[savedModel] {
                defaults.set(replacement, forKey: "CurrentTranscriptionModel")
            }

            // Remap selectedTranscriptionModelName inside each stored ModeConfig.
            // Check both the renamed key and the legacy key so older saved data is fixed
            // before ModeDataMigration copies it forward.
            // Uses JSONSerialization so the migration stays independent of the ModeConfig struct shape.
            remapStoredModeModels(defaults: defaults, mappings: removedModelMappings)

            defaults.set(true, forKey: "streaming-keys-migrated")
        }

        migrateSonioxV5(defaults: defaults)
    }

    private static func migrateSonioxV5(defaults: UserDefaults) {
        guard !defaults.bool(forKey: "soniox-v5-model-migrated") else { return }

        let mappings = [
            "stt-rt-v4": "stt-async-v5",
            "stt-async-v4": "stt-async-v5",
        ]

        if let savedModel = defaults.string(forKey: "CurrentTranscriptionModel"),
           let replacement = mappings[savedModel] {
            defaults.set(replacement, forKey: "CurrentTranscriptionModel")
        }

        remapStoredModeModels(defaults: defaults, mappings: mappings)
        defaults.set(true, forKey: "soniox-v5-model-migrated")
    }

    private static func remapStoredModeModels(defaults: UserDefaults, mappings: [String: String]) {
        for modeKey in ["modeConfigurationsV2", "powerModeConfigurationsV2"] {
            guard let data = defaults.data(forKey: modeKey),
                  var configs = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]]
            else { continue }

            var changed = false
            for index in configs.indices {
                guard let savedModel = configs[index]["selectedTranscriptionModelName"] as? String,
                      let replacement = mappings[savedModel] else { continue }
                configs[index]["selectedTranscriptionModelName"] = replacement
                changed = true
            }
            if changed, let newData = try? JSONSerialization.data(withJSONObject: configs) {
                defaults.set(newData, forKey: modeKey)
            }
        }
    }
}
