<div align="center">
  <img src="VoiceInk/Assets.xcassets/AppIcon.appiconset/256-mac.png" width="180" height="180" />
  <h1>Voco</h1>
  <p>Offline speech-to-text for macOS, with deep optimization for Traditional Chinese</p>

  [![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
  ![Platform](https://img.shields.io/badge/platform-macOS%2014.4%2B-brightgreen)
  [![Based on VoiceInk](https://img.shields.io/badge/based%20on-VoiceInk-orange)](https://github.com/Beingpax/VoiceInk)

  **[正體中文說明](README.zh-Hant.md)**
</div>

---

Voco is a fork of [VoiceInk](https://github.com/Beingpax/VoiceInk), built for **Traditional Chinese (Taiwan)** users. Speech recognition runs entirely on-device via whisper.cpp — only the optional AI refinement step talks to an LLM through your own API key.

## What's different from VoiceInk?

| Feature | VoiceInk | Voco |
|---------|----------|------|
| Traditional Chinese UI | English only | Full zh-Hant localization |
| Chinese post-processing | None | OpenCC simplified-to-traditional, homophone correction, colloquial punctuation, repetition detection |
| Taiwanese Chinese AI prompts | None | Built-in prompts for filler removal, auto-listing, homophone fixing |
| Qwen3-ASR engine | None | On-device Qwen3 speech recognition via MLX Swift |
| App context injection | None | Detects foreground app + window title, lets the LLM adapt tone |
| Edit Mode | None | Voice-driven editing of selected text |
| Voice commands | None | Voice commands (e.g. "select all + delete") |
| Confidence routing | None | Forces LLM pass when punctuation is missing in long segments |
| Licensing | Paid | All features unlocked |

## Features

- **Offline transcription** — whisper.cpp runs locally; audio never leaves your machine
- **Qwen3-ASR** — alternative on-device engine via MLX, optimized for Chinese
- **AI refinement** — post-process text through your own API key (OpenAI / Anthropic / Ollama / 11 providers)
- **Chinese optimization** — homophone correction, simplified-to-traditional conversion, colloquial punctuation, filler removal
- **Edit Mode** — select text, speak corrections, and Voco rewrites it in place
- **App-aware context** — automatically detects the active app to adjust output style
- **Voice commands** — hands-free control (e.g. "delete all")
- **Global hotkey** — customizable keyboard shortcut, supports hold-to-record
- **Personal dictionary** — custom vocabulary for names, jargon, technical terms
- **Power Mode** — per-app configuration presets

## Build

### Requirements

- macOS 14.4+
- Xcode (latest recommended)
- Git

### Quick start

```bash
git clone https://github.com/jason5545/Voco.git
cd Voco

# Full build (recommended for first time)
make all

# Build without Apple Developer certificate
make local

# Build + run
make dev
```

### Makefile targets

| Target | Description |
|--------|-------------|
| `make check` | Verify build prerequisites |
| `make whisper` | Build whisper.cpp XCFramework |
| `make build` | Standard build (requires Apple Developer account) |
| `make local` | Local build with ad-hoc signing (no certificate needed) |
| `make dev` | Build + launch |
| `make clean` | Remove build artifacts |

See [BUILDING.md](BUILDING.md) for detailed instructions.

### Local build limitations

Builds made with `make local` work fully except:
- No iCloud dictionary sync (requires CloudKit entitlement)
- No automatic updates (pull and rebuild to update)

## Forking

Voco uses `Bundle.main.bundleIdentifier` for all runtime identifiers (logging, data directories, Keychain, window IDs). To create your own fork:

1. Change `PRODUCT_BUNDLE_IDENTIFIER` in Xcode project settings (one place for each target)
2. Update the CloudKit container in `VoiceInk.entitlements` if you use iCloud sync
3. That's it — all Swift code adapts automatically

## Architecture

```
VoiceInk/
├── Views/              # SwiftUI views
├── Models/             # Data models (SwiftData)
├── Services/           # Business logic
│   ├── AIEnhancement/  #   AI refinement (multi-provider)
│   └── ChinesePostProcessing/  #   Chinese text pipeline
├── Whisper/            # whisper.cpp integration
├── Qwen3ASR/           # Qwen3 speech recognition (MLX)
├── PowerMode/          # App detection & auto-switching
└── Resources/          # Models, sounds
```

- **UI**: SwiftUI + AppKit
- **Data**: SwiftData
- **Transcription**: whisper.cpp (offline) + Qwen3-ASR (offline, MLX)
- **AI refinement**: OpenAI / Anthropic / Ollama / and 8 more providers (user's own API key)

## Privacy

- Speech recognition is 100% offline
- AI refinement sends **text only** (not audio) to the LLM provider you choose
- API keys are stored locally in Keychain, never routed through third-party servers
- App context sends only the application name and window title — no screen capture

## Credits

Voco is built on top of:

- [VoiceInk](https://github.com/Beingpax/VoiceInk) — upstream project by Pax
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — high-performance speech recognition
- [OpenCC](https://github.com/BYVoid/OpenCC) — Chinese simplified/traditional conversion (concept reference)
- [Sparkle](https://github.com/sparkle-project/Sparkle) — auto-updates
- [KeyboardShortcuts](https://github.com/sindresorhus/KeyboardShortcuts) — global hotkeys
- [SelectedTextKit](https://github.com/tisfeng/SelectedTextKit) — selected text extraction

## License

Licensed under the [GNU General Public License v3.0](LICENSE), consistent with upstream VoiceInk.
