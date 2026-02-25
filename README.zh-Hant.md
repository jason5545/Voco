<div align="center">
  <img src="VoiceInk/Assets.xcassets/AppIcon.appiconset/256-mac.png" width="180" height="180" />
  <h1>Voco</h1>
  <p>macOS 離線語音轉文字，針對臺灣正體中文深度優化</p>

  [![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
  ![Platform](https://img.shields.io/badge/platform-macOS%2014.4%2B-brightgreen)
  [![Based on VoiceInk](https://img.shields.io/badge/based%20on-VoiceInk-orange)](https://github.com/Beingpax/VoiceInk)

  **[English](README.md)**
</div>

---

Voco 是 [VoiceInk](https://github.com/Beingpax/VoiceInk) 的 fork，專為**臺灣正體中文使用者**打造。語音轉錄完全在本機執行（whisper.cpp），只有 AI 潤稿階段會透過你自己的 API key 與 LLM 溝通。

## 與上游 VoiceInk 的差異

| 功能 | VoiceInk | Voco |
|------|----------|------|
| 正體中文介面 | 英文 | 完整繁體中文本地化 |
| 中文後處理管線 | 無 | OpenCC 簡轉繁、拼音同音字修正、口語標點轉換、重複偵測 |
| 臺灣中文 AI 提示詞 | 無 | 內建臺灣用語提示詞（贅字過濾、自動條列化、同音字修正） |
| App Context 注入 | 無 | 自動偵測前景 App + 視窗標題，LLM 依情境調整語氣 |
| Qwen3-ASR 引擎 | 無 | 本機 Qwen3 語音辨識（透過 MLX Swift） |
| Edit Mode | 無 | 語音驅動的選取文字編輯 |
| 語音指令 | 無 | 支援語音指令（如「全部刪除」） |
| 信心度路由安全網 | 無 | 長段落無標點時強制走 LLM，防止漏標點 |
| RNNoise 噪音抑制 | 無 | 即時噪音抑制，提升輸入品質 |
| 授權模式 | 付費授權 | 所有功能解鎖 |

## 功能

- **離線轉錄** — whisper.cpp 本機 AI 模型，語音資料不離開你的電腦
- **Qwen3-ASR** — 替代轉錄引擎，透過 MLX 在本機執行，針對中文優化
- **AI 潤稿** — 透過你自己的 API key 進行後處理（12 個 provider：OpenAI、Anthropic、Ollama、Groq、Gemini 等）
- **臺灣中文優化** — 同音字修正、簡轉繁、口語標點、贅字過濾
- **Edit Mode** — 選取文字後用語音指令修改，Voco 自動改寫
- **App 情境感知** — 自動偵測目前使用的應用程式，調整輸出風格
- **語音指令** — 語音控制操作（如「全部刪除」）
- **信心度路由** — 長段落無標點時強制走 LLM，防止漏標點
- **RNNoise** — 即時噪音抑制，提升音訊輸入品質
- **全域快捷鍵** — 可自訂的鍵盤快捷鍵，支援按住錄音
- **個人詞典** — 自訂專有名詞、技術術語的辨識
- **Power Mode** — 根據不同 App 自動切換預設設定

## 建置

### 需求

- macOS 14.4+
- Xcode（建議最新版）
- Git

### 快速開始

```bash
git clone https://github.com/jason5545/Voco.git
cd Voco

# 完整建置（首次推薦）
make all

# 無 Apple Developer 憑證的本機建置
make local

# 開發用（建置 + 執行）
make dev
```

### Makefile 指令

| 指令 | 說明 |
|------|------|
| `make check` | 檢查建置環境 |
| `make whisper` | 建置 whisper.cpp XCFramework |
| `make build` | 標準建置 |
| `make local` | 無需憑證的本機建置 |
| `make dev` | 建置 + 啟動 |
| `make clean` | 清理建置產物 |

詳細建置說明請參考 [BUILDING.md](BUILDING.md)。

### 本機建置限制

使用 `make local` 建置的版本功能完整，但有以下例外：
- 無 iCloud 詞典同步（需要 CloudKit 權限）
- 無自動更新（需手動拉取並重新建置）

## Forking

Voco 使用 `Bundle.main.bundleIdentifier` 作為所有執行期識別碼（log、資料目錄、Keychain、視窗 ID）。如果你想建立自己的 fork：

1. 在 Xcode 專案設定中修改 `PRODUCT_BUNDLE_IDENTIFIER`（每個 target 一處）
2. 如果使用 iCloud 同步，更新 `VoiceInk.entitlements` 中的 CloudKit container
3. 完成 — 所有 Swift 程式碼會自動適應

## 架構

```
VoiceInk/
├── Views/              # SwiftUI 介面
├── Models/             # 資料模型（SwiftData）
├── Services/           # 服務層
│   ├── AIEnhancement/  #   AI 潤稿（多 provider 支援）
│   └── ChinesePostProcessing/  #   中文後處理管線
├── Whisper/            # whisper.cpp 整合
├── Qwen3ASR/           # Qwen3 語音辨識（MLX）
├── PowerMode/          # App 偵測與自動切換
└── Resources/          # 模型、音效
```

- **UI**：SwiftUI + AppKit
- **資料**：SwiftData
- **轉錄**：whisper.cpp（離線）+ Qwen3-ASR（離線，MLX）
- **AI 潤稿**：12 個 provider，包含 OpenAI、Anthropic、Ollama、Groq、Gemini（使用者自己的 API key）

## 隱私

- 語音轉錄 100% 離線執行
- AI 潤稿僅傳送**文字**（非音檔）到你選擇的 LLM provider
- API key 儲存在本機，不經過任何第三方伺服器
- App Context 只傳送應用程式名稱和視窗標題，不擷取畫面內容

## 致謝

Voco 建立在以下專案之上：

- [VoiceInk](https://github.com/Beingpax/VoiceInk) — 上游專案，由 Pax 開發
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — 高效能語音辨識引擎
- [SwiftyOpenCC](https://github.com/exgphe/SwiftyOpenCC) — 簡體轉正體中文（s2twp）
- [Sparkle](https://github.com/sparkle-project/Sparkle) — 自動更新
- [KeyboardShortcuts](https://github.com/sindresorhus/KeyboardShortcuts) — 全域快捷鍵
- [SelectedTextKit](https://github.com/tisfeng/SelectedTextKit) — 選取文字擷取

## 授權

本專案採用 [GNU General Public License v3.0](LICENSE) 授權，與上游 VoiceInk 一致。
