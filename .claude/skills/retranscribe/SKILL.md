---
name: retranscribe
description: 從 Voco 資料庫挖出最近的轉錄記錄，用指定的 ASR 模型重跑音檔，比對原文差異
argument-hint: "[天數或日期範圍] [筆數] [模型]"
allowed-tools: Bash(*), Read, Write, Edit, Grep, Glob
---

# 重跑轉錄比對工具

從 Voco 的 SwiftData 資料庫（`~/Library/Application Support/com.jasonchien.Voco/default.store`）挖出轉錄記錄，用指定的 ASR 模型重新跑音檔，然後比對結果。

## 參數解析

使用者可能用以下方式指定參數（都是可選的）：

- **時間範圍**：「今天」「昨天到今天」「最近3天」「3/20到3/22」→ 轉成 SQL 的 datetime 條件
- **筆數限制**：「20筆」「全部」→ 預設 20 筆
- **模型選擇**：「qwen3」「whisper」→ 預設 qwen3（1.7B-8bit）

原始參數：$ARGUMENTS

## 資料庫查詢

```sql
-- 資料庫位置
~/Library/Application Support/com.jasonchien.Voco/default.store

-- 關鍵欄位
ZTRANSCRIPTION 表：
  Z_PK          -- 主鍵
  ZTEXT         -- ASR 原始轉錄（增強前，目前引擎為 Qwen3-ASR 1.7B）
  ZENHANCEDTEXT -- LLM 增強後文字
  ZAUDIOFILEURL -- 音檔路徑（file:// URL，需 URL decode）
  ZTIMESTAMP    -- Core Data 時間戳（+ 978307200 = Unix timestamp）

-- 時間轉換
datetime(ZTIMESTAMP + 978307200, 'unixepoch', 'localtime')
```

## 執行步驟

### 1. 先查有多少筆符合條件的記錄

用 sqlite3 COUNT 查詢，告訴使用者總共有幾筆。

### 2. 寫 Python 批次腳本到 /tmp/voco_retranscribe.py

腳本需包含：

```python
#!/usr/bin/env python3
"""Batch re-transcribe Voco recordings."""
import sqlite3, os, time
from urllib.parse import unquote, urlparse

DB_PATH = os.path.expanduser(
    "~/Library/Application Support/com.jasonchien.Voco/default.store"
)

# 根據使用者選的模型決定
# Qwen3-ASR:
#   from mlx_audio.stt import load_model
#   model = load_model('mlx-community/Qwen3-ASR-1.7B-8bit')
#   output = model.generate(path, language="Chinese")
#   text = output.text.strip()
#
# Whisper MLX:
#   from mlx_audio.stt import load_model
#   model = load_model('mlx-community/whisper-large-v2-mlx-8bit')
#   output = model.generate(path, language="zh")
#   text = output["text"].strip()  # whisper 回傳 dict

# 查詢 → 逐筆跑 → 印出比對結果
# 格式：
#   [N/total] PK=xxx | datetime | elapsed
#     ZTEXT (ASR原文)    : ...
#     ZENHANCED (LLM增強): ...  （只在不同時顯示）
#     MODEL (重跑結果)   : ...
#     ⚡ 結果不同 / ✓ 結果相同

# 最後印統計摘要
```

### 3. 執行腳本

用 Bash 工具執行，timeout 設 600000（10 分鐘）。

### 4. 分析結果

跑完後，分析並摘要：
- 相同/不同比例
- 哪些案例模型表現更好或更差
- 有沒有系統性的問題模式（如同音字偏差、簡繁差異、code-switching 等）

## 可用模型（HF 快取中）

| 模型 ID | 類型 | 說明 |
|---------|------|------|
| `mlx-community/Qwen3-ASR-1.7B-8bit` | Qwen3-ASR | 預設選擇 |
| `mlx-community/Qwen3-ASR-0.6B-4bit` | Qwen3-ASR | 較小較快 |
| `mlx-community/whisper-large-v2-mlx-8bit` | Whisper MLX | 之前的引擎 |
| `mlx-community/whisper-large-asr-4bit` | Whisper MLX | 4-bit 版本 |

## 注意事項

- Qwen3-ASR 輸出是簡體中文，這是正常的（Voco 的 ChinesePostProcessingService 會用 OpenCC s2twp 轉正體）
- ZTEXT 存的是經過 ChinesePostProcessingService 處理後的文字（已轉正體），而重跑模型的輸出是簡體，比對時簡繁差異不算真正的「不同」
- 音檔路徑是 file:// URL，需要 URL decode
- 平均每筆約 0.3~2 秒，視音檔長度而定
