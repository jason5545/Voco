#!/usr/bin/env python3
"""
ASR 模型批次評測腳本。

用法：
  # 用 .venv (Python 3.14) 跑 sherpa-onnx / MLX 模型
  .venv/bin/python run_eval.py --models qwen3-1.7b,whisper-v3-turbo-mlx,firered-v1,sensevoice

  # 用 .venv310 (Python 3.10) 跑 FireRedASR2
  .venv310/bin/python run_eval.py --models firered2-aed

  # 跑所有（先用 .venv 跑前面的，再用 .venv310 跑 firered2）
  # 結果會追加到同一個 results.json

  # 指定測試集
  .venv/bin/python run_eval.py --test-set test_data/test_set.json --models sensevoice
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def run_qwen3(test_cases: list[dict], model_id: str) -> list[dict]:
    """Qwen3-ASR via mlx_audio."""
    from mlx_audio.stt.generate import generate_transcription

    model_map = {
        "qwen3-0.6b": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "qwen3-1.7b": "mlx-community/Qwen3-ASR-1.7B-8bit",
    }
    hf_id = model_map[model_id]
    print(f"  載入模型 {hf_id} ...")

    results = []
    for i, tc in enumerate(test_cases):
        print(f"  [{i+1}/{len(test_cases)}] {Path(tc['audio_path']).name} ({tc['duration']:.1f}s)")
        t0 = time.time()
        try:
            output = generate_transcription(
                model=hf_id,
                audio=tc["audio_path"],
            )
            text = output.text.strip() if hasattr(output, 'text') else str(output).strip()
        except Exception as e:
            text = f"[ERROR] {e}"
        elapsed = time.time() - t0
        results.append({
            "audio": Path(tc["audio_path"]).name,
            "model": model_id,
            "transcription": text,
            "elapsed_s": round(elapsed, 3),
            "rtf": round(elapsed / max(tc["duration"], 0.1), 4),
        })
    return results


def run_whisper_mlx(test_cases: list[dict], model_id: str) -> list[dict]:
    """Whisper via mlx_whisper."""
    import mlx_whisper

    model_map = {
        "whisper-v3-turbo-mlx": "mlx-community/whisper-large-v3-turbo",
        "whisper-v3-mlx": "mlx-community/whisper-large-v3",
    }
    hf_id = model_map[model_id]
    print(f"  載入模型 {hf_id} ...")
    # mlx_whisper 會在第一次呼叫時自動下載

    results = []
    for i, tc in enumerate(test_cases):
        print(f"  [{i+1}/{len(test_cases)}] {Path(tc['audio_path']).name} ({tc['duration']:.1f}s)")
        t0 = time.time()
        try:
            output = mlx_whisper.transcribe(
                tc["audio_path"],
                path_or_hf_repo=hf_id,
                language="zh",
            )
            text = output.get("text", "").strip()
        except Exception as e:
            text = f"[ERROR] {e}"
        elapsed = time.time() - t0
        results.append({
            "audio": Path(tc["audio_path"]).name,
            "model": model_id,
            "transcription": text,
            "elapsed_s": round(elapsed, 3),
            "rtf": round(elapsed / max(tc["duration"], 0.1), 4),
        })
    return results


def run_firered_v1(test_cases: list[dict]) -> list[dict]:
    """FireRedASR v1 AED-L via sherpa-onnx."""
    import sherpa_onnx

    model_dir = str(MODELS_DIR / "sherpa-onnx-fire-red-asr-large")
    print(f"  載入模型 {model_dir} ...")

    recognizer = sherpa_onnx.OfflineRecognizer.from_fire_red_asr(
        encoder=f"{model_dir}/encoder.int8.onnx",
        decoder=f"{model_dir}/decoder.int8.onnx",
        tokens=f"{model_dir}/tokens.txt",
        num_threads=4,
    )

    results = []
    for i, tc in enumerate(test_cases):
        print(f"  [{i+1}/{len(test_cases)}] {Path(tc['audio_path']).name} ({tc['duration']:.1f}s)")
        t0 = time.time()
        try:
            stream = recognizer.create_stream()
            import wave
            with wave.open(tc["audio_path"], "rb") as wf:
                assert wf.getsampwidth() == 2  # 16-bit
                assert wf.getnchannels() == 1   # mono
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            import numpy as np
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            stream.accept_waveform(sr, samples.tolist())
            recognizer.decode_stream(stream)
            text = stream.result.text.strip()
        except Exception as e:
            text = f"[ERROR] {e}"
        elapsed = time.time() - t0
        results.append({
            "audio": Path(tc["audio_path"]).name,
            "model": "firered-v1",
            "transcription": text,
            "elapsed_s": round(elapsed, 3),
            "rtf": round(elapsed / max(tc["duration"], 0.1), 4),
        })
    return results


def run_sensevoice(test_cases: list[dict]) -> list[dict]:
    """SenseVoice-Small via sherpa-onnx."""
    import sherpa_onnx

    model_dir = str(MODELS_DIR / "sherpa-onnx-sensevoice-small")
    print(f"  載入模型 {model_dir} ...")

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=f"{model_dir}/model.int8.onnx",
        tokens=f"{model_dir}/tokens.txt",
        language="auto",
        use_itn=True,
        num_threads=4,
    )

    results = []
    for i, tc in enumerate(test_cases):
        print(f"  [{i+1}/{len(test_cases)}] {Path(tc['audio_path']).name} ({tc['duration']:.1f}s)")
        t0 = time.time()
        try:
            stream = recognizer.create_stream()
            import wave
            with wave.open(tc["audio_path"], "rb") as wf:
                assert wf.getsampwidth() == 2
                assert wf.getnchannels() == 1
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            import numpy as np
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            stream.accept_waveform(sr, samples.tolist())
            recognizer.decode_stream(stream)
            text = stream.result.text.strip()
        except Exception as e:
            text = f"[ERROR] {e}"
        elapsed = time.time() - t0
        results.append({
            "audio": Path(tc["audio_path"]).name,
            "model": "sensevoice",
            "transcription": text,
            "elapsed_s": round(elapsed, 3),
            "rtf": round(elapsed / max(tc["duration"], 0.1), 4),
        })
    return results


def run_firered2_aed(test_cases: list[dict]) -> list[dict]:
    """FireRedASR2-AED via PyTorch (需要 Python 3.10)."""
    # 把 FireRedASR2S repo 加入 path
    repo_dir = str(MODELS_DIR / "FireRedASR2S")
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

    model_dir = str(MODELS_DIR / "FireRedASR2-AED")
    print(f"  載入模型 {model_dir} ...")

    config = FireRedAsr2Config(
        use_gpu=False,  # macOS 沒有 CUDA
        use_half=False,
        beam_size=3,
        return_timestamp=False,
    )
    model = FireRedAsr2.from_pretrained("aed", model_dir, config)

    results = []
    for i, tc in enumerate(test_cases):
        print(f"  [{i+1}/{len(test_cases)}] {Path(tc['audio_path']).name} ({tc['duration']:.1f}s)")
        t0 = time.time()
        try:
            output = model.transcribe(
                [f"utt_{i}"],
                [tc["audio_path"]],
            )
            text = output[0]["text"].strip() if output else ""
        except Exception as e:
            text = f"[ERROR] {e}"
        elapsed = time.time() - t0
        results.append({
            "audio": Path(tc["audio_path"]).name,
            "model": "firered2-aed",
            "transcription": text,
            "elapsed_s": round(elapsed, 3),
            "rtf": round(elapsed / max(tc["duration"], 0.1), 4),
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_MODELS = [
    "qwen3-0.6b", "qwen3-1.7b",
    "whisper-v3-turbo-mlx", "whisper-v3-mlx",
    "firered-v1", "sensevoice",
    "firered2-aed",
]

RUNNER_MAP = {
    "qwen3-0.6b": lambda tc: run_qwen3(tc, "qwen3-0.6b"),
    "qwen3-1.7b": lambda tc: run_qwen3(tc, "qwen3-1.7b"),
    "whisper-v3-turbo-mlx": lambda tc: run_whisper_mlx(tc, "whisper-v3-turbo-mlx"),
    "whisper-v3-mlx": lambda tc: run_whisper_mlx(tc, "whisper-v3-mlx"),
    "firered-v1": run_firered_v1,
    "sensevoice": run_sensevoice,
    "firered2-aed": run_firered2_aed,
}


def main():
    parser = argparse.ArgumentParser(description="ASR 模型批次評測")
    parser.add_argument(
        "--test-set",
        default=str(SCRIPT_DIR / "test_data" / "test_set.json"),
        help="測試集 JSON 路徑",
    )
    parser.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help=f"要測試的模型，逗號分隔。可選：{', '.join(ALL_MODELS)}",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "results.json"),
        help="輸出結果 JSON 路徑",
    )
    args = parser.parse_args()

    # 載入測試集
    with open(args.test_set, encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"測試集：{len(test_cases)} 筆\n")

    models_to_run = [m.strip() for m in args.models.split(",")]
    for m in models_to_run:
        if m not in RUNNER_MAP:
            print(f"未知模型：{m}，可選：{', '.join(ALL_MODELS)}")
            sys.exit(1)

    # 載入已有結果（追加模式）
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing_results = []
    if os.path.isfile(args.output):
        with open(args.output, encoding="utf-8") as f:
            existing_results = json.load(f)

    # 建立已完成的 (audio, model) 集合，跳過已測過的
    done_set = {(r["audio"], r["model"]) for r in existing_results}

    all_results = list(existing_results)

    for model_name in models_to_run:
        # 過濾出尚未測過的
        pending = [
            tc for tc in test_cases
            if (Path(tc["audio_path"]).name, model_name) not in done_set
        ]
        if not pending:
            print(f"[{model_name}] 已全部測完，跳過。\n")
            continue

        print(f"{'='*60}")
        print(f"模型：{model_name}（{len(pending)} 筆待測）")
        print(f"{'='*60}")

        t_model_start = time.time()
        runner = RUNNER_MAP[model_name]
        results = runner(pending)
        t_model_total = time.time() - t_model_start

        all_results.extend(results)

        # 每個模型測完就存一次（防中斷丟失）
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        # 摘要
        errors = [r for r in results if r["transcription"].startswith("[ERROR]")]
        avg_rtf = sum(r["rtf"] for r in results) / max(len(results), 1)
        print(f"\n  完成：{len(results)} 筆，錯誤：{len(errors)} 筆")
        print(f"  總耗時：{t_model_total:.1f}s，平均 RTF：{avg_rtf:.4f}")
        print()

    print(f"結果已寫入：{args.output}")
    print(f"總共 {len(all_results)} 筆結果。")


if __name__ == "__main__":
    main()
