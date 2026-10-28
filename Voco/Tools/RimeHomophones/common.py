"""離線模擬共用載入器；所有資料檔位置由呼叫端提供。"""

from __future__ import annotations

import collections
import json
import os
import re
import sqlite3
from pathlib import Path
from urllib.parse import quote

CJK = re.compile(r"[㐀-䶿一-鿿\U00020000-\U0002a6df]")
_T2TW: dict[str, str] = {}
_TW2T: dict[str, str] = {}


def required_path(value: str | None, env_name: str, label: str) -> Path:
    raw = (value or os.environ.get(env_name) or "").strip()
    if not raw:
        raise SystemExit(f"請用參數或環境變數 {env_name} 指定{label}路徑")
    return Path(raw).expanduser()


def is_cjk(ch: str) -> bool:
    return bool(CJK.fullmatch(ch))


def configure_variants(path: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    global _T2TW, _TW2T
    t2tw: dict[str, str] = {}
    tw2t: dict[str, str] = {}
    for line in Path(path).expanduser().open(encoding="utf-8"):
        if line.startswith("#") or "\t" not in line:
            continue
        key, value = line.rstrip("\n").split("\t")
        variants = value.split()
        if variants and variants[0] != key:
            t2tw.setdefault(key, variants[0])
        for tw in variants:
            if tw != key and tw not in tw2t and key != "樑":
                tw2t[tw] = key
    _T2TW, _TW2T = t2tw, tw2t
    return t2tw, tw2t


def to_t(value: str) -> str:
    return "".join(_TW2T.get(ch, ch) for ch in value)


def to_tw(value: str) -> str:
    return "".join(_T2TW.get(ch, ch) for ch in value)


def load_char_readings(rime_dir: str | Path) -> dict[str, dict[str, float]]:
    readings: dict[str, dict[str, float]] = collections.defaultdict(dict)
    body = False
    source = Path(rime_dir).expanduser() / "terra_pinyin.dict.yaml"
    for line in source.open(encoding="utf-8"):
        if line.startswith("..."):
            body = True
            continue
        if not body or line.startswith("#") or "\t" not in line:
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts[0]) != 1 or len(parts) < 2:
            continue
        weight = 100.0
        if len(parts) >= 3 and parts[2]:
            raw_weight = parts[2].strip()
            try:
                weight = float(raw_weight[:-1] if raw_weight.endswith("%") else raw_weight)
            except ValueError:
                weight = 100.0
        code = parts[1].strip()
        readings[parts[0]][code] = max(readings[parts[0]].get(code, 0.0), weight)
    return readings


def load_userdb(userdb_tsv: str | Path) -> list[tuple[str, str, int]]:
    result = []
    with Path(userdb_tsv).expanduser().open(encoding="utf-8") as source:
        for line in source:
            word, code, commits = line.rstrip("\n").split("\t")
            result.append((word, code, int(commits)))
    return result


def load_essay(essay_path: str | Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with Path(essay_path).expanduser().open(encoding="utf-8") as source:
        for line in source:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                try:
                    result[parts[0]] = int(parts[1])
                except ValueError:
                    continue
    return result


def _connect_readonly(path: str | Path) -> sqlite3.Connection:
    absolute = Path(path).expanduser().resolve()
    return sqlite3.connect(f"file:{quote(str(absolute))}?mode=ro", uri=True)


def voco_rows(store_path: str | Path) -> list[tuple[int, str]]:
    with _connect_readonly(store_path) as connection:
        rows = connection.execute(
            "select Z_PK, coalesce(ZFINALPASTEDTEXT, ZTEXT) from ZTRANSCRIPTION"
        ).fetchall()
    return [(pk, text) for pk, text in rows if text]


def word_replacements(dictionary_store_path: str | Path) -> list[tuple[str, str, str]]:
    with _connect_readonly(dictionary_store_path) as connection:
        return connection.execute(
            "select ZORIGINALTEXT, ZREPLACEMENTTEXT, ZSOURCE from ZWORDREPLACEMENT "
            "where ZSOURCE in ('editMode','correctionFeedback')"
        ).fetchall()


def control_gold(
    model_path: str | Path,
    evidence_path: str | Path,
) -> tuple[str, list[tuple[str, str, str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    model = json.loads(Path(model_path).expanduser().read_text(encoding="utf-8"))
    pairs = []
    negatives = []
    confirmed = []
    for policy in model["policies"]:
        if policy.get("autoApplyMode") != "apply":
            continue
        if policy.get("policyType") == "exactTrainablePair" and policy.get("inputText"):
            pairs.append((policy["inputText"], policy["targetText"], "exact", policy["policyId"]))
        else:
            pairs.append((policy["sourcePattern"], policy["targetText"], "scoped", policy["policyId"]))
        for guard in policy.get("negativeSourceGuards") or []:
            if guard.get("text"):
                negatives.append((guard["text"], policy["policyId"]))
    with Path(evidence_path).expanduser().open(encoding="utf-8") as source:
        for line in source:
            event = json.loads(line)
            examples = ((event.get("payload") or {}).get("examples") or {})
            for positive in examples.get("positive") or []:
                if positive.get("expectedText"):
                    confirmed.append((positive["expectedText"], event["eventId"]))
            for negative in examples.get("negative") or []:
                if negative.get("text"):
                    negatives.append((negative["text"], event["eventId"]))
    return model["autoApplyModelVersion"], pairs, negatives, confirmed
