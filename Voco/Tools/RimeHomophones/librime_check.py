"""用 Rime 副本的 librime 檢查候選；所有資料路徑由參數或環境變數提供。"""

import argparse
import json
import os
import re
import sys
import time


XFORMS = [(r"^m(\d)$", r"mu\1"), (r"^r5$", "er5"), ("iu", "iou"), ("ui", "uei"), ("ong", "ung"),
          (r"^yi?", "i"), (r"^wu?", "u"), ("iu", "v"), (r"^([jqx])u", r"\1v"), (r"([iuv])n", r"\1en"),
          (r"^zhi?", "Z"), (r"^chi?", "C"), (r"^shi?", "S"), (r"^([zcsr])i", r"\1"), ("ai", "A"),
          ("ei", "I"), ("ao", "O"), ("ou", "U"), ("ang", "K"), ("eng", "G"), ("an", "M"), ("en", "N"),
          ("er", "R"), ("eh", "E"), (r"([iv])e", r"\1E")]
TR = str.maketrans("bpmfdtnlgkhjqxZCSrzcsiuvaoeEAIOUMNKGR12345", "1qaz2wsxedcrfv5tgbyhnujm8ik,9ol.0p;/- 6347")
ZHUYIN = "ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ"
ZHUYIN_TONES = "ˊˇˋ˙"
ZHUYIN_KEYS = str.maketrans(ZHUYIN, "1qaz2wsxedcrfv5tgbyhnujm8ik,9ol.0p;/-")
TONE_KEYS = {"": " ", "ˊ": "6", "ˇ": "3", "ˋ": "4", "˙": "7"}
ZHUYIN_SYLLABLE = re.compile(f"([{ZHUYIN}]+)([{ZHUYIN_TONES}]?)")


def keys_for(syllables, profile):
    output = []
    if profile.code_format == "zhuyin":
        for syllable in syllables:
            if syllable.endswith("~"):
                body = syllable[:-1]
                if not body or any(char not in ZHUYIN for char in body):
                    return None
                output.append(body.translate(ZHUYIN_KEYS) + "~")
                continue
            match = ZHUYIN_SYLLABLE.fullmatch(syllable)
            if not match:
                return None
            body, tone = match.groups()
            output.append(body.translate(ZHUYIN_KEYS) + TONE_KEYS[tone])
        return "".join(output).replace(" ", "{space}") if output else None

    for syllable in syllables:
        for pattern, replacement in XFORMS:
            syllable = re.sub(pattern, replacement, syllable)
        output.append(syllable.translate(TR))
    return "".join(output).replace(" ", "{space}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("output")
    parser.add_argument("--rime-dir")
    parser.add_argument("--userdb-tsv")
    parser.add_argument("--tw-variants")
    parser.add_argument("--rimeh-dir")
    args = parser.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import active_rime_profile, configure_variants, load_char_readings, load_userdb, required_path, to_t

    rime_dir = required_path(args.rime_dir, "VOCO_RIME_DIR", "Rime 副本資料夾")
    profile = active_rime_profile(rime_dir)
    userdb_tsv = required_path(args.userdb_tsv, "VOCO_RIME_USERDB_TSV", "Rime 學習紀錄 TSV")
    variants_path = required_path(args.tw_variants, "VOCO_RIME_TW_VARIANTS", "TWVariants")
    rimeh_dir = required_path(args.rimeh_dir, "VOCO_RIME_PENGLAI_TOOLS", "rime-penglai 工具資料夾")
    configure_variants(variants_path)
    sys.path.insert(0, str(rimeh_dir))
    from rimeh import Rime

    with open(args.dump, encoding="utf-8") as source:
        dump = json.load(source)
    readings = load_char_readings(rime_dir, profile)
    top_reading = {char: max(values.items(), key=lambda item: item[1])[0] for char, values in readings.items()}
    typed_readings = {}
    for word, code, commits in load_userdb(userdb_tsv):
        if commits > 0:
            typed_readings.setdefault(word, code.split())

    rime = Rime(str(rime_dir), tw=True)
    result = []
    started = time.time()
    query_count = 0
    for item in dump["items"]:
        syllables = item.get("syls")
        if syllables is None:
            word = to_t(item["dst"])
            syllables = typed_readings.get(word)
            if syllables is None:
                syllables = [top_reading.get(char) for char in word]
        left = [top_reading.get(char) for char in to_t(item.get("ctxL", ""))]
        right = [top_reading.get(char) for char in to_t(item.get("ctxR", ""))]
        all_syllables = left + list(syllables) + right
        if any(syllable is None for syllable in all_syllables):
            item["top1"] = None
            result.append(item)
            continue
        key_sequence = keys_for(all_syllables, profile)
        if key_sequence is None:
            item["keys"] = None
            item["top1"] = None
            result.append(item)
            continue
        rime.clear()
        rime.keys(key_sequence)
        query_count += 1
        candidates = rime.cands()
        item["keys"] = key_sequence
        item["top1"] = candidates[0] if candidates else None
        item["top5"] = candidates[:5]
        result.append(item)

    elapsed = time.time() - started
    rime.clear()
    rime.close()
    temporary = args.output + ".tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump({"queries": query_count, "seconds": elapsed, "items": result}, output, ensure_ascii=False)
    os.replace(temporary, args.output)
    print(f"queries={query_count} seconds={elapsed:.2f} per_query_ms={elapsed / max(query_count, 1) * 1000:.2f}")


if __name__ == "__main__":
    main()
