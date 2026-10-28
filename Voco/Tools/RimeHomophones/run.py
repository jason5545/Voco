"""跑整個 Voco 歷史＋標準答案；資料路徑由參數或 VOCO_RIME_* 環境變數提供。"""
import argparse, sys, os, json, random, time, collections, difflib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (configure_variants, load_char_readings, load_userdb, load_essay,
                    voco_rows, word_replacements, control_gold, is_cjk, required_path)
from layer import Layer

CONFIGS = {
    # 名稱: Layer 參數
    "raw":        dict(guard_typed=False, guard_unique=False),
    "g":          dict(),                                            # 原文沒被 Jason 打過 + 唯一勝出
    "g_L3":       dict(min_len=3),
    "g_L3_c2":    dict(min_len=3, min_commits=2),
    "g_L3_ov":    dict(min_len=3, guard_typed_overlap=True),
    "g_L3_c2_ov": dict(min_len=3, min_commits=2, guard_typed_overlap=True),
    "g_L4_c2_ov": dict(min_len=4, min_commits=2, guard_typed_overlap=True),
    "g_notone":   dict(tone=False),
    "g_L3_notone": dict(min_len=3, tone=False),
    "g_L3_r5":    dict(min_len=3, min_reading_pct=5.0),
    "g_L3_c2_ov_r5": dict(min_len=3, min_commits=2, guard_typed_overlap=True, min_reading_pct=5.0),
    "s":          dict(seg_align=True),
    "s_ne":       dict(seg_align=True, essay_max=0),
    "s_ne_L3":    dict(seg_align=True, essay_max=0, min_len=3),
    "s_ne_L3_c2": dict(seg_align=True, essay_max=0, min_len=3, min_commits=2),
    "s_ne_L4":    dict(seg_align=True, essay_max=0, min_len=4),
    "s_ne_L4_c2": dict(seg_align=True, essay_max=0, min_len=4, min_commits=2),
    "s_ne_L3_notone": dict(seg_align=True, essay_max=0, min_len=3, tone=False),
    "s_ne_L3_c2_ov": dict(seg_align=True, essay_max=0, min_len=3, min_commits=2, guard_typed_overlap=True),
    "x_L2":    dict(seg_align=True, essay_max=0, min_len=2, min_reading_pct=5.0, exclude=True),
    "x_L2_c2": dict(seg_align=True, essay_max=0, min_len=2, min_commits=2, min_reading_pct=5.0, exclude=True),
    "x_L3":    dict(seg_align=True, essay_max=0, min_len=3, min_reading_pct=5.0, exclude=True),
    "x_L3_c2": dict(seg_align=True, essay_max=0, min_len=3, min_commits=2, min_reading_pct=5.0, exclude=True),
    "x_L3_c3": dict(seg_align=True, essay_max=0, min_len=3, min_commits=3, min_reading_pct=5.0, exclude=True),
    "x_L4_c2": dict(seg_align=True, essay_max=0, min_len=4, min_commits=2, min_reading_pct=5.0, exclude=True),
    "x_L3_c2_notone": dict(seg_align=True, essay_max=0, min_len=3, min_commits=2, min_reading_pct=5.0, exclude=True, tone=False),
    "x_L3_c2_r0": dict(seg_align=True, essay_max=0, min_len=3, min_commits=2, min_reading_pct=0.0, exclude=True),
    "x_L3_c2_noexcl": dict(seg_align=True, essay_max=0, min_len=3, min_commits=2, min_reading_pct=5.0),
    "x_L3_c2_noseg": dict(essay_max=0, min_len=3, min_commits=2, min_reading_pct=5.0, exclude=True),
    "x_L3_c2_noessay": dict(seg_align=True, min_len=3, min_commits=2, min_reading_pct=5.0, exclude=True),
    "x_L3_c2_r15": dict(seg_align=True, essay_max=0, min_len=3, min_commits=2, min_reading_pct=15.0, exclude=True),
}

def s2twp_rows(rows, cache=None):
    import subprocess, hashlib
    key = hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()
    if cache and os.path.exists(cache):
        d = json.load(open(cache))
        if d.get("key") == key:
            return [tuple(x) for x in d["rows"]]
    payload = "\n".join(json.dumps(t, ensure_ascii=False) for _, t in rows)
    opencc = os.environ.get("VOCO_RIME_OPENCC") or "opencc"
    out = subprocess.run([opencc, "-c", "s2twp.json"], input=payload, capture_output=True, text=True, check=True).stdout
    conv = [json.loads(l) for l in out.split("\n") if l.strip()]
    assert len(conv) == len(rows), (len(conv), len(rows))
    res = [(pk, t2) for (pk, _), t2 in zip(rows, conv)]
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        temporary = cache + ".tmp"
        with open(temporary, "w", encoding="utf-8") as output:
            json.dump({"key": key, "rows": res}, output, ensure_ascii=False)
        os.replace(temporary, cache)
    return res

def s2twp_list(texts):
    import subprocess
    payload = "\n".join(json.dumps(t, ensure_ascii=False) for t in texts)
    opencc = os.environ.get("VOCO_RIME_OPENCC") or "opencc"
    out = subprocess.run([opencc, "-c", "s2twp.json"], input=payload, capture_output=True, text=True, check=True).stdout
    conv = [json.loads(l) for l in out.split("\n") if l.strip()]
    assert len(conv) == len(texts)
    return conv

def diff_spans(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [(a[i1:i2], b[j1:j2]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", choices=CONFIGS)
    parser.add_argument("--rime-dir")
    parser.add_argument("--userdb-tsv")
    parser.add_argument("--essay")
    parser.add_argument("--tw-variants")
    parser.add_argument("--voco-store")
    parser.add_argument("--dictionary-store")
    parser.add_argument("--auto-apply-model")
    parser.add_argument("--evidence-jsonl")
    parser.add_argument("--rows-json")
    parser.add_argument("--s2twp-cache")
    parser.add_argument("--wordfreq")
    parser.add_argument("--dump")
    parser.add_argument("--defer", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--no-s2twp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.config
    rime_dir = required_path(args.rime_dir, "VOCO_RIME_DIR", "Rime 副本資料夾")
    userdb_tsv = required_path(args.userdb_tsv, "VOCO_RIME_USERDB_TSV", "Rime 學習紀錄 TSV")
    essay_path = required_path(args.essay, "VOCO_RIME_ESSAY", "essay 詞表")
    variants_path = required_path(args.tw_variants, "VOCO_RIME_TW_VARIANTS", "TWVariants")
    voco_store = required_path(args.voco_store, "VOCO_RIME_VOCO_STORE", "Voco 資料庫")
    dictionary_store = required_path(args.dictionary_store, "VOCO_RIME_DICTIONARY_STORE", "Voco 字典資料庫")
    model_path = required_path(args.auto_apply_model, "VOCO_RIME_AUTO_APPLY_MODEL", "auto-apply model")
    evidence_path = required_path(args.evidence_jsonl, "VOCO_RIME_EVIDENCE_JSONL", "auto-apply evidence")
    rows_json = args.rows_json or os.environ.get("VOCO_RIME_ROWS_JSON")
    cache_path = args.s2twp_cache or os.environ.get("VOCO_RIME_S2TWP_CACHE")
    wordfreq_path = args.wordfreq or os.environ.get("VOCO_RIME_WORDFREQ")
    configure_variants(variants_path)
    userdb = load_userdb(userdb_tsv)
    char_readings = load_char_readings(rime_dir)
    t0 = time.time()
    lex = load_essay(essay_path)
    if wordfreq_path:
        from common import to_t as _tt
        lex = {}
        with open(wordfreq_path, encoding="utf-8") as source:
            for line in source:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2 and p[1].isdigit():
                    k = _tt(p[0]); lex[k] = max(lex.get(k, 0), int(p[1]))
        print("lexicon=Voco word_freq.tsv", len(lex))
    if args.defer:
        _m = json.load(open(model_path, encoding="utf-8"))
        CONFIGS[name] = dict(CONFIGS[name], defer_patterns=sorted({p["sourcePattern"] for p in _m["policies"]
            if p.get("autoApplyMode") == "apply" and p.get("policyType") == "scopedReplacement" and p.get("contextRequired") and p.get("sourcePattern")}))
    if args.hybrid:
        layer = Layer(userdb, char_readings, load_essay(essay_path), **CONFIGS[name])
        layer.seg_lex = lex; layer.seg_total = float(sum(lex.values()))
        print("hybrid: guard=essay, segmentation=word_freq")
    else:
        layer = Layer(userdb, char_readings, lex, **CONFIGS[name])
    t_load = time.time() - t0
    if rows_json:
        with open(rows_json, encoding="utf-8") as source:
            value = json.load(source)
        rows = value.get("rows") if isinstance(value, dict) else value
        rows = [(int(pk), text) for pk, text in rows if text]
    else:
        rows = voco_rows(voco_store)
    if not args.no_s2twp:
        rows = s2twp_rows(rows, cache_path)
    t1 = time.time()
    changes = []   # (pk, start, src, dst, commits, text)
    changed_rows = set()
    for pk, text in rows:
        new, acc = layer.apply(text)
        for i, L, s, d, c in acc:
            changes.append((pk, i, s, d, c, text))
            changed_rows.add(pk)
    t_run = time.time() - t1
    total_chars = sum(len(t) for _, t in rows)
    print(f"== config {name} {layer.cfg}")
    print(f"rows={len(rows)} chars={total_chars} load={t_load:.2f}s run={t_run:.2f}s per_row={t_run/len(rows)*1000:.3f}ms per_100chars={t_run/total_chars*100*1000:.3f}ms")
    print(f"changed_rows={len(changed_rows)} changed_spans={len(changes)}")
    pair_counts = collections.Counter((s, d) for _, _, s, d, _, _ in changes)
    print(f"distinct_pairs={len(pair_counts)}")
    print("top_pairs:", "; ".join(f"{s}→{d}×{n}" for (s, d), n in pair_counts.most_common(25)))

    # ---- 一定要過的兩條 ----
    fz = [(pk, t) for pk, t in rows if "粉絲專業" in t]
    fz_occ = sum(t.count("粉絲專業") for _, t in fz)
    fz_fixed = 0
    print(f"-- 粉絲專業 rows={len(fz)} occ={fz_occ}")
    for pk, t in fz:
        new, acc = layer.apply(t)
        fixed = new.count("粉絲專頁") - t.count("粉絲專頁")
        fz_fixed += fixed
        print(f"  pk={pk} fixed={fixed}/{t.count('粉絲專業')} | {t}\n     → {new}")
    print(f"粉絲專業 fixed occ = {fz_fixed}/{fz_occ}")
    zy_other_total = 0
    zy_other_changed = 0
    for pk, t in rows:
        if "專業" not in t:
            continue
        new, acc = layer.apply(t)
        # 「專業」的位置（扣掉粉絲專業裡的）
        idx = [i for i in range(len(t) - 1) if t[i:i + 2] == "專業" and not (i >= 2 and t[i - 2:i] == "粉絲")]
        zy_other_total += len(idx)
        for i in idx:
            if any(a <= i + 1 and i < a + L for a, L, s, d, c in acc):
                zy_other_changed += 1
                print(f"  專業 touched pk={pk}: {t}")
    print(f"其他「專業」 total={zy_other_total} changed={zy_other_changed}")

    # ---- 隨機 30 處 ----
    rnd = random.Random(20261028)
    sample = rnd.sample(changes, min(30, len(changes)))
    print("-- random 30 changes (seed 20261028)")
    for pk, i, s, d, c, t in sample:
        ctx = t[max(0, i - 12):i] + "【" + s + "→" + d + "】" + t[i + len(s):i + len(s) + 12]
        print(f"  pk={pk} commits={c} {ctx}")

    # ---- 標準答案 ----
    wr = word_replacements(dictionary_store)
    ver, pairs, negatives, confirmed = control_gold(model_path, evidence_path)
    if not args.no_s2twp:
        flat = [x for o, rr, s in wr for x in (o, rr)] + [x for p in pairs for x in p[:2]] + [t for t, _ in negatives] + [t for t, _ in confirmed]
        conv = iter(s2twp_list(flat))
        wr = [(next(conv), next(conv), s) for o, rr, s in wr]
        pairs = [(next(conv), next(conv), *p[2:]) for p in pairs]
        negatives = [(next(conv), e) for t, e in negatives]
        confirmed = [(next(conv), e) for t, e in confirmed]
    def eval_pairs(label, plist):
        reproduce = harm = applicable = 0
        rep_list, harm_list, wrong_list = [], [], []
        wrong = 0
        for src, tgt, *meta in plist:
            gold = diff_spans(src, tgt)
            new, acc = layer.apply(src)
            mine = diff_spans(src, new)
            if gold and any(all(is_cjk(c) for c in g[0] + g[1]) and len(g[0]) == len(g[1]) and g[0] for g in gold):
                applicable += 1
            if mine:
                if new == tgt or (mine and all(m in gold for m in mine)):
                    reproduce += 1; rep_list.append((src, tgt, new))
                else:
                    wrong += 1; wrong_list.append((src, tgt, new))
            new2, acc2 = layer.apply(tgt)
            if new2 != tgt:
                harm += 1; harm_list.append((tgt, new2))
        print(f"-- gold {label}: pairs={len(plist)} same-length-CJK-diff pairs={applicable} reproduced={reproduce} changed-but-not-gold={wrong} harmed-confirmed-target={harm}")
        for x in rep_list[:20]: print("   REPRO", x)
        for x in wrong_list[:20]: print("   WRONG", x)
        for x in harm_list[:30]: print("   HARM", x)
    eval_pairs("ZWORDREPLACEMENT editMode/correctionFeedback", [(o, r) for o, r, s in wr])
    eval_pairs(f"auto-apply exact pairs ({ver})", [p for p in pairs if p[2] == "exact"])
    eval_pairs(f"auto-apply scoped/replacement rules ({ver})", [p for p in pairs if p[2] == "scoped"])
    h = [(t, layer.apply(t)[0]) for t, _ in confirmed]
    h = [(a, b) for a, b in h if a != b]
    print(f"-- evidence.jsonl positive expectedText (Jason 確認的正確句子) n={len(confirmed)} changed={len(h)}")
    for x in h[:30]: print("   HARM", x)
    hn = [(t, layer.apply(t)[0]) for t, _ in negatives]
    hn = [(a, b) for a, b in hn if a != b]
    print(f"-- negative examples/guards (不該改) n={len(negatives)} changed={len(hn)}")
    for x in hn[:30]: print("   NEG", x)
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as output:
            json.dump({"config": name, "changes": [(pk, i, s, d, c) for pk, i, s, d, c, _ in changes]}, output, ensure_ascii=False)

if __name__ == "__main__":
    main()
