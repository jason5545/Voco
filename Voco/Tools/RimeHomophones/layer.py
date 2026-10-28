"""Rime 學習紀錄重挑同音字：離線版規則（第 0 步模擬用，不是 Voco 程式）。

做法：Voco 文字逐字取 terra_pinyin（臺灣讀音版）的讀音；跟 Jason 在鼠鬚管送出過的詞（userdb commits > 0）
同字數、讀音完全相同（預設比聲調）但字不同時，候選換成 Jason 打過的那個詞。
"""
import collections
from common import is_cjk, to_t, to_tw

def strip_tone(code):
    return code[:-1] if code and code[-1].isdigit() else code

class Layer:
    def __init__(self, userdb, char_rd, essay, *, min_len=2, max_len=10, min_commits=1,
                 tone=True, min_reading_pct=0.0, guard_typed=True, guard_unique=True,
                 guard_typed_overlap=False, essay_max=None, max_diff_chars=None, seg_align=False, exclude=False, defer_patterns=None):
        self.cfg = dict(min_len=min_len, max_len=max_len, min_commits=min_commits, tone=tone,
                        min_reading_pct=min_reading_pct, guard_typed=guard_typed, guard_unique=guard_unique,
                        guard_typed_overlap=guard_typed_overlap, essay_max=essay_max, max_diff_chars=max_diff_chars,
                        seg_align=seg_align, exclude=exclude, defer=bool(defer_patterns))
        self.defer_patterns = [x for x in (defer_patterns or []) if x]
        self.essay = essay
        self.seg_lex = essay
        self.essay_total = float(sum(essay.values())) or 1.0
        norm = (lambda c: c) if tone else strip_tone
        self.norm = norm
        # 讀音表（只留比重 >= min_reading_pct 的讀音）
        self.rd = {}
        for ch, m in char_rd.items():
            s = {norm(c) for c, w in m.items() if w >= min_reading_pct}
            if not s:
                s = {norm(c) for c in m}
            self.rd[ch] = s
        # Jason 打過的詞：OpenCC 字形
        self.typed = collections.Counter()
        self.bucket = collections.defaultdict(dict)    # (L, code tuple) -> {word: commits}
        self.by_first = collections.defaultdict(list)  # (L, first syl) -> [(word, codes)]
        seen = set()
        for w, code, n in userdb:
            if n <= 0:
                continue
            self.typed[w] += n
            if not all(is_cjk(ch) for ch in w):
                continue
            syl = code.split()
            if len(syl) != len(w) or not (min_len <= len(w) <= max_len):
                continue
            key = tuple(norm(s) for s in syl)
            b = self.bucket[(len(w), key)]
            b[w] = b.get(w, 0) + n
            if (w, key) not in seen:
                seen.add((w, key))
                self.by_first[(len(w), key[0])].append((w, key))
        self.typed_set = {w for w in self.typed if len(w) >= 2}

    def boundaries(self, T):
        """essay 詞頻 + Jason 打過的詞，最大機率斷詞；回傳邊界位置集合"""
        import math
        n = len(T)
        best = [(-1e18, 0)] * (n + 1)
        best[0] = (0.0, 0)
        logt = math.log(self.seg_total if hasattr(self, "seg_total") else self.essay_total)
        for i in range(n):
            if best[i][0] <= -1e17:
                continue
            for L in range(1, min(8, n - i) + 1):
                w = T[i:i + L]
                if L == 1:
                    f = self.seg_lex.get(w, 0) + 1
                else:
                    f = self.seg_lex.get(w, 0) + (self.typed.get(w, 0) and 1000)
                    if f <= 0:
                        continue
                s = best[i][0] + math.log(f) - logt
                if s > best[i + L][0]:
                    best[i + L] = (s, L)
        b = {0, n}
        j = n
        while j > 0:
            L = best[j][1] or 1
            j -= L
            b.add(j)
        return b

    PRON = [set("你妳您"), set("他她它祂牠")]
    PARTICLE = set("的地得底")
    VARIANT = {frozenset("台臺"), frozenset("着著"), frozenset("裏裡"), frozenset("峯峰"), frozenset("綫線")}

    def excluded_pair(self, S, w):
        """語意差異不交給讀音層：人稱代名詞、的地得、異體字"""
        S2, w2 = to_tw(S), to_tw(w)
        for a, b in zip(S2, w2):
            if a == b:
                continue
            if any(a in g and b in g for g in self.PRON):
                return "pronoun"
            if a in self.PARTICLE or b in self.PARTICLE:
                return "particle"
            if frozenset((a, b)) in self.VARIANT:
                return "variant"
        return None

    def readings(self, ch):
        return self.rd.get(ch, ())

    def candidates(self, text_tw):
        """回傳 [(start, length, src_tw, dst_tw, commits, reason)]，已處理重疊（長的優先）"""
        T = to_t(text_tw)
        n = len(T)
        cands = []
        for i in range(n):
            if not is_cjk(T[i]):
                continue
            for r in self.readings(T[i]):
                for L in range(self.cfg["min_len"], self.cfg["max_len"] + 1):
                    if i + L > n:
                        break
                    lst = self.by_first.get((L, r))
                    if not lst:
                        continue
                    S = T[i:i + L]
                    if not all(is_cjk(c) for c in S):
                        continue
                    for w, key in lst:
                        if w == S:
                            continue
                        ok = True
                        for k in range(1, L):
                            if key[k] not in self.readings(S[k]):
                                ok = False
                                break
                        if ok:
                            cands.append((i, L, S, w, key))
        # 決策
        out = []
        dedup = set()
        for i, L, S, w, key in cands:
            if (i, L, w) in dedup:
                continue
            dedup.add((i, L, w))
            if to_tw(S) == to_tw(w):
                continue
            b = self.bucket[(L, key)]
            c = b.get(w, 0)
            if c < self.cfg["min_commits"]:
                continue
            if self.cfg["guard_typed"] and S in self.typed_set:
                continue
            if self.cfg["guard_unique"]:
                if any(v >= c for ww, v in b.items() if ww != w):
                    continue
            if self.cfg["essay_max"] is not None and self.essay.get(S, 0) > self.cfg["essay_max"]:
                continue
            if self.cfg["exclude"] and self.excluded_pair(S, w):
                continue
            if self.cfg["max_diff_chars"] is not None and sum(a != bch for a, bch in zip(S, w)) > self.cfg["max_diff_chars"]:
                continue
            out.append((i, L, S, w, c))
        if self.cfg["seg_align"] and out:
            B = self.boundaries(T)
            out = [o for o in out if o[0] in B and (o[0] + o[1]) in B]
        # Jason 打過、而且跟候選重疊、長度 >= 候選的原文片段：擋掉
        if self.cfg["guard_typed_overlap"] and out:
            protected = []
            for i in range(n):
                for L in range(2, self.cfg["max_len"] + 1):
                    if i + L > n:
                        break
                    if T[i:i + L] in self.typed_set:
                        protected.append((i, L))
            out = [o for o in out if not any(pi < o[0] + o[1] and o[0] < pi + pl and pl >= o[1] for pi, pl in protected)]
        # auto-apply 有語境鎖的 sourcePattern：讓給 auto-apply 決定
        if self.defer_patterns and out:
            S_tw = text_tw
            out = [o for o in out if not any(dp in S_tw[max(0, o[0] - len(dp) + 1):o[0] + o[1] + len(dp) - 1] and
                                             any(S_tw.find(dp, k) != -1 and S_tw.find(dp, k) < o[0] + o[1] and S_tw.find(dp, k) + len(dp) > o[0]
                                                 for k in [max(0, o[0] - len(dp) + 1)])
                                             for dp in self.defer_patterns)]
        # 重疊：長的優先，再比 commits
        out.sort(key=lambda x: (-x[1], -x[4], x[0]))
        taken = [False] * n
        acc = []
        for i, L, S, w, c in out:
            if any(taken[i:i + L]):
                continue
            for k in range(i, i + L):
                taken[k] = True
            acc.append((i, L, text_tw[i:i + L], to_tw(w), c))
        acc.sort()
        return acc

    def apply(self, text_tw):
        acc = self.candidates(text_tw)
        if not acc:
            return text_tw, acc
        out, cur = [], 0
        for i, L, s, d, c in acc:
            out.append(text_tw[cur:i]); out.append(d); cur = i + L
        out.append(text_tw[cur:])
        return "".join(out), acc
