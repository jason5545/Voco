#!/usr/bin/env python3
"""由 A oracle 與 rime-homophone 紀錄挖 B 規則，先驗收再呼叫 control CLI。"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import configure_variants, required_path, voco_rows
from layer import Layer
from run import CONFIGS, s2twp_rows


MINIMUM_OCCURRENCES = 2
EXCLUDED_PAIRS = {("有問題嘛", "有問題嗎")}
BOUNDARY_GUARDED_PAIRS = {"併沒有", "怎麼作", "作了什麼"}
REQUIRED_NEGATIVES = {
    "併沒有": ["合併沒有成功", "一併沒有成功", "吞併沒有成功", "兼併沒有成功"],
    "怎麼作": ["怎麼作業", "怎麼作為"],
    "作了什麼": [
        "工作了什麼", "操作了什麼", "動作了什麼", "合作了什麼", "創作了什麼", "製作了什麼"
    ],
}


def load_control_cli(path: Path):
    spec = importlib.util.spec_from_file_location("voco_auto_apply_control", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"無法載入 auto-apply control CLI：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_store_rows_and_rime_records(store_path: Path):
    absolute = store_path.resolve()
    connection = sqlite3.connect(f"file:{quote(str(absolute))}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "select Z_PK, coalesce(ZFINALPASTEDTEXT, ZTEXT) from ZTRANSCRIPTION"
        ).fetchall()
        records = connection.execute(
            "select Z_PK, ZCANONICALIZATIONREPLACEMENTSJSON from ZTRANSCRIPTION "
            "where ZCANONICALIZATIONREPLACEMENTSJSON is not null"
        ).fetchall()
    finally:
        connection.close()
    return [(int(pk), text) for pk, text in rows if text], records


def learned_inputs(generated: dict):
    userdb = [
        (entry["word"], " ".join(entry["syllables"]), int(entry["commits"]))
        for entry in generated["learnedWords"]
    ]
    readings = {
        character: {item["syllable"]: float(item["weightPercent"]) for item in values}
        for character, values in generated["characterReadings"].items()
    }
    return userdb, readings, generated["essayFrequencies"]


def oracle_changes(rows, generated, active_model, control):
    userdb, readings, essay = learned_inputs(generated)
    deferred = sorted({
        str(policy.get("sourcePattern"))
        for policy in active_model.get("policies", [])
        if policy.get("autoApplyMode") == "apply"
        and policy.get("policyType") == "scopedReplacement"
        and policy.get("contextRequired") is True
        and policy.get("sourcePattern")
    })
    layer = Layer(userdb, readings, essay, **dict(CONFIGS["x_L3_c2"], defer_patterns=deferred))
    changes = []
    for pk, text in rows:
        _output, fires = layer.apply(text)
        for offset, _length, source, target, commits in fires:
            changes.append((pk, offset, source, target, commits))
    return layer, changes


def rime_record_counts(records):
    counts = collections.Counter()
    examples = collections.defaultdict(list)
    for pk, raw in records:
        try:
            values = json.loads(raw or "[]")
        except json.JSONDecodeError:
            continue
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict) or item.get("reason") != "rime-homophone":
                continue
            source = str(item.get("originalText") or "")
            target = str(item.get("replacementText") or "")
            if source and target:
                counts[(source, target)] += 1
                examples[(source, target)].append(int(pk))
    return counts, examples


def active_literal_policies(model: dict, candidate_pairs: set[tuple[str, str]], events):
    selected = []
    covered = set()
    regex_sources = {
        (str(payload.get("sourcePattern") or ""), str(payload.get("targetText") or "")): str(payload.get("sourceText") or "")
        for event in events
        if event.get("action") == "addReplacementRule"
        for payload in [event.get("payload") if isinstance(event.get("payload"), dict) else {}]
        if payload.get("sourcePatternType") == "regex" and payload.get("sourceText") and payload.get("targetText")
    }
    for policy in model.get("policies", []):
        if policy.get("autoApplyMode") != "apply":
            continue
        pattern = str(policy.get("sourcePattern") or "")
        target = str(policy.get("targetText") or "")
        if policy.get("sourcePatternType") == "regex":
            source = regex_sources.get((pattern, target), "")
        else:
            source = pattern or str(policy.get("inputText") or "")
        pair = (source, target)
        if not source:
            continue
        if pair in candidate_pairs:
            selected.append(policy)
            covered.add(pair)
    return selected, covered


def boundary_words(source: str, essay: dict[str, int], taiwan_to_opencc: dict[str, str], opencc_to_taiwan: dict[str, str]):
    opencc_source = "".join(taiwan_to_opencc.get(char, char) for char in source)
    left = []
    right = []
    for word, frequency in essay.items():
        if len(word) != 2 or frequency <= 0:
            continue
        if word[-1] == opencc_source[0]:
            left.append((word, int(frequency)))
        if word[0] == opencc_source[-1]:
            right.append((word, int(frequency)))
    left.sort(key=lambda item: (-item[1], item[0]))
    right.sort(key=lambda item: (-item[1], item[0]))
    left = left[:8]
    right = right[:8]
    left_tw = [("".join(opencc_to_taiwan.get(char, char) for char in word), count) for word, count in left]
    right_tw = [("".join(opencc_to_taiwan.get(char, char) for char in word), count) for word, count in right]
    negatives = {word + source[1:] + "成功" for word, _count in left_tw}
    negatives.update(source[:-1] + word for word, _count in right_tw)
    negatives.update(REQUIRED_NEGATIVES.get(source, []))
    return left_tw, right_tw, sorted(negatives)


def replacement_pattern(source: str, pair_source: str, left, right) -> tuple[str, str | None]:
    if pair_source not in BOUNDARY_GUARDED_PAIRS:
        return source, None
    left_chars = sorted({word[0] for word, _count in left})
    right_chars = sorted({word[1] for word, _count in right})
    # Include the explicitly requested boundary negatives even if an essay variant is absent.
    for negative in REQUIRED_NEGATIVES.get(pair_source, []):
        if negative.endswith(source[1:] + "成功"):
            prefix = negative[: -len(source[1:] + "成功")]
            if len(prefix) >= 2:
                left_chars.append(prefix[-1])
        if negative.startswith(source[:-1]):
            suffix = negative[len(source[:-1]):]
            if len(suffix) >= 2:
                right_chars.append(suffix[1])
    left_class = "".join(sorted(set(left_chars)))
    right_class = "".join(sorted(set(right_chars)))
    prefix = f"(?<![{left_class}])" if left_class else ""
    suffix = f"(?![{right_class}])" if right_class else ""
    return prefix + re.escape(source) + suffix, "regex"


def make_candidate_policy(control, source, target, pattern, pattern_type, negative_specs, *, policy_id):
    payload = {
        "ruleType": "unlockedReplacement",
        "sourceText": source,
        "sourcePattern": pattern,
        "targetText": target,
        "ruleName": f"rime-homophone:{source}->{target}",
        "examples": {
            "positive": [],
            "negative": [
                {"text": text, "context": context, "expectedText": text}
                for text, context in negative_specs
            ],
        },
    }
    if pattern_type == "regex":
        payload["sourcePatternType"] = "regex"
        payload["targetTemplate"] = target
    event = {"eventId": policy_id, "action": "addReplacementRule", "payload": payload}
    return control.replacement_policy_from_event(event), payload


def apply_scoped_policies(text: str, policies: list[dict], control) -> tuple[str, list[dict]]:
    output = text
    fired = []
    for policy in policies:
        if not control.policy_fires(policy, output, text):
            continue
        updated = control.replace_policy_source_for_policy(output, policy)
        updated = control.apply_policy_result_transform(updated, policy, output)
        if updated != output:
            output = updated
            fired.append(policy)
    return output, fired


def expand_guards_until_negatives_are_safe(policies: list[dict], negatives: list[str], control) -> list[dict]:
    by_id = {policy["policyId"]: policy for policy in policies}
    for _iteration in range(len(policies) + 1):
        changed = False
        for text in negatives:
            output, fired = apply_scoped_policies(text, list(by_id.values()), control)
            if output == text:
                continue
            for policy in fired:
                guards = policy.setdefault("negativeSourceGuards", [])
                if not any(guard.get("text") == text for guard in guards):
                    guards.append({"text": text, "reason": "negativeExample"})
                    changed = True
        if not changed:
            return list(by_id.values())
    raise SystemExit("邊界負例仍會被 B 候選規則改動，無法安全加入規則")


def position_map(changes):
    expected = collections.defaultdict(dict)
    for pk, offset, source, target, _commits in changes:
        if len(source) != len(target):
            raise SystemExit(f"oracle 有不同字數的規則，無法逐字對拍：{source}→{target}")
        for index, (before, after) in enumerate(zip(source, target)):
            if before != after:
                expected[int(pk)][int(offset) + index] = after
    return expected


def simulate_b(rows, policies, expected_characters, control, approved_exception_pks):
    changed_rows = set()
    changed_characters = 0
    oracle_matched_rows = set()
    mismatches = []
    extra_rows = []
    exceptions = 0
    for pk, text in rows:
        output, _fired = apply_scoped_policies(text, policies, control)
        if output == text:
            continue
        before = list(text)
        after = list(output)
        if len(before) != len(after):
            mismatches.append((pk, "character-count", len(before), len(after)))
            continue
        row_changed = False
        for offset, (source_char, target_char) in enumerate(zip(before, after)):
            if source_char == target_char:
                continue
            row_changed = True
            changed_characters += 1
            expected_char = expected_characters.get(pk, {}).get(offset)
            if expected_char == target_char:
                oracle_matched_rows.add(pk)
                continue
            if (
                pk in approved_exception_pks
                and source_char == "作"
                and target_char == "做"
            ):
                exceptions += 1
                approved_exception_pks.add(pk)
                continue
            snippet = text[max(0, offset - 8):offset + 9]
            mismatches.append((pk, offset, source_char, target_char, expected_char, snippet))
            extra_rows.append(pk)
        if row_changed:
            changed_rows.add(pk)
    return {
        "changedRows": changed_rows,
        "changedCharacters": changed_characters,
        "oracleMatchedRows": oracle_matched_rows,
        "mismatches": mismatches,
        "extraRows": set(extra_rows),
        "exceptions": exceptions,
        "approvedExceptionPks": approved_exception_pks,
    }


def load_positive_sentences(evidence_path: Path):
    sentences = []
    with evidence_path.open(encoding="utf-8") as source:
        for line in source:
            event = json.loads(line)
            payload = event.get("payload") or {}
            if str(payload.get("ruleName") or "").startswith("rime-homophone:"):
                continue
            examples = (payload.get("examples") or {})
            for positive in examples.get("positive") or []:
                text = str(positive.get("expectedText") or "")
                if text:
                    sentences.append(text)
    return sentences


def negative_examples_for_control(negatives, base_policies, protected_guards, control):
    examples = []
    for text in negatives:
        baseline, _fires = control.replay_apply_policies(text, text, base_policies, protected_guards)
        if baseline == text:
            examples.append((text, ""))
        else:
            # Keep this phrase as a context-only guard: an existing runtime restart rule
            # normalizes the synthetic sentence even when the mined B policy is blocked.
            examples.append(("", text))
    return examples


def record_counts(records):
    counts = collections.Counter()
    row_ids = collections.defaultdict(list)
    for pk, raw in records:
        try:
            replacements = json.loads(raw or "[]")
        except json.JSONDecodeError:
            continue
        if not isinstance(replacements, list):
            continue
        for replacement in replacements:
            if not isinstance(replacement, dict) or replacement.get("reason") != "rime-homophone":
                continue
            source = str(replacement.get("originalText") or "")
            target = str(replacement.get("replacementText") or "")
            if source and target:
                counts[(source, target)] += 1
                row_ids[(source, target)].append(int(pk))
    return counts, row_ids


def load_approved_exception_pks(path: Path) -> set[int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "voco.rime-homophones-approved-subset-exceptions.v1":
        raise SystemExit(f"核准例外設定 schema 不符：{path}")
    raw_pks = value.get("approvedSingleCharacterMismatchPks")
    if not isinstance(raw_pks, list) or any(not isinstance(pk, int) or isinstance(pk, bool) or pk <= 0 for pk in raw_pks):
        raise SystemExit(f"核准例外設定必須是正整數清單：{path}")
    return set(raw_pks)


def run_cli(command, *, wordfreq_path: Path, capture=True, allow_failed=False):
    environment = os.environ.copy()
    environment["VOCO_WORD_FREQ_PATH"] = str(wordfreq_path)
    result = subprocess.run(command, check=False, capture_output=capture, text=True, env=environment)
    if result.returncode != 0:
        raise SystemExit(f"control CLI 失敗（exit={result.returncode}）：{result.stderr[-1000:] if capture else ''}")
    try:
        value = json.loads(result.stdout) if capture else {}
    except json.JSONDecodeError as error:
        raise SystemExit(f"control CLI 回傳非 JSON：{error}") from error
    if value.get("failed") is True and not allow_failed:
        raise SystemExit("control CLI 回報 failed=true")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def model_rule_signatures(model: dict) -> set[tuple[str, str]]:
    return {
        (str(policy.get("sourcePattern") or ""), str(policy.get("targetText") or ""))
        for policy in model.get("policies", [])
        if isinstance(policy, dict) and policy.get("sourcePattern") and policy.get("targetText")
    }


def release_gate_failures(
    *,
    new_rules: int,
    candidate: dict,
    formal: dict,
    validation: dict,
    wrong_positions: int,
    positive_changed: int,
    negative_changed: int,
) -> list[str]:
    failures = []
    candidate_signatures = model_rule_signatures(candidate)
    formal_signatures = model_rule_signatures(formal)
    if new_rules <= 0 and candidate_signatures == formal_signatures:
        failures.append("condition1:no_new_rules_or_model_rule_changes")
    if validation.get("ready") is not True:
        failures.append("condition2:ready_is_not_true")
    failure_metrics = {
        key: value
        for key, value in validation.items()
        if "failure" in key.lower() or key.lower().endswith("conflicts")
    }
    if not failure_metrics:
        failures.append("condition2:missing_failure_metrics")
    for key, value in sorted(failure_metrics.items()):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value != 0:
            failures.append(f"condition2:{key}={value!r}")
    if wrong_positions != 0:
        failures.append(f"condition3:wrong_positions={wrong_positions}")
    if positive_changed != 0:
        failures.append(f"condition3:positive_changed={positive_changed}")
    if negative_changed != 0:
        failures.append(f"condition3:negative_changed={negative_changed}")
    missing = sorted(formal_signatures - candidate_signatures)
    if missing:
        failures.append("condition4:missing=" + json.dumps(missing, ensure_ascii=False, separators=(",", ":")))
    return failures


def run_gated_release(
    *,
    gate_failures: list[str],
    current_formal: callable,
    candidate: dict,
    activate: callable,
    publish: callable,
    log: callable,
) -> bool:
    if gate_failures:
        log("AUTO_PUBLISH_BLOCKED " + "; ".join(gate_failures))
        return False
    log("AUTO_PUBLISH_GATE_PASSED activate=1 publish=1")
    activate()
    # Re-read the installed formal model immediately before publishing. This catches
    # any local model change made while activation ran.
    latest_formal = current_formal()
    latest_candidate_signatures = model_rule_signatures(candidate)
    latest_formal_signatures = model_rule_signatures(latest_formal)
    latest_missing = sorted(latest_formal_signatures - latest_candidate_signatures)
    log(
        f"AUTO_PUBLISH_CONDITION4_PRE_PUBLISH formal_signatures={len(latest_formal_signatures)} "
        f"candidate_signatures={len(latest_candidate_signatures)} missing={len(latest_missing)}"
    )
    if latest_missing:
        log(
            "AUTO_PUBLISH_BLOCKED condition4:missing="
            + json.dumps(latest_missing, ensure_ascii=False, separators=(",", ":"))
        )
        return False
    publish()
    log("AUTO_PUBLISH_PUBLISHED")
    return True


def self_test_auto_publish_gate() -> None:
    validation = {
        "ready": True,
        "positiveFailures": 0,
        "negativeFailures": 0,
        "exactApplyConflicts": 0,
        "manualContextLockFailures": 0,
        "familyMetadataFailures": 0,
        "corpusFailures": 0,
        "failureCount": 0,
    }
    candidate = {
        "policies": [
            {"sourcePattern": "既有", "targetText": "規則"},
            {"sourcePattern": "原文", "targetText": "改字"},
        ]
    }
    formal = {"policies": [{"sourcePattern": "既有", "targetText": "規則"}]}
    actions = []
    logs = []
    allowed = release_gate_failures(
        new_rules=1, candidate=candidate, formal=formal, validation=validation,
        wrong_positions=0, positive_changed=0, negative_changed=0,
    )
    passed = run_gated_release(
        gate_failures=allowed,
        current_formal=lambda: formal,
        candidate=candidate,
        activate=lambda: actions.append("activateModel"),
        publish=lambda: actions.append("publishWorkerRelease"),
        log=logs.append,
    )
    if not passed or actions != ["activateModel", "publishWorkerRelease"]:
        raise SystemExit(f"auto-publish self-test all-pass failed: {actions!r} {logs!r}")
    print(f"SELF_TEST all_pass actions={','.join(actions)} published=true")

    actions.clear()
    logs.clear()
    formal_with_missing = {
        "policies": [
            {"sourcePattern": "既有", "targetText": "規則"},
            {"sourcePattern": "正式模型獨有", "targetText": "保留"},
        ]
    }
    blocked = release_gate_failures(
        new_rules=1, candidate=candidate, formal=formal_with_missing, validation=validation,
        wrong_positions=0, positive_changed=0, negative_changed=0,
    )
    passed = run_gated_release(
        gate_failures=blocked,
        current_formal=lambda: formal_with_missing,
        candidate=candidate,
        activate=lambda: actions.append("activateModel"),
        publish=lambda: actions.append("publishWorkerRelease"),
        log=logs.append,
    )
    if passed or actions or not any("condition4:missing" in line for line in logs):
        raise SystemExit(f"auto-publish self-test condition4-block failed: {actions!r} {logs!r}")
    print(f"SELF_TEST condition4_fail actions=none published=false log={logs[0]}")


def log_auto_publish(message: str, log_path: Path) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as output:
        output.write(line + "\n")
        output.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test-auto-publish-gate", action="store_true")
    parser.add_argument("--generated-data")
    parser.add_argument("--voco-store")
    parser.add_argument("--tw-variants")
    parser.add_argument("--active-model")
    parser.add_argument("--evidence-store")
    parser.add_argument("--control-cli")
    parser.add_argument("--wordfreq")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--s2twp-cache")
    parser.add_argument("--approved-exceptions")
    parser.add_argument("--log-path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.self_test_auto_publish_gate:
        self_test_auto_publish_gate()
        return

    generated_path = required_path(args.generated_data, "VOCO_RIME_GENERATED_DATA", "產生器資料")
    store_path = required_path(args.voco_store, "VOCO_RIME_VOCO_STORE", "Voco 資料庫")
    variants_path = required_path(args.tw_variants, "VOCO_RIME_TW_VARIANTS", "TWVariants")
    active_model_path = required_path(args.active_model, "VOCO_RIME_ACTIVE_MODEL", "active auto-apply model")
    evidence_path = required_path(args.evidence_store, "VOCO_RIME_EVIDENCE_JSONL", "auto-apply evidence")
    control_cli_path = required_path(args.control_cli, "VOCO_AUTO_APPLY_CONTROL_CLI", "auto-apply control CLI")
    wordfreq_path = required_path(args.wordfreq, "VOCO_RIME_WORDFREQ", "Voco word frequency table")
    os.environ["VOCO_WORD_FREQ_PATH"] = str(wordfreq_path)
    artifact_dir = required_path(args.artifact_dir, "VOCO_RIME_ARTIFACT_DIR", "B 模型 artifact 資料夾")
    approved_exceptions_path = required_path(
        args.approved_exceptions,
        "VOCO_RIME_APPROVED_EXCEPTIONS",
        "B oracle 子集核准例外設定",
    )
    log_path = Path(args.log_path or os.environ.get(
        "VOCO_RIME_LOG_PATH", str(Path.home() / "Library/Logs/Voco-RimeHomophones.log")
    )).expanduser()
    cache_path = args.s2twp_cache or os.environ.get("VOCO_RIME_S2TWP_CACHE")

    generated = json.loads(generated_path.read_text(encoding="utf-8"))
    active_model = json.loads(active_model_path.read_text(encoding="utf-8"))
    control = load_control_cli(control_cli_path)
    configure_variants(variants_path)
    raw_rows, transcription_records = read_store_rows_and_rime_records(store_path)
    rows = s2twp_rows(raw_rows, cache_path)
    approved_exception_pks = load_approved_exception_pks(approved_exceptions_path)
    layer, oracle = oracle_changes(rows, generated, active_model, control)
    oracle_counts = collections.Counter((item[2], item[3]) for item in oracle)
    recorded_counts, record_rows = record_counts(transcription_records)
    totals = oracle_counts + recorded_counts

    candidate_pairs = {pair for pair, count in totals.items() if count >= MINIMUM_OCCURRENCES}
    events = control.load_events(evidence_path)
    active_policies, active_pairs = active_literal_policies(active_model, candidate_pairs, events)
    base_apply_policies = [p for p in active_model.get("policies", []) if p.get("autoApplyMode") == "apply"]
    protected_guards = control.protected_term_allowlist_guards(active_model)
    existing_events_by_pair = {}
    for event in events:
        if event.get("action") != "addReplacementRule":
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        pair = (
            str(payload.get("sourceText") or payload.get("sourcePattern") or ""),
            str(payload.get("targetText") or ""),
        )
        if all(pair):
            existing_events_by_pair[pair] = event
    new_pairs = sorted(
        pair for pair in candidate_pairs
        if pair not in active_pairs and pair not in EXCLUDED_PAIRS
    )

    expected_characters = position_map(oracle)
    opencc_to_taiwan = generated["openCCToTaiwan"]
    taiwan_to_opencc = generated["taiwanToOpenCC"]
    essay = generated["essayFrequencies"]
    definitions = []
    for index, (source, target) in enumerate(new_pairs):
        left, right, negatives = boundary_words(source, essay, taiwan_to_opencc, opencc_to_taiwan)
        validation_specs = negative_examples_for_control(
            negatives, base_apply_policies, protected_guards, control
        )
        existing_event = existing_events_by_pair.get((source, target))
        if existing_event:
            payload = existing_event.get("payload") or {}
            pattern = str(payload.get("sourcePattern") or source)
            pattern_type = payload.get("sourcePatternType")
            existing_specs = sorted({
                (str(item.get("text") or ""), str(item.get("context") or ""))
                for item in ((payload.get("examples") or {}).get("negative") or [])
                if isinstance(item, dict)
            })
            needs_reissue = existing_specs != sorted(set(validation_specs))
            if needs_reissue:
                regex_policy, payload = make_candidate_policy(
                    control, source, target, pattern, pattern_type, validation_specs,
                    policy_id=f"preflight-rime-{index}"
                )
            else:
                regex_policy = control.replacement_policy_from_event(existing_event)
        else:
            pattern, pattern_type = replacement_pattern(source, source, left, right)
            regex_policy, payload = make_candidate_policy(
                control, source, target, pattern, pattern_type, validation_specs,
                policy_id=f"preflight-rime-{index}"
            )
            needs_reissue = False
        definitions.append({
            "source": source,
            "target": target,
            "count": totals[(source, target)],
            "pattern": pattern,
            "patternType": pattern_type,
            "leftBoundaries": left,
            "rightBoundaries": right,
            "negatives": negatives,
            "validationNegativeSpecs": validation_specs,
            "policy": regex_policy,
            "payload": payload,
            "existingEvent": existing_event,
            "needsReissue": needs_reissue,
            "rowPks": sorted(set(
                [int(item[0]) for item in oracle if item[2] == source and item[3] == target]
                + record_rows.get((source, target), [])
            )),
        })

    all_negative_texts = sorted({text for definition in definitions for text in definition["negatives"]})
    new_policies = expand_guards_until_negatives_are_safe(
        [definition["policy"] for definition in definitions], all_negative_texts, control
    )
    policies_by_id = {policy["policyId"]: policy for policy in new_policies}
    for definition in definitions:
        definition["policy"] = policies_by_id[definition["policy"]["policyId"]]
        definition["negatives"] = sorted({
            str(guard.get("context") or guard.get("text"))
            for guard in definition["policy"].get("negativeSourceGuards") or []
            if guard.get("text") or guard.get("context")
        })

    b_summary = simulate_b(
        rows,
        active_policies + new_policies,
        expected_characters,
        control,
        approved_exception_pks,
    )
    positive_sentences = load_positive_sentences(evidence_path)
    positive_changes = [
        (index, text, apply_scoped_policies(text, new_policies, control)[0])
        for index, text in enumerate(positive_sentences)
        if apply_scoped_policies(text, new_policies, control)[0] != text
    ]
    negative_changes = [
        (text, apply_scoped_policies(text, new_policies, control)[0])
        for text in all_negative_texts
        if apply_scoped_policies(text, new_policies, control)[0] != text
    ]

    print(
        f"rows={len(rows)} oracle_rows={len({item[0] for item in oracle})} oracle_spans={len(oracle)} "
        f"oracle_pairs={len(oracle_counts)} A_rime_records={sum(recorded_counts.values())}"
    )
    print(f"candidate_pairs={len(candidate_pairs)} active_pairs={len(active_pairs)} excluded_pairs={len(EXCLUDED_PAIRS)} new_rules={len(definitions)}")
    for definition in definitions:
        print(
            f"RULE {definition['source']}→{definition['target']} count={definition['count']} "
            f"pattern_type={definition['patternType'] or 'literal'} "
            f"left={len(definition['leftBoundaries'])} right={len(definition['rightBoundaries'])} "
            f"negative={len(definition['negatives'])}"
        )
    for examples in REQUIRED_NEGATIVES.values():
        for text in examples:
            print(f"REQUIRED_BOUNDARY_NEGATIVE {text}")
    print(
        f"B rows={len(b_summary['changedRows'])} changed_codepoints={b_summary['changedCharacters']} "
        f"oracle_matched_rows={len(b_summary['oracleMatchedRows'])} approved_exceptions={b_summary['exceptions']} "
        f"wrong_positions={len(b_summary['mismatches'])} "
        f"approved_exception_rows={len(b_summary['approvedExceptionPks'])}/{len(approved_exception_pks)} "
        f"positive_sentences={len(positive_sentences)} positive_changed={len(positive_changes)} "
        f"boundary_negative_sentences={len(all_negative_texts)} negative_changed={len(negative_changes)} "
        f"control_validation_negatives={sum(len(d['validationNegativeSpecs']) for d in definitions)} "
        f"evidence_rules_to_reissue={sum(bool(d['needsReissue']) for d in definitions)}"
    )
    if b_summary["mismatches"]:
        for mismatch in b_summary["mismatches"]:
            print("WRONG", mismatch)
    if positive_changes:
        for index, text, output in positive_changes[:30]:
            print(f"POSITIVE_CHANGED index={index} input={text!r} output={output!r}")
    if negative_changes:
        for text, output in negative_changes:
            print(f"NEGATIVE_CHANGED input={text!r} output={output!r}")

    if b_summary["mismatches"] or positive_changes or negative_changes:
        raise SystemExit("B 預檢未通過；尚未寫入 evidence")
    if len(rows) == 34_190 and len(oracle) == 62:
        if (
            (len(b_summary["changedRows"]), b_summary["changedCharacters"], len(b_summary["oracleMatchedRows"]))
            != (40, 55, 39)
            or b_summary["exceptions"] != 1
            or b_summary["approvedExceptionPks"] != approved_exception_pks
        ):
            raise SystemExit("B 全歷史計數不符合裁決預期 40 筆／55 字／39 筆 oracle 同位")
        if b_summary["exceptions"] != 1:
            raise SystemExit("B 全歷史沒有唯一命中已核准的單筆例外")
    if args.dry_run:
        print("dry_run=1 evidence_written=0 compile=0 validate=0")
        return

    artifact_dir.mkdir(parents=True, exist_ok=True)
    event_count_before = len(events)
    event_changes_needed = any(
        definition["existingEvent"] is None or definition["needsReissue"]
        for definition in definitions
    )
    for definition in definitions:
        source = definition["source"]
        target = definition["target"]
        existing_event = definition["existingEvent"]
        if existing_event and not definition["needsReissue"]:
            continue
        if existing_event:
            old_policy = control.replacement_policy_from_event(existing_event)
            disable_command = [
                sys.executable, str(control_cli_path), "--json", "--evidence-store", str(evidence_path),
                "disableRule", "--control-plane", "local", "--policy-id", old_policy["policyId"],
                "--disposition", "replaced",
                "--reason", "Replace Rime boundary negative examples with runtime-context validation.",
            ]
            run_cli(disable_command, wordfreq_path=wordfreq_path)
        positive = f"{source}||Rime history pair count {definition['count']}||{target}"
        command = [
            sys.executable, str(control_cli_path), "--json", "--evidence-store", str(evidence_path),
            "addReplacementRule", "--control-plane", "local",
            "--source-pattern", definition["pattern"], "--target-text", target,
            "--source-text", source, "--rule-name", f"rime-homophone:{source}->{target}",
            "--positive", positive,
            "--note", "Auto-mined from Rime homophone history; local-only pending validation.",
        ]
        if definition["rowPks"]:
            command.extend(["--row-pk", str(definition["rowPks"][0])])
        if definition["patternType"] == "regex":
            command.extend(["--source-pattern-type", "regex", "--target-template", target])
        for negative_text, negative_context in definition["validationNegativeSpecs"]:
            negative_value = f"{negative_text}||{negative_context}" if negative_context else negative_text
            command.extend(["--negative", negative_value])
        result = run_cli(command, wordfreq_path=wordfreq_path)
        written = ((result.get("surfaces") or {}).get("macLocalCanonicalEvidence") or {}).get("written")
        if not written:
            raise SystemExit(f"addReplacementRule 沒有寫入本機 evidence：{source}→{target}")

    events_after = control.load_events(evidence_path)
    print(f"local_evidence_events_before={event_count_before} after={len(events_after)}")
    evidence_hash = sha256_file(evidence_path)
    active_model_hash = sha256_file(active_model_path)
    rules_hash = sha256_json([
        {
            "source": item["source"],
            "target": item["target"],
            "pattern": item["pattern"],
            "patternType": item["patternType"],
            "validationNegativeSpecs": item["validationNegativeSpecs"],
        }
        for item in definitions
    ])
    state_path = artifact_dir / "rime-homophones-state.json"
    candidate_path = None
    validation_report = None
    validation_value = None
    if not event_changes_needed and state_path.is_file():
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
            candidate = Path(previous["candidateModel"])
            report = Path(previous["validationReport"])
            report_value = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else {}
            if (
                previous.get("evidenceSha256") == evidence_hash
                and previous.get("activeModelSha256") == active_model_hash
                and previous.get("rulesSha256") == rules_hash
                and candidate.is_file()
                and report_value.get("ready") is True
            ):
                print(f"compile=skipped_unchanged candidate_model={candidate} validate=previously_ready")
                candidate_path = candidate
                validation_report = report
                validation_value = control.validation_summary(report_value)
        except (KeyError, OSError, json.JSONDecodeError):
            pass
    if candidate_path is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate_dir = artifact_dir / f"rime-homophones-{timestamp}"
        candidate_path = candidate_dir / "full-db.auto-apply-model.json"
        # This task's validation corpus is the scoped Voco history plus the local control examples.
        # ReplayLab corpora are a separate data source and are not part of this approval scope.
        common_validation_args = ["--skip-corpus-replay", "--skip-raw-input-replay"]
        compile_command = [
            sys.executable, str(control_cli_path), "--json", "--evidence-store", str(evidence_path),
            "compileModel", "--base-model", str(active_model_path), "--output-model", str(candidate_path),
            *common_validation_args,
        ]
        compiled = run_cli(compile_command, wordfreq_path=wordfreq_path)
        print(f"compiled_model={compiled.get('compiledModel')} validation={compiled.get('validation')}")
        validation_report = candidate_path.with_suffix(".validation.json")
        validate_command = [
            sys.executable, str(control_cli_path), "--json", "--evidence-store", str(evidence_path),
            "validateModel", "--model", str(candidate_path), "--base-model", str(active_model_path),
            "--report", str(validation_report), "--no-write-readiness", *common_validation_args,
        ]
        validated = run_cli(validate_command, wordfreq_path=wordfreq_path, allow_failed=True)
        validation_value = validated.get("validation") if isinstance(validated.get("validation"), dict) else {}
        print(f"validated_model={validated.get('model')} report={validated.get('report')} validation={validation_value}")
        if validation_value.get("ready") is True:
            atomic_json_write(
                state_path,
                {
                    "schema": "voco.rime-homophones-miner-state.v1",
                    "activeModelSha256": active_model_hash,
                    "evidenceSha256": evidence_hash,
                    "rulesSha256": rules_hash,
                    "candidateModel": str(candidate_path),
                    "validationReport": str(validation_report),
                },
            )

    if args.dry_run:
        print("auto_publish=skipped dry_run=1")
        return

    if candidate_path is None or validation_report is None or validation_value is None:
        raise SystemExit("候選模型或 validate 結果缺失，拒絕發布")
    candidate_model = json.loads(candidate_path.read_text(encoding="utf-8"))
    formal_model = json.loads(active_model_path.read_text(encoding="utf-8"))
    validation_failures = release_gate_failures(
        new_rules=len(definitions),
        candidate=candidate_model,
        formal=formal_model,
        validation=validation_value,
        wrong_positions=len(b_summary["mismatches"]),
        positive_changed=len(positive_changes),
        negative_changed=len(negative_changes),
    )
    candidate_signatures = model_rule_signatures(candidate_model)
    formal_signatures = model_rule_signatures(formal_model)
    validation_failure_metrics = {
        key: value
        for key, value in validation_value.items()
        if "failure" in key.lower() or key.lower().endswith("conflicts")
    }
    log_auto_publish(
        f"AUTO_PUBLISH_PREFLIGHT new_rules={len(definitions)} "
        f"candidate_signatures={len(candidate_signatures)} formal_signatures={len(formal_signatures)} "
        f"missing={len(formal_signatures - candidate_signatures)} ready={validation_value.get('ready')} "
        f"failures={json.dumps(validation_failure_metrics, ensure_ascii=False, sort_keys=True)} "
        f"wrong_positions={len(b_summary['mismatches'])} positive_changed={len(positive_changes)} "
        f"negative_changed={len(negative_changes)}"
    )
    def activate_candidate() -> None:
        command = [
            sys.executable, str(control_cli_path), "--json", "--evidence-store", str(evidence_path),
            "activateModel", "--model", str(candidate_path), "--active-model", str(active_model_path),
            "--base-model", str(active_model_path), "--backup-dir", str(artifact_dir / "activation-backups"),
            "--skip-corpus-replay", "--skip-raw-input-replay",
        ]
        try:
            result = run_cli(command, wordfreq_path=wordfreq_path)
        except SystemExit as error:
            log_auto_publish(f"AUTO_ACTIVATE_FAILED {error}", log_path)
            raise
        print("AUTO_ACTIVATE " + json.dumps(result, ensure_ascii=False, sort_keys=True))

    def current_formal_model() -> dict:
        return json.loads(active_model_path.read_text(encoding="utf-8"))

    def publish_candidate() -> None:
        # The active model is re-read directly before this call; run_gated_release checks
        # condition 4 at the last possible point before the Worker write.
        command = [
            sys.executable, str(control_cli_path), "--json", "--evidence-store", str(evidence_path),
            "publishWorkerRelease", "--model", str(candidate_path), "--base-model", str(active_model_path),
            "--skip-corpus-replay", "--skip-raw-input-replay",
        ]
        try:
            result = run_cli(command, wordfreq_path=wordfreq_path)
        except SystemExit as error:
            log_auto_publish(f"AUTO_PUBLISH_FAILED {error}", log_path)
            raise
        print("AUTO_PUBLISH_RESULT " + json.dumps(result, ensure_ascii=False, sort_keys=True))
        if result.get("published") is not True:
            log_auto_publish("AUTO_PUBLISH_FAILED " + json.dumps(result, ensure_ascii=False, sort_keys=True), log_path)
            raise SystemExit("Worker release 未成功發布")

    run_gated_release(
        gate_failures=validation_failures,
        current_formal=current_formal_model,
        candidate=candidate_model,
        activate=activate_candidate,
        publish=publish_candidate,
        log=lambda message: log_auto_publish(message, log_path),
    )


if __name__ == "__main__":
    main()
