"""Traffic-rule violation ontology + deterministic mapper for CAViAR.

Maps each free-text Violation-Identification answer to ONE primary rule family
using an ordered (priority) lexicon. Reports:
  - overall + train/test distribution by family
  - overlap diagnostic (answers matching >= 2 family lexicons)
  - other/unclear policy (rare-valid vs unspecified/low-quality)
  - per-family model performance (mean LLM-as-Judge score on the Nexar test split)
  - a reproducible random audit sample for author verification
"""
import json
import os
import random
from collections import Counter, defaultdict
import sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from caviar.paths import full_json, test_json, results_dir

import numpy as np

_full = full_json()
PATH = str(_full if _full.exists() else test_json())
EVAL_DIR = str(results_dir() / "16_fps") + "/"
EVAL_FILES = [
    "eval_Cosmos2-2B_base.json", "eval_Cosmos2-2B_merged.json",
    "eval_Cosmos2-8B_base.json", "eval_Cosmos2-8B_merged.json",
    "eval_InternVL3-2B_fps.json", "eval_InternVL3-2B_merged_fps.json",
    "eval_InternVL3-8B_fps.json", "eval_InternVL3-8B_merged_fps.json",
    "eval_Qwen3-VL-2B_fps.json", "eval_Qwen3-VL-2B_merged_fps.json",
    "eval_Qwen3-VL-8B_fps.json", "eval_Qwen3-VL-8B_merged_fps.json",
]

# ---------------------------------------------------------------------------
# ONTOLOGY: (code, name, keyword lexicon). ORDER = matching priority.
# The first family whose lexicon fires is assigned as the PRIMARY family, so
# specific control-device / right-of-way rules win over generic maneuver or
# attention cues. This precedence is the operational tie-break for overlaps.
# ---------------------------------------------------------------------------
ONTOLOGY = [
    ("SG", "Signal / sign violation", [
        "red light", "traffic light", "traffic signal", "stop sign",
        "ran the", "ran a red", "running the light", "running a red",
        "disobey", "ignored the signal", "ignored the light"]),
    ("RW", "Failure to yield / right-of-way", [
        "yield", "right of way", "right-of-way", "give way", "giving way",
        "priority", "main road", "side road", "right of passage",
        "failed to give", "stop and look", "before entering the intersection",
        "before entering an intersection"]),
    ("FD", "Unsafe following distance / rear-end", [
        "following distance", "safe distance", "safe following", "too close",
        "tailgat", "stopping distance", "brake in time", "rear-end",
        "rear end", "distance from the vehicle", "keep a safe distance",
        "keeping a safe distance", "maintain a safe distance",
        "safe braking distance"]),
    ("CT", "Loss of vehicle control", [
        "lose control", "lost control", "loss of control", "uncontrolled",
        "skid", "malfunction", "tilted", "control of the vehicle",
        "control of their vehicle", "spun", "rolled over"]),
    ("LC", "Improper lane change / merging", [
        "lane change", "change lane", "changing lane", "change lanes",
        "changed lane", "merg", "cut in", "cut off", "cutting",
        "lane discipline", "improper lane", "weav"]),
    ("OT", "Improper overtaking / passing", [
        "overtak", "overtook", "overtaking"]),
    ("TU", "Improper turn / U-turn / reversing", [
        "u-turn", "u turn", "turning", "while turning", "made a turn",
        "left turn", "right turn", "revers", "backing up", "back up"]),
    ("SP", "Unsafe speed / reckless driving", [
        "speed", "too fast", "reckless", "excessive", "overspeed",
        "slow down", "high velocity", "drove fast"]),
    ("ST", "Sudden stop / improper stopping or parking", [
        "suddenly stop", "sudden stop", "abrupt", "stopped without",
        "sudden brak", "hard brak", "emergency brak", "parking", "parked",
        "stopped in the middle", "stopping in the"]),
    ("PD", "Pedestrian / non-motorized / illegal crossing", [
        "pedestrian", "bicycle", "cyclist", "non-motor", "nonmotor",
        "crossed the road", "crossing the road", "jaywalk", "crosswalk",
        "illegally cross", "cross the road"]),
    ("AT", "Inattentive / improper observation", [
        "distract", "attention", "inattentive", "observe", "observation",
        "failed to notice", "lookout", "look out", "careless",
        "confirm the safety", "confirm safety", "check the rear",
        "checking the rear", "aware of", "vigilan", "pay attention"]),
]
CODE2NAME = {c: n for c, n, _ in ONTOLOGY}

UNSPEC_TOKENS = {"n/a", "na", "none", "unknown", "not applicable", "-", ""}


def norm(s):
    return (s or "").strip().lower()


def is_unspecified(a):
    n = norm(a)
    if n in UNSPEC_TOKENS:
        return True
    if len(n) < 25:
        return True
    return False


# Negative guards: a family does NOT fire if any of these phrases is present
# (they typically describe the ABSENCE of a control device, not a violation).
NEG_GUARDS = {
    "SG": ["without traffic", "no traffic light", "no traffic signal",
           "without a traffic", "no signal", "absence of traffic"],
}


def fires(code, kws, n):
    if not any(k in n for k in kws):
        return False
    for bad in NEG_GUARDS.get(code, []):
        if bad in n:
            return False
    return True


def all_matches(a):
    """Return list of family codes whose lexicon fires (for overlap stats)."""
    n = norm(a)
    return [code for code, _, kws in ONTOLOGY if fires(code, kws, n)]


def classify(a):
    """Return (code, bucket) where bucket in {family, OTHER, UNSPEC}."""
    if is_unspecified(a):
        return ("UNSPEC", "UNSPEC")
    n = norm(a)
    for code, _, kws in ONTOLOGY:
        if fires(code, kws, n):
            return (code, "family")
    return ("OTHER", "OTHER")


def load_dataset():
    data = json.load(open(PATH))
    rows = []  # (split, answer)
    for e in data:
        b = os.path.basename(e["video_path"])
        split = "train" if b.startswith("crash_1500") else "test"
        for qa in e["qa_pairs"]:
            if qa.get("benchmark") == "Violation Identification":
                rows.append((split, qa.get("answer", "")))
    return rows


def per_family_judge():
    """Mean judge score per family on the test split, aggregated over models."""
    scores = defaultdict(list)
    for fn in EVAL_FILES:
        path = os.path.join(EVAL_DIR, fn) if not EVAL_DIR.endswith("/") else EVAL_DIR + fn
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        for it in d.get("oe_judge_details", []):
            if it.get("benchmark") != "Violation Identification":
                continue
            js = it.get("judge_score")
            if not isinstance(js, (int, float)):
                continue
            code, bucket = classify(it.get("gt", ""))
            scores[code].append(js)
    return scores


def main():
    rows = load_dataset()
    total = len(rows)
    print(f"Total violation answers: {total}")

    # distribution
    by_code = Counter()
    by_split = {"train": Counter(), "test": Counter()}
    overlap_count = 0
    multi_hist = Counter()
    for split, a in rows:
        code, bucket = classify(a)
        by_code[code] += 1
        by_split[split][code] += 1
        m = all_matches(a)
        multi_hist[len(m)] += 1
        if len(m) >= 2:
            overlap_count += 1

    order = [c for c, _, _ in ONTOLOGY] + ["OTHER", "UNSPEC"]
    ntr = sum(by_split["train"].values())
    nte = sum(by_split["test"].values())

    judge = per_family_judge()

    print(f"\n{'Code':<6}{'Family':<46}{'All%':>7}{'Train%':>8}{'Test%':>7}"
          f"{'Judge':>8}{'Nj':>6}")
    for c in order:
        name = CODE2NAME.get(c, "Other (rare valid rules)" if c == "OTHER"
                             else "Unspecified / low-quality")
        allp = 100 * by_code[c] / total
        trp = 100 * by_split["train"][c] / ntr
        tep = 100 * by_split["test"][c] / nte
        js = judge.get(c, [])
        jm = np.mean(js) if js else float("nan")
        print(f"{c:<6}{name:<46}{allp:>6.1f}{trp:>7.1f}{tep:>6.1f}"
              f"{jm:>8.2f}{len(js):>6}")

    print(f"\nFamilies present: {sum(1 for c,_,_ in ONTOLOGY if by_code[c]>0)}")
    print(f"Overlap (answers firing >=2 family lexicons): "
          f"{overlap_count} ({100*overlap_count/total:.1f}%)")
    print(f"Multi-match histogram (#families fired -> count): "
          f"{dict(sorted(multi_hist.items()))}")
    print(f"OTHER (rare valid): {by_code['OTHER']} ({100*by_code['OTHER']/total:.1f}%)")
    print(f"UNSPEC (low-quality): {by_code['UNSPEC']} ({100*by_code['UNSPEC']/total:.1f}%)")

    # reproducible audit sample
    random.seed(7)
    sample = random.sample(rows, min(100, len(rows)))
    out_audit = str(results_dir() / "rule_audit_sample.txt")
    os.makedirs(os.path.dirname(out_audit), exist_ok=True)
    with open(out_audit, "w") as f:
        for i, (split, a) in enumerate(sample):
            code, _ = classify(a)
            f.write(f"{i:3d}\t{code}\t{a[:200]}\n")
    print(f"\nWrote {len(sample)}-item audit sample -> {out_audit}")


if __name__ == "__main__":
    main()
