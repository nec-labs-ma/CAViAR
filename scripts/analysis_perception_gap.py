import json
from collections import Counter
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
import sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from caviar.paths import results_dir

BASE = str(results_dir()) + "/"

BASE_FILES = {
    "Cosmos2-2B": "results_Cosmos2-2B.json",
    "Cosmos2-8B": "results_Cosmos2-8B.json",
    "InternVL3-2B": "results_InternVL3-2B_fps.json",
    "InternVL3-8B": "results_InternVL3-8B_fps.json",
    "Qwen3-VL-2B": "results_Qwen3-VL-2B_fps.json",
    "Qwen3-VL-8B": "results_Qwen3-VL-8B_fps.json",
}
FT_FILES = {
    "Cosmos2-2B": "results_Cosmos2-2B_merged.json",
    "Cosmos2-8B": "results_Cosmos2-8B_merged.json",
    "InternVL3-2B": "results_InternVL3-merged_2B_fps.json",
    "InternVL3-8B": "results_InternVL3-merged_fps.json",
    "Qwen3-VL-2B": "results_Qwen3-VL-2B_merged_fps.json",
    "Qwen3-VL-8B": "results_Qwen3-VL-8B_merged_fps.json",
}

# number of MCQ options per task (for random/uniform baseline)
NUM_CHOICES = {"Weather": 4, "Lighting": 2, "Road": 3, "Accident Type": 5}


def norm(s):
    s = (s or "").strip().lower()
    repl = {
        "rear-end": "rear end", "rearend": "rear end",
        "t bone": "t-bone", "tbone": "t-bone",
        "side by side": "side-by-side", "sidebyside": "side-by-side",
        "head on": "head-on", "headon": "head-on",
    }
    return repl.get(s, s)


def load_items(fn):
    with open(BASE + fn) as f:
        d = json.load(f)
    return d if isinstance(d, list) else d.get("results", d)


def task_of(item):
    b = item.get("benchmark")
    if b == "Weather & Light":
        q = item.get("question", "").lower()
        if "weather" in q:
            return "Weather"
        if "light" in q:
            return "Lighting"
        return None
    if b == "Road Conditions":
        return "Road"
    if b == "Accident Type":
        return "Accident Type"
    return None


def collect(fn):
    """Return dict task -> (list_gt, list_pred)."""
    out = {t: ([], []) for t in NUM_CHOICES}
    for it in load_items(fn):
        t = task_of(it)
        if t is None:
            continue
        out[t][0].append(norm(it.get("gt")))
        out[t][1].append(norm(it.get("pred")))
    return out


def metrics(gts, preds):
    labels = sorted(set(gts))
    acc = np.mean([g == p for g, p in zip(gts, preds)]) * 100
    # map preds not in label set to a sentinel so they count as wrong
    preds2 = [p if p in labels else "<invalid>" for p in preds]
    bal = balanced_accuracy_score(gts, preds2) * 100
    mf1 = f1_score(gts, preds2, labels=labels, average="macro", zero_division=0) * 100
    return acc, bal, mf1


def baselines(gts):
    c = Counter(gts)
    n = len(gts)
    maj = max(c.values()) / n * 100
    return maj


def aggregate(files):
    agg = {t: {"acc": [], "bal": [], "mf1": []} for t in NUM_CHOICES}
    gts_ref = {}
    for name, fn in files.items():
        data = collect(fn)
        for t, (g, p) in data.items():
            if not g:
                continue
            a, b, m = metrics(g, p)
            agg[t]["acc"].append(a)
            agg[t]["bal"].append(b)
            agg[t]["mf1"].append(m)
            gts_ref[t] = g
    return agg, gts_ref


def report(tag, files):
    print(f"\n================ {tag} (mean over {len(files)} models) ================")
    agg, gts_ref = aggregate(files)
    print(f"{'Task':<14}{'Random':>8}{'Major':>8}{'Acc':>8}{'BalAcc':>8}{'MacroF1':>9}")
    for t in NUM_CHOICES:
        rnd = 100.0 / NUM_CHOICES[t]
        maj = baselines(gts_ref[t])
        acc = np.mean(agg[t]["acc"])
        bal = np.mean(agg[t]["bal"])
        mf1 = np.mean(agg[t]["mf1"])
        print(f"{t:<14}{rnd:>8.1f}{maj:>8.1f}{acc:>8.1f}{bal:>8.1f}{mf1:>9.1f}")
    return agg, gts_ref


if __name__ == "__main__":
    report("BASE", BASE_FILES)
    report("FINE-TUNED", FT_FILES)
    # also show GT class distribution per task (from one file)
    print("\nTest-set GT class distribution:")
    data = collect(BASE_FILES["Cosmos2-2B"])
    for t, (g, _) in data.items():
        print(f"  {t}: {dict(Counter(g))}")
