import json
import numpy as np
import sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from caviar.paths import results_dir

BASE = str(results_dir()) + "/"
EVAL = BASE + "16_fps/"

# eval files -> (display, source raw file) for MCQ per-item
EVALS = {
    "Cosmos2-2B (B)": "eval_Cosmos2-2B_base.json",
    "Cosmos2-2B (FT)": "eval_Cosmos2-2B_merged.json",
    "Cosmos2-8B (B)": "eval_Cosmos2-8B_base.json",
    "Cosmos2-8B (FT)": "eval_Cosmos2-8B_merged.json",
    "InternVL3-2B (B)": "eval_InternVL3-2B_fps.json",
    "InternVL3-2B (FT)": "eval_InternVL3-2B_merged_fps.json",
    "InternVL3-8B (B)": "eval_InternVL3-8B_fps.json",
    "InternVL3-8B (FT)": "eval_InternVL3-8B_merged_fps.json",
    "Qwen3-VL-2B (B)": "eval_Qwen3-VL-2B_fps.json",
    "Qwen3-VL-2B (FT)": "eval_Qwen3-VL-2B_merged_fps.json",
    "Qwen3-VL-8B (B)": "eval_Qwen3-VL-8B_fps.json",
    "Qwen3-VL-8B (FT)": "eval_Qwen3-VL-8B_merged_fps.json",
}

rng = np.random.default_rng(0)
N = 10000


def boot_ci(values, fn=np.mean):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return (float("nan"),) * 3
    idx = rng.integers(0, n, size=(N, n))
    stats = fn(values[idx], axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return fn(values), lo, hi


def load(fn):
    with open(EVAL + fn) as f:
        return json.load(f)


def mcq_correct(d):
    """Per-item correctness from raw source file."""
    src = d["source_file"]
    with open(src) as f:
        raw = json.load(f)
    items = raw if isinstance(raw, list) else raw.get("results", raw)
    mc = [x for x in items if x.get("benchmark") in
          ("Weather & Light", "Road Conditions", "Accident Type")]
    return [1.0 if x.get("correct") else 0.0 for x in mc]


def judge_scores(d):
    return [it["judge_score"] for it in d.get("oe_judge_details", [])
            if isinstance(it.get("judge_score"), (int, float))]


print(f"{'Model':<20}{'MCQ Acc [95% CI]':<28}{'Judge [95% CI]':<24}")
for name, fn in EVALS.items():
    d = load(fn)
    mc = mcq_correct(d)
    js = judge_scores(d)
    m, mlo, mhi = boot_ci(mc)
    j, jlo, jhi = boot_ci(js)
    print(f"{name:<20}{100*m:5.1f} [{100*mlo:4.1f}, {100*mhi:4.1f}]        "
          f"{j:4.3f} [{jlo:.3f}, {jhi:.3f}]")

# typical CI half-widths
print("\nTypical 95% CI half-widths (percentage points / judge points):")
mcw, jcw = [], []
for fn in EVALS.values():
    d = load(fn)
    m, mlo, mhi = boot_ci(mcq_correct(d))
    j, jlo, jhi = boot_ci(judge_scores(d))
    mcw.append(100 * (mhi - mlo) / 2)
    jcw.append((jhi - jlo) / 2)
print(f"  MCQ acc: mean +/-{np.mean(mcw):.2f} pp (max +/-{np.max(mcw):.2f})")
print(f"  Judge:   mean +/-{np.mean(jcw):.3f}   (max +/-{np.max(jcw):.3f})")
