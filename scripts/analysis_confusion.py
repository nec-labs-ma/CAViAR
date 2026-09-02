import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from caviar.paths import results_dir

BASE = str(results_dir()) + "/"

BASE_FILES = [
    "results_Cosmos2-2B.json",
    "results_Cosmos2-8B.json",
    "results_InternVL3-2B_fps.json",
    "results_InternVL3-8B_fps.json",
    "results_Qwen3-VL-2B_fps.json",
    "results_Qwen3-VL-8B_fps.json",
]
FT_FILES = [
    "results_Cosmos2-2B_merged.json",
    "results_Cosmos2-8B_merged.json",
    "results_InternVL3-merged_2B_fps.json",
    "results_InternVL3-merged_fps.json",
    "results_Qwen3-VL-2B_merged_fps.json",
    "results_Qwen3-VL-8B_merged_fps.json",
]

CLASSES = ["T-Bone", "Rear End", "Side-by-Side", "head-on", "None"]
DISPLAY = ["T-Bone", "Rear-End", "Side-by-Side", "Head-on", "None"]


def norm(s):
    if s is None:
        return "None"
    s = str(s).strip().lower()
    m = {
        "t-bone": "T-Bone", "tbone": "T-Bone",
        "rear end": "Rear End", "rear-end": "Rear End",
        "side-by-side": "Side-by-Side", "side by side": "Side-by-Side",
        "head-on": "head-on", "head on": "head-on", "headon": "head-on",
        "none": "None", "no accident": "None",
    }
    return m.get(s, None)


def load_items(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k in ("results", "items", "data"):
            if k in data and isinstance(data[k], list):
                return data[k]
        return []
    return data


def build_cm(files):
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    idx = {c: i for i, c in enumerate(CLASSES)}
    for fn in files:
        items = load_items(BASE + fn)
        for it in items:
            if it.get("benchmark") != "Accident Type":
                continue
            gt = norm(it.get("gt"))
            pr = norm(it.get("pred"))
            if gt is None:
                continue
            if pr is None:
                pr = "None"
            cm[idx[gt], idx[pr]] += 1
    return cm


def report(name, cm):
    print(f"\n=== {name} ===")
    total = cm.sum()
    correct = np.trace(cm)
    print(f"Overall acc: {100*correct/total:.1f}%  (N={total})")
    for i, c in enumerate(DISPLAY):
        row = cm[i].sum()
        rec = 100 * cm[i, i] / row if row else 0
        print(f"  {c:14s} support={row:4d} recall={rec:5.1f}%  diag={cm[i,i]}")
    # rear-end overprediction ratio
    re_idx = CLASSES.index("Rear End")
    pred_re = cm[:, re_idx].sum()
    true_re = cm[re_idx, :].sum()
    print(f"  Rear-End predicted={pred_re} true={true_re} overpred={pred_re/true_re:.2f}x")
    return cm


def plot(cm_base, cm_ft, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, cm, title in zip(axes, [cm_base, cm_ft], ["Base (zero-shot)", "Fine-tuned"]):
        cmn = cm.astype(float)
        row = cmn.sum(axis=1, keepdims=True)
        row[row == 0] = 1
        cmn = cmn / row
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(DISPLAY)))
        ax.set_yticks(range(len(DISPLAY)))
        ax.set_xticklabels(DISPLAY, rotation=40, ha="right", fontsize=9)
        ax.set_yticklabels(DISPLAY, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        if title.startswith("Base"):
            ax.set_ylabel("Ground truth", fontsize=10)
        ax.set_title(title, fontsize=11)
        for i in range(len(DISPLAY)):
            for j in range(len(DISPLAY)):
                v = cmn[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    cmb = build_cm(BASE_FILES)
    cmf = build_cm(FT_FILES)
    report("BASE (aggregated 6 models)", cmb)
    report("FINE-TUNED (aggregated 6 models)", cmf)
    plot(cmb, cmf, str(results_dir() / "accident_type_confusion.png"))
