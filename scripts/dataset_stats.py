"""Print CAViAR annotation statistics for the public val/test (Nexar) release."""

import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from caviar.paths import test_json, train_json, full_json


def summarize(path: Path, split_name: str):
    if not path.exists():
        print(f"\n{split_name}: {path}  [not present in this release]")
        return None, None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    benches = Counter()
    qa_total = 0
    for item in data:
        for qa in item.get("qa_pairs", []):
            benches[qa.get("benchmark", "<missing>")] += 1
            qa_total += 1
    print(f"\n{split_name}: {path}")
    print(f"  videos: {len(data)}")
    print(f"  QA pairs: {qa_total}")
    for b, n in benches.most_common():
        print(f"    {b:<28} {n}")
    return len(data), qa_total


def main():
    n_test, qa_test = summarize(test_json(), "Val/Test (Nexar) — released")
    summarize(train_json(), "Train (CCD)")
    summarize(full_json(), "Full CAViAR")
    if n_test is not None:
        print("\nPublic release totals")
        print(f"  videos: {n_test}")
        print(f"  QA:     {qa_test}")


if __name__ == "__main__":
    main()
