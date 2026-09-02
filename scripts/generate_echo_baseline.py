"""
Generate a degenerate 'echo-question' baseline for metric sanity checks.

Requires released test annotations (data/annotations/test.json).

Usage:
  python scripts/generate_echo_baseline.py --test data/annotations/test.json --output results/results_echo_baseline.json
"""

import argparse
import json
import os
import random

random.seed(42)

MC_BENCHMARKS = ["Weather & Light", "Accident Type", "Road Conditions"]
OE_BENCHMARKS = ["Faulter Identification", "Victim Identification", "Violation Identification"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Path to test.json")
    parser.add_argument("--output", default="results/results_echo_baseline.json")
    args = parser.parse_args()

    with open(args.test, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    for item in test_data:
        video_path = item["video_path"]
        for qa in item["qa_pairs"]:
            benchmark = qa["benchmark"]
            if benchmark in MC_BENCHMARKS and "choices" in qa:
                pred = random.choice(qa["choices"])
                results.append({
                    "video_path": video_path,
                    "benchmark": benchmark,
                    "question": qa["question"],
                    "gt": qa["correct_answer"],
                    "choices": qa["choices"],
                    "pred": pred,
                    "pred_raw": pred,
                    "correct": pred.lower() == qa["correct_answer"].lower(),
                })
            elif benchmark in OE_BENCHMARKS:
                results.append({
                    "video_path": video_path,
                    "benchmark": benchmark,
                    "question": qa["question"],
                    "gt": qa["answer"],
                    "pred": qa["question"],
                    "pred_raw": qa["question"],
                })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(results)} entries -> {args.output}")


if __name__ == "__main__":
    main()
