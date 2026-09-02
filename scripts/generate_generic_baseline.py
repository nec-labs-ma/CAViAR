"""
Generate a majority-class / generic-template baseline for metric sanity checks.

Requires released test annotations (data/annotations/test.json).

Usage:
  python scripts/generate_generic_baseline.py --test data/annotations/test.json --output results/results_generic_baseline.json
"""

import argparse
import json
import os

GENERIC_ANSWERS = {
    "Faulter Identification": "The other vehicle is at fault for this accident.",
    "Victim Identification": "The other vehicle was the victim in this accident.",
    "Violation Identification": "The vehicle violated traffic rules by driving unsafely.",
}

# Majority classes on the CAViAR Nexar test split (paper statistics)
MAJORITY_MCQ = {
    "weather": "Sunny",
    "lighting": "Day",
    "road": "Dry",
    "accident_type": "none",
}

MC_BENCHMARKS = ["Weather & Light", "Accident Type", "Road Conditions"]
OE_BENCHMARKS = ["Faulter Identification", "Victim Identification", "Violation Identification"]


def get_majority_pred(question, choices):
    q_lower = question.lower()
    if "weather" in q_lower:
        target = MAJORITY_MCQ["weather"]
    elif "lighting" in q_lower:
        target = MAJORITY_MCQ["lighting"]
    elif "road" in q_lower:
        target = MAJORITY_MCQ["road"]
    else:
        target = MAJORITY_MCQ["accident_type"]
    for choice in choices:
        if choice.lower() == target.lower():
            return choice
    return choices[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Path to test.json")
    parser.add_argument("--output", default="results/results_generic_baseline.json")
    args = parser.parse_args()

    with open(args.test, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    for item in test_data:
        video_path = item["video_path"]
        for qa in item["qa_pairs"]:
            benchmark = qa["benchmark"]
            if benchmark in MC_BENCHMARKS and "choices" in qa:
                pred = get_majority_pred(qa["question"], qa["choices"])
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
                pred = GENERIC_ANSWERS[benchmark]
                results.append({
                    "video_path": video_path,
                    "benchmark": benchmark,
                    "question": qa["question"],
                    "gt": qa["answer"],
                    "pred": pred,
                    "pred_raw": pred,
                })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(results)} entries -> {args.output}")


if __name__ == "__main__":
    main()
