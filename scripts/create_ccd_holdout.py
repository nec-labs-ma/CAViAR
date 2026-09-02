"""
Create a CCD holdout split for the domain-shift ablation experiment.

Splits the 1,500 CCD videos in train.json into:
  - holdout/ccd_train.json  (1,200 videos, same format as train.json)
  - holdout/ccd_test.json   (300 videos, same format as test.json for eval scripts)
  - holdout/ccd_train.jsonl (1,200 videos in ShareGPT/JSONL format for SFT)

Usage:
  python create_ccd_holdout.py
"""

import json
import os
import random
import string

SEED = 42
TRAIN_SIZE = 1200
import sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from caviar.paths import train_json as _default_train_json, holdout_dir as _default_holdout_dir

INPUT_PATH = str(_default_train_json())
OUTPUT_DIR = str(_default_holdout_dir())


def format_question(qa):
    """Format a question, appending lettered choices for MCQ questions."""
    question = qa["question"]
    if "choices" in qa and qa["choices"]:
        choice_labels = string.ascii_uppercase
        choice_lines = [f"{choice_labels[i]}. {c}" for i, c in enumerate(qa["choices"])]
        question = question + "\n" + "\n".join(choice_lines)
    return question


def format_answer(qa):
    """Format the answer. For MCQ, return the letter + text."""
    if "choices" in qa and qa["choices"] and "correct_index" in qa:
        idx = qa["correct_index"]
        label = string.ascii_uppercase[idx]
        return f"{label}. {qa['answer']}"
    return qa["answer"]


def convert_entry_to_jsonl(entry, entry_id):
    """Convert a single video entry to ShareGPT JSONL format."""
    conversations = []
    for i, qa in enumerate(entry["qa_pairs"]):
        question = format_question(qa)
        if i == 0:
            question = "<video>\n" + question
        conversations.append({"from": "human", "value": question})
        conversations.append({"from": "gpt", "value": format_answer(qa)})
    return {
        "id": entry_id,
        "video": entry["video_path"],
        "conversations": conversations,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} CCD videos from {INPUT_PATH}")

    random.seed(SEED)
    indices = list(range(len(data)))
    random.shuffle(indices)

    train_indices = indices[:TRAIN_SIZE]
    test_indices = indices[TRAIN_SIZE:]

    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]

    print(f"Split: {len(train_data)} train / {len(test_data)} test")

    # Save ccd_train.json
    train_json_path = os.path.join(OUTPUT_DIR, "ccd_train.json")
    with open(train_json_path, "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {train_json_path}")

    # Save ccd_test.json
    test_json_path = os.path.join(OUTPUT_DIR, "ccd_test.json")
    with open(test_json_path, "w") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {test_json_path}")

    # Save ccd_train.jsonl (ShareGPT format for SFT)
    train_jsonl_path = os.path.join(OUTPUT_DIR, "ccd_train.jsonl")
    with open(train_jsonl_path, "w") as f:
        for idx, entry in enumerate(train_data):
            jsonl_entry = convert_entry_to_jsonl(entry, idx)
            f.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
    print(f"Saved {train_jsonl_path}")

    # Print summary statistics
    train_qa = sum(len(e["qa_pairs"]) for e in train_data)
    test_qa = sum(len(e["qa_pairs"]) for e in test_data)
    print(f"\nSummary:")
    print(f"  Train: {len(train_data)} videos, {train_qa} QA pairs")
    print(f"  Test:  {len(test_data)} videos, {test_qa} QA pairs")


if __name__ == "__main__":
    main()
