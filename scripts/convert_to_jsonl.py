"""
Convert CAViAR train.json to InternVL / ShareGPT-style train.jsonl.

Usage:
  python scripts/convert_to_jsonl.py --input data/annotations/train.json --output data/annotations/train.jsonl
"""

import argparse
import json
import string


def format_question(qa):
    question = qa["question"]
    if "choices" in qa and qa["choices"]:
        choice_lines = [
            f"{string.ascii_uppercase[i]}. {choice}"
            for i, choice in enumerate(qa["choices"])
        ]
        question = question + "\n" + "\n".join(choice_lines)
    return question


def format_answer(qa):
    if "choices" in qa and qa["choices"] and "correct_index" in qa:
        idx = qa["correct_index"]
        label = string.ascii_uppercase[idx]
        return f"{label}. {qa['answer']}"
    return qa["answer"]


def convert_entry(entry, entry_id):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to train.json")
    parser.add_argument("--output", required=True, help="Path to output train.jsonl")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(args.output, "w", encoding="utf-8") as f:
        for idx, entry in enumerate(data):
            f.write(json.dumps(convert_entry(entry, idx), ensure_ascii=False) + "\n")

    print(f"Converted {len(data)} entries -> {args.output}")


if __name__ == "__main__":
    main()
