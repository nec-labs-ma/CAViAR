"""
Caption-Conditioned Reasoning Experiment (Tier 3, #15).

Instead of feeding video frames, this script gives the model the
GROUND-TRUTH dense caption as textual context, then asks the responsibility
questions (fault, victim, violation). This disentangles:
  - If scores IMPROVE significantly → bottleneck is vision/perception
  - If scores stay LOW → bottleneck is reasoning itself (even with perfect info)

Uses Qwen3-VL (text-only mode) to keep the model architecture constant.

Usage:
  python evaluate_caption_conditioned.py --model 2B
  python evaluate_caption_conditioned.py --model 8B
  python evaluate_caption_conditioned.py --model-path /path/to/finetuned/merged

Then evaluate with:
  python evaluate_results.py --results results/results_caption_conditioned_Qwen3-VL-2B.json
"""

import os
import sys
import json
import argparse
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from caviar.paths import test_json as _default_test_json, results_dir as _default_results_dir
import time
from collections import defaultdict

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# ── Configuration ────────────────────────────────────────────────────────────
TEST_JSON = str(_default_test_json())
OUTPUT_DIR = str(_default_results_dir())

OE_BENCHMARKS = ["Faulter Identification", "Victim Identification", "Violation Identification"]

MODEL_MAP = {
    "2B": "Qwen/Qwen3-VL-2B",
    "8B": "Qwen/Qwen3-VL-8B",
}

OE_CONTEXT = {
    "Faulter Identification": {
        "context": (
            "You are analyzing a traffic incident based on the following description. "
            "Carefully consider the actions and movements of all vehicles described "
            "to determine which one caused the accident."
        ),
        "instruction": "Identify the vehicle at fault and briefly explain why.",
    },
    "Victim Identification": {
        "context": (
            "You are analyzing a traffic incident based on the following description. "
            "Carefully consider which vehicle was adversely affected by the actions "
            "of the at-fault vehicle."
        ),
        "instruction": "Identify the victim vehicle and briefly explain why.",
    },
    "Violation Identification": {
        "context": (
            "You are analyzing a traffic incident based on the following description. "
            "Analyze the driving behavior described and identify "
            "any traffic rule violations that led to the accident."
        ),
        "instruction": "Specify the traffic rule that was violated and which vehicle violated it.",
    },
}


def build_caption_conditioned_prompt(question, benchmark, dense_caption):
    """Build a text-only prompt using the ground-truth caption instead of video."""
    info = OE_CONTEXT.get(benchmark, {"context": "", "instruction": ""})
    prompt = (
        f"{info['context']}\n\n"
        f"Event description:\n\"{dense_caption}\"\n\n"
        f"{question}\n\n"
        f"{info['instruction']}"
    )
    return prompt


def run_evaluation(model_name, test_data):
    print(f"\n{'=' * 70}", flush=True)
    print(f"  Caption-Conditioned Reasoning Experiment", flush=True)
    print(f"  Model: {model_name}", flush=True)
    print(f"  Input: Ground-truth dense captions (NO video frames)", flush=True)
    print(f"  Tasks: {', '.join(OE_BENCHMARKS)}", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    except (ValueError, AssertionError, TypeError):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(model_name)

    results = []
    total = len(test_data)
    skipped = 0

    for vid_idx, item in enumerate(test_data):
        video_path = item["video_path"]

        # Extract the detailed dense caption for this video
        dense_caption = None
        for qa in item["qa_pairs"]:
            if qa["benchmark"] == "Dense Captioning" and "detailed" in qa["question"].lower():
                dense_caption = qa["answer"]
                break

        if dense_caption is None:
            # Fallback: try the summary caption
            for qa in item["qa_pairs"]:
                if qa["benchmark"] == "Dense Captioning":
                    dense_caption = qa["answer"]
                    break

        if dense_caption is None:
            skipped += 1
            continue

        # Process only responsibility tasks
        target_qas = [qa for qa in item["qa_pairs"] if qa["benchmark"] in OE_BENCHMARKS]

        for qa in target_qas:
            benchmark = qa["benchmark"]
            question_text = build_caption_conditioned_prompt(
                qa["question"], benchmark, dense_caption
            )

            # Text-only message (no images/video)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question_text},
                    ],
                }
            ]

            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                response_text = output_text[0].strip()

                # Strip <think> blocks if present (Qwen3 reasoning traces)
                if "<think>" in response_text:
                    import re
                    response_text = re.sub(
                        r"<think>.*?</think>", "", response_text, flags=re.DOTALL
                    ).strip()

            except Exception as e:
                print(f"  [ERROR] Inference failed for {video_path} / {benchmark}: {e}")
                response_text = ""

            entry = {
                "video_path": video_path,
                "benchmark": benchmark,
                "question": qa["question"],
                "gt": qa["answer"],
                "pred": response_text,
                "pred_raw": response_text,
                "input_type": "caption_conditioned",
                "caption_used": dense_caption[:200],
            }
            results.append(entry)

        if (vid_idx + 1) % 50 == 0 or (vid_idx + 1) == total:
            print(f"  Progress: {vid_idx + 1}/{total} videos processed", flush=True)

    del model, processor
    torch.cuda.empty_cache()

    if skipped > 0:
        print(f"  Skipped {skipped} videos (no dense caption found)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Caption-conditioned reasoning experiment: text-only responsibility QA"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["2B", "8B"],
        help="Which Qwen3-VL model to use (2B or 8B)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a finetuned model (overrides --model)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Custom label for output filename",
    )
    args = parser.parse_args()

    if args.model is None and args.model_path is None:
        parser.error("Either --model or --model-path is required.")

    print(f"Loading test data from: {TEST_JSON}")
    with open(TEST_JSON, "r") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} videos")

    # Determine model path and label
    if args.model_path:
        model_name = args.model_path
        if args.model_name:
            model_label = args.model_name
        else:
            model_label = args.model_path.rstrip("/").split("/")[-1]
    else:
        model_name = MODEL_MAP[args.model]
        model_label = f"Qwen3-VL-{args.model}"

    start = time.time()
    results = run_evaluation(model_name, test_data)
    elapsed = time.time() - start
    print(f"\n  Evaluation took {elapsed:.1f}s ({len(results)} predictions)")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"results_caption_conditioned_{model_label}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  Results saved to: {output_path}")
    print(f"\n  Next step: evaluate with LLM judge:")
    print(f"    python evaluate_results.py --results {output_path}")
    print(f"\n  Compare judge scores to video-based results to measure")
    print(f"  whether the bottleneck is vision (scores improve) or reasoning (scores stay low).")


if __name__ == "__main__":
    main()
