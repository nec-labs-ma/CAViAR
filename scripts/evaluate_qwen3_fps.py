"""
Evaluation script for Qwen3-VL (2B, 8B) on accident QA test set.

Uses native video input (file path + fps) supported by the Qwen3-VL processor,
instead of decord frame extraction. This matches the approach used in
evaluate_cosmos2.py for fairer cross-model comparison.

Tasks evaluated:
  Multiple-choice: Weather & Light, Accident Type, Road Conditions
  Open-ended:      Faulter Identification, Victim Identification, Violation Identification

Usage (base models):
  python evaluate_qwen3_fps.py --model 2B       # run with Qwen3-VL-2B
  python evaluate_qwen3_fps.py --model 8B       # run with Qwen3-VL-8B
  python evaluate_qwen3_fps.py --model all      # run both sequentially
  python evaluate_qwen3_fps.py --model 2B --fps 4   # custom fps

Usage (finetuned models):
  python evaluate_qwen3_fps.py --model-path /path/to/finetuned_qwen3vl_2B
  python evaluate_qwen3_fps.py --model-path /path/to/finetuned --model-name my_ft_2B --fps 16
"""

import os
import json
import argparse
import time
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

import sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from caviar.paths import resolve_video_path, test_json as _default_test_json, results_dir as _default_results_dir, holdout_dir as _default_holdout_dir

# ── Configuration ────────────────────────────────────────────────────────────
TEST_JSON = str(_default_test_json())
OUTPUT_DIR = str(_default_results_dir())

MC_BENCHMARKS = ["Weather & Light", "Accident Type", "Road Conditions"]
OE_BENCHMARKS = ["Faulter Identification", "Victim Identification", "Violation Identification"]
ALL_BENCHMARKS = MC_BENCHMARKS + OE_BENCHMARKS

MODEL_MAP = {
    "2B": "Qwen/Qwen3-VL-2B-Instruct",
    "8B": "Qwen/Qwen3-VL-8B-Instruct",
}

DEFAULT_FPS = 4


# ── Prompt construction ──────────────────────────────────────────────────────
MC_CONTEXT = {
    "Weather & Light": (
        "You are analyzing a dashcam video of a traffic incident. "
        "Based on the visual evidence in the video frames, answer the following question."
    ),
    "Accident Type": (
        "You are analyzing a dashcam video of a traffic incident. "
        "Carefully observe the movement and interaction of vehicles to determine the type of collision."
    ),
    "Road Conditions": (
        "You are analyzing a dashcam video of a traffic incident. "
        "Examine the road surface visible in the video frames to determine the road conditions."
    ),
}

OE_CONTEXT = {
    "Faulter Identification": {
        "context": (
            "You are analyzing a dashcam video of a traffic incident. "
            "Carefully observe the actions and movements of all vehicles involved "
            "to determine which one caused the accident."
        ),
        "instruction": "Identify the vehicle at fault and briefly explain why.",
    },
    "Victim Identification": {
        "context": (
            "You are analyzing a dashcam video of a traffic incident. "
            "Carefully observe which vehicle was adversely affected by the actions "
            "of the at-fault vehicle."
        ),
        "instruction": "Identify the victim vehicle and briefly explain why.",
    },
    "Violation Identification": {
        "context": (
            "You are analyzing a dashcam video of a traffic incident. "
            "Analyze the driving behavior of the vehicles involved and identify "
            "any traffic rule violations that led to the accident."
        ),
        "instruction": "Specify the traffic rule that was violated and which vehicle violated it.",
    },
}


def build_mc_prompt(question, choices, benchmark):
    context = MC_CONTEXT.get(benchmark, "")
    options_str = "\n".join(
        f"  {chr(65 + i)}. {c}" for i, c in enumerate(choices)
    )
    prompt = (
        f"{context}\n\n"
        f"{question}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Answer with ONLY the letter (e.g., A, B, C, ...) of the correct option."
    )
    return prompt


def build_oe_prompt(question, benchmark):
    info = OE_CONTEXT.get(benchmark, {"context": "", "instruction": "Provide a concise answer."})
    prompt = (
        f"{info['context']}\n\n"
        f"{question}\n\n"
        f"{info['instruction']}"
    )
    return prompt


def parse_mc_answer(response_text, choices):
    text = response_text.strip()
    for i, c in enumerate(choices):
        letter = chr(65 + i)
        if text.upper().startswith(letter):
            return c
    for c in choices:
        if c.lower() in text.lower():
            return c
    return text


# ── Main inference engine ────────────────────────────────────────────────────
def run_evaluation(model_key, test_data, fps=DEFAULT_FPS, model_path=None):
    model_name = model_path if model_path else MODEL_MAP[model_key]
    print(f"\n{'=' * 70}")
    print(f"  Loading model: {model_name}")
    if model_path:
        print(f"  (finetuned model from local path)")
    print(f"  Video FPS: {fps}")
    print(f"{'=' * 70}\n")

    # Load model — try flash_attention_2 first (Qwen3-VL default), then fallback
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

    for vid_idx, item in enumerate(tqdm(test_data, desc="Videos", unit="vid")):
        video_path = item["video_path"]
        video_path = resolve_video_path(video_path)

        if not os.path.exists(video_path):
            print(f"  [SKIP] Video not found: {video_path}")
            continue

        # Build native video content (processor handles frame sampling via fps)
        video_content = [
            {"type": "video", "video": video_path, "fps": fps},
        ]

        target_qas = [qa for qa in item["qa_pairs"] if qa["benchmark"] in ALL_BENCHMARKS]

        for qa in target_qas:
            benchmark = qa["benchmark"]
            is_mc = benchmark in MC_BENCHMARKS
            choices = qa.get("choices", [])

            if is_mc:
                question_text = build_mc_prompt(qa["question"], choices, benchmark)
            else:
                question_text = build_oe_prompt(qa["question"], benchmark)

            # Build messages with native video input + question text
            messages = [
                {
                    "role": "user",
                    "content": video_content + [
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
                    fps=fps,
                )
                inputs = inputs.to(model.device)

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=4096,
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
            except Exception as e:
                print(f"  [ERROR] Inference failed for {video_path} / {benchmark}: {e}")
                response_text = ""

            entry = {
                "video_path": item["video_path"],
                "benchmark": benchmark,
                "question": qa["question"],
                "gt": qa.get("correct_answer", qa["answer"]),
            }

            if is_mc:
                entry["choices"] = choices
                entry["pred"] = parse_mc_answer(response_text, choices)
                entry["pred_raw"] = response_text
                entry["correct"] = (entry["pred"].lower() == entry["gt"].lower())
            else:
                entry["pred"] = response_text

            results.append(entry)

    del model, processor
    torch.cuda.empty_cache()

    return results


def compute_mc_accuracy(results):
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if r["benchmark"] in MC_BENCHMARKS:
            stats[r["benchmark"]]["total"] += 1
            if r.get("correct", False):
                stats[r["benchmark"]]["correct"] += 1
    return stats


def save_results(results, model_label):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"results_Qwen3-VL-{model_label}_fps.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {output_path}")

    stats = compute_mc_accuracy(results)
    if stats:
        print(f"\n  Multiple-Choice Accuracy (Qwen3-VL-{model_label}):")
        print(f"  {'Benchmark':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
        print(f"  {'-' * 55}")
        total_c, total_t = 0, 0
        for bench in MC_BENCHMARKS:
            if bench in stats:
                c = stats[bench]["correct"]
                t = stats[bench]["total"]
                total_c += c
                total_t += t
                acc = c / t * 100 if t > 0 else 0
                print(f"  {bench:<25} {c:>8} {t:>8} {acc:>9.2f}%")
        if total_t > 0:
            print(f"  {'-' * 55}")
            print(f"  {'Overall MC':<25} {total_c:>8} {total_t:>8} {total_c / total_t * 100:>9.2f}%")

    oe_count = sum(1 for r in results if r["benchmark"] in OE_BENCHMARKS)
    print(f"\n  Open-ended responses collected: {oe_count}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL on accident QA test set (native video + fps)")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["2B", "8B", "all"],
        help="Which base Qwen3-VL model to evaluate (2B, 8B, or all)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a finetuned Qwen3-VL model (local directory). Overrides --model.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Custom label for the finetuned model (used in output filenames). "
             "Defaults to the directory name of --model-path.",
    )
    parser.add_argument(
        "--test-json",
        type=str,
        default=TEST_JSON,
        help="Path to the test JSON file",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Frames per second to sample from video (default: 4)",
    )
    args = parser.parse_args()

    if args.model is None and args.model_path is None:
        parser.error("Either --model (for base models) or --model-path (for finetuned models) is required.")

    print(f"Loading test data from: {args.test_json}")
    with open(args.test_json, "r") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} videos")

    if args.model_path:
        # ── Finetuned model mode ──
        model_label = args.model_name or os.path.basename(args.model_path.rstrip("/"))
        print(f"Running finetuned model: {args.model_path}  (label: {model_label})")
        start = time.time()
        results = run_evaluation(
            model_key=None, test_data=test_data,
            fps=args.fps, model_path=args.model_path,
        )
        elapsed = time.time() - start
        print(f"  Qwen3-VL-{model_label} evaluation took {elapsed:.1f}s")
        save_results(results, model_label)
    else:
        # ── Base model mode ──
        models_to_run = ["2B", "8B"] if args.model == "all" else [args.model]
        for model_key in models_to_run:
            start = time.time()
            results = run_evaluation(model_key, test_data, fps=args.fps)
            elapsed = time.time() - start
            print(f"  Qwen3-VL-{model_key} evaluation took {elapsed:.1f}s")
            save_results(results, model_key)


if __name__ == "__main__":
    main()
