"""
Evaluation script for InternVL3 (2B, 8B, 14B) on accident QA test set.

Uses fps-based frame sampling: the number of frames extracted per video
depends on the video duration (num_frames = duration * fps), matching the
approach used by evaluate_cosmos2.py and evaluate_qwen3_fps.py for fairer
cross-model comparison.

Tasks evaluated:
  Multiple-choice: Weather & Light, Accident Type, Road Conditions
  Open-ended:      Faulter Identification, Victim Identification, Violation Identification

Usage (base models):
  python evaluate_internvl3_fps.py --model 2B --fps 4
  python evaluate_internvl3_fps.py --model 8B --fps 16
  python evaluate_internvl3_fps.py --model 14B --fps 4
  python evaluate_internvl3_fps.py --model all --fps 4

Usage (finetuned models):
  python evaluate_internvl3_fps.py --model-path /path/to/finetuned --fps 16
  python evaluate_internvl3_fps.py --model-path /path/to/finetuned --model-name my_ft_8B --fps 4
"""

import os
import json
import argparse
import math
import time
import numpy as np
import torch
import torchvision.transforms as T
from collections import defaultdict
from tqdm import tqdm
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

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
    "2B":  "OpenGVLab/InternVL3-2B",
    "8B":  "OpenGVLab/InternVL3-8B",
}

DEFAULT_FPS = 4

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ── Image / Video preprocessing (from official InternVL3 repo) ───────────────
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


MAX_FRAMES = 64


def load_video_fps(video_path, target_fps=4, input_size=448, max_num=1, max_frames=MAX_FRAMES):
    """Load video frames using fps-based sampling and preprocess for InternVL3.

    Instead of a fixed number of segments, the number of frames is determined
    by: num_frames = video_duration_seconds * target_fps (minimum 1 frame),
    capped at max_frames to stay within InternVL3's context window.
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    video_fps = float(vr.get_avg_fps())
    duration = total_frames / video_fps

    num_segments = max(1, round(duration * target_fps))
    if num_segments > max_frames:
        num_segments = max_frames

    seg_size = float(total_frames) / num_segments
    frame_indices = np.array([
        int(seg_size / 2 + seg_size * idx) for idx in range(num_segments)
    ])
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


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
    """Build a multiple-choice prompt with benchmark-specific context."""
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
    """Build an open-ended prompt with benchmark-specific context."""
    info = OE_CONTEXT.get(benchmark, {"context": "", "instruction": "Provide a concise answer."})
    prompt = (
        f"{info['context']}\n\n"
        f"{question}\n\n"
        f"{info['instruction']}"
    )
    return prompt


def parse_mc_answer(response_text, choices):
    """Extract predicted choice from model response."""
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
def run_evaluation(model_key, test_data, fps=DEFAULT_FPS, max_frames=MAX_FRAMES, model_path=None):
    model_name = model_path if model_path else MODEL_MAP[model_key]
    print(f"\n{'=' * 70}")
    print(f"  Loading model: {model_name}")
    if model_path:
        print(f"  (finetuned model from local path)")
    print(f"  Video FPS: {fps}  |  Max frames: {max_frames}")
    print(f"{'=' * 70}\n")

    try:
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval().cuda()
    except TypeError:
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=False
    )

    generation_config = dict(max_new_tokens=512, do_sample=False)

    results = []
    total = len(test_data)

    for vid_idx, item in enumerate(tqdm(test_data, desc="Videos", unit="vid")):
        video_path = item["video_path"]
        video_path = resolve_video_path(video_path)

        if not os.path.exists(video_path):
            print(f"  [SKIP] Video not found: {video_path}")
            continue

        try:
            pixel_values, num_patches_list = load_video_fps(
                video_path, target_fps=fps, max_num=1, max_frames=max_frames
            )
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
        except Exception as e:
            print(f"  [ERROR] Failed to load {video_path}: {e}")
            continue

        video_prefix = ''.join([
            f'Frame{i + 1}: <image>\n' for i in range(len(num_patches_list))
        ])

        target_qas = [qa for qa in item["qa_pairs"] if qa["benchmark"] in ALL_BENCHMARKS]

        for qa in target_qas:
            benchmark = qa["benchmark"]
            is_mc = benchmark in MC_BENCHMARKS
            choices = qa.get("choices", [])

            if is_mc:
                question_text = build_mc_prompt(qa["question"], choices, benchmark)
            else:
                question_text = build_oe_prompt(qa["question"], benchmark)

            full_question = video_prefix + question_text

            try:
                response = model.chat(
                    tokenizer, pixel_values, full_question, generation_config,
                    num_patches_list=num_patches_list, history=None, return_history=False
                )
                response_text = response.strip() if isinstance(response, str) else response[0].strip()
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

    del model, tokenizer
    torch.cuda.empty_cache()

    return results


def compute_mc_accuracy(results):
    """Compute per-benchmark accuracy for multiple-choice tasks."""
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if r["benchmark"] in MC_BENCHMARKS:
            stats[r["benchmark"]]["total"] += 1
            if r.get("correct", False):
                stats[r["benchmark"]]["correct"] += 1
    return stats


def save_results(results, model_label):
    """Save results to JSON and print summary."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"results_InternVL3-{model_label}_8B_4fps.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {output_path}")

    stats = compute_mc_accuracy(results)
    if stats:
        print(f"\n  Multiple-Choice Accuracy (InternVL3-{model_label}):")
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


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate InternVL3 on accident QA test set (fps-based sampling)")
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["2B", "8B", "14B", "all"],
        help="Which base InternVL3 model to evaluate (2B, 8B, 14B, or all)"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to a finetuned InternVL3 model (local directory). Overrides --model."
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Custom label for the finetuned model (used in output filenames). "
             "Defaults to the directory name of --model-path."
    )
    parser.add_argument(
        "--test-json", type=str, default=TEST_JSON,
        help="Path to the test JSON file"
    )
    parser.add_argument(
        "--fps", type=int, default=DEFAULT_FPS,
        help="Frames per second to sample from video (default: 4)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=MAX_FRAMES,
        help="Maximum number of frames per video to avoid exceeding context window (default: 64)"
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
            fps=args.fps, max_frames=args.max_frames, model_path=args.model_path,
        )
        elapsed = time.time() - start
        print(f"  InternVL3-{model_label} evaluation took {elapsed:.1f}s")
        save_results(results, model_label)
    else:
        # ── Base model mode ──
        models_to_run = ["2B", "8B", "14B"] if args.model == "all" else [args.model]
        for model_key in models_to_run:
            start = time.time()
            results = run_evaluation(model_key, test_data, fps=args.fps, max_frames=args.max_frames)
            elapsed = time.time() - start
            print(f"  InternVL3-{model_key} evaluation took {elapsed:.1f}s")
            save_results(results, model_key)


if __name__ == "__main__":
    main()
