"""
Evaluation script for accident QA results using VLMEvalKit utilities.

Evaluates:
  Multiple-choice (Weather & Light, Accident Type, Road Conditions):
    - Accuracy (per-benchmark and overall)
  Open-ended (Faulter Identification, Victim Identification, Violation Identification):
    - Similarity metrics: BLEU-1/2/3/4, ROUGE-L, BERTScore
    - LLM-as-judge: GPT-4o (0/1/2 scoring via VLMEvalKit OpenAIWrapper)

Usage:
  conda activate vlmeval
  export OPENAI_API_KEY=YOUR_KEY
  export CAVIAR_VLMEVALKIT_PATH=/path/to/VLMEvalKit   # optional
  python evaluate_results.py --results results/results_InternVL3-2B.json
  python evaluate_results.py --results results/results_InternVL3-2B.json --skip-judge  # skip LLM judge
"""

import os
import json
import argparse
import re
import time
from collections import defaultdict
import sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from caviar.paths import results_dir as _default_results_dir

# ── Add VLMEvalKit to path ──────────────────────────────────────────────────
VLMEVALKIT_PATH = os.environ.get("CAVIAR_VLMEVALKIT_PATH", "")
if VLMEVALKIT_PATH:
    sys.path.insert(0, VLMEVALKIT_PATH)

# ── Configuration ────────────────────────────────────────────────────────────
MC_BENCHMARKS = ["Weather & Light", "Accident Type", "Road Conditions"]
OE_BENCHMARKS = ["Faulter Identification", "Victim Identification", "Violation Identification"]

OUTPUT_DIR = str(_default_results_dir())

# ── LLM-as-Judge prompt (adapted from VLMEvalKit qbench_video.py) ───────────
JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator for traffic accident analysis. "
    "You will compare a model's prediction against a ground truth answer "
    "and rate how well the prediction matches."
)

JUDGE_USER_PROMPT = """Given the following question about a traffic accident video:

Question: {question}

Ground truth answer: {gt}

Model prediction: {pred}

Rate the model's prediction on a scale of 0 to 5:
  5 = The prediction perfectly matches the ground truth in meaning, identifying the same vehicle(s) and the same reasoning.
  4 = The prediction almost completely matches — minor wording differences but all key facts are correct.
  3 = The prediction mostly matches — identifies the correct vehicle(s) but misses some reasoning details or includes minor inaccuracies.
  2 = The prediction partially matches — captures some correct aspects but misses key details or gets some elements wrong.
  1 = The prediction slightly matches — only a small part is relevant or correct, most is wrong or missing.
  0 = The prediction does not match the ground truth at all, or is irrelevant.

Respond with ONLY a single line in this exact format:
Score: <number>"""


# ══════════════════════════════════════════════════════════════════════════════
# MULTIPLE-CHOICE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_mc(results):
    """Compute per-benchmark and overall accuracy for multiple-choice tasks."""
    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        if r["benchmark"] not in MC_BENCHMARKS:
            continue
        bench = r["benchmark"]
        stats[bench]["total"] += 1
        if r.get("correct", False):
            stats[bench]["correct"] += 1

    mc_report = {}
    total_c, total_t = 0, 0
    for bench in MC_BENCHMARKS:
        if bench in stats:
            c = stats[bench]["correct"]
            t = stats[bench]["total"]
            total_c += c
            total_t += t
            mc_report[bench] = {
                "correct": c,
                "total": t,
                "accuracy": round(c / t * 100, 2) if t > 0 else 0.0,
            }
    if total_t > 0:
        mc_report["Overall"] = {
            "correct": total_c,
            "total": total_t,
            "accuracy": round(total_c / total_t * 100, 2),
        }
    return mc_report


# ══════════════════════════════════════════════════════════════════════════════
# SIMILARITY METRICS
# ══════════════════════════════════════════════════════════════════════════════
def compute_rouge_l(prediction, reference):
    """Compute ROUGE-L F1 score using longest common subsequence."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    # LCS via dynamic programming
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    precision = lcs_len / n if n > 0 else 0.0
    recall = lcs_len / m if m > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_bleu_scores(predictions, references):
    """Compute BLEU-1 through BLEU-4 using nltk."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    smoother = SmoothingFunction().method1
    bleu_scores = {f"BLEU-{i}": [] for i in range(1, 5)}

    for pred, ref in zip(predictions, references):
        ref_tokens = ref.lower().split()
        pred_tokens = pred.lower().split()

        if len(pred_tokens) == 0:
            for i in range(1, 5):
                bleu_scores[f"BLEU-{i}"].append(0.0)
            continue

        for i in range(1, 5):
            weights = tuple([1.0 / i] * i + [0.0] * (4 - i))
            score = sentence_bleu(
                [ref_tokens], pred_tokens,
                weights=weights,
                smoothing_function=smoother,
            )
            bleu_scores[f"BLEU-{i}"].append(score)

    avg_scores = {}
    for key, vals in bleu_scores.items():
        avg_scores[key] = round(sum(vals) / len(vals) * 100, 2) if vals else 0.0
    return avg_scores


def compute_bertscore(predictions, references, device=None):
    """Compute BERTScore using the bert_score package."""
    import torch
    from bert_score import score as bert_score_fn

    # Auto-select device: use CUDA_VISIBLE_DEVICES GPU if set, else CPU
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # Process in batches to avoid memory issues
    batch_size = 64
    all_P, all_R, all_F1 = [], [], []
    for i in range(0, len(predictions), batch_size):
        batch_preds = predictions[i:i + batch_size]
        batch_refs = references[i:i + batch_size]
        P, R, F1 = bert_score_fn(
            batch_preds, batch_refs,
            lang="en",
            verbose=False,
            rescale_with_baseline=True,
            device=device,
        )
        all_P.append(P)
        all_R.append(R)
        all_F1.append(F1)

    import torch as th
    all_P = th.cat(all_P)
    all_R = th.cat(all_R)
    all_F1 = th.cat(all_F1)

    return {
        "BERTScore-P": round(all_P.mean().item() * 100, 2),
        "BERTScore-R": round(all_R.mean().item() * 100, 2),
        "BERTScore-F1": round(all_F1.mean().item() * 100, 2),
    }


def compute_rouge_l_batch(predictions, references):
    """Compute average ROUGE-L over a batch."""
    scores = [compute_rouge_l(p, r) for p, r in zip(predictions, references)]
    avg = sum(scores) / len(scores) if scores else 0.0
    return {"ROUGE-L": round(avg * 100, 2)}


def evaluate_similarity(oe_entries):
    """Compute all similarity metrics for open-ended tasks, per-benchmark and overall."""
    bench_data = defaultdict(lambda: {"preds": [], "gts": []})

    for entry in oe_entries:
        bench = entry["benchmark"]
        bench_data[bench]["preds"].append(entry.get("pred", ""))
        bench_data[bench]["gts"].append(entry.get("gt", ""))

    # Also collect overall
    all_preds = [e.get("pred", "") for e in oe_entries]
    all_gts = [e.get("gt", "") for e in oe_entries]
    bench_data["Overall"] = {"preds": all_preds, "gts": all_gts}

    sim_report = {}
    for bench_name in OE_BENCHMARKS + ["Overall"]:
        if bench_name not in bench_data:
            continue
        preds = bench_data[bench_name]["preds"]
        gts = bench_data[bench_name]["gts"]
        if not preds:
            continue

        print(f"  Computing similarity for: {bench_name} ({len(preds)} samples)")

        scores = {}
        scores.update(compute_bleu_scores(preds, gts))
        scores.update(compute_rouge_l_batch(preds, gts))
        scores.update(compute_bertscore(preds, gts))
        scores["num_samples"] = len(preds)

        sim_report[bench_name] = scores

    return sim_report


# ══════════════════════════════════════════════════════════════════════════════
# LLM-AS-JUDGE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def parse_judge_score(response_text):
    """Extract numeric score (0-5) from judge response."""
    if not response_text:
        return -1
    # Try to find "Score: X" pattern
    match = re.search(r'Score:\s*(\d)', response_text)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 5:
            return score
    # Fallback: look for a standalone digit 0-5
    match = re.search(r'\b([0-5])\b', response_text)
    if match:
        return int(match.group(1))
    return -1


def evaluate_llm_judge(oe_entries, judge_model_name="gpt-4o"):
    """Use GPT-4o as judge to score open-ended predictions."""
    from vlmeval.api import OpenAIWrapper

    print(f"\n  Initializing LLM judge: {judge_model_name}")
    judge = OpenAIWrapper(
        judge_model_name,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        retry=3,
        verbose=False,
        temperature=0,
        max_tokens=64,
    )

    # Verify the judge is working
    if not judge.working():
        print("  [ERROR] LLM judge is not responding. Check your OPENAI_API_KEY.")
        print("  Skipping LLM-as-judge evaluation.")
        return {}, []

    bench_scores = defaultdict(list)
    details = []
    total = len(oe_entries)

    for idx, entry in enumerate(oe_entries):
        prompt = JUDGE_USER_PROMPT.format(
            question=entry["question"],
            gt=entry["gt"],
            pred=entry.get("pred", ""),
        )

        response = judge.generate(prompt)
        score = parse_judge_score(response)

        detail = {
            "video_path": entry.get("video_path", ""),
            "benchmark": entry["benchmark"],
            "question": entry["question"],
            "gt": entry["gt"],
            "pred": entry.get("pred", ""),
            "judge_response": response,
            "judge_score": score,
        }
        details.append(detail)
        bench_scores[entry["benchmark"]].append(score)

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f"    Judge progress: {idx + 1}/{total}")

    # Compute averages
    judge_report = {}
    all_scores = []
    for bench in OE_BENCHMARKS:
        if bench in bench_scores:
            valid = [s for s in bench_scores[bench] if s >= 0]
            invalid = len(bench_scores[bench]) - len(valid)
            avg = sum(valid) / len(valid) if valid else 0.0
            judge_report[bench] = {
                "avg_score": round(avg, 3),
                "max_score": 5.0,
                "num_samples": len(bench_scores[bench]),
                "num_valid": len(valid),
                "num_failed": invalid,
                "score_distribution": {
                    str(i): valid.count(i) for i in range(6)
                },
            }
            all_scores.extend(valid)

    if all_scores:
        judge_report["Overall"] = {
            "avg_score": round(sum(all_scores) / len(all_scores), 3),
            "max_score": 5.0,
            "num_samples": len(all_scores),
        }

    return judge_report, details


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════
def print_mc_report(mc_report):
    print(f"\n{'=' * 65}")
    print("  MULTIPLE-CHOICE ACCURACY")
    print(f"{'=' * 65}")
    print(f"  {'Benchmark':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-' * 55}")
    for bench in MC_BENCHMARKS + ["Overall"]:
        if bench in mc_report:
            r = mc_report[bench]
            print(f"  {bench:<25} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>9.2f}%")
            if bench == MC_BENCHMARKS[-1] and "Overall" in mc_report:
                print(f"  {'-' * 55}")
    print()


def print_similarity_report(sim_report):
    print(f"\n{'=' * 65}")
    print("  OPEN-ENDED SIMILARITY METRICS")
    print(f"{'=' * 65}")

    metrics = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L",
               "BERTScore-P", "BERTScore-R", "BERTScore-F1"]

    header = f"  {'Benchmark':<25}"
    for m in metrics:
        header += f" {m:>10}"
    print(header)
    print(f"  {'-' * (25 + 10 * len(metrics) + len(metrics))}")

    for bench in OE_BENCHMARKS + ["Overall"]:
        if bench in sim_report:
            row = f"  {bench:<25}"
            for m in metrics:
                val = sim_report[bench].get(m, 0.0)
                row += f" {val:>10.2f}"
            print(row)
            if bench == OE_BENCHMARKS[-1] and "Overall" in sim_report:
                print(f"  {'-' * (25 + 10 * len(metrics) + len(metrics))}")
    print()


def print_judge_report(judge_report):
    print(f"\n{'=' * 90}")
    print("  LLM-AS-JUDGE SCORES (GPT-4o, scale 0-5)")
    print(f"{'=' * 90}")
    header = f"  {'Benchmark':<25} {'Avg':>6} {'N':>6}"
    for i in range(6):
        header += f" {'S=' + str(i):>6}"
    print(header)
    print(f"  {'-' * 85}")

    for bench in OE_BENCHMARKS + ["Overall"]:
        if bench in judge_report:
            r = judge_report[bench]
            dist = r.get("score_distribution", {})
            n = r.get("num_samples", r.get("num_valid", 0))
            row = f"  {bench:<25} {r['avg_score']:>6.3f} {n:>6}"
            for i in range(6):
                val = dist.get(str(i), "-")
                row += f" {str(val):>6}"
            print(row)
            if bench == OE_BENCHMARKS[-1] and "Overall" in judge_report:
                print(f"  {'-' * 85}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Evaluate accident QA results")
    parser.add_argument(
        "--results", type=str, required=True,
        help="Path to results JSON file (e.g. results/results_InternVL3-2B.json)"
    )
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Skip LLM-as-judge evaluation (only compute accuracy and similarity)"
    )
    parser.add_argument(
        "--judge-model", type=str, default="gpt-4o",
        help="Judge model name (default: gpt-4o)"
    )
    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results}")
    with open(args.results, "r") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} entries")

    mc_entries = [r for r in results if r["benchmark"] in MC_BENCHMARKS]
    oe_entries = [r for r in results if r["benchmark"] in OE_BENCHMARKS]
    print(f"  MC entries: {len(mc_entries)}")
    print(f"  OE entries: {len(oe_entries)}")

    # ── 1. Multiple-choice accuracy ──────────────────────────────────────────
    print("\n[1/3] Computing multiple-choice accuracy...")
    mc_report = evaluate_mc(results)
    print_mc_report(mc_report)

    # ── 2. Similarity metrics ────────────────────────────────────────────────
    print("[2/3] Computing similarity metrics...")
    sim_report = evaluate_similarity(oe_entries)
    print_similarity_report(sim_report)

    # ── 3. LLM-as-judge ─────────────────────────────────────────────────────
    judge_report = {}
    judge_details = []
    if not args.skip_judge:
        print(f"[3/3] Running LLM-as-judge ({args.judge_model})...")
        judge_report, judge_details = evaluate_llm_judge(oe_entries, args.judge_model)
        print_judge_report(judge_report)
    else:
        print("[3/3] LLM-as-judge: SKIPPED (--skip-judge)")

    # ── Save report ──────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.results))[0]
    # e.g. results_InternVL3-2B -> eval_InternVL3-2B
    eval_name = basename.replace("results_", "eval_")
    output_path = os.path.join(OUTPUT_DIR, f"{eval_name}.json")

    report = {
        "source_file": os.path.abspath(args.results),
        "mc_results": mc_report,
        "oe_similarity": sim_report,
        "oe_judge": judge_report,
    }
    if judge_details:
        report["oe_judge_details"] = judge_details

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull evaluation report saved to: {output_path}")


if __name__ == "__main__":
    main()
