"""
Human validation UI for LLM-as-Judge scoring.

Samples 30–50 open-ended responses from Cosmos2-2B_merged, lets 2 humans score
with the same 0–5 rubric as GPT-4o, and reports agreement/correlation.

Usage:
  # 1. Generate sample and launch UI for annotation
  python human_validation.py --eval results/16_fps/eval_Cosmos2-2B_merged.json --sample 45

  # 2. Run analysis on saved annotations (after both humans have scored)
  python human_validation.py --analyze human_validation_Cosmos2-2B_merged.json
"""

import os
import json
import argparse
import random
from collections import defaultdict
import sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from caviar.paths import resolve_video_path, video_root

OE_BENCHMARKS = ["Faulter Identification", "Victim Identification", "Violation Identification"]

RUBRIC = """Rate the model's prediction on a scale of 0 to 5:
  5 = The prediction perfectly matches the ground truth in meaning, identifying the same vehicle(s) and the same reasoning.
  4 = The prediction almost completely matches — minor wording differences but all key facts are correct.
  3 = The prediction mostly matches — identifies the correct vehicle(s) but misses some reasoning details or includes minor inaccuracies.
  2 = The prediction partially matches — captures some correct aspects but misses key details or gets some elements wrong.
  1 = The prediction slightly matches — only a small part is relevant or correct, most is wrong or missing.
  0 = The prediction does not match the ground truth at all, or is irrelevant."""


def remap_video_path(path):
    """Map a dataset video_path to a local file under CAVIAR_VIDEO_ROOT."""
    return resolve_video_path(path)


def load_and_sample(eval_path, n_sample=45, seed=42):
    """Load eval file and sample n_sample items stratified across OE benchmarks."""
    with open(eval_path) as f:
        data = json.load(f)

    details = data.get("oe_judge_details", [])
    if not details:
        raise ValueError(f"No oe_judge_details in {eval_path}")

    # Stratify by benchmark
    by_bench = defaultdict(list)
    for d in details:
        bench = d.get("benchmark", "")
        if bench in OE_BENCHMARKS and d.get("judge_score") is not None:
            by_bench[bench].append(d)

    # Sample ~n_sample // 3 per benchmark
    per_bench = max(10, n_sample // 3)
    random.seed(seed)
    sampled = []
    for bench in OE_BENCHMARKS:
        pool = by_bench.get(bench, [])
        if len(pool) <= per_bench:
            sampled.extend(pool)
        else:
            sampled.extend(random.sample(pool, per_bench))

    # Shuffle to avoid benchmark clustering
    random.shuffle(sampled)

    # Add local video path and index
    for i, item in enumerate(sampled):
        item["_idx"] = i
        item["_video_local"] = remap_video_path(item.get("video_path", ""))

    return sampled


def run_ui(sampled, output_path, port=7860):
    """Launch Gradio UI for human annotation."""
    import gradio as gr

    annotations = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            annotations = json.load(f)
        # Ensure we have entries for all sampled
        while len(annotations) < len(sampled):
            annotations.append({
                "human1_score": None,
                "human2_score": None,
            })

    current_idx = [0]  # use list for closure

    def get_item(idx):
        if idx < 0 or idx >= len(sampled):
            return None, None, None, None, None, None, idx
        s = sampled[idx]
        video_path = s["_video_local"]
        orig_path = s.get("video_path", "")
        if not os.path.exists(video_path) and orig_path and os.path.exists(orig_path):
            video_path = orig_path
        elif not os.path.exists(video_path):
            video_path = None
        return (
            video_path,
            s.get("question", ""),
            s.get("gt", ""),
            s.get("pred", ""),
            s.get("judge_score", -1),
            annotations[idx]["human1_score"] if idx < len(annotations) else None,
            annotations[idx]["human2_score"] if idx < len(annotations) else None,
            idx,
        )

    def render(idx):
        out = get_item(idx)
        video, q, gt, pred, gpt_score, h1, h2, _ = out
        return video, q, gt, pred, str(gpt_score) if gpt_score >= 0 else "N/A", h1, h2, idx, idx

    def save_current(h1, h2, idx):
        try:
            h1_val = int(h1) if h1 is not None and str(h1).strip() != "" else None
            h2_val = int(h2) if h2 is not None and str(h2).strip() != "" else None
            if 0 <= idx < len(annotations):
                annotations[idx]["human1_score"] = h1_val
                annotations[idx]["human2_score"] = h2_val
        except (ValueError, TypeError):
            pass
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(annotations, f, indent=2)

    def save_and_next(h1, h2, idx):
        save_current(h1, h2, idx)
        next_idx = min(idx + 1, len(sampled) - 1)
        return list(render(next_idx))

    def save_and_prev(h1, h2, idx):
        save_current(h1, h2, idx)
        prev_idx = max(idx - 1, 0)
        return list(render(prev_idx))

    def go_to_idx(h1, h2, idx, new_idx):
        save_current(h1, h2, idx)
        try:
            target = min(max(0, int(new_idx)), len(sampled) - 1)
        except (ValueError, TypeError):
            target = 0
        return list(render(target))

    with gr.Blocks(title="Human Validation - LLM-as-Judge", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## Human Validation: Score model predictions (0–5)")
        gr.Markdown(f"**Rubric:** {RUBRIC}")

        with gr.Row():
            vid = gr.Video(label="Video", autoplay=False)
        with gr.Row():
            question = gr.Textbox(label="Question", lines=2, interactive=False)
        with gr.Row():
            gt = gr.Textbox(label="Ground Truth", lines=3, interactive=False)
        with gr.Row():
            pred = gr.Textbox(label="Model Prediction", lines=4, interactive=False)
        with gr.Row():
            gpt_score = gr.Textbox(label="GPT-4o Score", interactive=False)
        with gr.Row():
            h1_in = gr.Number(label="Human 1 Score (0–5)", value=None, precision=0, minimum=0, maximum=5)
            h2_in = gr.Number(label="Human 2 Score (0–5)", value=None, precision=0, minimum=0, maximum=5)
        with gr.Row():
            idx_num = gr.Number(label="Sample Index", value=0, precision=0, minimum=0, maximum=len(sampled) - 1)
        with gr.Row():
            prev_btn = gr.Button("← Previous")
            next_btn = gr.Button("Next →")
            go_btn = gr.Button("Go to index")

        idx_state = gr.State(0)

        def init_ui():
            return list(render(0))

        demo.load(init_ui, outputs=[vid, question, gt, pred, gpt_score, h1_in, h2_in, idx_state, idx_num])

        def on_next(h1, h2, idx):
            return save_and_next(h1, h2, idx)

        def on_prev(h1, h2, idx):
            return save_and_prev(h1, h2, idx)

        def on_go(h1, h2, idx, new_idx):
            return go_to_idx(h1, h2, idx, new_idx)

        next_btn.click(
            on_next,
            inputs=[h1_in, h2_in, idx_state],
            outputs=[vid, question, gt, pred, gpt_score, h1_in, h2_in, idx_state, idx_num],
        )
        prev_btn.click(
            on_prev,
            inputs=[h1_in, h2_in, idx_state],
            outputs=[vid, question, gt, pred, gpt_score, h1_in, h2_in, idx_state, idx_num],
        )
        go_btn.click(
            on_go,
            inputs=[h1_in, h2_in, idx_state, idx_num],
            outputs=[vid, question, gt, pred, gpt_score, h1_in, h2_in, idx_state, idx_num],
        )

        gr.Markdown(f"*Samples: 0–{len(sampled)-1} | Annotations saved to: {output_path}*")

    port = int(os.environ.get("GRADIO_SERVER_PORT", port))
    launch_kw = dict(
        server_name="0.0.0.0",
        server_port=port,
        share=True,
    )
    vroot = str(video_root())
    if os.path.isdir(vroot):
        launch_kw["allowed_paths"] = [vroot]
    demo.launch(**launch_kw)


def run_analysis(annot_path, sampled_path=None):
    """Compute agreement and correlation from saved annotations."""
    with open(annot_path) as f:
        annotations = json.load(f)

    # Load GPT scores from sampled items (same order as annotations)
    gpt_scores = []
    samples = []
    if sampled_path and os.path.exists(sampled_path):
        with open(sampled_path) as f:
            samples = json.load(f)
        gpt_scores = [s.get("judge_score") for s in samples]

    # Parse annotations (list of {human1_score, human2_score})
    h1 = []
    h2 = []
    gpt = []
    for i, ann in enumerate(annotations):
        if isinstance(ann, dict):
            h1v = ann.get("human1_score", ann.get("h1"))
            h2v = ann.get("human2_score", ann.get("h2"))
            gptv = gpt_scores[i] if i < len(gpt_scores) else None
        else:
            h1v, h2v, gptv = None, None, None
        if h1v is not None and h2v is not None and gptv is not None:
            h1.append(int(h1v))
            h2.append(int(h2v))
            gpt.append(int(gptv))

    # Filter to complete triplets
    valid = list(zip(h1, h2, gpt))
    if not valid:
        print("No valid (human1, human2, gpt) triplets found. Ensure annotations are complete.")
        return
    h1, h2, gpt = zip(*valid)
    n = len(h1)

    print("\n" + "=" * 70)
    print("  HUMAN VALIDATION REPORT (Cosmos2-2B_merged)")
    print("=" * 70)
    print(f"  Sample size: {n} (both humans + GPT-4o scores present)")
    print()

    # Cohen's kappa (human1 vs human2)
    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(h1, h2)
        print(f"  Inter-rater agreement (Human 1 vs Human 2):")
        print(f"    Cohen's κ = {kappa:.3f}")
        print()
    except ImportError:
        print("  [sklearn not installed] Cohen's kappa skipped.")
        kappa = None

    # Pearson and Spearman (humans vs GPT-4o)
    try:
        from scipy.stats import pearsonr, spearmanr
        r_pearson_h1, p1 = pearsonr(h1, gpt)
        r_pearson_h2, p2 = pearsonr(h2, gpt)
        r_spearman_h1, s1 = spearmanr(h1, gpt)
        r_spearman_h2, s2 = spearmanr(h2, gpt)
        h_avg = [(a + b) / 2 for a, b in zip(h1, h2)]
        r_pearson_avg, pa = pearsonr(h_avg, gpt)
        r_spearman_avg, sa = spearmanr(h_avg, gpt)

        print("  Correlation with GPT-4o (LLM-as-Judge):")
        print(f"    Human 1: Pearson r = {r_pearson_h1:.3f} (p={p1:.4f}), Spearman ρ = {r_spearman_h1:.3f}")
        print(f"    Human 2: Pearson r = {r_pearson_h2:.3f} (p={p2:.4f}), Spearman ρ = {r_spearman_h2:.3f}")
        print(f"    Human avg: Pearson r = {r_pearson_avg:.3f}, Spearman ρ = {r_spearman_avg:.3f}")
        print()
    except ImportError:
        print("  [scipy not installed] Correlation skipped.")

    # Per-benchmark if we have samples with benchmark info
    if samples and isinstance(samples[0], dict):
        by_bench = defaultdict(lambda: {"h1": [], "h2": [], "gpt": []})
        for i, s in enumerate(samples):
            if i >= len(annotations):
                break
            ann = annotations[i]
            if isinstance(ann, dict) and ann.get("human1_score") is not None and ann.get("human2_score") is not None and s.get("judge_score") is not None:
                bench = s.get("benchmark", "?")
                by_bench[bench]["h1"].append(ann["human1_score"])
                by_bench[bench]["h2"].append(ann["human2_score"])
                by_bench[bench]["gpt"].append(s["judge_score"])

        if by_bench:
            print("  Per-benchmark:")
            for bench in OE_BENCHMARKS:
                if bench in by_bench and len(by_bench[bench]["h1"]) >= 2:
                    try:
                        k = cohen_kappa_score(by_bench[bench]["h1"], by_bench[bench]["h2"])
                        r, _ = pearsonr(by_bench[bench]["h1"], by_bench[bench]["gpt"])
                        print(f"    {bench}: n={len(by_bench[bench]['h1'])}, κ={k:.3f}, r(H1,GPT)={r:.3f}")
                    except Exception:
                        pass

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Human validation for LLM-as-Judge")
    parser.add_argument(
        "--eval",
        type=str,
        default="results/16_fps/eval_Cosmos2-2B_merged.json",
        help="Path to eval JSON with oe_judge_details",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=45,
        help="Number of samples to draw (stratified, default: 45)",
    )
    parser.add_argument(
        "--analyze",
        type=str,
        default=None,
        help="Path to saved annotations JSON; run analysis instead of UI",
    )
    parser.add_argument(
        "--sample-file",
        type=str,
        default=None,
        help="Path to sampled items JSON (for analysis, to get GPT scores)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Gradio server port (default: 7861). Override with GRADIO_SERVER_PORT env.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_path = args.eval if os.path.isabs(args.eval) else os.path.join(script_dir, args.eval)

    if args.analyze:
        annot_path = args.analyze if os.path.isabs(args.analyze) else os.path.join(script_dir, args.analyze)
        sample_path = args.sample_file
        if not sample_path:
            # Default: human_validation_X.json -> human_validation_X_samples.json
            base = os.path.splitext(annot_path)[0]
            sample_path = base + "_samples.json"
        if sample_path and not os.path.isabs(sample_path):
            sample_path = os.path.join(script_dir, sample_path)
        run_analysis(annot_path, sample_path)
    else:
        sampled = load_and_sample(eval_path, n_sample=args.sample)
        # Save sampled items for analysis
        sample_output = os.path.join(
            script_dir,
            "results",
            "human_validation_Cosmos2-2B_merged_samples.json",
        )
        os.makedirs(os.path.dirname(sample_output), exist_ok=True)
        with open(sample_output, "w") as f:
            json.dump([{k: v for k, v in s.items() if not k.startswith("_")} for s in sampled], f, indent=2)
        print(f"Sampled {len(sampled)} items, saved to {sample_output}")

        annot_output = os.path.join(script_dir, "results", "human_validation_Cosmos2-2B_merged.json")
        run_ui(sampled, annot_output, port=args.port)


if __name__ == "__main__":
    main()
