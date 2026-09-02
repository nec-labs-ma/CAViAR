# Scripts

Run from the **repository root** (`CAViAR/`) so `caviar/` is importable:

```bash
python scripts/dataset_stats.py
python scripts/evaluate_results.py --results examples/sample_results.json --skip-judge
```

| Script | Purpose |
|--------|---------|
| `dataset_stats.py` | Print val/test (Nexar) video / QA counts |
| `evaluate_results.py` | MCQ accuracy, BLEU/ROUGE/BERTScore, optional GPT-4o judge |
| `generate_echo_baseline.py` | Degenerate echo baseline (`--test data/test.json`) |
| `generate_generic_baseline.py` | Majority / template baseline |
| `evaluate_qwen3.py` / `_fps.py` | Qwen3-VL inference on `data/test.json` |
| `evaluate_internvl3.py` / `_fps.py` | InternVL3 inference |
| `evaluate_cosmos2.py` | Cosmos-Reason2 inference |
| `evaluate_caption_conditioned.py` | Text-only GT-caption ablation |
| `analysis_*.py` | Paper diagnostics (need prediction dumps under `results/`) |
| `human_validation.py` | Gradio UI for judge validation |
| `convert_to_jsonl.py` | Train JSON → ShareGPT JSONL (**requires train annotations obtained separately**) |
| `create_ccd_holdout.py` | CCD holdout split (**requires train annotations obtained separately**) |
| `evaluate_qwen3_holdout.py` | Holdout inference (**requires holdout data obtained separately**) |

```bash
export CAVIAR_VIDEO_ROOT=/path/to/nexar_collision_prediction
export CAVIAR_VLMEVALKIT_PATH=/path/to/VLMEvalKit   # optional judge
```
