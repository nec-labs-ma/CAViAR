# Data access

## Released in this repository

| Asset | Location |
|-------|----------|
| Val/test annotations (Nexar) | `data/test.json` (749 videos); also [Hugging Face](https://huggingface.co/datasets/sparshgarg57/CAViAR) |
| Example preview + QA | `examples/nexar_00284.gif`, `examples/nexar_00284_preview.mp4`, `examples/nexar_00284_annotations.json` |
| Evaluation / analysis code | `scripts/` |
| Prompt templates & ontology | `docs/`, `caviar/ontology.py` |

**Not released here:** CCD train annotations, holdout JSON/JSONL, or raw video files.

## Source corpora

1. **Nexar** — CAViAR validation/test split (749 clips after filtering). **Released annotations correspond to this split.** Videos: [nexar-ai/nexar_collision_prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) (Nexar Open Data License).  
2. **Car Crash Dataset (CCD)** — CAViAR train split (1,500 clips). Annotations are **not** in this public repo.

For eval scripts, set `CAVIAR_VIDEO_ROOT` to your Nexar download. Filenames are `{id}.mp4` (e.g. `00284.mp4`); `video_path` in `data/test.json` is the numeric id only (`00284`). Nested `train/` / `test-public/` / `test-private/` folders are resolved automatically.

```
# Nexar HF layout (example)
nexar_collision_prediction/
  train/positive/00284.mp4
  train/negative/01924.mp4
  test-public/...
  test-private/...
```

You can also flatten mp4s into `data/videos/` as `00284.mp4`.

## Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CAVIAR_VIDEO_ROOT` | Directory of mp4 files (Nexar download root or a flat folder) | `data/videos/` |
| `CAVIAR_DATA_ROOT` | Annotation JSON directory | `data/` |
| `CAVIAR_RESULTS_DIR` | Prediction / eval outputs | `results/` |
| `CAVIAR_VLMEVALKIT_PATH` | Optional VLMEvalKit for LLM-as-Judge | unset |
| `OPENAI_API_KEY` | Required only if running GPT-4o judge | unset |

```bash
export CAVIAR_VIDEO_ROOT=/path/to/nexar_collision_prediction
python scripts/evaluate_qwen3.py --model 2B
python scripts/evaluate_results.py --results results/results_Qwen3-VL-2B.json --skip-judge
```
