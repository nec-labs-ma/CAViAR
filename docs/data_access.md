# Data access

## Released in this repository

| Asset | Location |
|-------|----------|
| Val/test annotations (Nexar) | `data/test.json` (749 videos) |
| Example preview + QA | `examples/nexar_00284.gif`, `examples/nexar_00284_preview.mp4`, `examples/nexar_00284_annotations.json` |
| Evaluation / analysis code | `scripts/` |
| Prompt templates & ontology | `docs/`, `caviar/ontology.py` |

**Not released here:** CCD train annotations, holdout JSON/JSONL, or raw video files.

## Source corpora

1. **Nexar** — CAViAR validation/test split (749 clips after filtering). **Released annotations correspond to this split.**  
2. **Car Crash Dataset (CCD)** — CAViAR train split (1,500 clips). Annotations are **not** in this public repo.

Download videos under each source’s original license. For the public eval scripts, place Nexar mp4 files in `data/videos/` (or any directory pointed to by `CAVIAR_VIDEO_ROOT`) using the filenames in `data/test.json` (e.g. `nexar_00284.mp4`).

```
data/videos/
  nexar_00001.mp4
  ...
  nexar_00284.mp4
  ...
```

## Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CAVIAR_VIDEO_ROOT` | Directory of mp4 files | `data/videos/` |
| `CAVIAR_DATA_ROOT` | Annotation JSON directory | `data/` |
| `CAVIAR_RESULTS_DIR` | Prediction / eval outputs | `results/` |
| `CAVIAR_VLMEVALKIT_PATH` | Optional VLMEvalKit for LLM-as-Judge | unset |
| `OPENAI_API_KEY` | Required only if running GPT-4o judge | unset |

```bash
export CAVIAR_VIDEO_ROOT=/path/to/nexar/videos
python scripts/evaluate_qwen3.py --model 2B
python scripts/evaluate_results.py --results results/results_Qwen3-VL-2B.json --skip-judge
```
