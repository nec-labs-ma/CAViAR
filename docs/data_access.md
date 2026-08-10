# Data access

## Current status

| Asset | Status |
|-------|--------|
| Evaluation / analysis code | **Available** in this repo |
| Prompt templates & ontology | **Available** |
| Illustrative schema samples | **Available** (`examples/`) |
| Full train/test annotations | **Pending** institutional approval |
| Raw dashcam videos | **Not redistributed** — obtain from CCD / Nexar |

## Source corpora

1. **Car Crash Dataset (CCD)** — used as the CAViAR train split (1,500 videos).  
2. **Nexar** — used as the CAViAR test split (749 videos after filtering).

Follow each source’s license and download instructions. Place videos under a local root, e.g.:

```
data/videos/
  crash_1500_000001.mp4
  nexar_00001.mp4
  ...
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `CAVIAR_VIDEO_ROOT` | Local directory containing video files |
| `CAVIAR_DATA_ROOT` | Directory with released annotation JSON (when available) |
| `CAVIAR_VLMEVALKIT_PATH` | Optional path to VLMEvalKit for LLM-as-Judge |
| `OPENAI_API_KEY` | Required only if running GPT-4o judge |

## After full annotation release

Expected files under `data/annotations/` (names may vary slightly):

- `train.json` — CCD split  
- `test.json` — Nexar split  
- Optional: `train.jsonl` (ShareGPT / InternVL format)

Then:

```bash
export CAVIAR_DATA_ROOT=./data/annotations
export CAVIAR_VIDEO_ROOT=./data/videos
python scripts/evaluate_results.py --results results/my_model.json --skip-judge
```
