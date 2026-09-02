# CAViAR

**Causal Accident Video and Incident Analysis Repository**

A human-annotated dashcam benchmark for fine-grained accident understanding and **responsibility reasoning** with vision-language models (VLMs).

Paper: *CAViAR: A Causal Video Dataset for Fine-Grained Accident Reasoning in Real-World Scenarios*.

Code and annotations: [https://github.com/nec-labs-ma/CAViAR](https://github.com/nec-labs-ma/CAViAR) · Val/test on Hugging Face: [sparshgarg57/CAViAR](https://huggingface.co/datasets/sparshgarg57/CAViAR)

> **Public release (this repository):** validation/test annotations for the **Nexar** split (**749 videos**). Train/holdout annotations are not redistributed here. Source videos come from [CCD](https://github.com/Cogito2012/CarCrashDataset) (train) and [nexar-ai/nexar_collision_prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) (val/test). This repository **does not redistribute the full video corpora**.

## What is CAViAR?

| | |
|---|---|
| **Full benchmark** | 2,249 real-world dashcam clips (CCD train / Nexar val–test) |
| **Released here** | Nexar validation/test annotations — **749 videos**, **7,407 QA pairs** |
| **Tasks** | Dense captioning, weather, lighting, road condition, accident type, at-fault agent, affected agent, rule-violation category |
| **Focus** | Observational responsibility attribution (not legal liability) |

CAViAR exposes a **Perception–Reasoning Gap**: VLMs can often recognize context (e.g., lighting) but struggle to apply traffic rules to infer responsibility.

## Example (Nexar)

Preview of Nexar clip **`00284`** (loops in the README):

![Preview of Nexar video 00284](examples/nexar_00284.gif)

**Source video:** id `00284` (`00284.mp4`) from [nexar-ai/nexar_collision_prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) ([Moura et al., 2025](https://arxiv.org/abs/2503.03848)). This is a short looping preview of the collision window; we do **not** redistribute the Nexar corpus. Place downloaded mp4s under `data/videos/` (or set `CAVIAR_VIDEO_ROOT`). Compact mp4 preview: [`examples/nexar_00284_preview.mp4`](examples/nexar_00284_preview.mp4). Full annotations: [`examples/nexar_00284_annotations.json`](examples/nexar_00284_annotations.json).

| Field | Annotation |
|-------|------------|
| **Summary** | The accident involved a collision between a straight-moving vehicle and a white sedan. |
| **Weather** | Sunny |
| **Lighting** | Day |
| **Road condition** | Dry |
| **Accident type** | T-Bone |
| **At-fault agent** | The accident was the fault of the driver of the straight-going vehicle. |
| **Affected agent** | The driver of the white car was the victim. |
| **Rule violation** | The driver of the straight-moving vehicle ran a red light, violating the traffic rule that motor vehicles must proceed in an orderly manner according to signal light instructions, resulting in the accident. |

## Repository layout

```
CAViAR/
├── caviar/                 # Ontology mapper + portable path helpers
├── configs/                # Example LoRA SFT hyperparameters (train path optional)
├── data/
│   ├── test.json           # Nexar val/test annotations (749 videos)
│   └── videos/             # Local mp4 root (not shipped; gitignored)
├── docs/                   # Schema, tasks, prompts, ontology
├── examples/
│   ├── nexar_00284.gif
│   ├── nexar_00284_preview.mp4
│   └── nexar_00284_annotations.json
└── scripts/                # Evaluation, inference, analysis
```

## Setup

```bash
git clone https://github.com/nec-labs-ma/CAViAR.git
cd CAViAR
pip install -r requirements.txt
```

Optional extras:

- VLM inference: `transformers`, `accelerate`, `peft`, `decord`, `pillow`
- LLM-as-Judge: install [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) and set `CAVIAR_VLMEVALKIT_PATH`
- Analysis plots: `scikit-learn`, `matplotlib`

## Quick start (no videos required)

```bash
# Val/test (Nexar) counts
python scripts/dataset_stats.py

# Map a free-text violation to a rule family
python -m caviar.ontology --text "failed to maintain a safe following distance"

# Inspect the Nexar example annotations
python -m json.tool examples/nexar_00284_annotations.json | head

# Score a tiny illustrative prediction file
python scripts/evaluate_results.py --results examples/sample_results.json --skip-judge
```

## Using the Nexar val/test videos

1. Download clips from [nexar-ai/nexar_collision_prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) under the Nexar Open Data License.
2. Point `CAVIAR_VIDEO_ROOT` at that download (the Nexar folder layout `train/` / `test-public/` / `test-private/` is fine). `video_path` in `data/test.json` is the numeric id (e.g. `00284` → `00284.mp4`).
3. Run a VLM on the released val/test split, then score predictions:

```bash
export CAVIAR_VIDEO_ROOT=/path/to/nexar_collision_prediction

python scripts/evaluate_qwen3.py --model 2B
python scripts/evaluate_results.py \
  --results results/results_Qwen3-VL-2B.json \
  --skip-judge
```

Train-oriented helpers (`convert_to_jsonl.py`, `create_ccd_holdout.py`, LoRA configs) remain for users who obtain CCD train annotations separately; those files are **not** included in this public release.

See [docs/data_access.md](docs/data_access.md) and [scripts/README.md](scripts/README.md).

## Tasks and metrics

| Task | Type | Metric |
|------|------|--------|
| Dense captioning | Open | BERTScore-F1 |
| Weather / lighting / road / accident type | MCQ | Accuracy (+ balanced acc, macro-F1) |
| At-fault agent, affected agent, rule violation | Open | LLM-as-Judge (0–5) |

Details: [docs/tasks.md](docs/tasks.md), [docs/prompts.md](docs/prompts.md), [docs/ontology.md](docs/ontology.md).

## Citation

```bibtex
@article{garg2026caviar,
  title={CAViAR: A Causal Video Dataset for Fine-Grained Accident Reasoning in Real-World Scenarios},
  author={Garg, Sparsh and Chen, Yi-Wen and Aich, Abhishek and others},
  journal={arXiv preprint arXiv:2608.19380},
  year={2026}
}
```

Please also cite **Nexar** when using the val/test videos ([nexar-ai/nexar_collision_prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction); Moura et al., 2025).

## Ethics

CAViAR labels are research annotations of *apparent* responsibility cues from video evidence. They are **not** legal determinations of liability and must not be used for adjudication, insurance, enforcement, or decisions about identifiable individuals.

For privacy concerns, annotation errors, or takedown requests, please open a GitHub issue.

## License

Code is released under [LICENSE](LICENSE) (MIT). Annotations in `data/test.json` are for academic research on accident understanding and rule-relevant multimodal reasoning. Raw CCD and Nexar videos remain under their original dataset licenses.
