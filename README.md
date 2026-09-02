# CAViAR

**Causal Accident Video and Incident Analysis Repository**

A human-annotated dashcam benchmark for fine-grained accident understanding and **responsibility reasoning** with vision-language models (VLMs).

Paper: *CAViAR: A Causal Video Dataset for Fine-Grained Accident Reasoning in Real-World Scenarios*.

Code and annotations: [https://github.com/nec-labs-ma/CAViAR](https://github.com/nec-labs-ma/CAViAR)

> **Public release (this repository):** validation/test annotations for the **Nexar** split (**749 videos**). Train/holdout annotations are not redistributed here. Source videos come from [CCD](https://github.com/Cogito2012/CarCrashDataset) (train) and [Nexar](https://www.nexar.com/) (val/test). This repository **does not redistribute the full video corpora**.

## What is CAViAR?

| | |
|---|---|
| **Full benchmark** | 2,249 real-world dashcam clips (CCD train / Nexar val–test) |
| **Released here** | Nexar validation/test annotations — **749 videos**, **7,407 QA pairs** |
| **Tasks** | Dense captioning, weather, lighting, road condition, accident type, at-fault agent, affected agent, rule-violation category |
| **Focus** | Observational responsibility attribution (not legal liability) |

CAViAR exposes a **Perception–Reasoning Gap**: VLMs can often recognize context (e.g., lighting) but struggle to apply traffic rules to infer responsibility.

## Example (Nexar)

Preview of **`nexar_00433.mp4`** (loops in the README):

![Preview of nexar_00433.mp4 from the Nexar dataset](examples/nexar_00433.gif)

**Source video:** `nexar_00433.mp4` from the **Nexar** dashcam dataset ([Moura et al., 2025](https://www.nexar.com/)). This is a short looping preview of the collision window; we do **not** redistribute the Nexar corpus. Obtain clips under the original Nexar license and place them under `data/videos/` (or set `CAVIAR_VIDEO_ROOT`). Compact mp4 preview: [`examples/nexar_00433_preview.mp4`](examples/nexar_00433_preview.mp4). Full annotations: [`examples/nexar_00433_annotations.json`](examples/nexar_00433_annotations.json).

| Field | Annotation |
|-------|------------|
| **Summary** | The accident involved a collision between a vehicle going straight and a yellow truck. |
| **Weather** | Sunny |
| **Lighting** | Day |
| **Road condition** | Dry |
| **Accident type** | Side-by-Side |
| **At-fault agent** | The accident was the fault of the yellow truck driver. |
| **Affected agent** | The victim was the driver of the straight-moving vehicle. |
| **Rule violation** | The driver of the yellow truck failed to activate the turn signal when making a turn and did not pay attention to the road conditions, violating the traffic rule that requires turn signals to be activated for at least three seconds when turning, which led to the accident. |

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
│   ├── nexar_00433.gif
│   ├── nexar_00433_preview.mp4
│   └── nexar_00433_annotations.json
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
python -m json.tool examples/nexar_00433_annotations.json | head

# Score a tiny illustrative prediction file
python scripts/evaluate_results.py --results examples/sample_results.json --skip-judge
```

## Using the Nexar val/test videos

1. Download Nexar clips under the original Nexar license.
2. Put mp4 files in `data/videos/` (or set `CAVIAR_VIDEO_ROOT`). Filenames must match `data/test.json` (e.g. `nexar_00433.mp4`).
3. Run a VLM on the released val/test split, then score predictions:

```bash
export CAVIAR_VIDEO_ROOT=/path/to/nexar/videos

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
@inproceedings{garg2026caviar,
  title={CAViAR: A Causal Video Dataset for Fine-Grained Accident Reasoning in Real-World Scenarios},
  author={Garg, Sparsh and Chen, Yi-Wen and Vijay and Aich, Abhishek},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

Please also cite the **Nexar** source dataset when using the released val/test videos or the example above (Moura et al., 2025).

## Ethics

CAViAR labels are research annotations of *apparent* responsibility cues from video evidence. They are **not** legal determinations of liability and must not be used for adjudication, insurance, enforcement, or decisions about identifiable individuals.

For privacy concerns, annotation errors, or takedown requests, please open a GitHub issue.

## License

Code is released under [LICENSE](LICENSE) (MIT). Annotations in `data/test.json` are for academic research on accident understanding and rule-relevant multimodal reasoning. Raw CCD and Nexar videos remain under their original dataset licenses.
