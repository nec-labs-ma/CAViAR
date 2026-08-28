# CAViAR

**Causal Accident Video and Incident Analysis Repository**

A human-annotated dashcam benchmark for fine-grained accident understanding and responsibility reasoning with vision-language models (VLMs).

> **Release status:** Evaluation code, prompts, configs, schema, and illustrative examples are available in this repository. **Full CAViAR annotations and train/test splits will be released upon paper acceptance pending institutional approval.** Source videos are obtained from [CCD](https://github.com/Cogito2012/CarCrashDataset) and [Nexar](https://www.nexar.com/); we do **not** redistribute raw video files.

Paper: *CAViAR: A Causal Video Dataset for Fine-Grained Accident Reasoning in Real-World Scenarios* (ECCV 2026 submission).

## What is CAViAR?

| | |
|---|---|
| **Videos** | 2,249 real-world dashcam clips (CCD train / Nexar test) |
| **QA pairs** | ~20,108 structured question–answer pairs |
| **Tasks** | Dense captioning, weather, lighting, road condition, accident type, at-fault agent, affected agent, rule-violation category |
| **Focus** | Observational responsibility attribution (not legal liability) |

CAViAR exposes a **Perception–Reasoning Gap**: VLMs can often recognize context (e.g., lighting) but struggle to apply traffic rules to infer responsibility.

## Repository contents (current)

```
CAViAR/
├── README.md
├── LICENSE
├── requirements.txt
├── caviar/                 # Shared utilities (rule ontology mapper)
├── configs/                # Example LoRA SFT hyperparameter configs
├── docs/                   # Schema, tasks, prompts, ontology, data access
├── examples/               # Illustrative QA samples (not the full dataset)
├── data/                   # Placeholder for upcoming full annotations
└── scripts/                # Evaluation, baselines, analysis utilities
```

**Not included yet (pending approval):** full `train.json` / `test.json` annotations, prediction dumps, and model checkpoints.

## Quick start

```bash
# Optional: install dependencies for metrics
pip install -r requirements.txt

# Inspect the data schema and illustrative sample
cat docs/data_schema.md
python -m json.tool examples/sample_qa.json

# Map free-text rule-violation answers to ontology families
python -m caviar.ontology --text "failed to maintain a safe following distance"

# After full annotations are released, evaluate model predictions:
# export OPENAI_API_KEY=...   # only if using LLM-as-Judge
# python scripts/evaluate_results.py --results path/to/results.json --skip-judge
```

## Getting source videos

1. Obtain CCD and Nexar videos under their original licenses.
2. Place them under a local directory, e.g. `data/videos/`.
3. When full CAViAR annotations are released, set `CAVIAR_VIDEO_ROOT` to that directory; scripts remap paths automatically.

See [docs/data_access.md](docs/data_access.md).

## Citation

```bibtex
@article{garg2026caviar,
  title={CAViAR: A Causal Video Dataset for Fine-Grained Accident Reasoning in Real-World Scenarios},
  author={Garg, Sparsh and Chen, Yi-Wen and Aich, Abhishek and others},
  journal={arXiv preprint arXiv:2608.19380},
  year={2026}
}
```

## Ethics

CAViAR labels are research annotations of *apparent* responsibility cues from video evidence. They are **not** legal determinations of liability and must not be used for adjudication, insurance, enforcement, or decisions about identifiable individuals.

## Contact

For privacy concerns, annotation errors, or takedown requests, please open a GitHub issue on this repository.

## License

Code in this repository is released under the terms in [LICENSE](LICENSE). Dataset annotations (when released) will carry a separate research-use license stated in `data/`.
