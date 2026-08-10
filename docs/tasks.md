# CAViAR tasks and metrics

| Task | Type | Metric |
|------|------|--------|
| Dense captioning (detailed + summary) | Open | BERTScore-F1 (+ BLEU / ROUGE-L) |
| Weather | MCQ | Accuracy, balanced accuracy, macro-F1 |
| Lighting | MCQ | Accuracy, balanced accuracy, macro-F1 |
| Road condition | MCQ | Accuracy, balanced accuracy, macro-F1 |
| Accident type | MCQ | Accuracy, balanced accuracy, macro-F1 |
| Apparent at-fault agent | Open | LLM-as-Judge (0–5) |
| Affected agent | Open | LLM-as-Judge (0–5) |
| Apparent rule-violation category | Open | LLM-as-Judge (0–5) |

## Judge rubric (0–5)

- **5** — Perfect match: same agent(s) and equivalent reasoning  
- **4** — All key facts correct; minor wording differences  
- **3** — Correct agent(s); missing reasoning details or minor inaccuracies  
- **2** — Partial match  
- **1** — Only a small part relevant/correct  
- **0** — Irrelevant or wrong  

See `docs/prompts.md` for the exact judge prompt template.

## Important notes

- Labels are *apparent* responsibility cues from dashcam evidence, not legal liability.
- Ambiguous clips may omit responsibility fields.
- Report majority/random baselines for MCQ tasks; class imbalance is substantial.
