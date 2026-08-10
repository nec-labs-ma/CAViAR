# Jurisdiction-agnostic rule-violation ontology

Free-text Violation Identification answers are mapped to **one primary family** with a deterministic, ordered keyword lexicon (`caviar/ontology.py`). No LLM is used in the mapping.

## Priority order (highest → lowest)

| Code | Family |
|------|--------|
| SG | Signal / sign violation |
| RW | Failure to yield / right-of-way |
| FD | Unsafe following distance / rear-end |
| CT | Loss of vehicle control |
| LC | Improper lane change / merging |
| OT | Improper overtaking / passing |
| TU | Improper turn / U-turn / reversing |
| SP | Unsafe speed / reckless driving |
| ST | Sudden stop / improper stopping |
| PD | Pedestrian / non-motorized crossing |
| AT | Inattentive / improper observation (generic catch-all) |

The first matching family wins. Residual buckets:

- **Other** — rare valid rules outside the eleven families  
- **Unspecified** — non-answers / low-quality fragments (not counted toward breadth)

## CLI

```bash
python -m caviar.ontology --text "failed to yield at the intersection"
python -m caviar.ontology --file answers.txt
```
