# CAViAR data schema

The public release includes the **Nexar val/test** split in `data/test.json`. See `examples/nexar_00433_annotations.json` and `examples/nexar_00433.gif` for a real annotated Nexar example.

## Top-level video record

```json
{
  "video_path": "nexar_00433.mp4",
  "qa_pairs": [ /* list of QA objects */ ]
}
```

`video_path` is the **filename** of a locally available Nexar clip (e.g. `nexar_00433.mp4`). Scripts resolve it via `CAVIAR_VIDEO_ROOT` (default `data/videos/`).

## QA object (common fields)

| Field | Type | Description |
|-------|------|-------------|
| `benchmark` | string | Task family name (see below) |
| `question` | string | Prompt shown to the model |
| `answer` | string | Human reference answer |

### Multiple-choice extras

| Field | Type | Description |
|-------|------|-------------|
| `choices` | string[] | Closed option set |
| `correct_answer` | string | Correct option text |
| `correct_index` | int | 0-based index into `choices` |

## Task families (`benchmark` values)

| ID | `benchmark` | Type |
|----|-------------|------|
| T1a/T1b | `Dense Captioning` | Open |
| T2/T3 | `Weather & Light` | MCQ (weather or lighting) |
| T4 | `Road Conditions` | MCQ |
| T5 | `Accident Type` | MCQ |
| T6 | `Faulter Identification` | Open (apparent at-fault agent) |
| T7 | `Victim Identification` | Open (affected agent) |
| T8 | `Violation Identification` | Open (rule-violation category) |
| — | `Accident Reason` | Open (folded into T8 in paper taxonomy) |

## Canonical MCQ label spaces

- **Weather:** sunny, rainy, cloudy, snowy  
- **Lighting:** day, night  
- **Road condition:** dry, wet, snowy  
- **Accident type:** rear-end, T-bone, side-by-side, head-on, none  

## Split design

| Split | Source | Role | In this repo? |
|-------|--------|------|---------------|
| Train | CCD | Fine-tuning | No |
| Val/Test | Nexar | Held-out evaluation | Yes (`data/test.json`) |

No shared video, scene, or device between CCD and Nexar splits.
