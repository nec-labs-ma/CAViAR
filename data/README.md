# CAViAR annotations (public release)

This directory ships the **Nexar validation/test** annotations only.

| File | Description |
|------|-------------|
| `test.json` | Nexar val/test split (**749 videos**, **7,407 QA pairs**). Same split: [Hugging Face](https://huggingface.co/datasets/sparshgarg57/CAViAR) |
| `videos/` | Local video root (gitignored). Set `CAVIAR_VIDEO_ROOT` if videos live elsewhere. |

**Not included in this repository:** CCD train annotations, holdout splits, or raw video files.

`video_path` is the numeric Nexar id (e.g. `00284` → `00284.mp4`). Scripts resolve it under `CAVIAR_VIDEO_ROOT` (default: `data/videos/`).

Videos: [nexar-ai/nexar_collision_prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) (Nexar Open Data License). See [docs/data_schema.md](../docs/data_schema.md) and [docs/data_access.md](../docs/data_access.md).
