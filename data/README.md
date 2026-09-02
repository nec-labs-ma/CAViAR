# CAViAR annotations (public release)

This directory ships the **Nexar validation/test** annotations only.

| File | Description |
|------|-------------|
| `test.json` | Nexar val/test split (**749 videos**, **7,407 QA pairs**) |
| `videos/` | Local video root (gitignored). Set `CAVIAR_VIDEO_ROOT` if videos live elsewhere. |

**Not included in this repository:** CCD train annotations, holdout splits, or raw video files.

`video_path` fields store **filenames only** (e.g. `nexar_00284.mp4`). Scripts join them with `CAVIAR_VIDEO_ROOT` (default: `data/videos/`).

Obtain Nexar clips under the original Nexar license. See [docs/data_schema.md](../docs/data_schema.md) and [docs/data_access.md](../docs/data_access.md).
