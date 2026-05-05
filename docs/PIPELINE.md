# End-to-End Angiography Pipeline

This document describes how to run the three-stage Sequential Angiography
workflow from raw frame sequences through case-level stenosis fusion.

Top-level project folders:

- `keyframes-extraction`
- `vessel-segmentation`
- `stenosis-detection`

Pipeline stages:

1. Keyframe extraction
2. Vessel segmentation / mask generation
3. Stenosis detection, temporal fusion, and multi-view fusion

Commands below use the root wrapper scripts and the subproject CLIs.
The package names are `angio_keyframes`, `drive_seg`, and
`stenosis_detection`.

## Expected Raw Input

A typical raw dataset is a case tree with one or more views. Each view contains
an ordered frame sequence. Case-level `views.json` and `patient.json` files can
live directly under the case folder. `views.json` describes the view geometry
used later by multi-view fusion.

Example:

```text
data/raw_cases/
  case_001/
    views.json
    patient.json
    view_01/
      frames/
        slice_0001.png
        slice_0002.png
        ...
    view_02/
      frames/
        slice_0001.png
        slice_0002.png
        ...
```

Keyframe extraction discovers nested image directories. If a directory named
`frames` contains images, the output mirrors the parent view path and does not
copy the literal `frames` directory into the output.

`views.json` and `patient.json` are expected to be case-level metadata. The
keyframe stage copies `<input-root>/<case_id>/views.json` and
`<input-root>/<case_id>/patient.json` to the corresponding output case folder so
the metadata remains available after keyframe extraction.

## Expected Intermediate Structure

The examples below use `work/` as the pipeline output directory from the
repository root:

```text
work/
  keyframes/
    case_001/
      views.json
      patient.json
      view_01/
        slice_0003.png
        slice_0004.png
        ...
      view_02/
        slice_0002.png
        slice_0003.png
        ...
  vessel_masks/
    case_001/
      views.json
      patient.json
      view_01/
        slice_0003_mask.png
        slice_0004_mask.png
        ...
      view_02/
        slice_0002_mask.png
        slice_0003_mask.png
        ...
  stenosis_frame_results/
    case_001/
      view_01/
        slice_0003_centerline.png
        slice_0003_segmentation_points.png
        slice_0003_stenosis_result_mask.png
        slice_0003_stenosis_result_original.png
        slice_0003_stenosis_results.json
        ...
      view_02/
        ...
    batch_summary.json
  stenosis_temporal_results/
    case_001/
      view_01/
        view_temporal_fusion.json
        view_temporal_fusion_summary.png
      view_02/
        view_temporal_fusion.json
        view_temporal_fusion_summary.png
```

Stenosis-compatible masks must be stored in a tree that mirrors the image tree.
For every keyframe image named:

```text
work/keyframes/case_001/view_01/slice_0003.png
```

the mask must be named:

```text
work/vessel_masks/case_001/view_01/slice_0003_mask.png
```

The required pattern is `<image_stem>_mask.png`. The extension can be another
supported image extension, but the `_mask` suffix is required.
CADICA frame names such as `p1_v1_00012.png` are also supported when the mask is
named `p1_v1_00012_mask.png` in the mirrored mask tree.

## Installation

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The editable install makes `angio_keyframes`, `drive_seg`, and
`stenosis_detection` importable from the monorepo.

## Stage 1: Keyframe Extraction

From the repository root:

```bash
python scripts/run_keyframes.py data/raw_cases --output-root work/keyframes --backend cpu --overwrite
```

Equivalent command from the subproject folder:

```bash
cd keyframes-extraction
python keyframes_extraction.py ../data/raw_cases --output-root ../work/keyframes --backend cpu --overwrite
```

Expected outputs:

- A mirrored keyframe tree under `work/keyframes`.
- Selected keyframe images only; original `frames` folders are not copied into
  the output tree.
- Case-level `views.json` and `patient.json` files copied from immediate child
  case folders under the input root into the corresponding output case folders.

If `--output-root` is omitted, the keyframe CLI creates its default sibling
output directory next to the input path.

CUDA note: `--backend cuda` requires a custom OpenCV build with CUDA support and
the required CUDA Python bindings. The standard `opencv-python` wheel is not
enough for CUDA acceleration.

## Stage 2: Vessel Segmentation / Mask Generation

Use a trained segmentation checkpoint to write stenosis-compatible masks for
the keyframes.

From the repository root:

```bash
python scripts/run_segmentation.py ^
  --input work/keyframes ^
  --checkpoint vessel-segmentation/checkpoints/coronary_finetuned.pt ^
  --stenosis-masks-root work/vessel_masks
```

Equivalent command from the subproject folder:

```bash
cd vessel-segmentation
python segment_retinal_images.py ^
  --input ../work/keyframes ^
  --checkpoint checkpoints/coronary_finetuned.pt ^
  --stenosis-masks-root ../work/vessel_masks
```

Expected outputs:

- A mirrored mask tree under `work/vessel_masks`.
- One mask per processed keyframe.
- Mask names in the form `<image_stem>_mask.png`, for example
  `slice_0003_mask.png`.
- JSON metadata files found in the input tree are copied to the output tree by
  the mask generation command, preserving metadata such as `views.json` and
  `patient.json` when present.

For model development rather than pipeline inference, the vessel project also
keeps its existing commands:

```bash
python vessel-segmentation/validate_dataset.py --dataset-root path/to/Datasets/Coronary
python vessel-segmentation/train.py --dataset-root path/to/Datasets/Coronary --train-split train --val-split val
python vessel-segmentation/predict.py --dataset-root path/to/Datasets/Coronary --split test
```

## Stage 3: Stenosis Detection

Run frame-level stenosis detection with the keyframes as images and the mirrored
vessel masks as masks.

From the repository root:

```bash
python scripts/run_stenosis_detection.py ^
  --images-root work/keyframes ^
  --masks-root work/vessel_masks ^
  --output-root work/stenosis_frame_results
```

Equivalent command from the subproject folder:

```bash
cd stenosis-detection
python run_stenosis_detection.py ^
  --images-root ../work/keyframes ^
  --masks-root ../work/vessel_masks ^
  --output-root ../work/stenosis_frame_results
```

Expected outputs for each frame:

- `<image_stem>_centerline.png`
- `<image_stem>_segmentation_points.png`
- `<image_stem>_stenosis_result_mask.png`
- `<image_stem>_stenosis_result_original.png`
- `<image_stem>_stenosis_results.json`

Batch mode also writes:

```text
work/stenosis_frame_results/batch_summary.json
```

The stenosis detector matches images and masks by mirrored relative path and
mask stem. If the image is `slice_0003.png`, the mask must be named
`slice_0003_mask.png`. CADICA names such as `p1_v1_00012.png` are accepted with
matching masks such as `p1_v1_00012_mask.png`.

## Temporal Fusion

Temporal fusion combines frame-level JSON outputs for one view into a stable
view-level result.

From the repository root, process all discovered views:

```bash
python scripts/run_temporal_fusion.py ^
  --results-root work/stenosis_frame_results ^
  --output-root work/stenosis_temporal_results ^
  --skip-existing
```

Equivalent command from the subproject folder:

```bash
cd stenosis-detection
python run_temporal_fusion.py ^
  --results-root ../work/stenosis_frame_results ^
  --output-root ../work/stenosis_temporal_results ^
  --skip-existing
```

Expected outputs for each view:

- `view_temporal_fusion.json`
- `view_temporal_fusion_summary.png`
- `view_temporal_fusion_demo.mp4` or `.gif` when `--write-video` is used

For one view only:

```bash
python scripts/run_temporal_fusion.py ^
  --results-root work/stenosis_frame_results/case_001/view_01 ^
  --output work/stenosis_temporal_results/case_001/view_01/view_temporal_fusion.json
```

## Multi-View Fusion

Multi-view fusion combines temporal fusion outputs for a case using view
metadata from `views.json`.

The multi-view input JSON should include one entry per view and point each view
at its temporal fusion output:

```json
{
  "case_id": "case_001",
  "views": [
    {
      "view_id": "view_01",
      "sequence_id": "view_01",
      "rao_lao": 30,
      "cra_cau": -10,
      "temporal_fusion_json": "../../stenosis_temporal_results/case_001/view_01/view_temporal_fusion.json"
    },
    {
      "view_id": "view_02",
      "sequence_id": "view_02",
      "rao_lao": -35,
      "cra_cau": 5,
      "temporal_fusion_json": "../../stenosis_temporal_results/case_001/view_02/view_temporal_fusion.json"
    }
  ]
}
```

If keyframe extraction copied `data/raw_cases/case_001/views.json` to
`work/keyframes/case_001/views.json`, you can use that copied file as the case
input after ensuring its `temporal_fusion_json` paths point to the temporal
fusion outputs you want to fuse.

From the repository root:

```bash
python scripts/run_multiview_fusion.py ^
  --input-json work/keyframes/case_001/views.json ^
  --output work/case_results/case_001/case_multiview_fusion.json
```

Or, if the case folder contains a file literally named `views.json`:

```bash
python scripts/run_multiview_fusion.py ^
  --case-root work/keyframes/case_001 ^
  --output work/case_results/case_001/case_multiview_fusion.json
```

If `views.json` contains stale or portable placeholder temporal paths, provide
the temporal results root separately:

```bash
python scripts/run_multiview_fusion.py ^
  --case-root-tree work/keyframes ^
  --temporal-results-root work/stenosis_temporal_results ^
  --output-root work/case_results
```

Expected output:

- One case-level JSON containing `case_id`, `view_count`, `views`,
  `per_view_summary`, `final_case_lesion`, `confidence`, `supporting_views`,
  and `fusion_metadata`.

## Common Path Mistakes

- Running a subproject command from the repository root but using paths as if
  the current directory were the subproject folder. For example, from the root
  use `vessel-segmentation/checkpoints/model.pt`; from
  `vessel-segmentation/`, use `checkpoints/model.pt`.
- Running from a subproject folder but forgetting `../` for shared root paths
  such as `../work/keyframes`.
- Omitting `--output-root` in batch stenosis detection. The batch command
  requires `--images-root`, `--masks-root`, and `--output-root`.
- Passing the original raw frames tree to stenosis detection after keyframe
  extraction. Stage 3 should usually use the keyframe output tree as
  `--images-root`.
- Saving masks without the `_mask` suffix. Stenosis detection expects
  `<image_stem>_mask.png`.
- Moving or regenerating `views.json` without updating `temporal_fusion_json`
  paths before multi-view fusion.
- Using relative paths inside `views.json` without checking what they are
  relative to. Relative `temporal_fusion_json` paths are resolved relative to
  the location of the `views.json` file.

## Minimal Smoke-Test Workflow

These commands check imports and CLI wiring without requiring real data:

```bash
python -m compileall scripts keyframes-extraction vessel-segmentation stenosis-detection
python -c "import angio_keyframes, drive_seg, stenosis_detection; print('ok')"

python scripts/run_keyframes.py --help
python scripts/run_segmentation.py --help
python scripts/run_stenosis_detection.py --help
python scripts/run_temporal_fusion.py --help
python scripts/run_multiview_fusion.py --help
```

```bash
python -m unittest discover keyframes-extraction/tests
python -m unittest discover stenosis-detection/tests
```

For a tiny data smoke test, use a small case with one or two views and a few
frames per view, then run the same stage commands with a small output root such
as `work/smoke/`.

## Troubleshooting

### Missing Masks

Symptoms:

- Stenosis detection reports missing mask files.
- Batch summary shows failures for every image.

Checks:

- Confirm the mask root mirrors the keyframe image tree.
- Confirm every mask has the `_mask` suffix, for example
  `slice_0003_mask.png`.
- Confirm `--images-root` points to `work/keyframes` and `--masks-root` points
  to `work/vessel_masks`.

### Missing `views.json`

Symptoms:

- Multi-view fusion cannot find `views.json` when using `--case-root`.
- Case-level fusion has no view geometry.

Checks:

- Confirm the raw case folder has `views.json` directly under the case folder,
  for example `data/raw_cases/case_001/views.json`.
- Confirm keyframe extraction was run on the dataset root, not a single image
  folder, so case-level `views.json` files could be copied.
- Confirm the copied file exists at `work/keyframes/case_001/views.json`.
- If the file is elsewhere, use `--input-json path/to/views.json` instead of
  `--case-root`.

### Import Errors

Symptoms:

- `ModuleNotFoundError: No module named 'angio_keyframes'`
- `ModuleNotFoundError: No module named 'drive_seg'`
- `ModuleNotFoundError: No module named 'stenosis_detection'`

Checks:

- From the repository root, run `pip install -e .`.
- Confirm the active Python environment is the one where dependencies were
  installed.
- For direct subproject scripts, run them by path from the root or from the
  matching subproject folder; the compatibility scripts add their local source
  paths where needed.

### CUDA / OpenCV Issues

Symptoms:

- `--backend cuda` fails during keyframe extraction.
- OpenCV reports missing CUDA functions.

Checks:

- Use `--backend cpu` for the standard `opencv-python` install.
- CUDA keyframe extraction requires a custom OpenCV build compiled with CUDA
  support and the needed CUDA Python bindings.
- Do not install CUDA-specific OpenCV packages unless you have verified they
  expose the required `cv2.cuda` APIs for this code path.

### Model Checkpoint Paths

Symptoms:

- Vessel segmentation cannot find a checkpoint.
- `torch.load` fails with a missing file path.

Checks:

- From the repository root, use paths such as
  `vessel-segmentation/checkpoints/coronary_finetuned.pt`.
- From `vessel-segmentation/`, use paths such as
  `checkpoints/coronary_finetuned.pt`.
- Confirm the checkpoint architecture matches the model configuration expected
  by the vessel segmentation code.
