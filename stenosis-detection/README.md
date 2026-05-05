# Python Stenosis Detection

This folder contains the stenosis detection and fusion stage of the Sequential Angiography monorepo. The importable package is `stenosis_detection`.

The pipeline detects vessel stenosis from angiography images and binary vessel masks.

The project supports:

- single-image processing
- whole-tree batch processing with mirrored output folders

## Installation

Install dependencies and the monorepo package from the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Batch Processing

Use batch mode when you have two mirrored data trees:

- one root for original images
- one root for masks

Example:

```bash
python run_stenosis_detection.py \
  --images-root path/to/original_root \
  --masks-root path/to/mask_root \
  --output-root path/to/output_root
```

The batch runner:

- walks the entire original-image tree recursively
- matches each supported frame image with a corresponding `<image_stem>_mask` file in the mirrored mask tree
- writes outputs into a mirrored tree under `--output-root`
- skips slices that already have a full output set unless `--overwrite` is used
- shows a `tqdm` progress bar during processing

### Expected Naming

Original-image files:

- `slice_0001.png`
- `slice_0123.jpg`

Mask files:

- `slice_0001_mask.png`
- `slice_0123_mask.jpg`
- `p1_v1_00012_mask.png`

The folder hierarchy under the image root and mask root should match.

## Frame + Temporal Parameter Sweeps

To run frame-level and temporal-fusion hyperparameter tests in one optimized
workflow:

```bash
python ../scripts/run_stenosis_temporal_sweep.py \
  --images-root ../work/cadica_keyframes \
  --masks-root ../work/cadica_vessel_masks \
  --output-root ../work/cadica_sweep \
  --workers 0 \
  --temporal-workers 0 \
  --no-debug-images \
  --allow-variable-frame-count \
  --stenosis-thresholds 0.25,0.35 \
  --average-radius-thresholds 4.0,5.0 \
  --radius-outside-fraction-thresholds 0.05,0.10 \
  --radius-min-outside-samples-values 2,3 \
  --min-supporting-frames-values 2,3 \
  --min-persistence-ratios 0.25,0.50
```

Frame outputs are grouped under `frame_results/<frame_variant>/...`; temporal
outputs are grouped under
`temporal_results/<frame_variant>/<temporal_variant>/...`. Existing outputs are
skipped by default; pass `--overwrite` to force a rerun.

### Mirrored Outputs

If an input image is located at:

```text
original_root/study_a/series_b/slice_0001.png
```

its outputs will be written to:

```text
output_root/study_a/series_b/slice_0001_centerline.png
output_root/study_a/series_b/slice_0001_segmentation_points.png
output_root/study_a/series_b/slice_0001_stenosis_result_mask.png
output_root/study_a/series_b/slice_0001_stenosis_result_original.png
output_root/study_a/series_b/slice_0001_stenosis_results.json
```

Batch mode also writes:

```text
output_root/batch_summary.json
```

### Skip Logic

Batch mode skips a slice when all expected output files for that slice already exist.

To force reprocessing:

```bash
python run_stenosis_detection.py \
  --images-root path/to/original_root \
  --masks-root path/to/mask_root \
  --output-root path/to/output_root \
  --overwrite
```

## Single-Image Processing

```bash
python run_stenosis_detection.py \
  --image path/to/image.png \
  --mask path/to/mask.png \
  --output-dir outputs
```

To display the figures after saving them:

```bash
python run_stenosis_detection.py \
  --image path/to/image.png \
  --mask path/to/mask.png \
  --output-dir outputs \
  --show
```

You can also run the package directly:

```bash
python -m stenosis_detection --image path/to/image.png --mask path/to/mask.png --output-dir outputs
```

## Inputs

Single-image mode:

- `--image`: original angiography image
- `--mask`: binary vessel mask image aligned to the same case
- `--output-dir`: directory where results will be written

Batch mode:

- `--images-root`: root directory of original images
- `--masks-root`: root directory of mask images
- `--output-root`: root directory for mirrored outputs

The mask should use vessel pixels on a dark background for best results.

## Outputs

Per slice, the pipeline writes:

- centerline visualization
- segmentation-point visualization
- stenosis result over the mask
- stenosis result over the original image
- JSON results with stenosis points, degree values, severity labels, and summary counts

## Temporal Fusion

After frame-level detection is finished, the temporal stage combines one 12-frame view into one stable view-level result.

The temporal pipeline:

- loads one ordered frame sequence from `*_stenosis_results.json` files
- selects a reference frame
- registers the remaining frames to that reference
- projects detections onto the reference centerline
- builds lesion tracks across frames
- filters tracks by persistence and saves persistent lesions plus one final lesion summary

### Temporal Input

The temporal stage expects frame-level JSON files produced by the current detector export. Each frame result must include:

- `frame` metadata with `view_id` and `frame_index`
- `skeleton_points`
- `stenosis_points`

### Temporal Usage

Fuse one view from a single folder:

```bash
python run_temporal_fusion.py \
  --results-root path/to/output_root/study_a/series_b \
  --output path/to/output_root/study_a/series_b/view_temporal_fusion.json
```

Write the optional demo video as well:

```bash
python run_temporal_fusion.py \
  --results-root path/to/output_root/study_a/series_b \
  --output path/to/output_root/study_a/series_b/view_temporal_fusion.json \
  --write-video \
  --video-fps 3 \
  --video-format mp4
```

Fuse one view from a larger output tree:

```bash
python run_temporal_fusion.py \
  --results-root path/to/output_root \
  --view-id study_a/series_b \
  --output path/to/output_root/study_a/series_b/view_temporal_fusion.json
```

Fuse every discovered view in one run and write results under a separate root:

```bash
python run_temporal_fusion.py \
  --results-root path/to/output_root \
  --output-root path/to/temporal_output_root
```

Skip already completed views when the full expected output set already exists:

```bash
python run_temporal_fusion.py \
  --results-root path/to/output_root \
  --output-root path/to/temporal_output_root \
  --skip-existing
```

Key arguments:

- `--results-root`: directory to scan for frame-level JSON files
- `--frame-results`: explicit list of frame-level JSON files
- `--output`: output JSON path for one selected view
- `--output-root`: root directory for one or more fused view outputs
- `--view-id`: optional exact `view_id` when the input contains multiple views
- `--expected-frame-count`: defaults to `12`
- `--min-supporting-frames`: persistence threshold for stable lesions
- `--min-persistence-ratio`: persistence threshold as a fraction of the full view
- `--write-video`: also writes a compact temporal demo video
- `--video-fps`: playback speed for the optional demo video
- `--video-format`: `mp4` by default, or `gif` when `imageio` is available
- `--skip-existing`: skip only views whose complete expected output set already exists; partial outputs are recomputed and overwritten

In batch mode, the runner shows a `tqdm` progress bar with processed and skipped counts.

### Temporal Output

The temporal runner writes one view-level JSON file containing:

- reference-frame selection details
- frame registrations
- lesion tracks
- persistent lesion summaries
- the final view-level lesion chosen from the persistent lesions

It also writes one summary PNG next to the JSON output, showing:

- the selected reference frame
- vessel mask/centerline context when available
- persistent lesion tracks and observations
- the final fused lesion highlight
- a compact text panel with fused stenosis statistics

When `--write-video` is enabled, the runner also writes one minimal demo video next to the JSON output. The video uses the original ordered frames and overlays only:

- the final chosen lesion with a strong marker
- other persistent lesions with small subtle markers
- a compact corner text box with frame number, fused degree, frame-level degree, support count, and severity

## Multi-View Fusion

After temporal fusion is finished for each view in a case, the multi-view stage combines those view-level JSON files into one case-level result.

### Multi-View Input

Each case folder should contain one `views.json` file:

```json
{
  "case_id": "case_001",
  "views": [
    {
      "view_id": "view_01",
      "sequence_id": "seq_01",
      "rao_lao": 30,
      "cra_cau": -10,
      "temporal_fusion_json": "outputs/case_001/view_01_temporal_fusion.json"
    }
  ]
}
```

### Multi-View Usage

```bash
python run_multiview_fusion.py \
  --case-root path/to/case_root \
  --output path/to/case_root/case_multiview_fusion.json
```

You can also point directly to the input JSON:

```bash
python run_multiview_fusion.py \
  --input-json path/to/case_root/views.json \
  --output path/to/case_root/case_multiview_fusion.json
```

For a tree of `views.json` files with temporal outputs stored under a separate
root:

```bash
python run_multiview_fusion.py \
  --case-root-tree path/to/keyframes_root \
  --temporal-results-root path/to/stenosis_temporal_results \
  --output-root path/to/case_results
```

### Multi-View Output

The case-level JSON includes:

- `case_id`
- `view_count`
- `views`
- `per_view_summary`
- `final_case_lesion`
- `confidence`
- `supporting_views`
- `fusion_metadata`

## Validation

Minimal temporal validation tests are available with Python's built-in `unittest` runner:

```bash
python -m unittest discover -s tests -q
```

They cover:

- loader parsing and frame ordering
- reference-frame selection
- point-to-centerline projection
- persistence filtering for lesion tracks

## Configuration

The CLI exposes the main detection thresholds:

- `--radius-search-range` default `110`
- `--segmentation-distance-threshold` default `8`
- `--stenosis-threshold` default `0.25`
- `--average-radius-threshold` default `4`
- `--final-point-distance-threshold` default `10`

## Pipeline Overview

The detection flow includes:

- mask preprocessing and grayscale conversion when needed
- resize to `800 x 600`
- centerline extraction by thinning
- 8-connected neighbor checks
- branch-point detection and nearby-point filtering
- shortest-path search between segmentation points
- radius estimation from the vessel mask
- stenosis severity calculation and final point filtering
- visualization and JSON export

## Repo Layout

- `run_stenosis_detection.py`
- `run_temporal_fusion.py`
- `run_multiview_fusion.py`
- `stenosis_detection/__init__.py`
- `stenosis_detection/__main__.py`
- `stenosis_detection/batch.py`
- `stenosis_detection/multiview/`
- `stenosis_detection/temporal/`
- `stenosis_detection/pipeline.py`
- `stenosis_detection/radius.py`
- `stenosis_detection/pathfinding.py`
- `stenosis_detection/neighbors.py`
- `stenosis_detection/queue_logic.py`
- `stenosis_detection/thinning.py`
- `stenosis_detection/visualization.py`
- `tests/`
