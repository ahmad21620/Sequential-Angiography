# CADICA Benchmarking

This page describes the practical workflow for preparing CADICA, running the
existing pipeline, and evaluating outputs against CADICA supervised
annotations. This is separate from the EHR weak-label benchmark.

## CADICA Files Used

The first CADICA integration uses only `selectedVideos/` for benchmark data.

Used files:

- `selectedVideos/pX/lesionVideos.txt`: marks videos with labeled lesions.
- `selectedVideos/pX/nonlesionVideos.txt`: marks selected videos without visible lesions.
- `selectedVideos/pX/vY/pX_vY_selectedFrames.txt`: marks clinically selected frames.
- `selectedVideos/pX/vY/groundtruth/*.txt`: frame-level lesion boxes in `[x, y, w, h]` form.
- `selectedVideos/pX/vY/input/*.png`: frame images.

Intentionally not used in the first implementation:

- `nonselectedVideos/` for official metrics.
- `groundTruthTable.mat`.
- `metadata.xlsx` for frame-level detection metrics.
- `CADICAprojections.json`, unless a later implementation uses it for real
  multi-view angles.

## 1. Prepare CADICA

This converts CADICA into the mirrored image tree expected by the pipeline and
writes `manifest.csv`, `manifest.jsonl`, and `preparation_summary.json`.

```bash
python scripts/prepare_cadica.py \
  --cadica-root data/CADICA \
  --output-root work/cadica_prepared \
  --frame-scope all_selected_videos \
  --negative-frame-scope selected \
  --copy-mode copy
```

Prepared images are written as:

```text
work/cadica_prepared/keyframes/p1/v1/slice_00012.png
```

## 2. Run Segmentation

```bash
python scripts/run_segmentation.py \
  --input work/cadica_prepared/keyframes \
  --stenosis-masks-root work/cadica_vessel_masks
```

## 3. Run Frame Stenosis Detection

```bash
python scripts/run_stenosis_detection.py \
  --images-root work/cadica_prepared/keyframes \
  --masks-root work/cadica_vessel_masks \
  --output-root work/cadica_frame_results \
  --workers 8 \
  --no-debug-images
```

Frame outputs are expected under:

```text
work/cadica_frame_results/p1/v1/slice_00012_stenosis_results.json
```

## 4. Run Temporal Fusion

CADICA videos can have variable frame counts, so use variable-count mode.

```bash
python scripts/run_temporal_fusion.py \
  --results-root work/cadica_frame_results \
  --output-root work/cadica_temporal_results \
  --allow-variable-frame-count \
  --min-supporting-frames 2 \
  --min-persistence-ratio 0.25 \
  --workers 8
```

Temporal outputs are optional for the supervised CADICA benchmark when
`--video-prediction-source frame_any` is used.

## 5. Run CADICA Benchmark

```bash
python scripts/run_cadica_benchmark.py \
  --manifest work/cadica_prepared/manifest.csv \
  --frame-results-root work/cadica_frame_results \
  --temporal-results-root work/cadica_temporal_results \
  --output-root work/cadica_benchmark \
  --frame-min-degree 0.0 \
  --box-margin-px 5 \
  --video-prediction-source frame_any
```

Main outputs:

- `cadica_frame_rows.csv` and `cadica_frame_rows.jsonl`
- `cadica_box_rows.csv`
- `cadica_video_rows.csv`
- `cadica_patient_rows.csv`
- `cadica_summary.json`
- Review queues:
  - `false_positive_frames.csv`
  - `false_negative_frames.csv`
  - `missed_gt_boxes.csv`
  - `unmatched_predicted_points.csv`

## Optional Review Images

Add `--write-review-images` to save simple overlay PNGs for manual review:

```bash
python scripts/run_cadica_benchmark.py \
  --manifest work/cadica_prepared/manifest.csv \
  --frame-results-root work/cadica_frame_results \
  --output-root work/cadica_benchmark \
  --write-review-images \
  --max-review-images 100
```

Images are written under `work/cadica_benchmark/review_images/` by default,
split into `false_positive/`, `false_negative/`,
`true_positive_localized/`, and `true_positive_nonlocalized/`. Each image
shows CADICA GT boxes, predicted stenosis points, and a small label/score
header.

## Interpretation

Frame binary metrics compare pipeline frame predictions against:

- Positive CADICA frames: frames with at least one CADICA GT box.
- Negative CADICA frames: frames from `nonlesionVideos.txt`, according to the
  preparation setting.

Lesion-video frames without GT boxes are labeled `unknown`, not negative. They
are written to the row files but excluded from binary frame metrics.

Localization uses point-in-box matching because the pipeline outputs stenosis
points, while CADICA provides bounding boxes. A predicted point is localized if
it falls inside a CADICA box expanded by `--box-margin-px`. IoU is not the main
metric for this benchmark.

Video metrics use `lesionVideos.txt` and `nonlesionVideos.txt`:

- `frame_any`: a video is predicted positive if any prepared frame is positive.
- `temporal_final`: a video is predicted positive if temporal fusion has a
  `final_lesion`.

Temporal metrics are optional and should be interpreted carefully if only
selected frames are prepared. Temporal fusion can improve stability analysis,
but CADICA frame selection affects the meaning of persistence.

The benchmark summary is supervised CADICA frame/video evaluation, not EHR
weak-label agreement.
