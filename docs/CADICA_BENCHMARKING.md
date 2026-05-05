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
- `selectedVideos/CADICAprojections.json`: categorical projection groups
  (`videosLCA`, `videosLCA2`, `videosRCA`) for multi-view fusion metadata.

Intentionally not used in the first implementation:

- `nonselectedVideos/` for official metrics.
- `groundTruthTable.mat`.
- `metadata.xlsx` for frame-level detection metrics.

CADICA does not provide numeric RAO/LAO or CRA/CAU angles in
`CADICAprojections.json`. The preparation step keeps `rao_lao=0.0` and
`cra_cau=0.0` only as backward-compatible placeholders, marks
`angle_status="missing"`, and writes the projection group and coronary side as
categorical metadata.

## CADICA Keyframe Extraction

To run the keyframe extractor directly on CADICA while preserving the original
non-CADICA behavior, opt into CADICA mode:

```bash
python scripts/run_keyframes.py \
  --input-root data/CADICA \
  --cadica-selected-frame-counts \
  --output-root work/cadica_keyframes \
  --overwrite
```

This reads `data/CADICA/selectedVideos/pX/vY/input/*.png`, writes
`work/cadica_keyframes/pX/vY/*.png`, and uses each video's
`pX_vY_selectedFrames.txt` count as that video's extraction limit. If you omit
`--cadica-selected-frame-counts`, the normal fixed `--limit` extraction path is
used.

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

To run a full CADICA parameter sweep after keyframe extraction and vessel
segmentation, use:

```bash
python scripts/run_stenosis_temporal_sweep.py \
  --images-root work/cadica_keyframes \
  --masks-root work/cadica_vessel_masks \
  --output-root work/cadica_sweep \
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

## 5. Run Multi-View Fusion

For CADICA, use categorical projection-group diversity rather than geometric
angle diversity:

```bash
python scripts/run_multiview_fusion.py \
  --case-root-tree work/cadica_prepared/keyframes \
  --temporal-results-root work/cadica_temporal_results \
  --output-root work/cadica_multiview_results \
  --view-diversity-mode projection_group \
  --split-by-coronary-side
```

`--view-diversity-mode projection_group` uses the CADICA groups as categorical
view metadata. It does not perform geometric angle-aware fusion on CADICA.
`--split-by-coronary-side` runs left and right coronary-side fusion separately
so LCA/LCA2 and RCA views cannot support each other. Views with missing or
unknown projection metadata are reported in the combined JSON but are not used
for side-specific fusion by default.

## 6. Run CADICA Benchmark

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
