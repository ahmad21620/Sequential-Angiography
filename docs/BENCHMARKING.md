# Weak-Label Benchmarking

This document explains how to benchmark existing stenosis pipeline outputs
against external EHR weak labels. The benchmark tools read JSON outputs that
already exist on disk. They do not rerun frame detection, temporal fusion, or
multi-view fusion, and they do not modify pipeline JSON schemas.

For supervised CADICA benchmarking against CADICA frame annotations, use the
separate workflow in [CADICA_BENCHMARKING.md](CADICA_BENCHMARKING.md).

## EHR Weak Label Format

Weak labels are loaded from a JSONL file with one JSON object per case:

```json
{
  "case_id": "case_001",
  "column_U": "...",
  "excel_row": 91749,
  "stenosis_exists": "yes",
  "severity": "moderate",
  "max_percent": 62,
  "evidence": "...",
  "confidence": "high",
  "raw_model_output": {}
}
```

Expected fields:

- `case_id`: case identifier used to match pipeline outputs.
- `stenosis_exists`: `yes`, `no`, or `unclear`.
- `severity`: `none`, `mild`, `moderate`, `severe`, `occlusion`, or `unclear`.
- `confidence`: weak-label confidence, typically `high`, `medium`, or `low`.
- Other fields are preserved in detailed benchmark rows when available.

The benchmark normalizes labels into binary targets:

- Positive if `stenosis_exists == "yes"` or `severity` is `mild`, `moderate`, `severe`, or `occlusion`.
- Negative if `stenosis_exists == "no"` or `severity == "none"`.
- Unclear otherwise, including conflicting positive and negative signals.

Unclear labels are excluded from binary metrics by default.

## Important Interpretation Note

The EHR label is a case-level weak label, not frame-level or lesion-location
ground truth. It says that a case likely does or does not contain stenosis
according to the EHR labeler. It does not identify which frame, view, vessel
segment, or pixel location contains disease.

Frame-level and sequence-level benchmark metrics should therefore be read as
weak agreement metrics against case-level labels, not true localization
accuracy. A frame false positive may still be a real visual finding that is not
reflected in the EHR label. A false negative may be caused by missing frames,
weak-label noise, or a case-level label that refers to a different view.

## Frame-Level Benchmarking

Frame benchmarking reads frame detector JSON files under:

```text
work/stenosis_frame_results/case_001/view_01/slice_0003_stenosis_results.json
```

For each frame JSON:

- `case_id` is derived from the first path component under `--results-root`.
- `sequence_id` is derived from the path between `case_id` and the filename.
- `predicted_positive` is true when `stenosis_points` is non-empty and the
  max stenosis degree is at least `--frame-min-degree`.
- `score` is the max stenosis degree for that frame.
- `predicted_severity` is the worst severity among detected stenosis points.

Frame outputs include:

- `frame_rows.csv`
- `frame_rows.jsonl`
- `frame_summary.json`
- `frame_summary.csv`

Example:

```bash
python scripts/run_benchmark.py \
  --level frame \
  --results-root work/stenosis_frame_results \
  --weak-labels path/to/weak_labels.jsonl \
  --output-root work/benchmarks/frame \
  --write-threshold-sweep
```

## Temporal Sequence-Level Benchmarking

Temporal benchmarking reads temporal fusion JSON files under:

```text
work/stenosis_temporal_results/case_001/view_01/view_temporal_fusion.json
```

For each temporal JSON:

- `case_id` is derived from the first path component under `--results-root`.
- `sequence_id` is derived from the path between `case_id` and the filename.
- `predicted_positive` is true when `final_lesion` exists and passes optional
  degree and persistence thresholds.
- `score` is `final_lesion.degrees.median` when available.
- `max_degree`, `persistence_ratio`, `supporting_frame_count`, and
  `total_frame_count` are copied from `final_lesion` when available.

Temporal outputs include:

- `temporal_sequence_rows.csv`
- `temporal_sequence_rows.jsonl`
- `temporal_sequence_summary.json`
- `temporal_sequence_summary.csv`

Example:

```bash
python scripts/run_benchmark.py \
  --level temporal \
  --results-root work/stenosis_temporal_results \
  --weak-labels path/to/weak_labels.jsonl \
  --output-root work/benchmarks/temporal \
  --write-threshold-sweep
```

## Multi-View Case-Level Benchmarking

Multi-view benchmarking reads case fusion JSON files under:

```text
work/case_results/case_001/case_multiview_fusion.json
```

For each case JSON:

- `case_id` is derived from the first path component under `--results-root`.
- If the JSON also has `case_id`, it must match the path-derived case id.
- `predicted_positive` is true when `final_case_lesion` exists and passes
  optional confidence, degree, and distinct-support requirements.
- `final_degree`, `final_max_degree`, `final_severity`, `confidence_score`,
  `confidence_label`, support counts, and candidate counts are copied from the
  case JSON when available.

Multi-view outputs include:

- `multiview_case_rows.csv`
- `multiview_case_rows.jsonl`
- `multiview_case_summary.json`
- `multiview_case_summary.csv`
- `false_positive_cases.csv`
- `false_negative_cases.csv`
- `true_positive_cases.csv`
- `true_negative_cases.csv`

The multi-view summary JSON also includes metric breakdowns by weak-label
severity, weak-label confidence, and pipeline confidence label.

Example:

```bash
python scripts/run_benchmark.py \
  --level multiview \
  --results-root work/case_results \
  --weak-labels path/to/weak_labels.jsonl \
  --output-root work/benchmarks/multiview \
  --write-threshold-sweep
```

## Reading False Positives And False Negatives

Treat false positives and false negatives as review queues, not final clinical
truth.

False positives can mean:

- The pipeline found a real image-level stenosis that the EHR weak label missed.
- The weak labeler had insufficient or ambiguous evidence.
- A frame or view contains an artifact or segmentation error.
- The case id mapping is wrong.

False negatives can mean:

- The weak label is noisy or refers to disease not visible in the processed
  frames.
- The relevant view was missing before temporal or multi-view fusion.
- The detector threshold is too strict.
- Temporal persistence or multi-view support rules filtered out a real lesion.

The most useful workflow is to inspect the detailed row files and outcome CSVs,
then tune thresholds only after reviewing a small sample of disagreements.

## Threshold Sweeps

Use `--write-threshold-sweep` to generate threshold reports before changing
pipeline parameters.

Sweep outputs:

- `threshold_sweep_frame.csv`
- `threshold_sweep_temporal.csv`
- `threshold_sweep_multiview.csv`
- `threshold_sweep_summary.json`

The sweep evaluates thresholds from `0.0` to `1.0` in steps of `0.05`.
Each row includes:

- threshold
- TP, FP, TN, FN
- accuracy
- precision
- recall
- specificity
- F1
- false positive rate
- false negative rate
- balanced accuracy
- predicted positive count

The summary JSON identifies suggested operating points:

- `best_f1`
- `best_balanced_accuracy`
- `highest_recall_with_specificity_at_least_0_80`
- `lowest_false_positive_rate_with_recall_at_least_0_70`

Sweep score sources:

- Frame: max stenosis degree per frame.
- Temporal: final lesion median degree and persistence ratio.
- Multi-view: confidence score and final case lesion total score when present.

Threshold sweeps are analysis reports only. They do not update detector,
temporal fusion, or multi-view fusion parameters.

## Final Sweep Benchmark Report

After a full stenosis sweep has been benchmarked, generate the compact final
binary detection report with:

```bash
python scripts/report_sweep_benchmark.py \
  --sweep-root work/cadica_sweep \
  --output-root work/cadica_sweep/final_benchmark_report \
  --top-k 25
```

If `--benchmark-root` is omitted, the report reads
`<sweep-root>/benchmark_results`. The report only uses existing benchmark
outputs; it does not rerun frame detection, temporal fusion, multi-view fusion,
or benchmarking.

This report is for final binary stenosis detection metrics. It ignores
severity-related summaries and level-of-stenosis statistics. Multi-view is the
primary final case-level result. Frame and temporal summaries are included only
as comparison stages.

When the sweep used `--multiview-split-by-coronary-side`, left and right side
fusion outputs are combined into one case-level binary prediction. A case is
positive if either side has a valid final lesion that passes the multi-view
benchmark thresholds.
