# Vessel Segmentation in PyTorch

This repository trains and evaluates a patch-based ResUNet for vessel segmentation from explicit on-disk dataset splits. The workflow now supports:

- retinal training from scratch with real FOV masks
- coronary fine-tuning from a retinal checkpoint
- explicit `train` / `val` / `test` splits only
- no HDF5 conversion, hidden validation split, or legacy mixed-dataset path

## Project Structure

```text
.
|-- train.py
|-- predict.py
|-- validate_dataset.py
|-- segment_retinal_images.py
|-- requirements.txt
`-- src
    `-- drive_seg
        |-- augmentations.py
        |-- __init__.py
        |-- config.py
        |-- dataset.py
        |-- engine.py
        |-- losses.py
        |-- metrics.py
        |-- model.py
        |-- optim.py
        |-- patching.py
        |-- preprocessing.py
        |-- utils.py
        `-- visualization.py
```

## Supported Dataset Layouts

### Retinal

```text
Datasets/
`-- Retinal/
    |-- train/
    |   |-- Original/
    |   |-- Mask/
    |   `-- FOV/
    |-- val/
    |   |-- Original/
    |   |-- Mask/
    |   `-- FOV/
    `-- test/
        |-- Original/
        |-- Mask/
        `-- FOV/
```

### Coronary

```text
Datasets/
`-- Coronary/
    |-- train/
    |   |-- Original/
    |   `-- Mask/
    |-- val/
    |   |-- Original/
    |   `-- Mask/
    `-- test/
        |-- Original/
        `-- Mask/
```

## Pairing Rules

- `Original` and `Mask` are required in every split.
- Files are paired by basename/stem, not by identical full filename.
  Example: `Original/0014.jpg` pairs with `Mask/0014.png`.
- If a split also contains `FOV/`, FOV files are paired by the same stem rule.
- Supported image suffixes are `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, and `.bmp`.
- Original images may be RGB or grayscale.
- Masks and FOV files are loaded as single-channel images.
- Every matched pair or triplet must have identical spatial size.

## FOV Behavior

- Retinal splits should include `FOV/`, and the loader uses the real FOV masks.
- Coronary splits do not need `FOV/`.
- If `FOV/` is missing for a split, the loader automatically creates a full-image valid mask of all ones.
- Validation and evaluation report whether each split uses provided FOV masks or auto-generated full-image masks.

## Installation

```bash
pip install -r requirements.txt
```

## Validate a Dataset Root

```bash
python validate_dataset.py --dataset-root path/to/Datasets/Retinal
python validate_dataset.py --dataset-root path/to/Datasets/Coronary
```

The validator prints counts for each split and reports:

- missing `Original` / `Mask` pairs
- duplicate stems in a directory
- unexpected extra files with unmatched stems
- size mismatches
- whether FOV is provided or auto-generated for each split

## Retinal Training From Scratch

```bash
python train.py \
  --dataset-root path/to/Datasets/Retinal \
  --train-split train \
  --val-split val \
  --checkpoint checkpoints/retinal_best.pt
```

Training data loading mode is configurable:

- default: `--lazy-train-loading true`
  Uses much less RAM by loading training images lazily from disk.
- optional: `--lazy-train-loading false`
  Preloads the training split into memory, which can be faster if the machine has enough RAM.

Full-image validation cadence is also configurable:

- default: `--full-image-validation-every 1`
  Runs full-image validation every epoch.
- example: `--full-image-validation-every 2`
  Keeps patch validation every epoch but runs full-image validation every second epoch.

## Retinal Test Evaluation

```bash
python predict.py \
  --dataset-root path/to/Datasets/Retinal \
  --split test \
  --checkpoint checkpoints/retinal_best.pt
```

## Coronary Fine-Tuning From Retinal Weights

```bash
python train.py \
  --dataset-root path/to/Datasets/Coronary \
  --train-split train \
  --val-split val \
  --init-checkpoint checkpoints/retinal_best.pt \
  --checkpoint checkpoints/coronary_finetuned.pt
```

`--init-checkpoint` initializes model weights for transfer learning. It does not resume optimizer or scheduler state.

## Resume Training

Use `--resume-checkpoint` to continue an interrupted run from a saved checkpoint state:

```bash
python train.py \
  --dataset-root path/to/Datasets/Retinal \
  --train-split train \
  --val-split val \
  --resume-checkpoint checkpoints/last_model.pt \
  --checkpoint checkpoints/retinal_best.pt
```

`--resume-checkpoint` restores:

- model weights
- optimizer state
- scheduler state
- training history
- best-checkpoint tracking state
- the next epoch number

## Coronary Test Evaluation

```bash
python predict.py \
  --dataset-root path/to/Datasets/Coronary \
  --split test \
  --checkpoint checkpoints/coronary_finetuned.pt
```

## Training and Evaluation Behavior

The shared pipeline keeps the existing patch-based behavior for both datasets:

- grayscale preprocessing with normalization, CLAHE, and gamma correction
- stratified patch sampling with vessel, hard-negative, and random-valid-region patches
- valid-region-aware patch filtering using real FOV masks when present and full-image masks otherwise
- optional lazy training-image loading to reduce RAM usage on large datasets
- conservative training-time augmentation
- configurable loss, optimizer, and scheduler
- patch validation every epoch and configurable full-image validation cadence during training
- ROC, metrics, and visualization outputs during evaluation

## Outputs

Training writes:

- `checkpoints/<name>.pt` for the best checkpoint selected by the configured validation metric
- `checkpoints/last_model.pt` for the most recent epoch, which can be used with `--resume-checkpoint`
- `outputs/training/learning_curve.png`
- `outputs/training/history.json`
- `outputs/training/dataset_summary.json`

Evaluation writes by default to `outputs/evaluation/<split>/`:

- `metrics.json`
- `roc_curve.png`
- `visualizations/*.png`

## Custom Inference

For ad hoc retinal images without masks:

```bash
python segment_retinal_images.py --input path/to/retinal_images --checkpoint checkpoints/retinal_best.pt
```

For stenosis-repo batch compatibility, write mirrored binary masks named
`<image_stem>_mask.png` under a separate mask root:

```bash
python segment_retinal_images.py \
  --input data/case_root \
  --checkpoint checkpoints/retinal_best.pt \
  --stenosis-masks-root outputs/stenosis_masks
```

If the input contains `data/case_root/study_a/series_b/slice_0001.png`, this produces
`outputs/stenosis_masks/study_a/series_b/slice_0001_mask.png`.

You can then use the same segmentation input tree as the stenosis repo `--images-root`
and the mirrored mask tree as `--masks-root`:

```bash
python path/to/stenosis_repo_batch_entry.py \
  --images-root data/case_root \
  --masks-root outputs/stenosis_masks
```

## Notes

- The package name remains `drive_seg`, but the dataset flow is now generic to explicit split-based vessel datasets.
- Checkpoints store model config, patch geometry, training config, and initialization metadata.
- The recommended workflow is retinal training first, then coronary fine-tuning from `--init-checkpoint`.
- Use `--resume-checkpoint` only when you want to continue the same training run, not for transfer learning.
