# Training Modifications

This log tracks high-level training pipeline changes over time so later fixes can append new sections without rewriting earlier history.

## Fix 2: Stratified Vessel-Aware Patch Sampling

### What changed

- Replaced shape-only patch sampling with a deterministic stratified sampler for both training and validation.
- Added vessel-centered patches, near-vessel hard-negative patches, and random valid FOV patches as the three sampling groups.
- Started using cropped FOV masks during training-time patch selection when they are available.
- Extended the training summary JSON to record patch-sampling settings and sampled patch counts by category.

### Why it changed

- Uniform patch sampling overproduced easy background patches and underemphasized thin-vessel supervision.
- Using valid-region-aware checks keeps sampled patches focused on relevant content instead of low-signal border regions.
- Hard-negative sampling adds more informative background examples close to vessels, which improves boundary learning pressure.

### New patch sampling behavior

- Vessel-centered patches receive the largest share of the sampling budget.
- Hard negatives are sampled from background pixels within a configurable band around vessel masks.
- The remaining budget is allocated to random valid FOV patches for broader context coverage.
- If a category has no valid candidates for a given pool, its quota is reassigned to other valid categories while keeping the run deterministic.

### New training options

- `--vessel-patch-ratio`
- `--hard-negative-patch-ratio`
- `--min-fov-coverage`
- `--hard-negative-band-width`

### Expected practical effect

- Training should spend more updates on vessel structure and vessel-adjacent mistakes instead of easy empty background.
- Validation patches now reflect the same sampling rules as training, which makes patch-level monitoring more aligned with the training objective.
- Minimum valid-region coverage should reduce wasted patches and improve patch quality near image borders.

## Fix 3: Conservative Training Patch Augmentation

### What changed

- Added a dedicated training-only augmentation pipeline for patch/image-mask pairs.
- Kept validation fully unaugmented.
- Extended the training summary JSON to record the augmentation configuration used for each run.

### Why it changed

- The patch pipeline needed more variation to reduce overfitting and improve robustness once the cleaner split and stratified sampling were in place.
- Vessel segmentation benefits from mild geometric variation and small appearance shifts without aggressive distortions.

### New augmentation categories

- Geometric transforms: horizontal flip, vertical flip, random 90-degree rotations, and mild free-angle rotations.
- Image-only photometric transforms: brightness, contrast, and gamma adjustments.
- Image-only degradations: mild Gaussian noise and mild blur.

### Configurable options

- `--enable-augmentation`
- `--hflip-prob`
- `--vflip-prob`
- `--rot90-prob`
- `--max-rotation-deg`
- `--rotation-prob`
- `--brightness-jitter`
- `--contrast-jitter`
- `--gamma-jitter`
- `--noise-prob`
- `--noise-std`
- `--blur-prob`
- `--blur-kernel-size`

### Expected practical effect

- Training should see a broader but still vessel-safe patch distribution.
- Mild appearance and geometry changes should improve robustness to acquisition differences and reduce early validation plateaus.
- Keeping validation unaugmented preserves a clean and stable monitoring signal.

## Fix 4: Optimizer and Scheduler Upgrade

### What changed

- Replaced the fixed inline optimizer setup with a configurable optimizer and scheduler stack.
- Added explicit support for `adamw` and `sgd` optimizers.
- Added explicit support for `none`, `reduce_on_plateau`, and `cosine` learning-rate schedules.
- Extended the training outputs to record optimizer/scheduler settings and learning-rate history.

### Why it changed

- The earlier fixed optimizer path was serviceable, but it left convergence quality and late-stage training control on the table.
- Small-data vessel segmentation benefits from stronger default regularization and a scheduler that reacts cleanly to validation behavior.

### Available optimizer and scheduler options

- Optimizers: `adamw`, `sgd`
- Schedulers: `none`, `reduce_on_plateau`, `cosine`

### New default behavior

- Training now defaults to `adamw`.
- The default learning rate is `1e-3` with `1e-4` weight decay.
- The default scheduler is `reduce_on_plateau`, driven by validation loss.
- `SGD` remains available with momentum as an explicit alternative.

### Expected practical effect

- The default stack should make early convergence steadier and late-stage refinement more responsive to validation stalls.
- Validation-aware learning-rate reduction should help training keep improving after the first plateau instead of staying on one fixed step size.
- Cosine annealing provides a simple alternative for full-horizon runs when a smooth epoch-based decay is preferred.

## Fix 5: Loss Upgrade for Thin-Vessel Segmentation

### What changed

- Replaced the fixed baseline loss path with a configurable vessel-segmentation loss stack.
- Added support for `bce_dice`, `weighted_bce_dice`, `tversky`, and `focal_tversky`.
- Extended the training summary JSON to record the selected loss and its relevant parameters.

### Why it changed

- Thin-vessel segmentation is highly imbalanced, and the old loss path did not give enough emphasis to hard positives.
- Vessel recall benefits from a loss setup that can explicitly rebalance positives or penalize false negatives more strongly.

### Available loss options

- `bce_dice`
- `weighted_bce_dice`
- `tversky`
- `focal_tversky`

### New default behavior

- Training now defaults to `weighted_bce_dice`.
- The default positive-class weight is `3.0`.
- The default combined-loss weights are kept simple with `1.0` BCE weight and `1.0` Dice weight.
- Tversky defaults remain conservative with `alpha=0.3`, `beta=0.7`, and focal Tversky uses `gamma=1.33`.

### Expected practical effect

- The default loss should push harder on sparse vessel pixels and improve thin-vessel recall compared with the old baseline.
- Tversky-based options give the training pipeline cleaner tools for handling strong foreground/background imbalance.
- Validation loss, scheduler decisions, and early stopping now stay aligned with the exact configured loss choice.

## Fix 6: Larger Patch Context and Patch-Size Cleanup

### What changed

- Replaced the split patch-height and patch-width training options with one square `patch_size` setting.
- Increased the default patch size to `64`.
- Updated training, validation sampling, dataset extraction, checkpointed patch geometry, and patch-based inference to read from the same patch-size configuration path.
- Extended the training summary JSON so the configured patch geometry is recorded explicitly.

### Why it changed

- The earlier small patch context limited the model's view of vessel continuity, branching structure, and junction neighborhoods.
- A larger square context gives the network more structural information without changing the architecture itself.

### New default patch size

- The default patch size is now `64`.

### Pipeline areas now tied to the configured patch size

- Train patch sampling
- Validation patch sampling
- Patch extraction in the dataset
- Checkpointed patch geometry
- Patch-based split evaluation and custom-image inference

### Expected practical effect

- The larger context should improve learning around vessel continuity, branch structure, and local junction geometry.
- Patch-based training and inference should now stay more internally consistent because every stage derives geometry from the same configured patch size.
- The cleaner configuration path reduces the chance of stale small-patch assumptions lingering in later fixes.

## Fix 7: Full-Image Validation and Checkpoint Selection Upgrade

### What changed

- Added deterministic full-image validation on the validation image pool during training.
- Started tracking full-image Dice, F1, precision, recall, IoU, and accuracy alongside the existing patch-level validation loss.
- Switched best-checkpoint selection and early stopping to a configurable validation-selection metric instead of assuming validation loss is always the deciding signal.
- Extended the saved training history so it records the validation-selection mode, threshold, and per-epoch full-image metrics.

### Why it changed

- Patch-level validation loss is useful for optimization, but it does not always reflect the checkpoint with the best full-image vessel segmentation quality.
- Vessel segmentation quality is judged on full-image structure, so checkpoint selection should follow a full-image metric by default.

### Full-image metrics now tracked

- Dice
- F1
- Precision
- Recall
- IoU
- Accuracy

### New default behavior

- Full-image validation now runs by default during training.
- The default checkpoint-selection metric is `val_dice`.
- Patch-level validation loss is still retained for loss tracking and the default reduce-on-plateau scheduler.

### New validation options

- `--selection-metric`
- `--val-threshold`
- `--run-full-image-validation`

### Expected practical effect

- The saved best checkpoint should be more closely aligned with real full-image vessel segmentation quality instead of only patch-level loss behavior.
- Validation reporting should now give a clearer picture of thin-vessel continuity, recall, and overall segmentation quality on held-out images.
- The configurable threshold and selection metric make checkpoint selection more explicit and easier to reason about during experiments.

## Fix 8: ResUNet Architecture Upgrade

### What changed

- Replaced the plain U-Net model with a compact ResUNet built from residual encoder blocks, a residual bottleneck, and residual decoder refinement after skip fusion.
- Added explicit `architecture` metadata to the saved model configuration so checkpoints record the network family they were trained with.
- Switched the default model width to `48` base channels while keeping `64` available through the existing width setting.
- Centralized model construction so training, split evaluation, and custom-image inference all build the network from the same configuration path.

### Why it changed

- Residual feature paths make deeper patch models easier to optimize and help preserve fine vessel evidence across encoder and decoder stages.
- The older plain U-Net path had become the main architectural bottleneck after the surrounding training pipeline improvements.
- A `48`-channel default gives the upgraded network more representational capacity without jumping straight to the memory cost of `64`.

### New default width

- The default model width is now `48` base channels.
- `64` base channels is the recommended next width step when memory allows.

### Expected practical effect

- Residual feature reuse should improve vessel continuity modeling and make thin-structure evidence easier to preserve through the network.
- Optimization should be steadier than the plain U-Net baseline, especially once the wider patch context and full-image validation signals are already in place.
- The `48`-channel default should offer a better capacity-to-VRAM balance, while `64` remains the natural higher-capacity option for stronger hardware.
