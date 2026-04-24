# Sequential Angiography Pipeline

## Project Overview

This repository contains a three-stage Sequential Angiography pipeline for extracting angiography keyframes, generating vessel masks, and detecting stenosis with temporal and multi-view fusion.

The three stages are kept in separate project folders so each component can be run and tested independently while sharing one root installation and one set of root-level project configuration files.

## Repository Structure

```text
project-root/
|-- .gitattributes
|-- .gitignore
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- docs/
|   `-- PIPELINE.md
|-- scripts/
|   |-- _wrapper_utils.py
|   |-- run_keyframes.py
|   |-- run_segmentation.py
|   |-- run_stenosis_detection.py
|   |-- run_temporal_fusion.py
|   `-- run_multiview_fusion.py
|-- keyframes-extraction/
|   |-- keyframes_extraction.py
|   |-- src/angio_keyframes/
|   `-- tests/
|-- vessel-segmentation/
|   |-- train.py
|   |-- predict.py
|   |-- validate_dataset.py
|   |-- segment_retinal_images.py
|   `-- src/drive_seg/
`-- stenosis-detection/
|   |-- run_stenosis_detection.py
|   |-- run_temporal_fusion.py
|   |-- run_multiview_fusion.py
|   |-- stenosis_detection/
|   `-- tests/
```

Package names:

- `angio_keyframes`
- `drive_seg`
- `stenosis_detection`

## Installation

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Pipeline Stages

1. **Keyframe extraction**
   - Input: raw angiography frame sequences.
   - Output: mirrored keyframe image folders.
   - Project folder: `keyframes-extraction`.

2. **Vessel segmentation / mask generation**
   - Input: extracted keyframes and a trained segmentation checkpoint.
   - Output: mirrored binary vessel masks named `<image_stem>_mask.png`.
   - Project folder: `vessel-segmentation`.

3. **Stenosis detection and fusion**
   - Input: keyframes, mirrored vessel masks, and case/view metadata.
   - Output: frame-level stenosis results, view-level temporal fusion results, and case-level multi-view fusion results.
   - Project folder: `stenosis-detection`.

## How To Run Each Stage

The root `scripts/` wrappers run from the repository root and forward arguments to the existing subproject CLIs.

### Stage 1: Keyframe Extraction

```bash
python scripts/run_keyframes.py data/raw_cases --output-root work/keyframes --backend cpu --overwrite
```

Subproject command:

```bash
cd keyframes-extraction
python keyframes_extraction.py ../data/raw_cases --output-root ../work/keyframes --backend cpu --overwrite
```

### Stage 2: Vessel Mask Generation

```bash
python scripts/run_segmentation.py ^
  --input work/keyframes ^
  --checkpoint vessel-segmentation/checkpoints/coronary_finetuned.pt ^
  --stenosis-masks-root work/vessel_masks
```

Subproject command:

```bash
cd vessel-segmentation
python segment_retinal_images.py ^
  --input ../work/keyframes ^
  --checkpoint checkpoints/coronary_finetuned.pt ^
  --stenosis-masks-root ../work/vessel_masks
```

### Stage 3: Frame-Level Stenosis Detection

```bash
python scripts/run_stenosis_detection.py ^
  --images-root work/keyframes ^
  --masks-root work/vessel_masks ^
  --output-root work/stenosis_frame_results
```

Subproject command:

```bash
cd stenosis-detection
python run_stenosis_detection.py ^
  --images-root ../work/keyframes ^
  --masks-root ../work/vessel_masks ^
  --output-root ../work/stenosis_frame_results
```

### Temporal Fusion

```bash
python scripts/run_temporal_fusion.py ^
  --results-root work/stenosis_frame_results ^
  --output-root work/stenosis_temporal_results ^
  --skip-existing
```

### Multi-View Fusion

```bash
python scripts/run_multiview_fusion.py ^
  --case-root work/keyframes/case_001 ^
  --output work/case_results/case_001/case_multiview_fusion.json
```

Detailed input/output layout and troubleshooting notes are available in `docs/PIPELINE.md`.

## Tests And Smoke Checks

From the repository root:

```bash
python -m compileall scripts keyframes-extraction vessel-segmentation stenosis-detection
python -m unittest discover keyframes-extraction/tests
python -m unittest discover stenosis-detection/tests
python -c "import angio_keyframes, drive_seg, stenosis_detection; print('ok')"
```

Wrapper help checks:

```bash
python scripts/run_keyframes.py --help
python scripts/run_segmentation.py --help
python scripts/run_stenosis_detection.py --help
python scripts/run_temporal_fusion.py --help
python scripts/run_multiview_fusion.py --help
```

## CUDA OpenCV Requirement

The default keyframe extraction backend is CPU and works with the standard `opencv-python` package.

GPU keyframe extraction with `--backend cuda` requires a custom OpenCV build compiled with CUDA support and the required CUDA Python bindings. The standard `opencv-python` wheel does not provide this CUDA backend.

## Expected Inputs And Outputs

Expected raw input:

```text
data/raw_cases/
  case_001/
    views.json
    view_01/
      frames/
        slice_0001.png
        slice_0002.png
    view_02/
      frames/
        slice_0001.png
        slice_0002.png
```

Expected pipeline outputs:

```text
work/keyframes/
  case_001/
    views.json
    view_01/
      slice_0003.png
      slice_0004.png
work/vessel_masks/
  case_001/
    view_01/
      slice_0003_mask.png
      slice_0004_mask.png
work/stenosis_frame_results/
  case_001/
    view_01/
      slice_0003_stenosis_results.json
      slice_0003_centerline.png
      slice_0003_segmentation_points.png
      slice_0003_stenosis_result_mask.png
      slice_0003_stenosis_result_original.png
work/stenosis_temporal_results/
  case_001/
    view_01/
      view_temporal_fusion.json
      view_temporal_fusion_summary.png
work/case_results/
  case_001/
    case_multiview_fusion.json
```

Keyframe extraction preserves case-level `views.json` files in the keyframe output tree. Multi-view fusion uses `views.json` to combine temporal fusion outputs across views.
