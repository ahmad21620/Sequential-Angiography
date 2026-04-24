# Coronary Angiogram Keyframes Extraction

This repository is now organized as a Python package instead of a notebook-driven project.

The original extraction notebook has been migrated into importable modules under `src/`, so there is one maintained code path and no notebook-specific runtime logic left behind.

## Project layout

```text
.
|-- keyframes_extraction.py
|-- pyproject.toml
|-- requirements.txt
|-- src/
|   `-- angio_keyframes/
|       |-- __init__.py
|       |-- __main__.py
|       |-- backends.py
|       |-- cli.py
|       |-- discovery.py
|       |-- images.py
|       |-- models.py
|       `-- pipeline.py
|-- tests/
|   `-- test_pipeline.py
|-- oneSampleCORO/
`-- all_dataSet_50_videos/
```

## Extraction logic

The extraction rule follows one temporal scoring pipeline per sequence:

1. Read each frame in filename order.
2. Convert it to grayscale.
3. Enhance vessel-like dark tubular structures with CLAHE followed by multi-scale black-hat filtering.
4. Build a pre-contrast baseline from the first `N` enhanced frames using a pixelwise median.
5. Score every frame by subtracting that baseline from the enhanced frame, clamping negative values to `0`, and taking the mean response.
6. Smooth the score curve over time with a centered moving average.
7. Find the strongest smoothed peak and keep one contiguous `limit`-frame window centered around it as much as possible.
8. Save the original grayscale keyframes into a mirrored output tree in temporal order, without copying the original `frames` directories.

The default is `6` keyframes per sequence, the default baseline uses the first `3` frames, and the default smoothing window is `5`. If a sequence is shorter than the baseline or output window, the available frames are used.

## Installation

Create a virtual environment and install the runtime dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you want the package import path and console script available in the environment, install the project in editable mode:

```bash
pip install -e .
```

## Usage

Run the compatibility entrypoint from the repo root:

```bash
python keyframes_extraction.py all_dataSet_50_videos --backend cpu --output-root extracted_keyframes --overwrite
```

After `pip install -e .`, run the package directly:

```bash
python -m angio_keyframes all_dataSet_50_videos --backend cpu --output-root extracted_keyframes --overwrite
```

You can also use the installed console script:

```bash
angio-keyframes all_dataSet_50_videos --backend cpu --output-root extracted_keyframes --overwrite
```

CUDA example:

```bash
angio-keyframes all_dataSet_50_videos --backend cuda --output-root extracted_keyframes_cuda --overwrite
```

Useful flags:

- `--limit 8`: keep a different number of keyframes.
- `--baseline-frames 5`: use the first 5 frames to build the pre-contrast baseline.
- `--smoothing-window 7`: use a larger odd centered moving-average window for score smoothing.
- `--backend cuda`: run the heavy image-processing path on a CUDA-capable OpenCV build.
- `--workers 4`: process up to 4 discovered frame directories in parallel.
- `--output-root extracted_keyframes`: write keyframes into a mirrored tree rooted at `extracted_keyframes`.
- `--frames-dirname frames`: change the directory name used for discovery.
- `--overwrite`: replace an existing output directory.
- `--skip-existing`: skip sequences whose output directory already exists.

## Development

Run the test suite with:

```bash
python -m unittest discover -s tests
```

## Performance

- Frames are decoded directly in grayscale with `cv2.IMREAD_GRAYSCALE`, which avoids an extra BGR-to-gray conversion step.
- Keyframe candidates keep only metadata and scores in memory, so full grayscale images are not retained for the entire sequence.
- CLAHE and black-hat kernels are created once per process and reused across frames.
- `--workers` enables folder-level parallelism across discovered sequences for the cpu backend while keeping each sequence itself deterministic.
- The CLI shows a `tqdm` progress bar over discovered sequences so long dataset runs stay visible.

## Backends

- `--backend cpu` is the default and remains fully supported on any OpenCV build.
- `--backend cuda` accelerates the heavy per-frame enhancement and scoring path with OpenCV CUDA primitives while keeping selection, traversal, and output writing unchanged.
- The cuda backend requires OpenCV to be built with CUDA support and to expose the needed CUDA Python bindings.
- If you explicitly request `--backend cuda` and CUDA is unavailable, the command exits with a clear error instead of silently falling back to cpu.

## Notes

- When the input path already points to a directory of image files, the selected keyframes are written into the output root directly.
- When the input path is a dataset root, every nested directory that contains supported image files is treated as a sequence and written directly into the mirrored output tree as `<output-root>/<relative-sequence-path>`.
- Marker images such as `.extract_complete.png` are ignored during discovery, scoring, and output writing.
- `--overwrite` and `--skip-existing` are mutually exclusive output modes.
- The dataset directories under `all_dataSet_50_videos/` are preserved as project assets. The package code lives only under `src/`.
