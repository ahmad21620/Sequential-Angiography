# Coronary Angiogram Keyframes Extraction

This folder contains the keyframe extraction stage of the Sequential Angiography monorepo. The importable package is `angio_keyframes`, and the compatibility script `keyframes_extraction.py` is kept for direct subproject use.

## Project layout

```text
.
|-- keyframes_extraction.py
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
`-- README.md
```

## Extraction logic

The extraction rule follows one temporal scoring pipeline per sequence:

1. Read each frame in filename order.
2. Convert it to grayscale.
3. Enhance vessel-like dark tubular structures with CLAHE followed by multi-scale black-hat filtering.
4. Build a pre-contrast baseline from the first `N` enhanced frames using a pixelwise median.
5. Score every frame by subtracting that baseline from the enhanced frame, clamping negative values to `0`, and taking the mean response.
6. Smooth the score curve over time with a centered moving average.
7. Find the strongest smoothed peak and keep one contiguous `limit`-frame window around it. The default `centered` mode keeps the peak centered as much as possible; `leading` mode ends the window at the peak.
8. Save the original grayscale keyframes into a mirrored output tree in temporal order, without copying the original `frames` directories.

The default is `6` keyframes per sequence, the default baseline uses the first `3` frames, and the default smoothing window is `5`. If a sequence is shorter than the baseline or output window, the available frames are used.

## Installation

Install dependencies and the monorepo package from the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Usage

Run the root wrapper from the repository root:

```bash
python scripts/run_keyframes.py data/raw_cases --backend cpu --output-root work/keyframes --overwrite
```

Run the compatibility entrypoint from this subproject folder:

```bash
cd keyframes-extraction
python keyframes_extraction.py ../data/raw_cases --backend cpu --output-root ../work/keyframes --overwrite
```

After `pip install -e .`, run the package directly:

```bash
python -m angio_keyframes data/raw_cases --backend cpu --output-root work/keyframes --overwrite
```

You can also use the installed console script:

```bash
angio-keyframes data/raw_cases --backend cpu --output-root work/keyframes --overwrite
```

CUDA example:

```bash
angio-keyframes data/raw_cases --backend cuda --output-root work/keyframes_cuda --overwrite
```

Leading-window example, useful for the initial contrast-flow phase before maximum vessel filling:

```bash
python scripts/run_keyframes.py data/raw_cases --limit 8 --window-mode leading --backend cpu --output-root work/keyframes_leading --overwrite
```

CADICA example, run from the repository root:

```bash
python scripts/run_keyframes.py \
  --input-root data/CADICA \
  --cadica-selected-frame-counts \
  --output-root work/cadica_keyframes \
  --overwrite
```

With `--cadica-selected-frame-counts`, a CADICA root is resolved to
`selectedVideos/`, frame directories default to CADICA `input/` folders, outputs
are written as `pX/vY/*.png`, and each video keeps the same number of frames as
its `pX_vY_selectedFrames.txt` reference. Without that flag, the original fixed
`--limit` extraction behavior is unchanged.

Useful flags:

- `--limit 8`: keep a different number of keyframes.
- `--window-mode centered|leading`: choose whether the peak is centered in the output window or the final frame of the output window. Default: `centered`.
- `--baseline-frames 5`: use the first 5 frames to build the pre-contrast baseline.
- `--smoothing-window 7`: use a larger odd centered moving-average window for score smoothing.
- `--backend cuda`: run the heavy image-processing path on a CUDA-capable OpenCV build.
- `--workers 4`: process up to 4 discovered frame directories in parallel.
- `--input-root data/CADICA`: named alias for the input path, useful in scripted runs.
- `--output-root extracted_keyframes`: write keyframes into a mirrored tree rooted at `extracted_keyframes`.
- `--frames-dirname frames`: change the directory name used for discovery.
- `--cadica-selected-frame-counts`: opt into CADICA `selectedVideos/pX/vY/input` discovery and per-video selected-frame counts.
- `--overwrite`: replace an existing output directory.
- `--skip-existing`: skip sequences whose output directory already exists.

## Development

Run the test suite with:

```bash
python -m unittest discover keyframes-extraction/tests
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
