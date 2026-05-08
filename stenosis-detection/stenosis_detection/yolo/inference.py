from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..thinning import thin_binary_mask
from .schema import SCORE_SEMANTICS, YoloDetection, YoloDetectorMetadata, build_yolo_frame_payload

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is absent.
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, *, total=None, desc=None, unit=None, dynamic_ncols=None):
            self.iterable = iterable
            self.total = total
            self.desc = desc or "Progress"
            self.unit = unit or "item"
            self.count = 0
            print(f"{self.desc}: 0/{self.total if self.total is not None else '?'} {self.unit}")

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            for item in self.iterable:
                yield item
                self.update(1)

        def update(self, n=1):
            self.count += n

        def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
            return None

        def write(self, message):
            print(message)

        def close(self):
            print(f"{self.desc}: {self.count}/{self.total if self.total is not None else '?'} {self.unit}")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ORIGINAL_SLICE_PATTERN = re.compile(r"^slice_(\d+)$", re.IGNORECASE)
CADICA_FRAME_PATTERN = re.compile(r"^p\d+_v\d+_\d+$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class YoloInferenceConfig:
    weights: str
    imgsz: int | None = None
    conf: float = 0.25
    iou: float = 0.7
    device: str | None = None
    mask_threshold: int = 127
    batch_size: int = 1

    def detector_metadata(self) -> YoloDetectorMetadata:
        return YoloDetectorMetadata(
            name="yolov8",
            weights=str(self.weights),
            imgsz=self.imgsz,
            conf=float(self.conf),
            iou=float(self.iou),
            device=None if self.device is None else str(self.device),
            score_semantics=SCORE_SEMANTICS,
        )


@dataclass(frozen=True, slots=True)
class YoloTreeJob:
    image_path: Path
    relative_dir: Path
    image_stem: str
    mask_path: Path | None = None


@dataclass(frozen=True, slots=True)
class YoloBatchFailure:
    image_path: str
    error: str


@dataclass(frozen=True, slots=True)
class YoloBatchSummary:
    images_root: Path
    masks_root: Path | None
    output_root: Path
    total_jobs: int
    workers: int
    processed: int
    skipped_existing: int
    failed: int
    failures: list[YoloBatchFailure]
    review_images_saved: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "images_root": str(self.images_root),
            "masks_root": None if self.masks_root is None else str(self.masks_root),
            "output_root": str(self.output_root),
            "total_jobs": int(self.total_jobs),
            "workers": int(self.workers),
            "processed": int(self.processed),
            "skipped_existing": int(self.skipped_existing),
            "failed": int(self.failed),
            "failures": [asdict(failure) for failure in self.failures],
            "review_images_saved": bool(self.review_images_saved),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 frame-level stenosis detection and write temporal-compatible JSON outputs.",
    )
    parser.add_argument("--images-root", required=True, help="Root containing extracted keyframe images.")
    parser.add_argument("--weights", required=True, help="Path to YOLO weights, usually best.pt.")
    parser.add_argument("--output-root", required=True, help="Root where mirrored YOLO frame JSON outputs are written.")
    parser.add_argument("--imgsz", type=int, default=1024, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--device", help="Device string passed to Ultralytics, e.g. 0, cpu, or cuda.")
    parser.add_argument("--batch-size", type=int, default=1, help="YOLO inference batch size. Use with --workers 1 for GPU.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes. Default: 1. Use 0 for all CPU cores; this loads one model per worker.",
    )
    output_behavior = parser.add_mutually_exclusive_group()
    output_behavior.add_argument("--overwrite", action="store_true", help="Re-run frames even when outputs exist.")
    output_behavior.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip frames whose expected output set already exists. This is the default behavior.",
    )
    parser.add_argument(
        "--no-debug-images",
        action="store_true",
        help="Write JSON only. YOLO review images are already opt-in via --save-review-images.",
    )
    parser.add_argument(
        "--masks-root",
        help="Optional mirrored vessel mask root. Masks are used only to populate skeleton_points for registration.",
    )
    parser.add_argument(
        "--save-review-images",
        action="store_true",
        help="Also write simple YOLO bbox overlay PNGs next to the JSON outputs.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=127,
        help="Threshold used when optional masks are converted into skeleton_points.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be 0 or greater.")
    if args.imgsz is not None and args.imgsz <= 0:
        parser.error("--imgsz must be positive.")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be in the range [0.0, 1.0].")
    if not 0.0 <= args.iou <= 1.0:
        parser.error("--iou must be in the range [0.0, 1.0].")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.save_review_images and args.no_debug_images:
        parser.error("--save-review-images cannot be combined with --no-debug-images.")

    try:
        jobs = discover_yolo_tree_jobs(args.images_root, masks_root=args.masks_root)
        summary = process_yolo_tree(
            jobs,
            args.output_root,
            images_root=args.images_root,
            masks_root=args.masks_root,
            config=YoloInferenceConfig(
                weights=args.weights,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                mask_threshold=args.mask_threshold,
                batch_size=args.batch_size,
            ),
            skip_existing=not args.overwrite,
            workers=args.workers,
            write_review_images=args.save_review_images,
        )
    except (FileNotFoundError, NotADirectoryError, FileExistsError, ValueError, OSError) as exc:
        print(f"YOLO stenosis detection failed: {exc}", file=sys.stderr)
        return 2

    print("YOLO stenosis detection completed.")
    print(f"Total frames discovered: {summary.total_jobs}")
    print(f"Workers: {summary.workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Processed: {summary.processed}")
    print(f"Skipped existing: {summary.skipped_existing}")
    print(f"Failed: {summary.failed}")
    print(f"Review images: {'yes' if summary.review_images_saved else 'no'}")
    print(f"Summary JSON: {Path(args.output_root) / 'batch_summary.json'}")
    return 0 if summary.failed == 0 else 1


def discover_yolo_tree_jobs(images_root: str | Path, *, masks_root: str | Path | None = None) -> list[YoloTreeJob]:
    resolved_images_root = Path(images_root)
    if not resolved_images_root.is_dir():
        raise NotADirectoryError(f"Images root does not exist or is not a directory: {resolved_images_root}")

    mask_index = _build_mask_index(Path(masks_root)) if masks_root is not None else {}
    jobs: list[YoloTreeJob] = []
    for image_path in sorted(_iter_candidate_images(resolved_images_root)):
        relative_path = image_path.relative_to(resolved_images_root)
        mask_path = mask_index.get((relative_path.parent.as_posix(), f"{image_path.stem}_mask"))
        jobs.append(
            YoloTreeJob(
                image_path=image_path,
                relative_dir=relative_path.parent,
                image_stem=image_path.stem,
                mask_path=mask_path,
            )
        )

    if not jobs:
        raise FileNotFoundError(
            f"No supported frame image files were found under: {resolved_images_root}. "
            "Expected names like slice_0001.png or p1_v1_00012.png."
        )
    return jobs


def process_yolo_tree(
    jobs: list[YoloTreeJob],
    output_root: str | Path,
    *,
    images_root: str | Path,
    masks_root: str | Path | None = None,
    config: YoloInferenceConfig,
    skip_existing: bool = True,
    workers: int = 1,
    write_review_images: bool = False,
    model: Any | None = None,
) -> YoloBatchSummary:
    worker_count = _resolve_worker_count(workers)
    if model is not None and worker_count != 1:
        raise ValueError("An injected YOLO model can only be used with workers=1.")

    resolved_output_root = Path(output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    pending_jobs = _filter_pending_jobs(jobs, resolved_output_root, skip_existing, write_review_images)
    skipped_existing = len(jobs) - len(pending_jobs)
    processed = 0
    failures: list[YoloBatchFailure] = []

    with tqdm(total=len(jobs), desc="YOLO frames", unit="frame", dynamic_ncols=True) as progress:
        if skipped_existing:
            progress.update(skipped_existing)
            progress.set_postfix(processed=processed, skipped=skipped_existing, failed=len(failures))

        if worker_count == 1:
            active_model = model if model is not None else load_yolo_model(config.weights)
            for batch_jobs in _batched(pending_jobs, config.batch_size):
                batch_failures = _process_yolo_tree_batch(
                    batch_jobs,
                    resolved_output_root,
                    config,
                    write_review_images=write_review_images,
                    model=active_model,
                )
                processed += len(batch_jobs) - len(batch_failures)
                for failure in batch_failures:
                    failures.append(failure)
                    progress.write(f"Failed: {failure.image_path} -> {failure.error}")
                progress.update(len(batch_jobs))
                progress.set_postfix(processed=processed, skipped=skipped_existing, failed=len(failures))
        elif pending_jobs:
            with ProcessPoolExecutor(max_workers=worker_count, initializer=_prepare_worker_process) as executor:
                future_to_job = {
                    executor.submit(
                        _process_yolo_tree_job,
                        job,
                        resolved_output_root,
                        config,
                        write_review_images,
                        None,
                    ): job
                    for job in pending_jobs
                }
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        failure = future.result()
                    except Exception as exc:  # pragma: no cover - protects parent progress for long runs.
                        failure = YoloBatchFailure(image_path=str(job.image_path), error=str(exc))
                    if failure is None:
                        processed += 1
                    else:
                        failures.append(failure)
                        progress.write(f"Failed: {job.image_path} -> {failure.error}")
                    progress.update(1)
                    progress.set_postfix(processed=processed, skipped=skipped_existing, failed=len(failures))

    summary = YoloBatchSummary(
        images_root=Path(images_root),
        masks_root=None if masks_root is None else Path(masks_root),
        output_root=resolved_output_root,
        total_jobs=len(jobs),
        workers=worker_count,
        processed=processed,
        skipped_existing=skipped_existing,
        failed=len(failures),
        failures=failures,
        review_images_saved=write_review_images,
    )
    (resolved_output_root / "batch_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2),
        encoding="utf-8",
    )
    return summary


def load_yolo_model(weights: str | Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: ultralytics. Install YOLO dependencies with "
            "`pip install ultralytics pyyaml` or your project-specific YOLO environment."
        ) from exc
    return YOLO(str(weights))


def run_yolo_frame_job(
    job: YoloTreeJob,
    *,
    model: Any,
    config: YoloInferenceConfig,
) -> dict[str, Any]:
    image_bgr = _load_image_bgr(job.image_path)
    image_height, image_width = image_bgr.shape[:2]
    raw_results = _predict(model, job.image_path, config)
    detections = _detections_from_results(
        raw_results,
        image_width=image_width,
        image_height=image_height,
        model=model,
    )
    skeleton_points = (
        []
        if job.mask_path is None
        else load_skeleton_points_from_mask(
            job.mask_path,
            image_width=image_width,
            image_height=image_height,
            mask_threshold=config.mask_threshold,
        )
    )
    return build_yolo_frame_payload(
        image_path=job.image_path,
        mask_path=job.mask_path,
        detector=config.detector_metadata(),
        view_id=_view_id_from_relative_dir(job.relative_dir),
        image_width=image_width,
        image_height=image_height,
        detections=detections,
        skeleton_points_xy=skeleton_points,
    )


def run_yolo_frame_batch_jobs(
    jobs: list[YoloTreeJob],
    *,
    model: Any,
    config: YoloInferenceConfig,
) -> tuple[list[dict[str, Any]], list[YoloBatchFailure]]:
    valid_jobs: list[YoloTreeJob] = []
    image_shapes: dict[Path, tuple[int, int]] = {}
    failures: list[YoloBatchFailure] = []

    for job in jobs:
        try:
            image_bgr = _load_image_bgr(job.image_path)
        except Exception as exc:
            failures.append(YoloBatchFailure(image_path=str(job.image_path), error=str(exc)))
            continue
        image_height, image_width = image_bgr.shape[:2]
        image_shapes[job.image_path] = (int(image_width), int(image_height))
        valid_jobs.append(job)

    if not valid_jobs:
        return [], failures

    try:
        raw_results = _predict_batch(model, valid_jobs, config)
    except Exception as exc:
        failures.extend(YoloBatchFailure(image_path=str(job.image_path), error=str(exc)) for job in valid_jobs)
        return [], failures

    payloads: list[dict[str, Any]] = []
    for index, job in enumerate(valid_jobs):
        try:
            image_width, image_height = image_shapes[job.image_path]
            result = raw_results[index] if index < len(raw_results) else None
            detections = _detections_from_result(
                result,
                image_width=image_width,
                image_height=image_height,
                model=model,
            )
            skeleton_points = (
                []
                if job.mask_path is None
                else load_skeleton_points_from_mask(
                    job.mask_path,
                    image_width=image_width,
                    image_height=image_height,
                    mask_threshold=config.mask_threshold,
                )
            )
            payloads.append(
                build_yolo_frame_payload(
                    image_path=job.image_path,
                    mask_path=job.mask_path,
                    detector=config.detector_metadata(),
                    view_id=_view_id_from_relative_dir(job.relative_dir),
                    image_width=image_width,
                    image_height=image_height,
                    detections=detections,
                    skeleton_points_xy=skeleton_points,
                )
            )
        except Exception as exc:
            failures.append(YoloBatchFailure(image_path=str(job.image_path), error=str(exc)))
    return payloads, failures


def load_skeleton_points_from_mask(
    mask_path: str | Path,
    *,
    image_width: int,
    image_height: int,
    mask_threshold: int = 127,
) -> list[list[int]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Unable to read optional vessel mask: {mask_path}")
    mask = _ensure_uint8(mask)
    if mask.ndim == 3 and mask.shape[2] == 4:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2BGR)
    if mask.ndim == 3:
        mask_gray = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
    elif mask.ndim == 2:
        mask_gray = mask
    else:
        raise ValueError(f"Unsupported mask shape for skeleton support: {mask.shape}")

    if mask_gray.shape[:2] != (int(image_height), int(image_width)):
        mask_gray = cv2.resize(mask_gray, (int(image_width), int(image_height)), interpolation=cv2.INTER_NEAREST)

    if mask_gray.max(initial=0) <= 1:
        mask_gray = np.clip(mask_gray.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    skeleton_mask = thin_binary_mask(mask_gray > int(mask_threshold))
    rows, cols = np.where(skeleton_mask)
    if rows.size == 0:
        return []
    points_xy = np.column_stack((cols + 1, rows + 1)).astype(np.int32)
    return [[int(point[0]), int(point[1])] for point in points_xy]


def save_review_image(image_path: Path, output_path: Path, payload: dict[str, Any]) -> Path:
    image = _load_image_bgr(image_path)
    for detection in payload.get("yolo_detections", []):
        if not isinstance(detection, dict):
            continue
        bbox = detection.get("bbox")
        if not isinstance(bbox, dict):
            continue
        x1 = max(0, int(bbox.get("x1", 1)) - 1)
        y1 = max(0, int(bbox.get("y1", 1)) - 1)
        x2 = max(0, int(bbox.get("x2", 1)) - 1)
        y2 = max(0, int(bbox.get("y2", 1)) - 1)
        confidence = detection.get("confidence")
        label = f"{detection.get('class_name', 'Stenosis')} {float(confidence):.2f}" if confidence is not None else "Stenosis"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Failed to write review image: {output_path}")
    return output_path


def _process_yolo_tree_job(
    job: YoloTreeJob,
    output_root: Path,
    config: YoloInferenceConfig,
    write_review_images: bool = False,
    model: Any | None = None,
) -> YoloBatchFailure | None:
    try:
        active_model = model if model is not None else load_yolo_model(config.weights)
        output_dir = output_root / job.relative_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = run_yolo_frame_job(job, model=active_model, config=config)
        output_paths = _build_output_paths(output_dir, job.image_stem)
        output_paths["results_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if write_review_images:
            save_review_image(job.image_path, output_paths["review_image"], payload)
    except Exception as exc:  # pragma: no cover - depends on external models and data.
        return YoloBatchFailure(image_path=str(job.image_path), error=str(exc))
    return None


def _process_yolo_tree_batch(
    jobs: list[YoloTreeJob],
    output_root: Path,
    config: YoloInferenceConfig,
    write_review_images: bool = False,
    model: Any | None = None,
) -> list[YoloBatchFailure]:
    try:
        active_model = model if model is not None else load_yolo_model(config.weights)
        payloads, failures = run_yolo_frame_batch_jobs(jobs, model=active_model, config=config)
        failed_paths = {failure.image_path for failure in failures}
        payload_index = 0
        for job in jobs:
            if str(job.image_path) in failed_paths:
                continue
            payload = payloads[payload_index]
            payload_index += 1
            output_dir = output_root / job.relative_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_paths = _build_output_paths(output_dir, job.image_stem)
            output_paths["results_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
            if write_review_images:
                save_review_image(job.image_path, output_paths["review_image"], payload)
        return failures
    except Exception as exc:  # pragma: no cover - protects long batch runs.
        return [YoloBatchFailure(image_path=str(job.image_path), error=str(exc)) for job in jobs]


def _predict(model: Any, image_path: Path, config: YoloInferenceConfig) -> list[Any]:
    kwargs = {
        "source": str(image_path),
        "imgsz": config.imgsz,
        "conf": config.conf,
        "iou": config.iou,
        "device": config.device,
        "batch": config.batch_size,
        "save": False,
        "verbose": False,
        "task": "detect",
    }
    compact_kwargs = {key: value for key, value in kwargs.items() if value is not None}
    results = model.predict(**compact_kwargs)
    if results is None:
        return []
    if isinstance(results, list):
        return results
    return list(results)


def _predict_batch(model: Any, jobs: list[YoloTreeJob], config: YoloInferenceConfig) -> list[Any]:
    kwargs = {
        "source": [str(job.image_path) for job in jobs],
        "imgsz": config.imgsz,
        "conf": config.conf,
        "iou": config.iou,
        "device": config.device,
        "batch": config.batch_size,
        "save": False,
        "verbose": False,
        "task": "detect",
    }
    compact_kwargs = {key: value for key, value in kwargs.items() if value is not None}
    results = model.predict(**compact_kwargs)
    if results is None:
        return []
    if isinstance(results, list):
        return results
    return list(results)


def _detections_from_results(
    results: list[Any],
    *,
    image_width: int,
    image_height: int,
    model: Any,
) -> list[YoloDetection]:
    if not results:
        return []
    result = results[0]
    return _detections_from_result(
        result,
        image_width=image_width,
        image_height=image_height,
        model=model,
    )


def _detections_from_result(
    result: Any,
    *,
    image_width: int,
    image_height: int,
    model: Any,
) -> list[YoloDetection]:
    if result is None:
        return []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy_values = _to_nested_list(getattr(boxes, "xyxy", []))
    confidence_values = _to_flat_list(getattr(boxes, "conf", []))
    class_values = _to_flat_list(getattr(boxes, "cls", []))
    names = _resolve_class_names(result, model)

    detections: list[YoloDetection] = []
    for index, raw_box in enumerate(xyxy_values):
        if len(raw_box) < 4:
            continue
        confidence = float(confidence_values[index]) if index < len(confidence_values) else 0.0
        class_id = int(float(class_values[index])) if index < len(class_values) else 0
        detections.append(
            YoloDetection(
                bbox_xyxy_zero_based=(
                    float(raw_box[0]),
                    float(raw_box[1]),
                    float(raw_box[2]),
                    float(raw_box[3]),
                ),
                confidence=confidence,
                class_id=class_id,
                class_name=_class_name(names, class_id),
                image_width=int(image_width),
                image_height=int(image_height),
            )
        )
    return detections


def _resolve_class_names(result: Any, model: Any) -> dict[int, str]:
    raw_names = getattr(result, "names", None)
    if raw_names is None:
        raw_names = getattr(model, "names", None)
    if isinstance(raw_names, dict):
        names: dict[int, str] = {}
        for key, value in raw_names.items():
            try:
                names[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return names
    if isinstance(raw_names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(raw_names)}
    return {}


def _class_name(names: dict[int, str], class_id: int) -> str:
    return names.get(int(class_id), str(class_id))


def _to_nested_list(value: Any) -> list[list[float]]:
    converted = _to_python(value)
    if converted is None:
        return []
    if isinstance(converted, np.ndarray):
        converted = converted.tolist()
    if not isinstance(converted, list):
        return []
    if converted and not isinstance(converted[0], list):
        return [converted]
    return converted


def _to_flat_list(value: Any) -> list[float]:
    converted = _to_python(value)
    if converted is None:
        return []
    if isinstance(converted, np.ndarray):
        converted = converted.tolist()
    if isinstance(converted, list):
        if converted and isinstance(converted[0], list):
            return [float(item) for row in converted for item in row]
        return [float(item) for item in converted]
    return [float(converted)]


def _to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (TypeError, ValueError):
            pass
    return value


def _iter_candidate_images(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        if _is_supported_frame_stem(path.stem):
            yield path


def _is_supported_frame_stem(stem: str) -> bool:
    return ORIGINAL_SLICE_PATTERN.match(stem) is not None or CADICA_FRAME_PATTERN.match(stem) is not None


def _build_mask_index(masks_root: Path) -> dict[tuple[str, str], Path]:
    if not masks_root.is_dir():
        raise NotADirectoryError(f"Masks root does not exist or is not a directory: {masks_root}")
    mask_index: dict[tuple[str, str], Path] = {}
    for path in masks_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        if not path.stem.lower().endswith("_mask"):
            continue
        relative_path = path.relative_to(masks_root)
        key = (relative_path.parent.as_posix(), path.stem)
        if key in mask_index:
            raise FileExistsError(
                f"Duplicate mask candidates found for {relative_path.parent / path.stem}: "
                f"{mask_index[key]} and {path}"
            )
        mask_index[key] = path
    return mask_index


def _filter_pending_jobs(
    jobs: list[YoloTreeJob],
    output_root: Path,
    skip_existing: bool,
    write_review_images: bool,
) -> list[YoloTreeJob]:
    if not skip_existing:
        return jobs
    pending_jobs: list[YoloTreeJob] = []
    for job in jobs:
        expected_outputs = _expected_outputs(output_root / job.relative_dir, job.image_stem, write_review_images)
        if not all(path.exists() and path.stat().st_size > 0 for path in expected_outputs):
            pending_jobs.append(job)
    return pending_jobs


def _expected_outputs(output_dir: Path, image_stem: str, write_review_images: bool) -> list[Path]:
    output_paths = _build_output_paths(output_dir, image_stem)
    expected = [output_paths["results_json"]]
    if write_review_images:
        expected.append(output_paths["review_image"])
    return expected


def _build_output_paths(output_dir: Path, image_stem: str) -> dict[str, Path]:
    return {
        "results_json": output_dir / f"{image_stem}_stenosis_results.json",
        "review_image": output_dir / f"{image_stem}_yolo_review.png",
    }


def _view_id_from_relative_dir(relative_dir: Path) -> str:
    clean_parts = [part for part in relative_dir.parts if part not in {"", "."}]
    if not clean_parts:
        return "view"
    return Path(*clean_parts).as_posix()


def _load_image_bgr(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return image


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image_float = image.astype(np.float32)
    if image_float.size == 0:
        return image_float.astype(np.uint8)
    if image_float.min(initial=0.0) >= 0.0 and image_float.max(initial=0.0) <= 1.0:
        image_float = image_float * 255.0
    elif image_float.max(initial=0.0) > 255.0 or image_float.min(initial=0.0) < 0.0:
        image_float = cv2.normalize(image_float, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(np.round(image_float), 0, 255).astype(np.uint8)


def _batched(items: list[YoloTreeJob], batch_size: int):
    size = max(1, int(batch_size))
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _resolve_worker_count(workers: int) -> int:
    if workers < 0:
        raise ValueError("workers must be 0 or greater.")
    if workers == 0:
        return max(1, os.cpu_count() or 1)
    return workers


def _prepare_worker_process() -> None:
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
