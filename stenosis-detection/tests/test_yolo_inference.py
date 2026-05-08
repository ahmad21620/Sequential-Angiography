"""Tests for pipeline-compatible YOLO frame-level stenosis inference."""

from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stenosis_detection.temporal.loader import load_frame_result
from stenosis_detection.yolo.inference import (
    YoloBatchSummary,
    YoloInferenceConfig,
    discover_yolo_tree_jobs,
    process_yolo_tree,
)
from stenosis_detection.yolo.schema import SCORE_SEMANTICS, YoloDetection, build_yolo_frame_payload
import stenosis_detection.cli as cli_module


class FakeBoxes:
    xyxy = [[99.0, 199.0, 144.0, 239.0]]
    conf = [0.83]
    cls = [0]


class FakeResult:
    boxes = FakeBoxes()
    names = {0: "Stenosis"}


class FakeModel:
    names = {0: "Stenosis"}

    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult()]


class YoloInferenceTests(unittest.TestCase):
    def test_detection_converts_bbox_to_one_based_point(self):
        detection = YoloDetection(
            bbox_xyxy_zero_based=(99.0, 199.0, 144.0, 239.0),
            confidence=0.83,
            class_id=0,
            class_name="Stenosis",
            image_width=512,
            image_height=512,
        )

        point = detection.to_stenosis_point_dict()

        self.assertEqual(point["x"], 123)
        self.assertEqual(point["y"], 220)
        self.assertEqual(
            point["bbox"],
            {
                "x1": 100,
                "y1": 200,
                "x2": 145,
                "y2": 240,
                "width": 45,
                "height": 40,
            },
        )
        self.assertEqual(point["degree"], 0.83)
        self.assertEqual(point["confidence"], 0.83)
        self.assertEqual(point["severity"], "severe")
        self.assertEqual(point["score_semantics"], SCORE_SEMANTICS)

    def test_writes_mirrored_pipeline_json_without_masks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_root = root / "keyframes"
            output_root = root / "yolo_frame_results"
            image_path = images_root / "case_001" / "view_01" / "slice_0003.png"
            _write_image(image_path, width=512, height=512)
            fake_model = FakeModel()

            jobs = discover_yolo_tree_jobs(images_root)
            summary = process_yolo_tree(
                jobs,
                output_root=output_root,
                images_root=images_root,
                config=YoloInferenceConfig(
                    weights="runs/stenosis/best.pt",
                    imgsz=1024,
                    conf=0.25,
                    iou=0.7,
                    device="0",
                ),
                model=fake_model,
            )

            expected_json = (
                output_root
                / "case_001"
                / "view_01"
                / "slice_0003_stenosis_results.json"
            )
            self.assertTrue(expected_json.exists())
            self.assertEqual(summary.processed, 1)
            self.assertEqual(summary.failed, 0)
            self.assertEqual(summary.skipped_existing, 0)
            self.assertEqual(fake_model.calls[0]["imgsz"], 1024)
            self.assertEqual(fake_model.calls[0]["conf"], 0.25)
            self.assertEqual(fake_model.calls[0]["iou"], 0.7)
            self.assertEqual(fake_model.calls[0]["device"], "0")

            payload = json.loads(expected_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["image_path"], str(image_path.resolve()))
            self.assertIsNone(payload["mask_path"])
            self.assertEqual(payload["detector"]["name"], "yolov8")
            self.assertEqual(payload["detector"]["weights"], "runs/stenosis/best.pt")
            self.assertEqual(payload["detector"]["score_semantics"], SCORE_SEMANTICS)
            self.assertEqual(payload["frame"]["image_name"], "slice_0003.png")
            self.assertEqual(payload["frame"]["image_stem"], "slice_0003")
            self.assertEqual(payload["frame"]["frame_index"], 3)
            self.assertEqual(payload["frame"]["view_id"], "case_001/view_01")
            self.assertEqual(payload["frame"]["width"], 512)
            self.assertEqual(payload["frame"]["height"], 512)
            self.assertEqual(payload["counts"]["skeleton_points"], 0)
            self.assertEqual(payload["counts"]["stenosis_points"], 1)
            self.assertEqual(payload["counts"]["detections"], 1)
            self.assertEqual(payload["skeleton_points"], [])
            self.assertEqual(payload["stenosis_points"][0]["x"], 123)
            self.assertEqual(payload["stenosis_points"][0]["y"], 220)
            self.assertEqual(payload["stenosis_points"][0]["degree"], 0.83)
            self.assertEqual(payload["stenosis_points"][0]["confidence"], 0.83)
            self.assertEqual(payload["yolo_detections"][0]["bbox"]["x1"], 100)

    def test_cadica_layout_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_root = root / "cadica_prepared" / "keyframes"
            output_root = root / "yolo_frame_results"
            image_path = images_root / "p1" / "v1" / "slice_00012.png"
            _write_image(image_path, width=64, height=48)

            jobs = discover_yolo_tree_jobs(images_root)
            process_yolo_tree(
                jobs,
                output_root=output_root,
                images_root=images_root,
                config=YoloInferenceConfig(weights="fake.pt"),
                model=FakeModel(),
            )

            expected_json = output_root / "p1" / "v1" / "slice_00012_stenosis_results.json"
            payload = json.loads(expected_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["frame"]["view_id"], "p1/v1")
            self.assertEqual(payload["frame"]["frame_index"], 12)
            self.assertEqual(payload["frame"]["width"], 64)
            self.assertEqual(payload["frame"]["height"], 48)

    def test_temporal_loader_accepts_yolo_json_without_masks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_root = root / "keyframes"
            output_root = root / "results"
            image_path = images_root / "case_001" / "view_01" / "slice_0003.png"
            _write_image(image_path)

            jobs = discover_yolo_tree_jobs(images_root)
            process_yolo_tree(
                jobs,
                output_root=output_root,
                images_root=images_root,
                config=YoloInferenceConfig(weights="fake.pt"),
                model=FakeModel(),
            )

            frame_result = load_frame_result(
                output_root / "case_001" / "view_01" / "slice_0003_stenosis_results.json"
            )
            self.assertEqual(frame_result.observation_count, 1)
            self.assertEqual(frame_result.skeleton_points_xy.shape, (0, 2))
            self.assertEqual(frame_result.stenosis_points_xy.shape, (1, 2))
            self.assertIsNone(frame_result.mask_path)
            self.assertEqual(frame_result.stenosis_degrees.tolist(), [0.83])

    def test_optional_masks_populate_skeleton_points(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_root = root / "keyframes"
            masks_root = root / "masks"
            output_root = root / "results"
            image_path = images_root / "case_001" / "view_01" / "slice_0001.png"
            mask_path = masks_root / "case_001" / "view_01" / "slice_0001_mask.png"
            _write_image(image_path, width=20, height=20)
            _write_mask(mask_path)

            jobs = discover_yolo_tree_jobs(images_root, masks_root=masks_root)
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].mask_path, mask_path.resolve())
            process_yolo_tree(
                jobs,
                output_root=output_root,
                images_root=images_root,
                masks_root=masks_root,
                config=YoloInferenceConfig(weights="fake.pt"),
                model=FakeModel(),
            )

            payload = json.loads(
                (
                    output_root
                    / "case_001"
                    / "view_01"
                    / "slice_0001_stenosis_results.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["mask_path"], str(mask_path.resolve()))
            self.assertGreater(payload["counts"]["skeleton_points"], 0)
            self.assertEqual(
                payload["counts"]["skeleton_points"],
                len(payload["skeleton_points"]),
            )
            self.assertTrue(all(point[0] >= 1 and point[1] >= 1 for point in payload["skeleton_points"]))

    def test_integrated_cli_runs_yolo_without_masks_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_root = root / "keyframes"
            output_root = root / "results"
            weights_path = root / "best.pt"
            image_path = images_root / "case_001" / "view_01" / "slice_0001.png"
            _write_image(image_path, width=32, height=32)
            weights_path.write_bytes(b"mock weights")

            original_process_yolo_tree = cli_module.process_yolo_tree
            calls = []
            cli_module.process_yolo_tree = _fake_cli_yolo_process_tree(calls)
            try:
                exit_code = cli_module.main(
                    [
                        "--detector",
                        "yolo",
                        "--images-root",
                        str(images_root),
                        "--yolo-weights",
                        str(weights_path),
                        "--output-root",
                        str(output_root),
                        "--yolo-imgsz",
                        "1024",
                        "--yolo-conf",
                        "0.25",
                        "--yolo-iou",
                        "0.7",
                        "--device",
                        "0",
                    ]
                )
            finally:
                cli_module.process_yolo_tree = original_process_yolo_tree

            self.assertEqual(exit_code, 0)
            self.assertEqual(calls[0]["masks_root"], None)
            self.assertEqual(calls[0]["config"].device, "0")
            self.assertTrue(
                (
                    output_root
                    / "case_001"
                    / "view_01"
                    / "slice_0001_stenosis_results.json"
                ).is_file()
            )

    def test_integrated_cli_requires_masks_for_vessel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_root = root / "keyframes"
            images_root.mkdir()

            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as context:
                    cli_module.main(
                        [
                            "--detector",
                            "vessel",
                            "--images-root",
                            str(images_root),
                            "--output-root",
                            str(root / "results"),
                        ]
                    )
            self.assertEqual(context.exception.code, 2)


def _write_image(path: Path, width: int = 512, height: int = 512) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write test image: {path}")


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:18, 10] = 255
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"Failed to write test mask: {path}")


def _fake_cli_yolo_process_tree(calls: list[dict[str, object]]):
    def fake_process_yolo_tree(
        jobs,
        output_root,
        *,
        images_root,
        masks_root=None,
        config,
        skip_existing=True,
        workers=1,
        write_review_images=False,
        model=None,
    ):
        output_root = Path(output_root)
        calls.append(
            {
                "masks_root": masks_root,
                "config": config,
                "write_review_images": write_review_images,
            }
        )
        for job in jobs:
            detection = YoloDetection(
                bbox_xyxy_zero_based=(9.0, 9.0, 11.0, 11.0),
                confidence=float(config.conf),
                class_id=0,
                class_name="Stenosis",
                image_width=32,
                image_height=32,
            )
            payload = build_yolo_frame_payload(
                image_path=job.image_path,
                mask_path=job.mask_path,
                detector=config.detector_metadata(),
                view_id=job.relative_dir.as_posix(),
                image_width=32,
                image_height=32,
                skeleton_points_xy=[],
                detections=[detection],
            )
            output_path = output_root / job.relative_dir / f"{job.image_stem}_stenosis_results.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return YoloBatchSummary(
            images_root=Path(images_root),
            masks_root=None if masks_root is None else Path(masks_root),
            output_root=output_root,
            total_jobs=len(jobs),
            workers=1,
            processed=len(jobs),
            skipped_existing=0,
            failed=0,
            failures=[],
            review_images_saved=write_review_images,
        )

    return fake_process_yolo_tree


if __name__ == "__main__":
    unittest.main()
