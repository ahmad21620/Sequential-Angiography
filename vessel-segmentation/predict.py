from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drive_seg.config import PatchConfig
from drive_seg.dataset import (
    DatasetValidationError,
    format_validation_report,
    load_dataset_split,
    load_sample_fov_mask,
    load_sample_image,
    load_sample_mask,
    validate_dataset_root,
)
from drive_seg.engine import predict_full_image
from drive_seg.metrics import compute_segmentation_metrics
from drive_seg.model import build_model
from drive_seg.preprocessing import binarize_masks, preprocess_image
from drive_seg.utils import ensure_dir, format_metrics, load_checkpoint, resolve_device, save_json
from drive_seg.visualization import save_evaluation_prediction_panels, save_roc_curve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PyTorch ResUNet on an explicit vessel-segmentation dataset split."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing split folders such as train/, val/, and test/.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Name of the dataset split to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "best_model.pt",
        help="Path to a trained PyTorch checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for metrics, ROC curve, and visualizations. Defaults to outputs/evaluation/<split>/.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of ordered patches processed per forward pass.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used to derive binary predictions from probability maps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers used during patch inference.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of images to evaluate for quick smoke tests.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = args.output_dir if args.output_dir is not None else ROOT / "outputs" / "evaluation" / args.split
    output_dir = ensure_dir(output_root)
    device = resolve_device(args.device)

    dataset_root = args.dataset_root.resolve()
    dataset_report = validate_dataset_root(dataset_root, splits=(args.split,))
    print(format_validation_report(dataset_report))
    if not dataset_report.is_valid:
        raise DatasetValidationError("Dataset validation failed. Fix the reported issues and retry.")

    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    patch_config = PatchConfig(**checkpoint["patch_config"])
    patch_config.validate()

    split = load_dataset_split(dataset_root, args.split)
    selected_samples = split.samples[: args.limit] if args.limit is not None else split.samples
    if not selected_samples:
        raise ValueError("No images selected for evaluation. Increase --limit or choose a non-empty split.")

    preprocessed_images: list = []
    truth_masks: list = []
    fov_masks: list = []
    probability_maps: list = []
    image_ids: list[str] = []
    total_images = len(selected_samples)
    for index, sample in enumerate(selected_samples, start=1):
        image = preprocess_image(load_sample_image(sample))
        truth_mask = binarize_masks(load_sample_mask(sample))
        fov_mask = binarize_masks(load_sample_fov_mask(sample, spatial_shape=truth_mask.shape[-2:]))
        preprocessed_images.append(image)
        truth_masks.append(truth_mask)
        fov_masks.append(fov_mask)
        image_ids.append(sample.image_id)

        print(f"Predicting {sample.image_id} ({index}/{total_images})")
        probability_map = predict_full_image(
            model=model,
            image=image,
            patch_size=patch_config.patch_shape,
            stride=patch_config.stride,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=device.type == "cuda",
        )
        probability_maps.append(probability_map)

    metrics, roc_data = compute_segmentation_metrics(
        probability_maps=probability_maps,
        truth_masks=truth_masks,
        fov_masks=fov_masks,
        threshold=args.threshold,
    )

    save_json(output_dir / "metrics.json", metrics)
    save_roc_curve(
        fpr=roc_data["fpr"],
        tpr=roc_data["tpr"],
        auc_value=float(metrics["auc"]),
        output_path=output_dir / "roc_curve.png",
    )
    save_evaluation_prediction_panels(
        images=preprocessed_images,
        truth_masks=truth_masks,
        probability_maps=probability_maps,
        image_ids=image_ids,
        output_dir=output_dir / "visualizations",
    )

    print("Evaluation metrics:")
    for line in format_metrics(metrics):
        print(f"  {line}")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
