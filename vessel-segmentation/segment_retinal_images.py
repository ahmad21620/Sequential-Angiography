from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drive_seg.config import PatchConfig
from drive_seg.dataset import load_rgb_image, resolve_custom_image_paths
from drive_seg.engine import predict_full_image
from drive_seg.model import build_model
from drive_seg.preprocessing import preprocess_images
from drive_seg.utils import ensure_dir, load_checkpoint, resolve_device
from drive_seg.visualization import (
    custom_prediction_outputs_exist,
    get_stenosis_mask_output_path,
    save_custom_prediction,
    save_stenosis_mask,
    stenosis_mask_output_exists,
)

IGNORED_INPUT_FILENAMES = {"extract_complete.png", ".extract_complete.png"}
MIRRORED_METADATA_FILENAMES = {"views.json", "patient.json"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run retinal vessel segmentation on custom retinal images."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a single retinal image or a directory of images.",
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
        default=ROOT / "outputs" / "custom",
        help="Directory for probability maps, binary masks, and preview panels.",
    )
    parser.add_argument(
        "--stenosis-masks-root",
        type=Path,
        default=None,
        help=(
            "Optional root directory for stenosis-compatible binary masks. "
            "When set, saves only mirrored <image_stem>_mask.png files."
        ),
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
        help="Threshold used to derive binary masks from probability maps.",
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
    return parser


def validate_single_image_stenosis_output_path(
    input_path: Path,
    masks_root: Path,
    image_id: Path,
) -> None:
    if input_path.is_dir():
        return
    output_path = get_stenosis_mask_output_path(
        masks_root=masks_root,
        image_id=image_id,
    )
    expected_name = f"{input_path.stem}_mask.png"
    if output_path.name != expected_name:
        raise ValueError(
            "Stenosis-compatible single-image output must be named "
            f"{expected_name}, got {output_path.name}."
        )


def is_mirrored_metadata_json_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.lower() in MIRRORED_METADATA_FILENAMES:
        return True
    return path.suffix.lower() == ".json"


def copy_mirrored_metadata_json_files(input_root: Path, output_root: Path) -> int:
    if not input_root.is_dir():
        return 0

    resolved_output_root = output_root.resolve()
    copied_files = 0
    for json_path in sorted(
        path
        for path in input_root.rglob("*")
        if is_mirrored_metadata_json_file(path)
    ):
        resolved_json_path = json_path.resolve()
        if (
            resolved_output_root == resolved_json_path
            or resolved_output_root in resolved_json_path.parents
        ):
            continue

        destination_path = output_root / json_path.relative_to(input_root)
        if destination_path.resolve() == resolved_json_path:
            continue

        ensure_dir(destination_path.parent)
        shutil.copy2(json_path, destination_path)
        copied_files += 1

    return copied_files


def copy_mirrored_json_files(input_root: Path, output_root: Path) -> int:
    return copy_mirrored_metadata_json_files(input_root, output_root)


def main() -> None:
    args = build_parser().parse_args()
    stenosis_masks_root = (
        ensure_dir(args.stenosis_masks_root)
        if args.stenosis_masks_root is not None
        else None
    )
    output_dir = ensure_dir(args.output_dir) if stenosis_masks_root is None else None
    output_root = (
        stenosis_masks_root if stenosis_masks_root is not None else output_dir
    )
    device = resolve_device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    patch_config = PatchConfig(**checkpoint["patch_config"])
    patch_config.validate()

    image_names, image_paths = resolve_custom_image_paths(
        args.input,
        ignore_names=IGNORED_INPUT_FILENAMES,
    )
    if stenosis_masks_root is not None and len(image_names) == 1:
        validate_single_image_stenosis_output_path(
            input_path=args.input,
            masks_root=stenosis_masks_root,
            image_id=image_names[0],
        )
    copied_json_files = copy_mirrored_metadata_json_files(
        input_root=args.input,
        output_root=output_root,
    )
    total_images = len(image_names)
    processed_images = 0
    skipped_images = 0

    progress = tqdm(
        zip(image_names, image_paths),
        total=total_images,
        desc="Segmenting images",
        unit="image",
    )
    for image_name, image_path in progress:
        progress.set_postfix_str(image_name.as_posix())
        if (
            output_dir is not None
            and custom_prediction_outputs_exist(output_dir=output_dir, image_id=image_name)
        ):
            skipped_images += 1
            continue
        if (
            stenosis_masks_root is not None
            and stenosis_mask_output_exists(masks_root=stenosis_masks_root, image_id=image_name)
        ):
            skipped_images += 1
            continue
        rgb_image = load_rgb_image(image_path)
        preprocessed = preprocess_images(np.expand_dims(rgb_image, axis=0))[0]
        probability_map = predict_full_image(
            model=model,
            image=preprocessed,
            patch_size=patch_config.patch_shape,
            stride=patch_config.stride,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=device.type == "cuda",
        )
        if stenosis_masks_root is None:
            save_custom_prediction(
                original_rgb=rgb_image,
                probability_map=probability_map,
                output_dir=output_dir,
                image_id=image_name,
                threshold=args.threshold,
            )
        else:
            save_stenosis_mask(
                probability_map=probability_map,
                masks_root=stenosis_masks_root,
                image_id=image_name,
                threshold=args.threshold,
            )
        processed_images += 1

    if stenosis_masks_root is None:
        print(
            f"Saved custom segmentation outputs to {output_dir} "
            "(processed="
            f"{processed_images}, skipped={skipped_images}, copied_json={copied_json_files})"
        )
    else:
        print(
            f"Saved stenosis-compatible masks to {stenosis_masks_root} "
            "(processed="
            f"{processed_images}, skipped={skipped_images}, copied_json={copied_json_files})"
        )


if __name__ == "__main__":
    main()
