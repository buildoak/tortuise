#!/usr/bin/env python3
"""
export_sharp_onnx.py — Export Apple SHARP model to ONNX format.

Developer tool. Not shipped with the binary.

Prerequisites:
    pip install torch torchvision onnx onnxruntime

Usage:
    python3 export_sharp_onnx.py [--checkpoint PATH] [--output sharp.onnx]

If no checkpoint is provided, downloads the official weights from Apple CDN.

Exit codes:
    0 — success
    1 — error
    2 — usage error
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

APPLE_CHECKPOINT_URL = (
    "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
DEFAULT_CHECKPOINT_NAME = "sharp_2572gikvuh.pt"
INPUT_SIZE = 1536


def download_checkpoint(dest: Path) -> None:
    """Download SHARP checkpoint from Apple CDN."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Checkpoint already cached: {dest}")
        return

    print(f"Downloading SHARP checkpoint to {dest}...")
    urllib.request.urlretrieve(APPLE_CHECKPOINT_URL, str(dest))
    print(f"Download complete ({dest.stat().st_size / 1e6:.1f} MB)")


def export_to_onnx(checkpoint_path: Path, output_path: Path) -> None:
    """Load SHARP model and export to ONNX format."""
    try:
        import torch
    except ImportError:
        print("Error: PyTorch is required. Install with: pip install torch", file=sys.stderr)
        sys.exit(1)

    try:
        import onnx
    except ImportError:
        print("Error: onnx is required. Install with: pip install onnx", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")

    # The SHARP repo must be cloned for the model classes to be available.
    # Try importing from ml-sharp source tree.
    sharp_repo = os.environ.get("SHARP_REPO")
    if sharp_repo:
        sys.path.insert(0, sharp_repo)

    try:
        from src.sharp.model import SharpModel
    except ImportError:
        print(
            "Error: Cannot import SHARP model classes.\n"
            "Clone https://github.com/apple/ml-sharp and either:\n"
            "  1. Run this script from the ml-sharp directory, or\n"
            "  2. Set SHARP_REPO=/path/to/ml-sharp",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model = SharpModel()

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device).eval()

    # Create dummy inputs matching expected shapes
    dummy_image = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    dummy_disparity = torch.tensor([1.0], device=device)

    print(f"Tracing model with input shape {list(dummy_image.shape)}...")

    # Export to ONNX
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_image, dummy_disparity),
        str(output_path),
        input_names=["image", "disparity_factor"],
        output_names=[
            "mean_vectors_3d_positions",
            "singular_values_scales",
            "quaternions_rotations",
            "colors_rgb_linear",
            "opacities_alpha_channel",
        ],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "disparity_factor": {0: "batch"},
            "mean_vectors_3d_positions": {0: "batch", 1: "num_gaussians"},
            "singular_values_scales": {0: "batch", 1: "num_gaussians"},
            "quaternions_rotations": {0: "batch", 1: "num_gaussians"},
            "colors_rgb_linear": {0: "batch", 1: "num_gaussians"},
            "opacities_alpha_channel": {0: "batch", 1: "num_gaussians"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify the exported model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    size_mb = output_path.stat().st_size / 1e6
    print(f"Exported ONNX model: {output_path} ({size_mb:.1f} MB)")
    print("Model verification passed.")

    # Print model info
    print("\nModel inputs:")
    for inp in onnx_model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")

    print("\nModel outputs:")
    for out in onnx_model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Apple SHARP model to ONNX format",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=Path,
        default=None,
        help="Path to SHARP checkpoint (.pt). Downloads from Apple CDN if not provided.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("sharp.onnx"),
        help="Output ONNX file path (default: sharp.onnx)",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CACHE_DIR / DEFAULT_CHECKPOINT_NAME
        download_checkpoint(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    export_to_onnx(checkpoint_path, args.output)
    print("\nDone. Place the ONNX file at ~/.tortuise/models/sharp.onnx for tortuise to use.")


if __name__ == "__main__":
    main()
