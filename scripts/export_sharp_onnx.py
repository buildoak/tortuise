#!/usr/bin/env python3
"""
export_sharp_onnx.py — Export Apple SHARP model to ONNX format.

Developer tool. Not shipped with the binary.

Prerequisites:
    pip install torch onnx

Usage:
    # From ml-sharp repo root (or set SHARP_REPO):
    python3 export_sharp_onnx.py [--checkpoint PATH] [--output sharp.onnx]

    # With explicit repo path:
    SHARP_REPO=/path/to/ml-sharp python3 export_sharp_onnx.py

If no checkpoint is provided, downloads the official weights from Apple CDN.

Exit codes:
    0 — success
    1 — error
    2 — usage error
"""

import argparse
import os
import sys
from pathlib import Path

APPLE_CHECKPOINT_URL = (
    "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
DEFAULT_CHECKPOINT_NAME = "sharp_2572gikvuh.pt"
INPUT_SIZE = 1536


def setup_sharp_imports() -> None:
    """Add ml-sharp source tree to sys.path so we can import sharp modules."""
    sharp_repo = os.environ.get("SHARP_REPO")
    if sharp_repo:
        # Expand ~ to home directory (env vars set via shell may contain literal ~)
        sharp_repo = os.path.expanduser(sharp_repo)
        # ml-sharp uses src layout: ml-sharp/src/sharp/...
        src_dir = os.path.join(sharp_repo, "src")
        if os.path.isdir(src_dir):
            sys.path.insert(0, src_dir)
        else:
            sys.path.insert(0, sharp_repo)
    else:
        # Assume running from ml-sharp repo root
        src_dir = os.path.join(os.getcwd(), "src")
        if os.path.isdir(src_dir):
            sys.path.insert(0, src_dir)


def download_checkpoint(dest: Path) -> None:
    """Download SHARP checkpoint from Apple CDN."""
    import urllib.request

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

    setup_sharp_imports()

    try:
        from sharp.models import PredictorParams, create_predictor
    except ImportError:
        print(
            "Error: Cannot import SHARP model classes.\n"
            "Clone https://github.com/apple/ml-sharp and either:\n"
            "  1. Run this script from the ml-sharp/src directory, or\n"
            "  2. Set SHARP_REPO=/path/to/ml-sharp\n"
            "\nAlso ensure ml-sharp dependencies are installed:\n"
            "  pip install -r /path/to/ml-sharp/requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device("cpu")
    print(f"Loading checkpoint: {checkpoint_path}")

    state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=True)

    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()

    # SHARP's forward() returns a Gaussians3D NamedTuple, which torch.onnx.export
    # can't handle directly. Wrap it to return a flat tuple of tensors.
    class SharpOnnxWrapper(torch.nn.Module):
        def __init__(self, predictor):
            super().__init__()
            self.predictor = predictor

        def forward(self, image, disparity_factor):
            g = self.predictor(image, disparity_factor)
            # Gaussians3D fields: mean_vectors, singular_values, quaternions, colors, opacities
            return g.mean_vectors, g.singular_values, g.quaternions, g.colors, g.opacities

    wrapper = SharpOnnxWrapper(gaussian_predictor).to(device)

    # Create dummy inputs matching expected shapes
    dummy_image = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    dummy_disparity = torch.tensor([1.0], device=device)

    print(f"Tracing model with input shape {list(dummy_image.shape)}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Output names must match what tortuise's Rust runtime expects
    output_names = [
        "mean_vectors_3d_positions",
        "singular_values_scales",
        "quaternions_rotations",
        "colors_rgb_linear",
        "opacities_alpha_channel",
    ]

    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_disparity),
        str(output_path),
        input_names=["image", "disparity_factor"],
        output_names=output_names,
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "disparity_factor": {0: "batch"},
            output_names[0]: {0: "batch", 1: "num_gaussians"},
            output_names[1]: {0: "batch", 1: "num_gaussians"},
            output_names[2]: {0: "batch", 1: "num_gaussians"},
            output_names[3]: {0: "batch", 1: "num_gaussians"},
            output_names[4]: {0: "batch", 1: "num_gaussians"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # Use legacy TorchScript exporter; dynamo chokes on SHARP's dynamic shapes
    )

    # The legacy exporter produces many small external data files for large models.
    # Consolidate into model + single .data file for clean deployment.
    onnx_model = onnx.load(str(output_path), load_external_data=True)

    # Clean up the scattered external data files from the initial export
    for f in output_path.parent.iterdir():
        if f.is_file() and f != output_path and not f.suffix:
            if f.name.startswith("onnx__"):
                f.unlink()

    # Save with a single external data file (model exceeds protobuf 2GB limit)
    data_filename = output_path.name + ".data"
    onnx.save_model(
        onnx_model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_filename,
    )

    # Verify by checking the model file on disk (supports external data references)
    onnx.checker.check_model(str(output_path))

    data_path = output_path.parent / data_filename
    model_mb = output_path.stat().st_size / 1e6
    data_mb = data_path.stat().st_size / 1e6
    print(f"Exported ONNX model: {output_path} ({model_mb:.1f} MB)")
    print(f"External weights:    {data_path} ({data_mb:.1f} MB)")
    print("Model verification passed.")

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
