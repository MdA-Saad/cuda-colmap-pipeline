import pycolmap
from pathlib import Path
import argparse


# --- CLI args ---
parser = argparse.ArgumentParser(description="COLMAP reconstruction pipeline")
parser.add_argument(
    "--device",
    choices=["cpu", "cuda"],
    default="cuda",
    help="Device for feature extraction and matching (default: cuda)",
)
args = parser.parse_args()

device = pycolmap.Device.cuda if args.device == "cuda" else pycolmap.Device.cpu

# --- Paths ---
root_dir = Path("data/south-building")
image_dir = root_dir / "images"
output_path = Path("data/workspace")

output_path.mkdir(exist_ok=True)
database_path = output_path / "database.db"
mvs_path = output_path / "mvs"

# Extract Features
print("Extracting features ({args.device.upper()} Mode)...")
extraction_options = pycolmap.FeatureExtractionOptions()
extraction_options.max_image_size = 1024

fusion_options = pycolmap.StereoFusionOptions()
fusion_options.max_image_size = 1024   # memory limit

pycolmap.extract_features(
    database_path,
    image_dir,
    extraction_options=extraction_options,
    device=device,
)

# Match Features
print("Matching features ({args.device.upper()} Mode)...")
pycolmap.match_exhaustive(
    database_path,
    device=device
)

# sparse reconstruction
print("Running incremental mapping (SfM)...")
maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)

if maps:
    best_reconstruction = max(maps.items(), key=lambda kv: len(kv[1].images))[1]
    best_reconstruction.write(output_path)

    print(f"Sparse reconstruction saved. Registered {len(best_reconstruction.images)} images.")
    print(f"Number of 3D points: {len(best_reconstruction.points3D)}")

    # Dense Reconstruction
    print("\nRunning dense reconstruction (MVS)...")
    pycolmap.undistort_images(mvs_path, output_path, image_dir)

    print("Running patch match stereo matching...")
    pycolmap.patch_match_stereo(mvs_path)

    print("Fusing depth maps into point cloud...")
    fusion_options = pycolmap.StereoFusionOptions()
    fusion_options.max_image_size = 1024
    pycolmap.stereo_fusion(
            mvs_path / "fused.ply",
            mvs_path,
            options=fusion_options,
            output_type="ply")
    print("Dense reconstruction complete! Point cloud saved to dense.ply")
else:
    print("No reconstruction generated. Check image overlap and quality.")
