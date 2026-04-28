import sys
sys.path.insert(0, "/opt/depth-anything")

import os
from pathlib import Path
import cv2
import torch
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2

INPUT_DIR = Path(os.environ.get("INPUT_IMAGES", "/data/input/images"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DEPTH", "/data/input/mono_depth"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

encoder = os.environ.get("DEPTH_ENCODER", "vitb")
model = DepthAnythingV2(**model_configs[encoder])

ckpt = f"/opt/depth-anything/checkpoints/depth_anything_v2_{encoder}.pth"
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model = model.to(DEVICE).eval()

print(f"Input images: {INPUT_DIR}")
print(f"Output depth: {OUTPUT_DIR}")
print(f"Encoder: {encoder}")
print(f"Device: {DEVICE}")

for img_path in sorted(INPUT_DIR.glob("*")):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    out_path = OUTPUT_DIR / f"{img_path.stem}.npy"
    if out_path.exists():
        continue

    print(f"Depth: {img_path.name}")

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load: {img_path}")
        continue

    depth = model.infer_image(img).astype(np.float32)
    np.save(out_path, depth)

print("Depth generation done")