import os
from PIL import Image

CROP_LEFT = 300
CROP_RIGHT = 300
DIRS = ["datasets/eye/train_A", "datasets/eye/test_A"]

for d in DIRS:
    for fname in sorted(os.listdir(d)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            continue
        path = os.path.join(d, fname)
        img = Image.open(path)
        w, h = img.size
        if w <= CROP_LEFT + CROP_RIGHT:
            print(f"Skipping {fname}: width {w} too small to crop")
            continue
        cropped = img.crop((CROP_LEFT, 0, w - CROP_RIGHT, h))
        cropped.save(path)
        print(f"Cropped {path} ({w}x{h} -> {cropped.size[0]}x{cropped.size[1]})")
