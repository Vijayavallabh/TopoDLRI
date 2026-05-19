import os
import numpy as np
from PIL import Image

DIRS = ["datasets/eye/train_B", "datasets/eye/test_B"]

for d in DIRS:
    for fname in sorted(os.listdir(d)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            continue
        path = os.path.join(d, fname)
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        h, w = arr.shape[:2]

        mask = np.zeros((h, w), dtype=bool)

        mask[h-300:, :600] = np.all(arr[h-300:, :600] > 20, axis=2)
        mask[h-200:, -600:] = np.all(arr[h-200:, -600:] > 20, axis=2)
        
        arr[mask] = [0, 0, 0]

        Image.fromarray(arr).save(path)
        count = np.sum(mask)
        print(f"Cleaned {path}: {count} watermark pixels removed")
