"""
Process the 3 Gemini walking frames + original walking.png:
- Remove white background
- Crop to character bounds
- Resize to Player dimensions (38 x 64)
- Save as sprites/assassin/walk_0.png, walk_1.png, walk_2.png, walk_3.png
"""
import os
import shutil
import numpy as np
from PIL import Image

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "sprites", "assassin")
OUT_DIR = SRC_DIR

W, H = 38, 64  # Player.W, Player.H

GEMINI_SOURCES = [
    "Gemini_Generated_Image_3j8s7q3j8s7q3j8s.png",
    "Gemini_Generated_Image_3j8s7q3j8s7q3j8s (1).png",
    "Gemini_Generated_Image_3j8s7q3j8s7q3j8s (2).png",
]

def process_and_save(path, out_path):
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)

    # Remove white background
    r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
    white_mask = (r > 230) & (g > 220) & (b > 200)
    arr[white_mask, 3] = 0

    # Crop to non-transparent bounding box
    visible = arr[:,:,3] > 10
    rows = np.any(visible, axis=1)
    cols = np.any(visible, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    padding = 4
    rmin = max(0, rmin - padding)
    rmax = min(arr.shape[0] - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(arr.shape[1] - 1, cmax + padding)
    cropped = Image.fromarray(arr[rmin:rmax+1, cmin:cmax+1])

    resized = cropped.resize((W, H), Image.LANCZOS)
    resized.save(out_path)
    print(f"Saved {out_path}")

# frames that need to be flipped to face left (0-indexed)
FLIP = {0, 1}  # Gemini frames 0,1 face left — flip to right (game default convention)

# Process 3 Gemini frames
for i, fname in enumerate(GEMINI_SOURCES):
    process_and_save(
        os.path.join(SRC_DIR, fname),
        os.path.join(OUT_DIR, f"walk_{i}.png")
    )

# Frame 3 = original walking.png (already clean, just resize to match)
process_and_save(
    os.path.join(SRC_DIR, "walking.png"),
    os.path.join(OUT_DIR, "walk_3.png")
)

# Flip any frames that came out facing right
for i in FLIP:
    path = os.path.join(OUT_DIR, f"walk_{i}.png")
    img = Image.open(path)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.save(path)
    print(f"Flipped walk_{i}.png to face left")

print("Done — walk_0.png, walk_1.png, walk_2.png, walk_3.png ready.")
