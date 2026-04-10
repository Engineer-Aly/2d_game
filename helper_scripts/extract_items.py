from PIL import Image
import numpy as np

img = Image.open("/home/aly/Documents/proj/2d_game/ChatGPT Image Apr 10, 2026, 05_13_53 PM.png").convert("RGBA")
data = np.array(img)

R = data[:,:,0].astype(float)
G = data[:,:,1].astype(float)
B = data[:,:,2].astype(float)

# Foreground detection: pixels that deviate from neutral gray (colorful items)
avg = (R + G + B) / 3.0
color_dev = np.sqrt((R - avg)**2 + (G - avg)**2 + (B - avg)**2)

# Items are colorful; background gradient is neutral gray (low color_dev)
# Threshold 25 gives clean zero-gaps between items
fg_colorful = color_dev > 25

col_profile = fg_colorful.sum(axis=0)

# Find gaps: contiguous runs where column foreground count < 5
width = col_profile.shape[0]
GAP_THRESH = 5

in_gap = False
gaps = []
gap_start = None

for x in range(width):
    if col_profile[x] < GAP_THRESH:
        if not in_gap:
            in_gap = True
            gap_start = x
    else:
        if in_gap:
            in_gap = False
            gaps.append((gap_start, x - 1))

if in_gap:
    gaps.append((gap_start, width - 1))

print(f"Found {len(gaps)} gaps (fg < {GAP_THRESH}): {gaps}")

# Filter gaps: must be at least 15px wide (to exclude single-pixel noise)
meaningful_gaps = [(s, e) for s, e in gaps if (e - s) >= 15]
print(f"Meaningful gaps (>=20px wide): {meaningful_gaps}")

# We expect 2 interior gaps (plus possibly leading/trailing gaps at edges)
# Remove leading gap (starts at col 0) and trailing gap (ends at last col)
interior_gaps = [(s, e) for s, e in meaningful_gaps if s > 50 and e < (width - 50)]
print(f"Interior gaps: {interior_gaps}")

# Take 2 largest interior gaps (separating the 3 items)
two_gaps = sorted(interior_gaps, key=lambda g: g[1] - g[0], reverse=True)[:2]
two_gaps = sorted(two_gaps, key=lambda g: g[0])  # sort by position
print(f"Using gaps: {two_gaps}")

assert len(two_gaps) == 2, f"Expected 2 gaps, found {len(two_gaps)}"

# Build 3 column sections
section_bounds = [
    (0, two_gaps[0][0] - 1),
    (two_gaps[0][1] + 1, two_gaps[1][0] - 1),
    (two_gaps[1][1] + 1, width - 1),
]
print(f"Section column bounds: {section_bounds}")

PAD = 6
names = ["item_1", "item_2", "item_3"]
labels = ["potion", "spider", "incense burner"]

for (col_start, col_end), name, label in zip(section_bounds, names, labels):
    # Tight bounding box using colorful foreground within this section
    section_fg = fg_colorful[:, col_start:col_end + 1]
    rows_with_fg = np.where(section_fg.any(axis=1))[0]
    cols_with_fg = np.where(section_fg.any(axis=0))[0]

    assert len(rows_with_fg) > 0 and len(cols_with_fg) > 0, f"No foreground found in {name}"

    row_min = rows_with_fg[0]
    row_max = rows_with_fg[-1]
    col_min = col_start + cols_with_fg[0]
    col_max = col_start + cols_with_fg[-1]

    # Apply padding
    h, w = data.shape[:2]
    r0 = max(0, row_min - PAD)
    r1 = min(h - 1, row_max + PAD)
    c0 = max(0, col_min - PAD)
    c1 = min(w - 1, col_max + PAD)

    crop = data[r0:r1+1, c0:c1+1].copy()

    # Remove background: pixels where R>230 AND G>230 AND B>230 (white background)
    cr = crop[:,:,0].astype(int)
    cg = crop[:,:,1].astype(int)
    cb = crop[:,:,2].astype(int)
    white_mask = (cr > 230) & (cg > 230) & (cb > 230)
    crop[white_mask, 3] = 0

    out_img = Image.fromarray(crop, "RGBA")
    out_path = f"/home/aly/Documents/proj/2d_game/sprites/{name}.png"
    out_img.save(out_path)
    print(f"{name} ({label}): {out_img.width}x{out_img.height}px -> {out_path}")

print("Done.")
