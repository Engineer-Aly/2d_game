from PIL import Image
import numpy as np

img = Image.open("/home/aly/Documents/proj/2d_game/Gemini_Generated_Image_pzdw1cpzdw1cpzdw.png").convert("RGBA")
data = np.array(img)

R = data[:, :, 0].astype(int)
G = data[:, :, 1].astype(int)
B = data[:, :, 2].astype(int)

# Checkerboard: gray pixels where R≈G≈B and R is ~125 or ~184
# Foreground: color deviates from gray
is_fg = (
    (np.abs(R - G) > 15) |
    (np.abs(R - B) > 15) |
    (np.abs(G - B) > 15)
)

# Also treat very dark pixels (outline/shadow) as foreground if they're not the neutral gray
# Dark gray checkerboard squares are ~125, light are ~184
# Very dark pixels (R<80) that aren't checkerboard
is_dark_fg = (R < 80) & (np.abs(R - G) <= 15) & (np.abs(R - B) <= 15)
# These dark near-gray could be outlines — include them only if surrounded by color
# For simplicity: include them as fg since checkerboard min is ~125
is_fg = is_fg | is_dark_fg

# Column profile: count foreground pixels per column
col_profile = is_fg.sum(axis=0)

print("Column profile (non-zero range):", np.where(col_profile > 0)[0][[0, -1]])

# Find gaps: columns with very few foreground pixels between characters
# Smooth the profile to find clear valleys
threshold = 3  # fewer than this = gap candidate
gap_cols = np.where(col_profile <= threshold)[0]

# Find contiguous gap regions
gaps = []
if len(gap_cols) > 0:
    start = gap_cols[0]
    prev = gap_cols[0]
    for c in gap_cols[1:]:
        if c - prev > 1:
            gaps.append((start, prev))
            start = c
        prev = c
    gaps.append((start, prev))

print("Gap regions (col ranges):", gaps)

# We expect 2 significant gaps separating 3 characters
# Filter gaps that are wide enough (> 5px wide) and not at image edges
h, w = data.shape[:2]
sig_gaps = [(s, e) for s, e in gaps if (e - s) >= 3 and s > 5 and e < w - 5]
print("Significant gaps:", sig_gaps)

# Pick the 2 largest gaps
sig_gaps.sort(key=lambda x: x[1] - x[0], reverse=True)
split_gaps = sorted(sig_gaps[:2], key=lambda x: x[0])
print("Split gaps used:", split_gaps)

# Define character column ranges
fg_cols = np.where(col_profile > 0)[0]
x_start = int(fg_cols[0])
x_end = int(fg_cols[-1])

boundaries = [x_start, split_gaps[0][1], split_gaps[1][1], x_end]
char_ranges = [
    (boundaries[0], boundaries[1]),
    (split_gaps[0][1] + 1, split_gaps[1][0]),
    (split_gaps[1][1] + 1, boundaries[3]),
]
print("Character column ranges:", char_ranges)

names = ["animal_1", "animal_2", "animal_3"]
labels = ["tiger", "lion", "wolf"]
PAD = 4

for i, (cx0, cx1) in enumerate(char_ranges):
    # Find tight row bounds within this column range
    region_fg = is_fg[:, cx0:cx1+1]
    row_profile = region_fg.sum(axis=1)
    fg_rows = np.where(row_profile > 0)[0]
    if len(fg_rows) == 0:
        print(f"{labels[i]}: no foreground pixels found!")
        continue
    ry0 = max(0, int(fg_rows[0]) - PAD)
    ry1 = min(h - 1, int(fg_rows[-1]) + PAD)
    rx0 = max(0, cx0 - PAD)
    rx1 = min(w - 1, cx1 + PAD)

    crop = data[ry0:ry1+1, rx0:rx1+1].copy()

    # Remove checkerboard background: set alpha=0 where gray
    cR = crop[:, :, 0].astype(int)
    cG = crop[:, :, 1].astype(int)
    cB = crop[:, :, 2].astype(int)

    is_bg = (
        (np.abs(cR - cG) <= 15) &
        (np.abs(cR - cB) <= 15) &
        (np.abs(cG - cB) <= 15) &
        (cR >= 80)  # not super dark (keep dark outlines)
    )

    crop[:, :, 3] = np.where(is_bg, 0, 255)

    out = Image.fromarray(crop, "RGBA")
    out_path = f"/home/aly/Documents/proj/2d_game/sprites/{names[i]}.png"
    out.save(out_path)
    print(f"{labels[i]} ({names[i]}.png): {out.width}x{out.height}px  [cols {rx0}-{rx1}, rows {ry0}-{ry1}]")

print("Done.")
