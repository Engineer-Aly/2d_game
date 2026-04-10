"""
extract_sprites.py — extract labeled sprites from ChatGPT-generated sprite sheets.
Uses PIL + numpy only (no scipy).
"""

import numpy as np
from PIL import Image
import os

SPRITES_DIR = "/home/aly/Documents/proj/2d_game/sprites"
IMG1 = "/home/aly/Documents/proj/2d_game/ChatGPT Image Apr 10, 2026, 02_07_32 AM.png"
IMG2 = "/home/aly/Documents/proj/2d_game/ChatGPT Image Apr 10, 2026, 02_07_40 AM.png"


def find_bboxes(arr):
    """
    Given an RGBA or RGB numpy array, find non-white bounding boxes by:
    1. Building a binary mask of non-white pixels (R<=230 OR G<=230 OR B<=230)
    2. Projecting onto rows and columns to find gaps (all-white rows/cols)
    3. Merging contiguous non-gap spans into bounding boxes
    4. Splitting each row-span by column projection within that row band
    Returns list of (row_start, row_end, col_start, col_end) tuples.
    """
    if arr.shape[2] == 4:
        rgb = arr[:, :, :3]
    else:
        rgb = arr

    # Non-white mask
    mask = ~((rgb[:, :, 0] > 230) & (rgb[:, :, 1] > 230) & (rgb[:, :, 2] > 230))

    # Row projection: rows that have any non-white pixel
    row_has_content = mask.any(axis=1)   # shape (H,)
    col_has_content = mask.any(axis=0)   # shape (W,)

    def spans_from_bool(arr_1d, min_gap=5):
        """Find contiguous True-spans, ignoring gaps smaller than min_gap."""
        spans = []
        in_span = False
        start = 0
        n = len(arr_1d)
        for i in range(n):
            if arr_1d[i] and not in_span:
                start = i
                in_span = True
            elif not arr_1d[i] and in_span:
                # Check if this is a real gap or a tiny noise gap
                # Look ahead for next True
                gap_end = i
                while gap_end < n and not arr_1d[gap_end]:
                    gap_end += 1
                gap_size = gap_end - i
                if gap_size >= min_gap or gap_end == n:
                    spans.append((start, i - 1))
                    in_span = False
                    # If there's more content after the gap, we'll catch it in the loop
        if in_span:
            spans.append((start, n - 1))
        return spans

    row_spans = spans_from_bool(row_has_content, min_gap=8)

    bboxes = []
    for (r0, r1) in row_spans:
        # Within this row band, find column spans
        band_mask = mask[r0:r1+1, :]
        band_col = band_mask.any(axis=0)
        col_spans = spans_from_bool(band_col, min_gap=8)
        for (c0, c1) in col_spans:
            # Filter out tiny noise boxes (< 10px in either dimension)
            if (r1 - r0 + 1) >= 10 and (c1 - c0 + 1) >= 10:
                bboxes.append((r0, r1, c0, c1))

    return bboxes


def remove_white_bg(crop_arr, r_thresh=230, g_thresh=220, b_thresh=210):
    """
    Convert crop to RGBA, set alpha=0 for near-white pixels.
    """
    if crop_arr.shape[2] == 4:
        out = crop_arr.copy()
    else:
        h, w = crop_arr.shape[:2]
        out = np.zeros((h, w, 4), dtype=np.uint8)
        out[:, :, :3] = crop_arr
        out[:, :, 3] = 255

    white = (out[:, :, 0] > r_thresh) & (out[:, :, 1] > g_thresh) & (out[:, :, 2] > b_thresh)
    out[white, 3] = 0
    return out


def process_image(path, label):
    print(f"\n{'='*60}")
    print(f"Processing {label}: {os.path.basename(path)}")
    print(f"{'='*60}")

    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    print(f"  Image size: {img.size[0]}x{img.size[1]}")

    bboxes = find_bboxes(arr)
    print(f"  Found {len(bboxes)} bounding box(es):")
    for i, (r0, r1, c0, c1) in enumerate(bboxes):
        print(f"    [{i}] rows {r0}-{r1} ({r1-r0+1}px tall), cols {c0}-{c1} ({c1-c0+1}px wide)")

    return arr, bboxes


# ── Process Image 2 (the labeled sprite sheet) ────────────────────────────────

arr2, bboxes2 = process_image(IMG2, "Image 2 (sprite sheet)")

# Expected order after bbox detection (sorted top-left → bottom-right by row then col):
# [0] wall tile  (top-left, ~48x48)
# [1] floor tile (top-center, ~48x48)
# [2] bg sky     (top-right, large)
# [3] torch      (bottom-left, ~24x48)
#
# We sort bboxes by (row_start, col_start) to get stable ordering.
bboxes2_sorted = sorted(bboxes2, key=lambda b: (b[0], b[2]))

print("\n  Sorted bboxes (row_start, col_start order):")
for i, b in enumerate(bboxes2_sorted):
    r0, r1, c0, c1 = b
    print(f"    [{i}] rows {r0}-{r1}, cols {c0}-{c1}  size={c1-c0+1}x{r1-r0+1}")

# Map by size heuristic: largest area = sky background
areas = [(i, (b[1]-b[0]+1)*(b[3]-b[2]+1), b) for i, b in enumerate(bboxes2_sorted)]
areas_sorted_by_area = sorted(areas, key=lambda x: x[1], reverse=True)

print("\n  Bboxes sorted by area (largest first):")
for idx, area, b in areas_sorted_by_area:
    r0, r1, c0, c1 = b
    print(f"    original_idx={idx} area={area} size={c1-c0+1}x{r1-r0+1}")

# Identify sky (largest), then remaining sorted by (row, col)
sky_idx = areas_sorted_by_area[0][0]
non_sky = [b for i, (_, _, b) in enumerate(areas_sorted_by_area) if areas_sorted_by_area[i][0] != sky_idx]
# Sort non-sky by row then col
non_sky_sorted = sorted(non_sky, key=lambda b: (b[0], b[2]))

print(f"\n  Sky bbox index (among sorted): {sky_idx}")
print(f"  Non-sky bboxes (sorted by position):")
for i, b in enumerate(non_sky_sorted):
    r0, r1, c0, c1 = b
    print(f"    [{i}] rows {r0}-{r1}, cols {c0}-{c1}  size={c1-c0+1}x{r1-r0+1}")

def save_sprite(arr, bbox, out_path, remove_white=True, scale=None):
    r0, r1, c0, c1 = bbox
    crop = arr[r0:r1+1, c0:c1+1]
    if remove_white:
        crop = remove_white_bg(crop)
    img = Image.fromarray(crop, "RGBA")
    if scale:
        img = img.resize(scale, Image.LANCZOS)
    img.save(out_path)
    print(f"  Saved: {out_path}  (final size: {img.size[0]}x{img.size[1]})")

print("\n  Saving sprites from Image 2...")

if len(non_sky_sorted) >= 3:
    # non_sky_sorted[0] = wall tile (top-left of non-sky), [1] = floor tile (top-right of non-sky or nearby), [2] = torch
    # Actually depends on layout — print to verify, use index 0,1 as wall/floor, last as torch
    # Sort all non-sky by col within same row band for wall/floor, torch is lowest row
    # Find which are on top row vs bottom row
    top_row_items = []
    bottom_row_items = []
    if len(non_sky_sorted) > 0:
        row_threshold = non_sky_sorted[0][0] + (non_sky_sorted[-1][0] - non_sky_sorted[0][0]) // 2
        for b in non_sky_sorted:
            if b[0] <= row_threshold:
                top_row_items.append(b)
            else:
                bottom_row_items.append(b)

    print(f"  Top-row non-sky items: {len(top_row_items)}, bottom-row: {len(bottom_row_items)}")

    if len(top_row_items) >= 2:
        wall_bbox = top_row_items[0]   # leftmost on top row
        floor_bbox = top_row_items[1]  # next on top row
        r0,r1,c0,c1 = wall_bbox
        print(f"  wall_tile: size={c1-c0+1}x{r1-r0+1}")
        save_sprite(arr2, wall_bbox,  f"{SPRITES_DIR}/wall_tile_outdoor.png",  scale=(48,48))

        r0,r1,c0,c1 = floor_bbox
        print(f"  floor_tile: size={c1-c0+1}x{r1-r0+1}")
        save_sprite(arr2, floor_bbox, f"{SPRITES_DIR}/floor_tile_outdoor.png", scale=(48,48))
    elif len(top_row_items) == 1:
        print("  WARNING: only 1 top-row non-sky item found, using it as wall_tile")
        save_sprite(arr2, top_row_items[0], f"{SPRITES_DIR}/wall_tile_outdoor.png", scale=(48,48))

    if len(bottom_row_items) >= 1:
        torch_bbox = bottom_row_items[0]
        r0,r1,c0,c1 = torch_bbox
        print(f"  torch: size={c1-c0+1}x{r1-r0+1}")
        save_sprite(arr2, torch_bbox, f"{SPRITES_DIR}/torch.png", scale=(24,48))
    else:
        # Torch might be in top row if layout differs
        print("  WARNING: no bottom-row items, checking if last non-sky item is torch")
        if len(non_sky_sorted) >= 3:
            save_sprite(arr2, non_sky_sorted[-1], f"{SPRITES_DIR}/torch.png", scale=(24,48))

elif len(non_sky_sorted) == 2:
    save_sprite(arr2, non_sky_sorted[0], f"{SPRITES_DIR}/wall_tile_outdoor.png",  scale=(48,48))
    save_sprite(arr2, non_sky_sorted[1], f"{SPRITES_DIR}/floor_tile_outdoor.png", scale=(48,48))
    print("  WARNING: Only 2 non-sky bboxes found, no torch saved")
elif len(non_sky_sorted) == 1:
    save_sprite(arr2, non_sky_sorted[0], f"{SPRITES_DIR}/wall_tile_outdoor.png",  scale=(48,48))
    print("  WARNING: Only 1 non-sky bbox found")

# Save sky (no white removal, it's dark)
sky_bbox = bboxes2_sorted[sky_idx]
r0,r1,c0,c1 = sky_bbox
print(f"  bg_night sky: size={c1-c0+1}x{r1-r0+1}")
save_sprite(arr2, sky_bbox, f"{SPRITES_DIR}/bg_night.png", remove_white=False)


# ── Process Image 1 ───────────────────────────────────────────────────────────

arr1, bboxes1 = process_image(IMG1, "Image 1")
bboxes1_sorted = sorted(bboxes1, key=lambda b: (b[0], b[2]))
print("\n  Image 1 bboxes (sorted):")
for i, b in enumerate(bboxes1_sorted):
    r0, r1, c0, c1 = b
    print(f"    [{i}] rows {r0}-{r1}, cols {c0}-{c1}  size={c1-c0+1}x{r1-r0+1}")

print("\nDone.")
