#!/usr/bin/env python3
"""Generate level_3.txt, render level_3_preview.png, and update levels.json."""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

GAME_DIR = Path("/home/aly/Documents/proj/2d_game")
SPRITES_DIR = GAME_DIR / "sprites"
TILE = 48

# ---------------------------------------------------------------------------
# Level layout
# ---------------------------------------------------------------------------
# 60 cols wide, 18 rows tall (rows 0–17)
# Row 0  = ceiling (W/G tiles)
# Row 17 = floor row (F tiles with gaps)
# Rows 1–16 = open air with floating platforms

LEVEL = [
    # Row 0 — ceiling with 3 guards (62 cols)
    "WWWGWWWWWWWWWWWWWWWWWWWWWWWWWWWGWWWWWWWWWWWWWWWWWWWWWGWWWWWWWW",
    # Row 1 — open air, side walls only
    "W............................................................W",
    # Row 2 — open air, Vlad near top-right
    "W.........................................................V..W",
    # Row 3 — high platform on right for Vlad to stand on
    "W.....................................................WWWWWWWW",
    # Row 4 — open air
    "W............................................................W",
    # Row 5 — left platform cluster
    "W....WWWWWWW.................................................W",
    # Row 6 — dagger hovering above left platform / right upper platform
    "W....D.............................................WWWWWWWWW.W",
    # Row 7 — mid-high platform centre-left
    "W....................WWWWWWWWWW...............................W",
    # Row 8 — open air
    "W............................................................W",
    # Row 9 — left-centre platform
    "W........WWWWWWWW............................................W",
    # Row 10 — open air, dagger above mid platform
    "W........................D....................................W",
    # Row 11 — mid platform
    "W......................WWWWWWWWWWWW...........................W",
    # Row 12 — open air
    "W............................................................W",
    # Row 13 — right-centre platform
    "W..................................WWWWWWWW...................W",
    # Row 14 — open air, dagger above right platform
    "W.............................................D...............W",
    # Row 15 — lower right platform
    "W.............................................WWWWWWWWWWWWWWW",
    # Row 16 — open air near floor, player start bottom-left
    "W.P..........................................................W",
    # Row 17 — floor with 3 gaps/holes
    "WFFFFFFFFFFFF.........FFFFFFFF.........FFFFFFFFFFFF.........FW",
]

# Verify col counts (informational)
for i, row in enumerate(LEVEL):
    print(f"Row {i:02d}: {len(row):3d} chars  | {row}")

# ---------------------------------------------------------------------------
# Normalise all rows to the same width (pad with '.')
# ---------------------------------------------------------------------------
WIDTH = max(len(r) for r in LEVEL)
HEIGHT = len(LEVEL)
LEVEL = [row.ljust(WIDTH, '.') for row in LEVEL]

print(f"\nNormalised grid: {WIDTH} cols × {HEIGHT} rows  →  {WIDTH*TILE}×{HEIGHT*TILE} px")

# ---------------------------------------------------------------------------
# Save level_3.txt
# ---------------------------------------------------------------------------
level_txt = GAME_DIR / "level_3.txt"
header = (
    "# Outdoor castle courtyard — night\n"
    "# W = wall   . = empty air\n"
    "# F = floor tile   D = dagger\n"
    "# V = Vlad   P = player start\n"
    "# G = guard (ceiling row only)\n"
    "#\n"
)
with open(level_txt, "w") as f:
    f.write(header)
    for row in LEVEL:
        f.write(row + "\n")

print(f"\nSaved: {level_txt}")

# ---------------------------------------------------------------------------
# Render preview PNG
# ---------------------------------------------------------------------------
IMG_W = WIDTH * TILE
IMG_H = HEIGHT * TILE

# Background
bg_src = Image.open(SPRITES_DIR / "bg_night_wide.png").convert("RGBA")
bg = bg_src.resize((IMG_W, IMG_H), Image.LANCZOS)
canvas = bg.copy()

# Load tileset sprites
wall_tile = Image.open(SPRITES_DIR / "wall_tile_outdoor2.png").convert("RGBA").resize((TILE, TILE))
floor_tile = Image.open(SPRITES_DIR / "floor_tile_outdoor2.png").convert("RGBA").resize((TILE, TILE))

draw = ImageDraw.Draw(canvas)

# Try to get a font; fall back to default
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
except Exception:
    font = ImageFont.load_default()

for row_idx, row_str in enumerate(LEVEL):
    for col_idx, ch in enumerate(row_str):
        x = col_idx * TILE
        y = row_idx * TILE

        if ch in ('W', 'G'):
            canvas.paste(wall_tile, (x, y), wall_tile)
        elif ch == 'F':
            canvas.paste(floor_tile, (x, y), floor_tile)
        elif ch == 'D':
            # Gold diamond
            cx, cy = x + TILE // 2, y + TILE // 2
            half = 10
            diamond = [(cx, cy - half), (cx + half, cy), (cx, cy + half), (cx - half, cy)]
            draw.polygon(diamond, fill=(255, 200, 0), outline=(180, 140, 0))
        elif ch == 'J':
            # Purple circle
            r = 10
            cx, cy = x + TILE // 2, y + TILE // 2
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(148, 0, 211), outline=(80, 0, 130))
        elif ch == 'V':
            # Red "V"
            draw.text((x + TILE // 4, y + TILE // 8), "V", font=font, fill=(220, 30, 30))
        elif ch == 'P':
            # Green "P"
            draw.text((x + TILE // 4, y + TILE // 8), "P", font=font, fill=(0, 200, 60))

preview_path = GAME_DIR / "level_3_preview.png"
canvas.save(preview_path)
print(f"Saved preview: {preview_path}")

# ---------------------------------------------------------------------------
# Update levels.json
# ---------------------------------------------------------------------------
levels_json = GAME_DIR / "levels.json"
with open(levels_json) as f:
    data = json.load(f)

# Remove any existing level_3 entry to avoid duplicates
data["levels"] = [lvl for lvl in data["levels"] if lvl.get("file") != "level_3.txt"]
data["levels"].append({"file": "level_3.txt", "name": "Level 3 — The Courtyard"})

with open(levels_json, "w") as f:
    json.dump(data, f, indent=2)

print(f"Updated: {levels_json}")
print("\nDone!")
