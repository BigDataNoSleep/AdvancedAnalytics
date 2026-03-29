"""
Lego Minifig Dataset — Preprocessing
=====================================
Steps:
  1. Load JSON
  2. Drop irrelevant columns
  3. Match rows to available images
  4. Reorder rows to follow the image file list (alphabetical by filename)
  5. Save Data/processed/minifigs_clean.csv

Run from the project root (where the Data/ folder lives).
"""

import json
import os
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
JSON_PATH  = "Data/minifigs.json"
IMAGES_DIR = r"C:\Users\thoma\OneDrive\Documents\KUL\KUL 2025-2026\Advanced Analytics in Business\Assignment 2\images"
OUTPUT_DIR = "Data/processed"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
with open(JSON_PATH, encoding="utf-8") as f:
    df = pd.DataFrame(json.load(f))
print(f"Loaded {len(df)} records | columns: {list(df.columns)}")

# ── 2. DROP IRRELEVANT COLUMNS ────────────────────────────────────────────────
DROP_COLS = [
    "id",                # sequential row ID
    "name",              # verbose text description
    "link",              # BrickLink URL
    "img_url",           # remote image URL
    "year_released",     # duplicate of 'year' (string vs int)
    "set_id",            # e.g. "3 sets" — not useful
    "current_value_new",
    "current_value_used",
    "character_name",    # >90 % null
    "img_local_path",    # stale relative path; replaced by image_path below
]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
print(f"Columns after drop: {list(df.columns)}")

# ── 3. MATCH ROWS TO AVAILABLE IMAGES ────────────────────────────────────────
# Image filenames match minifig_number (e.g. OLD040 → OLD040.jpg).
# Build a case-insensitive lookup: uppercase stem → actual filename.
image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".jpg")]
img_lookup  = {os.path.splitext(f)[0].upper(): f for f in image_files}

df["image_path"] = df["minifig_number"].apply(
    lambda num: os.path.join(IMAGES_DIR, img_lookup[num.upper()])
    if num.upper() in img_lookup else None
)

n_found   = df["image_path"].notna().sum()
n_missing = df["image_path"].isna().sum()
print(f"\nImages matched : {n_found}")
print(f"No image found : {n_missing}")

df = df[df["image_path"].notna()].reset_index(drop=True)
print(f"Rows kept      : {len(df)}")

# ── 4. REORDER ROWS TO FOLLOW THE IMAGE FILE LIST ────────────────────────────
# Sort image filenames alphabetically (same order as a directory listing),
# then reindex the dataframe to match that order.
sorted_filenames = sorted(image_files)
stem_to_order    = {os.path.splitext(f)[0].upper(): i
                    for i, f in enumerate(sorted_filenames)}

df["_sort_key"] = df["minifig_number"].apply(lambda n: stem_to_order.get(n.upper(), 99_999))
df.sort_values("_sort_key", inplace=True)
df.drop(columns=["_sort_key"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\nFirst 5 minifig_numbers after reorder: {df['minifig_number'].head().tolist()}")

