"""
Lego Minifig Dataset — Train / Val / Test Split
================================================
Input  : Data/processed/minifigs_clean.json
Output : Data/processed/train.json  (70 %)
         Data/processed/val.json    (20 %)
         Data/processed/test.json   (10 %)

Split is stratified on 'category' to preserve class distribution.
Classes with fewer than 3 samples cannot be stratified and are
assigned entirely to the training set.

Run from the project root (where the Data/ folder lives).
"""

import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_PATH = "Data/processed/minifigs_clean.json"
OUTPUT_DIR = "Data/processed"
TRAIN_SIZE = 0.70
VAL_SIZE   = 0.20
TEST_SIZE  = 0.10
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

assert abs(TRAIN_SIZE + VAL_SIZE + TEST_SIZE - 1.0) < 1e-9, "Sizes must sum to 1."

# ── LOAD ──────────────────────────────────────────────────────────────────────
with open(INPUT_PATH, encoding="utf-8") as f:
    df = pd.DataFrame(json.load(f))
print(f"Loaded {len(df)} records")

# ── SPLIT ─────────────────────────────────────────────────────────────────────
# Classes with < 3 samples cannot be stratified — put them all in train.
counts       = df["category"].value_counts()
df_ok        = df[df["category"].isin(counts[counts >= 3].index)]
df_tiny      = df[~df.index.isin(df_ok.index)]

if len(df_tiny):
    print(f"  {len(df_tiny)} rows from under-represented classes → train only")

# Step 1: split off test (10 %)
train_val, test = train_test_split(
    df_ok,
    test_size=TEST_SIZE,
    stratify=df_ok["category"],
    random_state=RANDOM_SEED,
)

# Step 2: split remaining 90 % into train (70 %) and val (20 %)
# Relative val size within the 90 % remainder = 0.20 / 0.90
train, val = train_test_split(
    train_val,
    test_size=VAL_SIZE / (TRAIN_SIZE + VAL_SIZE),
    stratify=train_val["category"],
    random_state=RANDOM_SEED,
)

# Append tiny-class rows to train
train = pd.concat([train, df_tiny], ignore_index=True)

print(f"Train : {len(train):>5}  ({len(train)/len(df)*100:.1f} %)")
print(f"Val   : {len(val):>5}  ({len(val)/len(df)*100:.1f} %)")
print(f"Test  : {len(test):>5}  ({len(test)/len(df)*100:.1f} %)")

print("Split complete (sklearn guarantees no row overlap between splits)")

# ── SAVE ──────────────────────────────────────────────────────────────────────
splits = {"train": train, "val": val, "test": test}

for name, split_df in splits.items():
    out_path = os.path.join(OUTPUT_DIR, f"{name}.json")
    split_df.to_json(out_path, orient="records", indent=2, force_ascii=False)
    print(f"Saved → {out_path}")
