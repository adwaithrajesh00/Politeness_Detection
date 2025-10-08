# Defining input CSV and Output paths
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parent.parent   # project root
RAW  = BASE / "data" / "raw"                   # Changing the dataset (With added abnormalities)
PROC = BASE / "data" / "processed"

CSV_PATH = RAW / "data.csv"                    # data.csv(dataset)
STICKER_NPY_PATH = PROC / "sticker_features.npy"
STICKER_MAP_CSV  = PROC / "sticker_tag_index.csv"

PROC.mkdir(parents=True, exist_ok=True)  

# Loading Dataset

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print("Loaded:", CSV_PATH, "| Rows:", len(df))
print("Columns:", df.columns.tolist()[:10])

# Splitting Sticker tags (if two sticker tags are there in a row like "angry smile" it will split it into 2)

def split_tags(cell):
    if pd.isna(cell):
        return []
    s = str(cell).lower().strip()
    # If multiple tags, split by comma, else by space
    parts = [p.strip() for p in (s.split(",") if "," in s else s.split()) if p.strip()]
    return parts

# Apply to dataset
sticker_lists = df["sticker_tags"].apply(split_tags)

# Building the vocabulary

all_tags = sorted({t for tags in sticker_lists for t in tags})
tag_to_idx = {t: i for i, t in enumerate(all_tags)}

print("Unique sticker tags:", len(all_tags))
print("Example tags:", all_tags[:10])

# Creating the one hot matrix

N = len(df)
K = len(all_tags)
onehot = np.zeros((N, K), dtype=np.float32)

for i, tags in enumerate(sticker_lists):
    for t in tags:
        j = tag_to_idx.get(t)
        if j is not None:
            onehot[i, j] = 1.0

print("One-hot shape:", onehot.shape)
print("Row 0 example:", onehot[0])

# Saving the Matrix

np.save(STICKER_NPY_PATH, onehot)

# Save the tag index mapping
pd.DataFrame({"index": range(K), "tag": all_tags}).to_csv(STICKER_MAP_CSV, index=False)

print("Saved:", STICKER_NPY_PATH)
print("Saved:", STICKER_MAP_CSV)

