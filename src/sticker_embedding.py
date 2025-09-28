# src/sticker_embedding_sbert.py
import os
import re
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Paths 

DATA_CSV   = "data/raw/data.csv"    
OUT_DIR    = "data/processed"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  

os.makedirs(OUT_DIR, exist_ok=True)


#  Loading dataset

df = pd.read_csv(DATA_CSV)
assert "sticker_tags" in df.columns, "CSV must have a 'sticker_tags' column."
print(f"Loaded: {DATA_CSV}  rows={len(df)}")


# Clean & parse tags

def clean_tags(s: str):
    if not isinstance(s, str):
        return []
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9,\s]+", "", s)
    parts = re.split(r"[,\s]+", s)
    return [p for p in parts if p]

tag_lists = df["sticker_tags"].fillna("").apply(clean_tags)


# Unique vocab

unique_tags = sorted({t for tags in tag_lists for t in tags})
print(f"Unique sticker tags: {len(unique_tags)} → {unique_tags[:10]}")


#  Load SBERT

print(f"Loading SBERT: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

#  Encode all tags

if unique_tags:
    tag_matrix = model.encode(unique_tags, batch_size=64, convert_to_numpy=True,
                              show_progress_bar=True, normalize_embeddings=False)
else:
    tag_matrix = np.zeros((0, 384), dtype=np.float32)

tag2vec = {tag: tag_matrix[i] for i, tag in enumerate(unique_tags)}

#  Row-level vectors

EMB_DIM = 384
def row_vector(tags):
    if not tags:
        return np.zeros(EMB_DIM, dtype=np.float32)
    vecs = [tag2vec[t] for t in tags if t in tag2vec]
    if not vecs:
        return np.zeros(EMB_DIM, dtype=np.float32)
    v = np.mean(vecs, axis=0)
    norm = np.linalg.norm(v) + 1e-12
    return (v / norm).astype(np.float32)

rows = [row_vector(tags) for tags in tqdm(tag_lists, desc="Building row vectors")]
X_sticker_bert = np.vstack(rows).astype(np.float32)
print("Sticker SBERT features shape:", X_sticker_bert.shape)


#  Saving the  outputs

OUT_NPY   = os.path.join(OUT_DIR, "sticker_features.npy")
OUT_TAGS  = os.path.join(OUT_DIR, "sticker_tag_vocab.csv")
OUT_INFO  = os.path.join(OUT_DIR, "sticker_sbert_info.json")

np.save(OUT_NPY, X_sticker_bert)
pd.DataFrame({"tag": unique_tags}).to_csv(OUT_TAGS, index=False)
with open(OUT_INFO, "w", encoding="utf-8") as f:
    json.dump({
        "model": MODEL_NAME,
        "input_csv": DATA_CSV,
        "num_rows": int(len(df)),
        "embedding_dim": EMB_DIM,
        "num_unique_tags": int(len(unique_tags)),
        "feature_file": OUT_NPY,
        "tag_vocab_csv": OUT_TAGS
    }, f, indent=2)

print(f"Saved: {OUT_NPY}")


