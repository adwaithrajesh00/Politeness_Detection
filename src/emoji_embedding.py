# Setting the project path and importing necassary libraries
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

# project-relative paths
DATA_CSV_PATH  = Path("data/raw/data.csv")
EMOJI2VEC_PATH = Path("data/external/emoji2vec.txt")
OUT_NPY_PATH   = Path("data/processed/emoji_features.npy")

EMOJI_COL = "emojis"  

# Controlled feature group "3 categories"

SOFTENERS    = set("🙂😊😇🙏✨🙌🥰🤗💖👍")
HOSTILITY    = set("🙄😒😑😠💢👎💀🤬😡")
INTENSIFIERS = set("😂🤣🔥🤯🥵😤👏🎉✨")

# Loading emoji2vec table

def load_emoji_vectors(path: Path) -> dict[str, np.ndarray]:
    table: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            emoji = parts[0]
            vec = np.fromiter((float(x) for x in parts[1:]), dtype=np.float32)
            table[emoji] = vec
    if not table:
        raise ValueError(f"Emoji vector table is empty. Check: {path}")
    dims = {v.shape[0] for v in table.values()}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent emoji2vec dims found: {dims}")
    return table


# Emoji2vec (emoji pooling same sematics together)

def emoji2vec_mean(emoji_string: str, table: dict[str, np.ndarray], dim: int) -> np.ndarray:
    if not isinstance(emoji_string, str) or not emoji_string:
        return np.zeros(dim, dtype=np.float32)
    emjs = list(emoji_string)                
    vecs = [table[e] for e in emjs if e in table]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(np.stack(vecs, axis=0), axis=0)

# counting the controlled categories (soft, hostile, intense)
def controlled_counts(emoji_string: str) -> np.ndarray:
    if not isinstance(emoji_string, str) or not emoji_string:
        return np.array([0, 0, 0], dtype=np.float32)
    c = Counter(list(emoji_string))
    soft    = sum(c[e] for e in SOFTENERS if e in c)
    hostile = sum(c[e] for e in HOSTILITY if e in c)
    intense = sum(c[e] for e in INTENSIFIERS if e in c)
    return np.array([soft, hostile, intense], dtype=np.float32)

#  combining the 2 embeddings for one cell → [emoji2vec... , soft, hostile, intense] "emoji2vec is 300-D, this returns 303-D".
def embed_emoji_cell(emoji_string: str, table: dict[str, np.ndarray], dim: int) -> np.ndarray:
    dense = emoji2vec_mean(emoji_string, table, dim)  
    ctrl  = controlled_counts(emoji_string)            
    return np.concatenate([dense, ctrl], axis=0)       


#Vectorizing the entire emoji columns

def embed_emoji_column(df: pd.DataFrame, col: str, table: dict[str, np.ndarray], dim: int) -> np.ndarray:
    out = np.zeros((len(df), dim + 3), dtype=np.float32)
    for i, s in enumerate(df[col].astype(str).fillna("")):
        out[i] = embed_emoji_cell(s, table, dim)
    return out

#Main function
if __name__ == "__main__":
    # 1) load CSV
    if not DATA_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_CSV_PATH}")
    df = pd.read_csv(DATA_CSV_PATH)
    if EMOJI_COL not in df.columns:
        raise KeyError(f"Column '{EMOJI_COL}' not found in {DATA_CSV_PATH}")

    # 2) load emoji2vec
    if not EMOJI2VEC_PATH.exists():
        raise FileNotFoundError(f"Missing emoji2vec file: {EMOJI2VEC_PATH}")
    EMO_TABLE = load_emoji_vectors(EMOJI2VEC_PATH)
    EMO_DIM = next(iter(EMO_TABLE.values())).shape[0]

    # 3) build features
    E_emoji = embed_emoji_column(df, EMOJI_COL, EMO_TABLE, EMO_DIM)  # (N, EMO_DIM+3)

    # 4) ensure output dir exists & save
    OUT_NPY_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NPY_PATH, E_emoji)
    print(f"Saved: {OUT_NPY_PATH}  shape: {E_emoji.shape}")
