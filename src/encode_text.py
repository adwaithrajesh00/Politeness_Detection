import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Config

INPUT_CSV  = os.path.join("data","raw","data.csv")
OUTPUT_DIR = os.path.join("data","processed")
EMB_NPY    = os.path.join(OUTPUT_DIR, "text_embeddings.npy")
IDS_CSV    = os.path.join(OUTPUT_DIR, "ids.csv")
INFO_TXT   = os.path.join(OUTPUT_DIR, "embedding_info.txt")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Reading CSV

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}")

    df = pd.read_csv(path)

    required = ["id", "text", "emojis", "sticker_tags", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    df["text"] = df["text"].fillna("").astype(str)
    return df

# Loading Model (Turning sentences into 384-dim vectors.)

def load_model(name):
    model = SentenceTransformer(name)
    return model

# Encoding
def make_embeddings(model, texts):
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    )
    return embeddings

# Saving the outputs

def save_outputs(embeddings, df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(EMB_NPY, embeddings)
    df[["id"]].to_csv(IDS_CSV, index=False)

    with open(INFO_TXT, "w", encoding="utf-8") as f:
        f.write(f"model={MODEL_NAME}\n")
        f.write(f"num_rows={len(df)}\n")
        f.write(f"vector_dim={embeddings.shape[1]}\n")

# Main Function

def main():
    df = load_data(INPUT_CSV)
    texts = df["text"].tolist()

    model = load_model(MODEL_NAME)
    embeddings = make_embeddings(model, texts)

    save_outputs(embeddings, df)

    print("Embeddings shape:", embeddings.shape)
    if len(embeddings) > 0:
        print("First vector (first 5 values):", np.round(embeddings[0][:5], 4).tolist())

if __name__ == "__main__":
    main()

    