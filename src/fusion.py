# Importing Libraries

import os
import sys
import numpy as np

#  organizing all input and output file locations in one place

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__)) 
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")

TEXT_FP   = os.path.join(DATA_DIR, "text_embeddings.npy")
EMOJI_FP  = os.path.join(DATA_DIR, "emoji_features.npy")
STICK_FP  = os.path.join(DATA_DIR, "sticker_features.npy")

OUT_TEXT        = os.path.join(DATA_DIR, "X_text.npy")
OUT_TEXT_EMOJI  = os.path.join(DATA_DIR, "X_text_emoji.npy")
OUT_FUSION      = os.path.join(DATA_DIR, "X_text_emoji_sticker.npy")


# Helper function to safely load files (eg:embedding already there)
def safe_load(fp, name):
    if not os.path.exists(fp):
        sys.exit(f"[ERROR] Missing {name} at: {fp}")
    arr = np.load(fp)
    if arr.ndim != 2:
        sys.exit(f"[ERROR] {name} must be a 2D array (got shape {arr.shape})")
    return arr

# Loading all 3 Matrices

def load_inputs():
    text  = safe_load(TEXT_FP,  "text_features.npy")    # (N, 384)
    emoji = safe_load(EMOJI_FP, "emoji_features.npy")   # (N, 303)
    stick = safe_load(STICK_FP, "sticker_features.npy") # (N, 14)

    # Row alignment check: all must have same N rows
    n_text, n_emoji, n_stick = text.shape[0], emoji.shape[0], stick.shape[0]
    if not (n_text == n_emoji == n_stick):
        sys.exit(f"[ERROR] Row count mismatch: text={n_text}, emoji={n_emoji}, sticker={n_stick}.\n"
                 "Ensure they were generated from the SAME data.csv ordering.")

    for name, arr in [("text", text), ("emoji", emoji), ("sticker", stick)]:
        if not np.all(np.isfinite(arr)):
            sys.exit(f"[ERROR] {name} contains NaN/Inf values. Clean before fusion.")

    return text, emoji, stick

# Building the 3 concatinated feature set

def build_feature_sets(text, emoji, stick):
    X_text = text
    X_text_emoji = np.concatenate([text, emoji], axis=1)

    X_fusion = np.concatenate([text, emoji, stick], axis=1)

    return X_text, X_text_emoji, X_fusion

# Saving Outputs

def save_outputs(X_text, X_text_emoji, X_fusion):
    np.save(OUT_TEXT,       X_text)
    np.save(OUT_TEXT_EMOJI, X_text_emoji)
    np.save(OUT_FUSION,     X_fusion)

    print("== Fusion Summary ==")
    print(f"Saved: {OUT_TEXT}                 shape={X_text.shape}")
    print(f"Saved: {OUT_TEXT_EMOJI}           shape={X_text_emoji.shape}")
    print(f"Saved: {OUT_FUSION}               shape={X_fusion.shape}")

# Main Function

if __name__ == "__main__":
    text, emoji, stick = load_inputs()
    X_text, X_text_emoji, X_fusion = build_feature_sets(text, emoji, stick)
    save_outputs(X_text, X_text_emoji, X_fusion)



