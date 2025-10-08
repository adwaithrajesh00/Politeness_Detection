# Importing the necassary Modules
import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Code for enabling to slect the .npy file we wanted ([text],[text+emoji]+[text+emoji+sticker_tags])
def parse_args():
    ap = argparse.ArgumentParser(description="Logistic Regression on a single feature matrix.")
    ap.add_argument("--X", required=True, help="Path to .npy feature file (e.g., data/processed/X_text.npy)")
    ap.add_argument("--csv", required=True, help="Path to labels CSV (e.g., data/raw/data.csv)")
    ap.add_argument("--label-col", default="label", help="Label column name in CSV (default: label)")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test fraction if no 'split' column (default: 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    ap.add_argument("--save-model", action="store_true", help="Save trained model (optional)")
    ap.add_argument("--save-metrics", action="store_true", help="Save metrics JSON (optional)")
    return ap.parse_args()

# Loading labels in the CSV files (option to ignore split , but my data have it (70%/20%))
def load_labels(csv_path: str, label_col: str = "label"):
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {csv_path}.")

    # Accept 0/1 OR strings 'polite'/'impolite'
    y_raw = df[label_col].astype(str).str.lower().str.strip()
    try:
        y = y_raw.astype(int).to_numpy()
    except ValueError:
        mapping = {"polite": 1, "impolite": 0}
        if not set(y_raw.unique()).issubset(mapping.keys()):
            raise ValueError(
                f"Labels must be 0/1 or one of {list(mapping.keys())}. Found: {y_raw.unique()}"
            )
        y = y_raw.map(mapping).to_numpy()

    split = None
    if "split" in df.columns:
        split = df["split"].astype(str).str.lower().str.strip().to_numpy()

    return y, split


def make_model():
    # StandardScaler: makes features comparable (mean 0, std 1).
    # LogisticRegression: linear classifier that outputs probabilities.
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

# Main training and Evaluation Logic

def run_once(X_path, csv_path, label_col, test_size, seed, save_model, save_metrics):
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Feature file not found: {X_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    #  Load features and labels
    X = np.load(X_path)
    y, split = load_labels(csv_path, label_col)

    if X.shape[0] != len(y):
        raise ValueError(f"Row mismatch: X has {X.shape[0]} rows but labels have {len(y)}.")

    print("== Logistic Regression (single run) ==")
    print(f"Features: {X_path}  shape={X.shape}")
    print(f"Labels:   {csv_path}  n={len(y)}")

    #  Train/test split
    if split is not None and (("train" in split) or ("test" in split)):
        train_idx = np.where(split == "train")[0]
        test_idx  = np.where(split == "test")[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError("split column exists but no 'train'/'test' rows found.")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"Using CSV 'split' → train={len(train_idx)}, test={len(test_idx)}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        print(f"Random split (stratified) → train={len(y_train)}, test={len(y_test)}")

    #  Train
    model = make_model()
    model.fit(X_train, y_train)

    #  Predict + metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["impolite(0)", "polite(1)"], digits=4
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\n== Results ==")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix [rows=true, cols=pred]:")
    print(cm)

    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    if save_metrics:
        metrics_path = os.path.join(out_dir, "metrics_single_run.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "feature_file": X_path,
                "accuracy": float(acc),
                "classification_report": report,
                "confusion_matrix": cm.tolist()
            }, f, ensure_ascii=False, indent=2)
        print(f"\nSaved metrics → {metrics_path}")

    if save_model:
        pipe = model
        clf = pipe.named_steps["clf"]
        scaler = pipe.named_steps["scaler"]
        model_blob = {
            "feature_file": X_path,
            "coef": clf.coef_.tolist(),
            "intercept": clf.intercept_.tolist(),
            "classes": clf.classes_.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }
        model_path = os.path.join(out_dir, "lr_model_single_run.json")
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(model_blob, f, ensure_ascii=False, indent=2)
        print(f"Saved model → {model_path}")


def main():
    args = parse_args()
    run_once(
        X_path=args.X,
        csv_path=args.csv,
        label_col=args.label_col,
        test_size=args.test_size,
        seed=args.seed,
        save_model=args.save_model,
        save_metrics=args.save_metrics,
    )

if __name__ == "__main__":
    main()


