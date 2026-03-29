"""
src/evaluate.py — Evaluate the trained model on the test set
Run: python src/evaluate.py
"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.data_loader import build_datasets


def plot_confusion_matrix(cm, class_names, save_path=None):
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix — ASL Gesture Classifier", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[Evaluate] Confusion matrix saved to {save_path}")
    plt.show()


def evaluate():
    print("=" * 60)
    print("  ASL Gesture Classifier — Evaluation")
    print("=" * 60)

    # ── 1. Load model ──────────────────────────────────────────────
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"[ERROR] Model not found at {config.MODEL_SAVE_PATH}")
        print("Please run 'python src/train.py' first.")
        sys.exit(1)

    print(f"[Evaluate] Loading model from {config.MODEL_SAVE_PATH}...")
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)

    # ── 2. Load label map ──────────────────────────────────────────
    with open(config.LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    # Invert: index -> class name
    idx_to_class = {v: k for k, v in label_map.items()}

    # ── 3. Build test dataset ──────────────────────────────────────
    _, _, test_ds, class_names = build_datasets()

    # ── 4. Predict ────────────────────────────────────────────────
    print("[Evaluate] Running predictions on test set...")
    y_true, y_pred = [], []
    for imgs, labels in test_ds:
        preds = model.predict(imgs, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── 5. Metrics ────────────────────────────────────────────────
    acc = np.mean(y_true == y_pred)
    print(f"\n✅ Test Accuracy: {acc * 100:.2f}%\n")

    label_names = [idx_to_class[i] for i in range(len(class_names))]
    print(classification_report(y_true, y_pred, target_names=label_names))

    # ── 6. Confusion matrix ───────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(config.MODELS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, label_names, save_path=cm_path)


if __name__ == "__main__":
    evaluate()
