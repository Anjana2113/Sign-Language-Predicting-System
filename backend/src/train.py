"""
src/train.py — Optimized training script for the QuickModel ASL classifier
"""
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.data_loader import build_datasets
from src.model import build_model, compile_model


def train():
    print("=" * 60)
    print("  ASL Gesture Classifier — Optimized Training (128x128)")
    print("=" * 60)

    # 1. Data
    train_ds, val_ds, test_ds, class_names = build_datasets()

    # 2. Model
    model = build_model(num_classes=len(class_names))
    compile_model(model, learning_rate=config.LEARNING_RATE)
    model.summary()

    # 3. Callbacks
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    # 4. Training (Single Phase - no base to freeze)
    print(f"\n[Train] Starting {config.EPOCHS} epochs on CPU...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Final Evaluation
    print("\n[Train] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

    print(f"\n✅ Model saved to: {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
