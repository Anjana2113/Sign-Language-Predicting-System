"""
src/train_landmarks.py — Train on coordinate data (with Augmentation)
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.model import build_landmark_model

def augment_landmarks(data, labels):
    """
    Augments the landmark dataset by mirroring (X -> -X) 
    to handle both Left and Right hands equally.
    """
    print(" [Train] Augmenting data (Mirroring)...")
    mirrored_data = data.copy()
    # Landmarks are [x0, y0, x1, y1, ...]
    # Flip X coordinates (indices 0, 2, 4, ...)
    mirrored_data[:, 0::2] = -mirrored_data[:, 0::2]
    
    combined_data = np.vstack((data, mirrored_data))
    combined_labels = np.hstack((labels, labels))
    
    # Add small Gaussian noise
    noise = np.random.normal(0, 0.01, combined_data.shape)
    noisy_data = combined_data + noise
    
    return np.vstack((combined_data, noisy_data)), np.hstack((combined_labels, combined_labels))

def train():
    print(" [Train] Loading extracted landmarks...")
    try:
        X = np.load("landmarks_data.npy")
        y = np.load("landmarks_labels.npy")
    except FileNotFoundError:
        print(" [ERROR] landmarks_data.npy not found.")
        return

    # Augment
    X_aug, y_aug = augment_landmarks(X, y)
    print(f" [Train] Augmented dataset size: {len(X_aug)}")

    # One-hot
    y_onehot = tf.keras.utils.to_categorical(y_aug, num_classes=config.NUM_CLASSES)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_onehot, test_size=0.15, random_state=42, stratify=y_aug
    )

    model = build_landmark_model()
    
    checkpoint = ModelCheckpoint(
        config.LANDMARK_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    print(f" [Train] Starting training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200, 
        batch_size=64,
        callbacks=[checkpoint, early_stop]
    )

    # Save final
    model.save(config.LANDMARK_MODEL_PATH)
    print(f" [Success] Landmark model saved to {config.LANDMARK_MODEL_PATH}")

if __name__ == "__main__":
    train()
