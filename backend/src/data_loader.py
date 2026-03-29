"""
src/data_loader.py — Dataset loading, augmentation, and splitting
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def get_image_paths_and_labels():
    """Scan dataset directory and return (paths, integer_labels, class_names)."""
    image_paths = []
    labels = []

    class_dirs = sorted([
        d for d in os.listdir(config.DATASET_PATH)
        if os.path.isdir(os.path.join(config.DATASET_PATH, d))
    ])

    # Build label map: class_name -> integer
    label_map = {cls: idx for idx, cls in enumerate(class_dirs)}

    # Save label map for inference
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(config.LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    for cls_name in class_dirs:
        cls_dir = os.path.join(config.DATASET_PATH, cls_name)
        for img_file in os.listdir(cls_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(cls_dir, img_file))
                labels.append(label_map[cls_name])

    print(f"[DataLoader] Found {len(image_paths)} images across {len(class_dirs)} classes.")
    return image_paths, labels, class_dirs


def load_and_preprocess(path, label):
    """TF-compatible image loading and preprocessing."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=config.IMG_CHANNELS)
    img = tf.image.resize(img, config.IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0  # normalize to [0, 1]
    label = tf.one_hot(label, depth=config.NUM_CLASSES)
    return img, label


def augment(img, label):
    """Apply data augmentation for training."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def build_datasets():
    """
    Returns (train_ds, val_ds, test_ds) as tf.data.Dataset objects.
    """
    paths, labels, class_names = get_image_paths_and_labels()

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        paths, labels,
        test_size=config.TEST_RATIO,
        stratify=labels,
        random_state=42
    )

    # Second split: train vs val
    val_size = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size,
        stratify=y_trainval,
        random_state=42
    )

    print(f"[DataLoader] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    def make_ds(paths_list, labels_list, augment_data=False, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices(
            (tf.constant(paths_list), tf.constant(labels_list))
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths_list), seed=42)
        ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if augment_data:
            ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_ds(X_train, y_train, augment_data=True, shuffle=True)
    val_ds   = make_ds(X_val,   y_val,   augment_data=False, shuffle=False)
    test_ds  = make_ds(X_test,  y_test,  augment_data=False, shuffle=False)

    return train_ds, val_ds, test_ds, class_names


if __name__ == "__main__":
    train_ds, val_ds, test_ds, classes = build_datasets()
    print(f"Classes: {classes}")
    for imgs, lbls in train_ds.take(1):
        print(f"Batch shape: {imgs.shape}, Labels shape: {lbls.shape}")
