"""
src/model.py — Landmark Classifier (Pivot Architecture)
"""
import tensorflow as tf
from tensorflow.keras import layers, models
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

def build_landmark_model(num_classes=config.NUM_CLASSES):
    """
    Builds a simple Dense model for vector (42,) input.
    """
    model = models.Sequential([
        layers.Input(shape=(config.LANDMARK_INPUT_SIZE,)),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    model = build_landmark_model()
    model.summary()
