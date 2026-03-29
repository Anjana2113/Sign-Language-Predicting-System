"""
src/gesture_detector.py — Landmarks-Only Feature Extractor
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class GestureDetector:
    """
    Extracts 21 hand landmarks and normalizes them for the coordinate classifier.
    """

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),           # Index
        (9, 10), (10, 11), (11, 12),              # Middle
        (13, 14), (14, 15), (15, 16),             # Ring
        (17, 18), (18, 19), (19, 20),             # Pinky
        (0, 5), (5, 9), (9, 13), (13, 17), (17, 0) # Palm
    ]

    def __init__(self, max_hands: int = config.MAX_HANDS, detection_confidence: float = 0.5):
        model_path = os.path.join(config.MODELS_DIR, "hand_landmarker.task")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray):
        """
        Returns landmark vectors (42 features) for each hand.
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)

        if not results.hand_landmarks:
            return [], [], [], annotated

        features_list = []
        bboxes = []
        handedness_list = []

        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            # 1. Handedness
            if idx < len(results.handedness):
                side = results.handedness[idx][0].category_name
            else:
                side = "Unknown"
            handedness_list.append(side)

            # 2. Extract and Normalize Coordinates
            pts = np.array([[lm.x, lm.y] for lm in hand_landmarks])
            
            # NORMALIZATION MATCHING EXTRACT_LANDMARKS.PY
            # a. Zero-center at wrist
            pts_norm = pts - pts[0]
            # b. Max-abs scaling to [-1, 1]
            max_val = np.abs(pts_norm).max()
            if max_val > 0:
                pts_norm = pts_norm / max_val
            
            # 3. Features vector
            features_list.append(pts_norm.flatten())
            
            # 4. Bounding box for UI
            px_pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks])
            x, y, wb, hb = cv2.boundingRect(px_pts)
            bboxes.append((x-10, y-10, x+wb+10, y+hb+10))

            # 5. Visuals
            color = (0, 255, 0) if side == "Right" else (255, 200, 0)
            for pt in px_pts:
                cv2.circle(annotated, tuple(pt), 4, color, -1)
            for start_idx, end_idx in self.HAND_CONNECTIONS:
                start = tuple(px_pts[start_idx])
                end = tuple(px_pts[end_idx])
                cv2.line(annotated, start, end, (255, 255, 255), 1)

        return features_list, bboxes, handedness_list, annotated

    def release(self):
        self.detector.close()
