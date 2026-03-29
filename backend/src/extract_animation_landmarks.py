"""
src/extract_animation_landmarks.py — Parallel Dataset joint extraction (TWO HANDS)
Designed specifically for the Text->Sign Animation feature.
"""
import os
import sys
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

def process_class(cls_name, label_idx):
    """
    Worker function to process all images in a single class directory.
    Extracts up to 2 hands (84 features).
    """
    model_path = os.path.join(config.MODELS_DIR, "hand_landmarker.task")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2, # <-- Changed to 2 for animation support
        min_hand_detection_confidence=0.3, # low confidence to catch tough angles
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cls_dir = os.path.join(config.DATASET_PATH, cls_name)
    if not os.path.exists(cls_dir):
        return [], [], cls_name
        
    img_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    local_landmarks = []
    local_labels = []

    for img_file in img_files:
        img_path = os.path.join(cls_dir, img_file)
        frame = cv2.imread(img_path)
        if frame is None: continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)

        if results.hand_landmarks:
            # We want to store exactly 84 coordinates (2 hands * 21 landmarks * 2 coords)
            # hand0 is the first detected hand, hand1 is the second (or zeros if none)
            
            # --- Hand 0 ---
            lm_list_0 = results.hand_landmarks[0]
            pts_0 = np.array([[lm.x, lm.y] for lm in lm_list_0])
            pts_0 = pts_0 - pts_0[0] # origin at wrist
            max_val_0 = np.abs(pts_0).max()
            if max_val_0 > 0:
                pts_0 = pts_0 / max_val_0
            flat_0 = pts_0.flatten()
            
            # --- Hand 1 ---
            if len(results.hand_landmarks) > 1:
                lm_list_1 = results.hand_landmarks[1]
                pts_1 = np.array([[lm.x, lm.y] for lm in lm_list_1])
                pts_1 = pts_1 - pts_1[0] # origin at wrist
                max_val_1 = np.abs(pts_1).max()
                if max_val_1 > 0:
                    pts_1 = pts_1 / max_val_1
                flat_1 = pts_1.flatten()
            else:
                flat_1 = np.zeros(42) # Padding for null second hand
                
            combined = np.concatenate((flat_0, flat_1)) # 84 features
            
            local_landmarks.append(combined)
            local_labels.append(label_idx)
    
    detector.close()
    return local_landmarks, local_labels, cls_name

def extract_landmarks_parallel():
    if not os.path.exists(config.DATASET_PATH):
        print(f" [Error] Dataset path not found: {config.DATASET_PATH}")
        return
        
    class_dirs = sorted([
        d for d in os.listdir(config.DATASET_PATH)
        if os.path.isdir(os.path.join(config.DATASET_PATH, d))
    ])
    label_map = {cls: idx for idx, cls in enumerate(class_dirs)}

    all_landmarks = []
    all_labels = []

    print(f" [Extract (Animation)] Parallel processing {len(class_dirs)} classes using {multiprocessing.cpu_count()} cores...")

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {executor.submit(process_class, cls, label_map[cls]): cls for cls in class_dirs}
        
        for future in as_completed(futures):
            cls_name = futures[future]
            try:
                lms, lbls, name = future.result()
                all_landmarks.extend(lms)
                all_labels.extend(lbls)
                print(f"  [DONE] {name}: {len(lms)} total samples so far: {len(all_landmarks)}")
            except Exception as e:
                print(f"  [ERROR] {cls_name}: {e}")

    # Save to a completely separate file so the trained model is untampered
    np.save("animation_landmarks_data.npy", np.array(all_landmarks))
    np.save("animation_landmarks_labels.npy", np.array(all_labels))
    print(f"\n [Success] Extracted {len(all_landmarks)} samples (84 features each) for animation.")

if __name__ == "__main__":
    extract_landmarks_parallel()
