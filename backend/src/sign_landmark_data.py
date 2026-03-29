import numpy as np
import os
import json
import config

class SignLandmarkData:
    """
    Loads the trained landmark data and provides median landmark skeletons 
    for each sign language letter class.
    """
    def __init__(self):
        self.label_to_landmarks = {}
        self._load_data()

    def _load_data(self):
        data_path = os.path.join(config.BASE_DIR, "animation_landmarks_data.npy")
        labels_path = os.path.join(config.BASE_DIR, "animation_landmarks_labels.npy")
        
        is_two_hand = True
        if not os.path.exists(data_path) or not os.path.exists(labels_path):
            print(f" [SignLandmarkData] WARNING: Missing {data_path}. Falling back to 1-hand data.")
            data_path = os.path.join(config.BASE_DIR, config.LANDMARKS_DATA_FILE)
            labels_path = os.path.join(config.BASE_DIR, config.LANDMARKS_LABELS_FILE)
            is_two_hand = False
            
        if not os.path.exists(data_path):
            return

        print(f" [SignLandmarkData] Loading landmark dataset for letter animation (Two-hand={is_two_hand})...")
        X = np.load(data_path)
        y = np.load(labels_path)
        
        with open(config.LABEL_MAP_PATH) as f:
            label_map = json.load(f)
            
        # Reverse label map: index string -> letter
        idx_to_class = {int(v): k for k, v in label_map.items()}
        
        # Group landmarks by class
        class_data = {label: [] for label in idx_to_class.values()}
        
        for i, label_idx in enumerate(y):
            if label_idx in idx_to_class:
                class_label = idx_to_class[label_idx]
                class_data[class_label].append(X[i])
                
        # Calculate median landmark for each class to get a representative skeleton
        for label, items in class_data.items():
            if items:
                # Median helps reduce outlier noise in the dataset
                median_vector = np.median(items, axis=0)
                
                if is_two_hand:
                    # 84 features -> hand0 and hand1
                    flat_0 = median_vector[:42]
                    flat_1 = median_vector[42:]
                    hand0 = flat_0.reshape(21, 2).tolist()
                    
                    if np.all(flat_1 == 0):
                        hand1 = None
                    else:
                        hand1 = flat_1.reshape(21, 2).tolist()
                        
                    self.label_to_landmarks[label] = {"hand0": hand0, "hand1": hand1}
                else:
                    # Fallback to 42 features -> hand0 only
                    hand0 = median_vector.reshape(21, 2).tolist()
                    self.label_to_landmarks[label] = {"hand0": hand0, "hand1": None}
                
        print(f" [SignLandmarkData] Loaded median landmarks for {len(self.label_to_landmarks)} classes.")

    def get_landmarks(self, char):
        """
        Returns {"hand0": [...], "hand1": [...]} for a given letter/digit,
        or None if not found in the dataset.
        """
        char = str(char).upper()
        return self.label_to_landmarks.get(char, None)

# Global singleton instance loaded once
_instance = None

def get_landmark_provider():
    global _instance
    if _instance is None:
        _instance = SignLandmarkData()
    return _instance
