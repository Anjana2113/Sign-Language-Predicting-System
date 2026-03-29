"""
src/app.py — Sign Interpreter (Autocomplete & Interactive Mode)
"""
import cv2
import numpy as np
import time
import json
import os
import sys
from pathlib import Path
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.gesture_detector import GestureDetector
from src.tts import TTSEngine
from src.autocomplete import Autocomplete

def get_best_camera():
    for idx in [0, 1, 2, 3]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret: return cap
            cap.release()
    return None

cap = get_best_camera()

class SignLanguageApp:
    def __init__(self, cap):
        self.cap = cap
        self.window_name = "Sign Language Interpreter"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_click)
        
        # Load Model
        if os.path.exists(config.LANDMARK_MODEL_PATH):
            try:
                import tf_keras as tfk
                self.model = tfk.models.load_model(config.LANDMARK_MODEL_PATH)
            except ImportError:
                self.model = tf.keras.models.load_model(config.LANDMARK_MODEL_PATH)
        else:
            print(" [ERROR] Landmark model not found.")
            sys.exit(1)

        with open(config.LABEL_MAP_PATH) as f:
            label_map = json.load(f)
        self.idx_to_class = {v: k for k, v in label_map.items()}
        self.detector = GestureDetector()
        self.tts = TTSEngine()
        self.ac = Autocomplete()

        # State
        self.current_sentence = ""
        self.latest_label = "?"
        self.mirror_view = True
        self.suggestions = []
        
        # UI Buttons (x, y, w, h)
        # Row 1: Word Suggestions (Dynamic Labels)
        self.suggest_buttons = [
            (10, 370, 185, 35), # Sugg 1
            (210, 370, 185, 35),# Sugg 2
            (410, 370, 185, 35) # Sugg 3
        ]
        
        # Row 2: Control Buttons
        self.buttons = {
            "ADD":    (10, 420, 110, 40),
            "SPACE":  (130, 420, 110, 40),
            "SPEAK":  (250, 420, 110, 40),
            "DELETE": (370, 420, 110, 40),
            "CLEAR":  (490, 420, 110, 40)
        }

    def _mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check Suggestions
            for i, (bx, by, bw, bh) in enumerate(self.suggest_buttons):
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    if i < len(self.suggestions):
                        self._apply_suggestion(self.suggestions[i])
                        return
            
            # Check Control Buttons
            for name, (bx, by, bw, bh) in self.buttons.items():
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._handle_button(name)
                    return

    def _apply_suggestion(self, word):
        # Find the start of the current word
        parts = self.current_sentence.split(" ")
        parts[-1] = word
        self.current_sentence = " ".join(parts) + " "
        self.suggestions = []

    def _handle_button(self, name):
        if name == "ADD":
            if self.latest_label and self.latest_label != "?":
                self.current_sentence += self.latest_label
        elif name == "SPACE":
            self.current_sentence += " "
        elif name == "SPEAK":
            self.tts.speak(self.current_sentence)
        elif name == "DELETE":
            self.current_sentence = self.current_sentence[:-1]
        elif name == "CLEAR":
            self.current_sentence = ""

    def run(self):
        print(" [App] Interactive GUI with Autocomplete Live.")
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None: continue
            
            if self.mirror_view:
                frame = cv2.flip(frame, 1)
            
            features, bboxes, sides, annotated_frame = self.detector.process(frame)

            best_label = "?"
            best_confidence = 0.0
            best_side = ""
            best_bbox = None

            for i in range(len(features)):
                feat = features[i]
                input_batch = np.expand_dims(feat, axis=0)
                preds = self.model.predict(input_batch, verbose=0)
                idx = np.argmax(preds[0])
                confidence = preds[0][idx]
                label = self.idx_to_class[idx]

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_label = label
                    best_side = sides[i]
                    best_bbox = bboxes[i]

            if best_confidence > 0.85:
                self.latest_label = best_label
                color = (0, 255, 0)
            else:
                self.latest_label = "?"
                color = (0, 165, 255)

            if best_bbox:
                cv2.rectangle(annotated_frame, (best_bbox[0], best_bbox[1]), (best_bbox[2], best_bbox[3]), color, 2)
                cv2.putText(annotated_frame, f"{best_side}: {best_label} ({best_confidence:.2f})", 
                            (best_bbox[0], best_bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- Update Suggestions ---
            current_word = self.current_sentence.split(" ")[-1]
            self.suggestions = self.ac.get_suggestions(current_word)

            self._draw_gui(annotated_frame)
            
            cv2.imshow(self.window_name, annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('m'): self.mirror_view = not self.mirror_view

        self.cap.release()
        cv2.destroyAllWindows()
        self.detector.release()

    def _draw_gui(self, frame):
        h, w = frame.shape[:2]
        # Text Display Area
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"TEXT: {self.current_sentence}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Autocomplete Buttons (Row 1)
        for i, (x, y, bw, bh) in enumerate(self.suggest_buttons):
            label = ""
            if i < len(self.suggestions):
                label = self.suggestions[i]
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 60, 0), -1)
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 1)
                cv2.putText(frame, label, (x+10, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Control Buttons (Row 2)
        for name, (x, y, bw, bh) in self.buttons.items():
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (40, 40, 40), -1)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (200, 200, 200), 2)
            cv2.putText(frame, name, (x + 20, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

if __name__ == "__main__":
    if cap:
        app = SignLanguageApp(cap)
        app.run()
