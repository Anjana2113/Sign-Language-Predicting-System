from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os
import sys
import json
from pathlib import Path

# Fix path to include the root directory
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.gesture_detector import GestureDetector
from src.autocomplete import Autocomplete
from src.tts import TTSEngine
from src.gemini_service import GeminiService
from src.sign_landmark_data import get_landmark_provider
from flask import request

app = Flask(__name__, 
            template_folder=os.path.join(Path(__file__).parent.parent.parent, 'frontend', 'templates'),
            static_folder=os.path.join(Path(__file__).parent.parent.parent, 'frontend', 'static'))

# Initialize components
detector = GestureDetector()
ac = Autocomplete()
tts = TTSEngine()
gemini = GeminiService()
landmark_provider = get_landmark_provider()

# Load Model
try:
    import tf_keras as tfk
    model = tfk.models.load_model(config.LANDMARK_MODEL_PATH)
except ImportError:
    model = tf.keras.models.load_model(config.LANDMARK_MODEL_PATH)
except Exception as e:
    print(f" [ERROR] Landmark model failed to load: {e}")
    sys.exit(1)

with open(config.LABEL_MAP_PATH) as f:
    label_map = json.load(f)
idx_to_class = {int(v): k for k, v in label_map.items()}

# Global State
current_sentence = ""
ai_sentence = ""
latest_label = "?"
latest_confidence = 0.0

def get_best_camera():
    for idx in [0, 1, 2, 3]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret: return cap
            cap.release()
    return None

def generate_frames():
    global latest_label, latest_confidence
    cap = get_best_camera()
    if not cap:
        print(" [ERROR] No camera found.")
        return
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Process frame for gestures
            features, bboxes, sides, annotated_frame = detector.process(frame)
            
            best_label = "?"
            best_conf = 0.0
            
            for i in range(len(features)):
                feat = features[i]
                input_batch = np.expand_dims(feat, axis=0)
                preds = model.predict(input_batch, verbose=0)
                idx = np.argmax(preds[0])
                confidence = float(preds[0][idx])
                label = idx_to_class[idx]

                if confidence > best_conf:
                    best_conf = confidence
                    best_label = label

            latest_label = best_label
            latest_confidence = best_conf

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_state')
def get_state():
    global current_sentence, latest_label, latest_confidence
    word = current_sentence.split(" ")[-1]
    suggestions = ac.get_suggestions(word)
    return jsonify({
        "sentence": current_sentence,
        "ai_sentence": ai_sentence,
        "label": latest_label,
        "confidence": f"{latest_confidence:.2f}",
        "suggestions": suggestions
    })

@app.route('/action/<name>')
def handle_action(name):
    global current_sentence, latest_label
    if name == "ADD":
        if latest_label != "?":
            current_sentence += latest_label
    elif name == "SPACE":
        current_sentence += " "
    elif name == "DELETE":
        current_sentence = current_sentence[:-1]
    elif name == "CLEAR":
        current_sentence = ""
        ai_sentence = ""
    return jsonify({"status": "ok", "sentence": current_sentence, "ai_sentence": ai_sentence})

@app.route('/apply_suggestion/<word>')
def apply_suggestion(word):
    global current_sentence
    parts = current_sentence.split(" ")
    parts[-1] = word
    current_sentence = " ".join(parts) + " "
    return jsonify({"status": "ok", "sentence": current_sentence})

@app.route('/speak')
def speak_sentence():
    global current_sentence, ai_sentence
    text_to_speak = ai_sentence if ai_sentence.strip() else current_sentence
    if text_to_speak.strip():
        tts.speak(text_to_speak)
    return jsonify({"status": "ok"})

@app.route('/refine_sentence')
def refine_sentence():
    global current_sentence, ai_sentence
    if current_sentence.strip():
        ai_sentence = gemini.refine_sentence(current_sentence)
    return jsonify({"status": "ok", "ai_sentence": ai_sentence})

@app.route('/text_to_sign', methods=['POST'])
def text_to_sign():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
        
    user_text = data['text']
    
    # 1. Ask Gemini to extract key words
    refined_text = gemini.refine_text_to_keywords(user_text)
    
    # 2. Split into words and then into a flat list of letters (including spaces)
    words = refined_text.split()
    letters = []
    for i, word in enumerate(words):
        for char in word:
            letters.append(char)
        if i < len(words) - 1:
            letters.append(" ") # Add space between words

    return jsonify({
        "status": "ok",
        "refined_text": refined_text,
        "words": words,
        "letters": letters
    })

@app.route('/get_landmark/<char>')
def get_landmark(char):
    if char == " " or not char:
        return jsonify({"char": char, "hand0": None, "hand1": None})
        
    landmarks_data = landmark_provider.get_landmarks(char)
    if not landmarks_data:
        return jsonify({"char": char, "hand0": None, "hand1": None}), 404
        
    return jsonify({
        "char": char, 
        "hand0": landmarks_data.get("hand0"), 
        "hand1": landmarks_data.get("hand1")
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
