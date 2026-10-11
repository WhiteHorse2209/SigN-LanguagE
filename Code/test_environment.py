"""
Diagnostic Test Script
Validates all required dependencies and hardware/model components
in the current Python environment.
"""

import sys
import os

def check_environment():
    print("=" * 60)
    print("  SIGN LANGUAGE & NLP DIAGNOSTIC CHECK")
    print("=" * 60)
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version   : {sys.version.split()[0]}")
    print("-" * 60)

    # 1. MediaPipe
    try:
        import mediapipe as mp
        hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
        print("[OK] MediaPipe is installed and initialized.")
    except Exception as e:
        print(f"[FAIL] MediaPipe error: {e}")

    # 2. OpenCV
    try:
        import cv2
        print(f"[OK] OpenCV is installed (version {cv2.__version__}).")
    except Exception as e:
        print(f"[FAIL] OpenCV error: {e}")

    # 3. Scikit-learn & Model
    try:
        import pickle
        import sklearn
        model_path = os.path.join(os.path.dirname(__file__), 'mp_mlp_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                bundle = pickle.load(f)
            classes = list(bundle['label_encoder'].classes_)
            print(f"[OK] Trained MLP Model loaded successfully! Known signs ({len(classes)}): {classes[:6]}...")
        else:
            print(f"[WARNING] Model file not found at {model_path}.")
    except Exception as e:
        print(f"[FAIL] Model loading error: {e}")

    # 4. NLP Next-Word Predictor
    try:
        from nlp_predictor import NextWordPredictor
        pred = NextWordPredictor()
        res = pred.predict(["I"])
        print(f"[OK] NLP Predictor initialized successfully (predicts {res} after 'I').")
    except Exception as e:
        print(f"[FAIL] NLP Predictor error: {e}")

    # 5. Streamlit
    try:
        import streamlit
        print(f"[OK] Streamlit is installed (version {streamlit.__version__}).")
    except Exception as e:
        print(f"[FAIL] Streamlit error: {e}")

    # 6. Pyttsx3 TTS
    try:
        import pyttsx3
        tts = pyttsx3.init()
        tts.stop()
        del tts
        print("[OK] Text-to-Speech (pyttsx3) engine initialized successfully.")
    except Exception as e:
        print(f"[WARNING] Text-to-Speech engine notice: {e}")

    # 7. Ollama
    try:
        import ollama
        print("[OK] Ollama client library is installed.")
    except Exception as e:
        print(f"[FAIL] Ollama library error: {e}")

    print("=" * 60)
    print("  ALL CORE SYSTEM CHECKS COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    check_environment()
