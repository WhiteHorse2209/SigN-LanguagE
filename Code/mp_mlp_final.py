import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
from threading import Thread
import threading
import ollama

# Global lock to prevent overlapping voice calls
tts_lock = threading.Lock()

def say_text(text):
    """Create a fresh TTS engine for every word to avoid getting stuck."""
    # Use a lock to ensure only one thread speaks at a time
    if tts_lock.locked():
        return
        
    with tts_lock:
        try:
            tts = pyttsx3.init()
            tts.setProperty('rate', 150)
            tts.say(text)
            tts.runAndWait()
            tts.stop()
            # On some systems, we need to explicitly del the object
            del tts
        except Exception as e:
            # Silently ignore loop errors
            if "run loop already started" not in str(e):
                print(f"TTS error: {e}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load the MLP model bundle (model + scaler + label encoder)
try:
    with open('mp_mlp_model.pkl', 'rb') as f:
        bundle = pickle.load(f)
    model = bundle['model']
    scaler = bundle['scaler']
    le = bundle['label_encoder']
    print(f"Model loaded! Knows signs: {list(le.classes_)}")
except FileNotFoundError:
    print("Error: mp_mlp_model.pkl not found. Run mp_mlp_trainer.py first!")
    exit()

def get_prediction(landmarks_row):
    """Scale and predict from raw landmark coordinates."""
    scaled = scaler.transform([landmarks_row])
    pred_encoded = model.predict(scaled)[0]
    probabilities = model.predict_proba(scaled)[0]
    
    pred_label = le.inverse_transform([pred_encoded])[0]
    confidence = max(probabilities)
    
    return pred_label, confidence

def form_sentence(words):
    """Use Ollama AI to generate a meaningful sentence from the given words."""
    if not words:
        return ""

    # Remove consecutive duplicates
    cleaned = [words[0]]
    for w in words[1:]:
        if w.lower() != cleaned[-1].lower():
            cleaned.append(w)

    word_list = ", ".join(cleaned)
    prompt = (
        f"You are a sentence generator. "
        f"Using ONLY these words (you can change their form slightly, e.g. 'love' -> 'loves'): [{word_list}], "
        f"generate ONE short, natural, meaningful English sentence. "
        f"Do NOT add any extra words that are not related to the given words. "
        f"Reply with ONLY the sentence, nothing else."
    )

    try:
        response = ollama.chat(
            model='qwen2:0.5b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        sentence = response['message']['content'].strip()
        # Clean up any quotes the model may add
        sentence = sentence.strip('"').strip("'")
        if not sentence.endswith(('.', '!', '?')):
            sentence += '.'
        return sentence
    except Exception as e:
        print(f"Ollama error: {e}")
        # Fallback: just join the words
        return " ".join(cleaned) + "."


def recognize_realtime():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found!")
        return

    print("=" * 50)
    print("  SIGN LANGUAGE INTERPRETER [MLP]")
    print("=" * 50)
    print("Controls:")
    print("  SPACE = Add current word to sentence")
    print("  S     = Form sentence and speak it")
    print("  C     = Clear all words")
    print("  Q     = Quit")
    print("=" * 50)
    
    current_word = ""
    frames_stable = 0
    sentence = []
    formed_sentence = ""       # The final formed sentence
    show_sentence_timer = 0    # How long to show the formed sentence

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        display_text = "No Hand Detected"
        confidence_display = ""
        
        if results.multi_hand_landmarks:
            landmarks_row = []
            
            for hand_landmarks in results.multi_hand_landmarks[:2]:
                # Draw styled skeleton on both hands
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                for lm in hand_landmarks.landmark:
                    landmarks_row.extend([lm.x, lm.y, lm.z])

            # Pad if only 1 hand
            while len(landmarks_row) < (21 * 3 * 2):
                landmarks_row.extend([0.0, 0.0, 0.0])

            pred_label, conf = get_prediction(landmarks_row)
            
            if conf > 0.75:
                display_text = pred_label
                confidence_display = f"{conf*100:.1f}%"
                
                if pred_label == current_word:
                    frames_stable += 1
                    if frames_stable == 25:
                        # Auto-add word to sentence (no duplicate at end)
                        if not sentence or sentence[-1] != pred_label:
                            sentence.append(pred_label)
                else:
                    current_word = pred_label
                    frames_stable = 0
            else:
                display_text = "..."
                confidence_display = f"{conf*100:.1f}%"
                frames_stable = 0

        # Draw UI
        # Top bar - current prediction
        cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.putText(frame, display_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 255, 100), 2)
        if confidence_display:
            cv2.putText(frame, confidence_display, (w - 110, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

        # Middle area - show formed sentence (if recently formed)
        if show_sentence_timer > 0:
            # Draw a highlighted sentence box in the center
            box_h = 60
            box_y = (h // 2) - (box_h // 2)
            cv2.rectangle(frame, (20, box_y), (w - 20, box_y + box_h), (0, 120, 0), -1)
            cv2.rectangle(frame, (20, box_y), (w - 20, box_y + box_h), (100, 255, 100), 2)
            
            # Fit text size to box
            font_scale = 0.8
            text_size = cv2.getTextSize(formed_sentence, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            # Shrink font if too wide
            while text_size[0] > (w - 60) and font_scale > 0.4:
                font_scale -= 0.1
                text_size = cv2.getTextSize(formed_sentence, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            
            text_x = (w - text_size[0]) // 2
            text_y = box_y + (box_h + text_size[1]) // 2
            cv2.putText(frame, formed_sentence, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
            show_sentence_timer -= 1

        # Bottom bar - collected words
        words_str = " | ".join(sentence[-8:])  # Show last 8 words
        cv2.rectangle(frame, (0, h - 70), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, "Words:", (10, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(frame, words_str, (10, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 50), 2)
        
        # Controls hint at bottom-right
        cv2.putText(frame, "SPACE:Add  S:Sentence  C:Clear  Q:Quit", (10, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)

        cv2.imshow("Sign Language Interpreter [MLP]", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):  # Clear all words
            sentence = []
            formed_sentence = ""
            show_sentence_timer = 0
            print("[Cleared]")
        elif key == ord(' '):  # Space = manually add current word
            if current_word and (not sentence or sentence[-1] != current_word):
                sentence.append(current_word)
                print(f"[Added] {current_word}")
        elif key == ord('s'):  # S = Form sentence and speak it
            if sentence:
                formed_sentence = form_sentence(sentence)
                show_sentence_timer = 120  # Show for ~4 seconds
                print(f"\n>>> SENTENCE: {formed_sentence}")
                Thread(target=say_text, args=(formed_sentence,)).start()
            else:
                print("[No words collected yet]")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_realtime()
