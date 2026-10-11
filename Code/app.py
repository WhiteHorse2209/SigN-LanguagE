import streamlit as st
import cv2
import mediapipe as mp
import pickle
import threading
import pyttsx3
import ollama
import os
from nlp_predictor import NextWordPredictor, get_default_ollama_model

OLLAMA_MODEL = get_default_ollama_model()

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="SignBridge - Real-Time Sign Language Interpreter",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Global Styles */
    .block-container {
        padding-top: 1.5rem;
    }
    
    /* Title Styling */
    h1 {
        text-align: center;
        background: linear-gradient(90deg, #bb86fc, #03dac6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        margin-bottom: 30px;
    }
    
    /* Glassmorphism Containers */
    .glass-panel {
        background: rgba(30, 30, 30, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    .status-text {
        color: #bb86fc;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .large-pred {
        font-size: 3.5rem;
        font-weight: 800;
        color: #03dac6;
        text-align: center;
        line-height: 1.2;
        padding: 10px;
    }
    
    .sentence-display {
        font-size: 1.8rem;
        color: #ffffff;
        background: linear-gradient(145deg, #1e1e1e, #2a2a2a);
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #03dac6;
        margin-top: 20px;
        min-height: 90px;
        display: flex;
        align-items: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    /* Video border */
    .video-canvas img {
        border-radius: 20px;
        border: 2px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    
    /* Styling Buttons */
    .stButton>button {
        border-radius: 12px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .nlp-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(187, 134, 252, 0.2), rgba(3, 218, 198, 0.2));
        border: 1px solid rgba(187, 134, 252, 0.4);
        border-radius: 10px;
        padding: 5px 10px;
        font-size: 0.9rem;
        color: #03dac6;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Core application state initialization
# -----------------------------------
if 'collected_words' not in st.session_state:
    st.session_state.collected_words = []
if 'formed_sentence' not in st.session_state:
    st.session_state.formed_sentence = ""
if 'current_word' not in st.session_state:
    st.session_state.current_word = ""
if 'last_stable_word' not in st.session_state:
    st.session_state.last_stable_word = ""
if 'frames_stable' not in st.session_state:
    st.session_state.frames_stable = 0
if 'run' not in st.session_state:
    st.session_state.run = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# Variables for TTS to prevent multiple overlapping calls
tts_lock = threading.Lock()

def say_text_async(text):
    """Speaks text in a background thread."""
    def _speak():
        if tts_lock.locked(): return
        with tts_lock:
            try:
                tts = pyttsx3.init()
                tts.setProperty('rate', 150)
                tts.say(text)
                tts.runAndWait()
                tts.stop()
                del tts
            except Exception:
                pass
    threading.Thread(target=_speak, daemon=True).start()

# Load models safely
@st.cache_resource
def get_ml_components():
    model_path = os.path.join(os.path.dirname(__file__), 'mp_mlp_model.pkl')
    try:
        with open(model_path, 'rb') as f:
            bundle = pickle.load(f)
        m = bundle['model']
        s = bundle['scaler']
        l = bundle['label_encoder']
    except FileNotFoundError:
        return None, None, None, None, None, None, None
        
    m_h = mp.solutions.hands
    h = m_h.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    m_d = mp.solutions.drawing_utils
    m_ds = mp.solutions.drawing_styles
    return m, s, l, h, m_h, m_d, m_ds

model, scaler, le, hands, mp_hands, mp_drawing, mp_drawing_styles = get_ml_components()

@st.cache_resource
def get_nlp_engine(vocab_classes):
    return NextWordPredictor(vocab=list(vocab_classes) if vocab_classes else None)

nlp_engine = get_nlp_engine(tuple(le.classes_)) if le is not None else get_nlp_engine(())

def form_sentence_via_ollama(words):
    if not words: return ""
    # Filter duplicates
    cleaned = [words[0]]
    for w in words[1:]:
        if w.lower() != cleaned[-1].lower():
            cleaned.append(w)
    word_list = ", ".join(cleaned)
    prompt = (
        f"You are a sentence generator. "
        f"Using ONLY these words (you can change their form slightly): [{word_list}], "
        f"generate ONE short, natural, meaningful English sentence. "
        f"Reply with ONLY the final sentence, nothing else."
    )
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': prompt}])
        sentence = response['message']['content'].strip(' "\'')
        if not sentence.endswith(('.', '!', '?')): sentence += '.'
        return sentence
    except Exception:
        return " ".join(cleaned) + "."

# -----------------------------------
# Layout & UI
# -----------------------------------
st.markdown("<h1>✨ SignBridge Real-Time AI ✨</h1>", unsafe_allow_html=True)

if not model:
    st.error("Model file 'mp_mlp_model.pkl' not found! Please train the model in your original project first.")
    st.stop()

col_video, col_controls = st.columns([2.5, 1.2])

with col_controls:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("<div class='status-text'>Action Center</div>", unsafe_allow_html=True)
    
    # Run toggle
    if st.session_state.run:
        if st.button("🛑 Stop Camera", use_container_width=True):
            st.session_state.run = False
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.rerun()
    else:
        if st.button("🎦 Start Camera", use_container_width=True, type="primary"):
            st.session_state.run = True
            st.session_state.cap = cv2.VideoCapture(0)
            st.rerun()

    st.write("---")
    
    # Interaction Buttons
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("➕ Add Word", use_container_width=True):
            if st.session_state.current_word:
                if not st.session_state.collected_words or st.session_state.collected_words[-1] != st.session_state.current_word:
                    nlp_engine.record_selection(st.session_state.collected_words, st.session_state.current_word)
                    st.session_state.collected_words.append(st.session_state.current_word)
    with col_b:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.collected_words = []
            st.session_state.formed_sentence = ""
            
    if st.button("🎙️ Form AI Sentence & Speak", use_container_width=True, type="primary"):
        if st.session_state.collected_words:
            with st.spinner("Refining grammar via Local LLM..."):
                sent = form_sentence_via_ollama(st.session_state.collected_words)
                st.session_state.formed_sentence = sent
                say_text_async(sent)
                
    st.markdown('</div>', unsafe_allow_html=True)
    
    # NLP Next-Word Prediction Suggestions Panel
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("<div class='status-text'>💡 Next Word Suggestions</div>", unsafe_allow_html=True)
    next_preds = nlp_engine.predict(st.session_state.collected_words, top_k=3)
    
    # Context Preview
    context_str = ' '.join(st.session_state.collected_words) if st.session_state.collected_words else '(Start signing)'
    st.markdown(f"<div style='font-size:0.85rem; color:#aaa; margin-bottom:8px;'>Context: <span style='color:#bb86fc; font-weight:600;'>{context_str}</span> ➔ <span style='color:#03dac6; font-style:italic;'>[?]</span></div>", unsafe_allow_html=True)
    
    pred_cols = st.columns(len(next_preds))
    for idx, p_w in enumerate(next_preds):
        with pred_cols[idx]:
            if st.button(f"⚡ {idx+1}. {p_w}", key=f"sugg_{idx}_{p_w}", use_container_width=True):
                if not st.session_state.collected_words or st.session_state.collected_words[-1] != p_w:
                    nlp_engine.record_selection(st.session_state.collected_words, p_w)
                    st.session_state.collected_words.append(p_w)
                    st.session_state.frames_stable = -15
                    st.rerun()
    st.caption("⚡ Click any suggestion chip above to insert immediately without signing.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction Status
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("<div class='status-text'>Detected Sign</div>", unsafe_allow_html=True)
    pred_ui = st.empty()
    conf_progress = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Words Box
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("<div class='status-text'>Assembled Queue</div>", unsafe_allow_html=True)
    words_ui = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with col_video:
    st.markdown('<div class="video-canvas">', unsafe_allow_html=True)
    video_ui = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.formed_sentence:
        st.markdown(f'<div class="sentence-display">{"🤖 " + st.session_state.formed_sentence}</div>', unsafe_allow_html=True)

# -----------------------------------
# Backend Execution Loop
# -----------------------------------
if not st.session_state.run:
    # State when paused
    video_ui.info("Camera is currently paused. Use the Action Center to start it.")
    pred_ui.markdown("<div class='large-pred' style='color: #666;'>⏸️</div>", unsafe_allow_html=True)
    words_ui.markdown(f"**{' | '.join(st.session_state.collected_words) if st.session_state.collected_words else 'No words yet.'}**")
else:
    # Update current word UI (so it dynamically reflects array changes before next frame)
    words_ui.markdown(f"**{' | '.join(st.session_state.collected_words) if st.session_state.collected_words else 'Waiting for signs...'}**")
    
    if st.session_state.cap and st.session_state.cap.isOpened():
        ret, frame = st.session_state.cap.read()
        if ret:
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hands.process(rgb_frame)
            
            display_text = "None"
            conf_val = 0.0
            
            if results.multi_hand_landmarks:
                landmarks_row = []
                for hl in results.multi_hand_landmarks[:2]:
                    # Draw elegant connections
                    mp_drawing.draw_landmarks(
                        frame, hl, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    for lm in hl.landmark:
                        landmarks_row.extend([lm.x, lm.y, lm.z])
                
                # Padding for single hand or no hands
                while len(landmarks_row) < 126:
                    landmarks_row.extend([0.0, 0.0, 0.0])
                    
                # Predictions
                scaled = scaler.transform([landmarks_row])
                pred_encoded = model.predict(scaled)[0]
                probs = model.predict_proba(scaled)[0]
                pred_label = le.inverse_transform([pred_encoded])[0]
                conf = max(probs)
                
                if conf > 0.75:
                    display_text = pred_label
                    conf_val = conf
                    st.session_state.current_word = pred_label
                    
                    # Auto-add Logic (20 consecutive frames)
                    if pred_label == st.session_state.last_stable_word:
                        st.session_state.frames_stable += 1
                        if st.session_state.frames_stable == 20: 
                            if not st.session_state.collected_words or st.session_state.collected_words[-1] != pred_label:
                                nlp_engine.record_selection(st.session_state.collected_words, pred_label)
                                st.session_state.collected_words.append(pred_label)
                                st.session_state.frames_stable = -15 # Cooldown period
                    else:
                        st.session_state.last_stable_word = pred_label
                        st.session_state.frames_stable = 0
                else:
                    st.session_state.frames_stable = 0
            
            # Draw NLP Suggestion HUD on live webcam frame
            h_f, w_f, _ = frame.shape
            cv2.rectangle(frame, (0, h_f - 46), (w_f, h_f), (28, 22, 38), -1)
            cv2.putText(frame, "SUGGESTIONS:", (10, h_f - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (187, 134, 252), 1)
            hud_preds = nlp_engine.predict(st.session_state.collected_words, top_k=3)
            pill_x = 145
            for i, wrd in enumerate(hud_preds):
                pill_text = f"[{i+1}] {wrd}"
                is_match = (display_text == wrd and conf_val > 0.75)
                bg_col = (0, 100, 60) if is_match else (45, 36, 62)
                bdr_col = (0, 255, 150) if is_match else (3, 218, 198)
                txt_col = (255, 255, 255) if is_match else (3, 218, 198)
                t_w = cv2.getTextSize(pill_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)[0][0]
                if pill_x + t_w + 10 < w_f:
                    cv2.rectangle(frame, (pill_x - 5, h_f - 38), (pill_x + t_w + 5, h_f - 10), bg_col, -1)
                    cv2.rectangle(frame, (pill_x - 5, h_f - 38), (pill_x + t_w + 5, h_f - 10), bdr_col, 1)
                    cv2.putText(frame, pill_text, (pill_x, h_f - 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, txt_col, 2)
                    pill_x += t_w + 18
            
            # Show Feed in UI container
            frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_ui.image(frame_disp, use_container_width=True)
            
            # Update Prediction UI components
            color = "#03dac6" if conf_val > 0.75 else "#ff6b6b"
            pred_ui.markdown(f"<div class='large-pred' style='color: {color};'>{display_text}</div>", unsafe_allow_html=True)
            if conf_val > 0.1:
                conf_progress.progress(float(conf_val), text=f"Confidence: {conf_val*100:.1f}%")
            else:
                conf_progress.empty()
                
            # Loop recursively while 'run' is true to update video continuously
            st.rerun()
        else:
            video_ui.error("Error reading frame from webcam hardware.")
