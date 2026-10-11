# SigN-LanguagE: Real-Time Hand Sign Interpreter 🤟

Welcome to **SigN-LanguagE**, an end-to-end, real-time AI system designed to bridge communication gaps by translating hand signs into spoken and written English sentences. The project leverages computer vision, neural networks, and local Large Language Models (LLMs) to seamlessly interpret gestures on the fly!

---

## ✨ Features

- **Real-Time Landmark Detection:** Utilizes **Google's MediaPipe** to extract 126 skeletal joints (x, y, z coordinates for two hands) from live webcam feed.
- **Custom MLP Classifier Model:** Built with `scikit-learn`, a fast Multi-Layer Perceptron (MLP) Neural Network predicts hand signs instantly. 
- **NLP Next-Word Prediction (`nlp_predictor.py`):** Anticipates what word you will sign next before it appears on screen! Uses a hybrid 0ms-latency statistical N-gram Markov engine backed by an asynchronous local Ollama LLM (`llama3.2:3b`, `qwen3`, etc.). Choose suggestions via 1-click chips in Streamlit or hotkeys `[1]`, `[2]`, `[3]` in OpenCV.
- **AI Sentence Generation:** Replaces simple word queues with fluent English. Integrates with **Ollama** (`llama3.2:3b` or similar) to form grammatically correct and meaningful sentences from predicted words.
- **Text-to-Speech (TTS):** Uses `pyttsx3` to locally vocalize the formed sentences without relying on internet APIs! 
- **Two User Interfaces:** 
  - **OpenCV Window (`mp_mlp_final.py`):** Simple desktop HUD app with real-time video overlay and number hotkeys (`1-3`) for next-word autofill.
  - **Streamlit Web-UI (`app.py`):** A beautiful, glassmorphism-styled dashboard running locally in your browser with interactive suggestion chips.

---

## 🛠️ Project Architecture

This project is separated into distinct pipelines:

### 1. Data Collection
- `mp_dataset_creator.py` : Mass-extracts hand landmarks from recorded `.mp4` video files within sorted subfolders.
- `mp_record_live.py` : Alternatively, lets you record your own signs via webcam directly into the `landmarks_data_2hands.csv` dataset.

### 2. Model Training
- `mp_mlp_trainer.py` : Splits your dataset, scales features with `StandardScaler`, and trains a 3-layer MLP Network. It automatically saves a bundle: `mp_mlp_model.pkl` containing the Model, Scaler, and Label Encoder.

### 3. NLP & Predictive Engine
- `train_nlp.py` : Trains a multi-order statistical N-gram language model (4-gram, trigram, bigram, unigram) on external text files or `training_corpus.txt`, saving `nlp_model.pkl`.
- `training_corpus.txt` : Customizable conversational corpus used to train next-word prediction probabilities.
- `nlp_predictor.py` : Fast hybrid predictor that loads `nlp_model.pkl` for zero-latency inference, paired with background Ollama semantic modeling.
- `test_nlp_suggestions.py` : Interactive CLI test script to verify suggestions on custom sequences without a webcam.

### 4. Inference
- `app.py` : Run using `streamlit run Code/app.py`. Displays live video HUD, prediction chips, gesture recognition, and sentence speech synthesis.
- `mp_mlp_final.py` : Lightweight OpenCV window with keyboard shortcuts (`1-3` for predictive autofill, `SPACE` to add, `S` to synthesize sentence).

---

## 🚀 How to Run

### **Prerequisites**
You need Python 3.9+ installed along with the following libraries:
```bash
pip install opencv-python mediapipe pandas numpy scikit-learn pyttsx3 streamlit ollama
```
*(Also ensure you have [Ollama](https://ollama.com/) installed locally on your system with a model like `llama3.2:3b` using `ollama run llama3.2:3b`).*

### **1. Build/Add Your Dataset**
If you want to train it on your own face/hands:
```bash
python Code/mp_record_live.py
```
Follow the console prompts to record your gestures.

### **2. Train the Vision Model**
Once you have your `landmarks_data_2hands.csv` filled:
```bash
python Code/mp_mlp_trainer.py
```
*(This generates the `mp_mlp_model.pkl` weights)*.

### **3. Train the NLP Next-Word Predictor on Custom Data**
Train on your sentences text file (or external corpus):
```bash
python Code/train_nlp.py
```
Or train on any custom external file:
```bash
python Code/train_nlp.py --corpus path/to/my_sentences.txt
```
*(This compiles transition statistics into `nlp_model.pkl` for instant prediction)*.

### **4. Start Translating!**

#### Step 1: Activate Virtual Environment
In PowerShell (from the project folder):
```powershell
.\venv\Scripts\Activate.ps1
```

#### Step 2: Test Environment & NLP Suggestions (Optional / Instant Verification)
- **Check all packages & models:**
  ```powershell
  python Code/test_environment.py
  ```
- **Test next-word prediction simulator in terminal:**
  ```powershell
  python Code/test_nlp_suggestions.py
  ```

#### Step 3: Run the Interpreter
- **Option A: Streamlit Web Dashboard**
  ```powershell
  streamlit run Code/app.py
  ```
  *(Features modern autocomplete chips with 1-click insertion and webcam suggestion pills)*

- **Option B: OpenCV Desktop Window**
  ```powershell
  python Code/mp_mlp_final.py
  ```
  *(Features on-screen suggestion pills: press **1**, **2**, or **3** to autofill without signing, **SPACE** to add sign, **S** to synthesize & speak sentence, **C** to clear).*

---

## 🤝 Collaborating

1. Ask your collaborator to clone the repo: `git clone https://github.com/WhiteHorse2209/SigN-LanguagE.git`
2. Create feature branches: `git checkout -b feature-new-sign`
3. Push changes and review them together!

Enjoy bridging the gap of communication!
