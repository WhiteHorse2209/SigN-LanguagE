# SigN-LanguagE: Real-Time Hand Sign Interpreter 🤟

Welcome to **SigN-LanguagE**, an end-to-end, real-time AI system designed to bridge communication gaps by translating hand signs into spoken and written English sentences. The project leverages computer vision, neural networks, and local Large Language Models (LLMs) to seamlessly interpret gestures on the fly!

---

## ✨ Features

- **Real-Time Landmark Detection:** Utilizes **Google's MediaPipe** to extract 126 skeletal joints (x, y, z coordinates for two hands) from live webcam feed.
- **Custom MLP Classifier Model:** Built with `scikit-learn`, a fast Multi-Layer Perceptron (MLP) Neural Network predicts hand signs instantly. 
- **AI Sentence Generation:** Replaces simple word queues with fluent English. Integrates with **Ollama** (`qwen2:0.5b` or similar) to form grammatically correct and meaningful sentences from predicted words.
- **Text-to-Speech (TTS):** Uses `pyttsx3` to locally vocalize the formed sentences without relying on internet APIs! 
- **Two User Interfaces:** 
  - **OpenCV Window (`mp_mlp_final.py`):** Simple bounding box style desktop app.
  - **Streamlit Web-UI (`app.py`):** A beautiful, glassmorphism-styled dashboard running locally in your browser.

---

## 🛠️ Project Architecture

This project is separated into distinct pipelines:

### 1. Data Collection
- `mp_dataset_creator.py` : Mass-extracts hand landmarks from recorded `.mp4` video files within sorted subfolders.
- `mp_record_live.py` : Alternatively, lets you record your own signs via webcam directly into the `landmarks_data_2hands.csv` dataset.

### 2. Model Training
- `mp_mlp_trainer.py` : Splits your dataset, scales features with `StandardScaler`, and trains a 3-layer MLP Network. It automatically saves a bundle: `mp_mlp_model.pkl` containing the Model, Scaler, and Label Encoder.

### 3. Inference
- `app.py` : Run using `streamlit run app.py`. It uses Streamlit and OpenCV to give you a real-time responsive browser window.
- `mp_mlp_final.py` : Run locally from node/terminal to get a lightweight OpenCV window interface purely reliant on keyboard shortcuts.

---

## 🚀 How to Run

### **Prerequisites**
You need Python 3.9+ installed along with the following libraries:
```bash
pip install opencv-python mediapipe pandas numpy scikit-learn pyttsx3 streamlit ollama
```
*(Also ensure you have [Ollama](https://ollama.com/) installed locally on your system and have downloaded a lightweight model like `qwen2:0.5b` using `ollama run qwen2:0.5b`).*

### **1. Build/Add Your Dataset**
If you want to train it on your own face/hands:
```bash
python Code/mp_record_live.py
```
Follow the console prompts to record your gestures.

### **2. Train the Model**
Once you have your `landmarks_data_2hands.csv` filled:
```bash
python Code/mp_mlp_trainer.py
```
*(This generates the `mp_mlp_model.pkl` weights)*.

### **3. Start Translating!**
To see the glossy front-end Streamlit app:
```bash
cd Code
streamlit run app.py
```

---

## 🤝 Collaborating

1. Ask your collaborator to clone the repo: `git clone https://github.com/WhiteHorse2209/SigN-LanguagE.git`
2. Create feature branches: `git checkout -b feature-new-sign`
3. Push changes and review them together!

Enjoy bridging the gap of communication!
