# 🎭 Speech Emotion Recognition

An AI-powered application that detects emotions from voice/speech using Machine Learning.

---

## 📌 Features

- ✅ Detects **8 emotions**: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
- ✅ **Web Interface** using Streamlit
- ✅ **Adaptive Learning** - Model improves from user feedback
- ✅ **Waveform Visualization** of uploaded audio
- ✅ **Confidence Scores** for all emotions

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| ML Model | MLP Classifier (scikit-learn) |
| Audio Processing | Librosa |
| Web Framework | Streamlit |
| Visualization | Plotly |
| Dataset | RAVDESS (1,441 audio samples) |


---

## 🚀 Installation

### 1. Clone or Download the project

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Web App (Streamlit)
```bash
streamlit run app.py
```
Then open: http://localhost:8501

### CLI Version
```bash
python speech_emotion_recognition.py
```

---

## 📖 How to Use (Web App)

1. **Train Model** → Click "🚀 Train Model" in sidebar
2. **Upload Audio** → Drag & drop WAV/MP3 file
3. **Analyze** → Click "🔮 Analyze Emotion"
4. **View Results** → See predicted emotion with confidence

### Adaptive Learning (Optional)
- If prediction is wrong, select correct emotion
- Click "Save for Retraining"
- Click "🔄 Retrain with Feedback" to improve model

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~55-70% |
| Dataset Size | 1,441 samples |
| Emotions | 8 classes |
| Features | MFCC, Chroma, Mel, Spectral |

---

## 🎵 Dataset Info (RAVDESS)

- **Full Name**: Ryerson Audio-Visual Database of Emotional Speech and Song
- **Samples**: 1,441 audio files
- **Actors**: 24 professional actors (12 male, 12 female)
- **Emotions**: 8 (neutral, calm, happy, sad, angry, fearful, disgust, surprised)

### File Naming Convention
```
03-01-03-02-01-02-20.wav
│  │  │  │  │  │  │
│  │  │  │  │  │  └── Actor (01-24)
│  │  │  │  │  └──── Repetition (01-02)
│  │  │  │  └─────── Statement (01-02)
│  │  │  └────────── Intensity (01=normal, 02=strong)
│  │  └───────────── Emotion (01-08)
│  └──────────────── Vocal channel (01=speech, 02=song)
└─────────────────── Modality (03=audio-only)
```

**Emotion Codes:**
- 01 = Neutral
- 02 = Calm
- 03 = Happy
- 04 = Sad
- 05 = Angry
- 06 = Fearful
- 07 = Disgust
- 08 = Surprised

---

## 🔧 Requirements

```
streamlit
numpy<2.0
librosa
soundfile
scikit-learn
plotly
joblib
```