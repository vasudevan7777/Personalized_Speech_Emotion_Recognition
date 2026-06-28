# 🎭 Speech Emotion Recognition & Wellness Dashboard

Detect emotions from voice and speech, analyze acoustic features, and provide personalized mental wellness recommendations using Deep Learning.

---

## 📖 Project Description

This is an end-to-end Machine Learning and audio digital signal processing (DSP) application designed to analyze spoken voice recordings, identify emotional states, and provide actionable mental wellness coping strategies. 

By analyzing raw audio clips, extracting high-dimensional speech features, and running predictions through a customized deep neural network, the system detects primary emotions with their confidence probabilities. Additionally, it computes acoustic arousal metrics (like RMS energy and pitch variance) to evaluate an emotional risk score, helping users track and monitor mental wellness trends over time.

---

## 💡 Solution Approach

Our solution follows a modular speech processing and deep learning pipeline:

1. **Data Preprocessing & Feature Extraction** ([features.py](file:///e:/SER%20PROJECT/features.py)):
   - Normalizes audio amplitude and trims silent regions to avoid noise bias using [preprocess_audio](file:///e:/SER%20PROJECT/features.py#L10).
   - Runs audio quality diagnostics via [audio_quality_check](file:///e:/SER%20PROJECT/features.py#L24) (detects low volume, duration, silence, and background noise).
   - Extracts a robust **549-dimensional** feature vector containing means and standard deviations of MFCCs (40), Chroma (12), Spectral Contrast (7), Zero Crossing Rate (1), RMS Energy (1), and Mel Spectrogram (128).
2. **Deep Learning Model Training** ([train_model.py](file:///e:/SER%20PROJECT/train_model.py)):
   - Loads and augments the RAVDESS dataset in parallel using [joblib.Parallel](file:///e:/SER%20PROJECT/train_model.py#L150-L153) across all CPU cores (noise insertion + speed stretching).
   - Trains a deep Multi-Layer Perceptron (MLP) Classifier using TensorFlow/Keras.
   - Core Architecture: 512 → 256 → 128 → 64 dense layers with Batch Normalization, Dropout (0.35–0.25), and L2 regularization to prevent overfitting.
   - Optimizations: EarlyStopping and ReduceLROnPlateau learning rate decay for fast, stable convergence (~8–15 minutes on CPU).
3. **Core Evaluation & Inference** ([speech_emotion_recognition.py](file:///e:/SER%20PROJECT/speech_emotion_recognition.py)):
   - Implements the [KerasEmotionModel](file:///e:/SER%20PROJECT/speech_emotion_recognition.py#L112) wrapper along with custom [SimpleStandardScaler](file:///e:/SER%20PROJECT/speech_emotion_recognition.py#L34) and [SimpleLabelEncoder](file:///e:/SER%20PROJECT/speech_emotion_recognition.py#L19).
   - Sets low (0.45) and moderate (0.60) confidence thresholds to provide visual warnings on uncertain classifications.
4. **Interactive Dashboard & Wellness Engine** ([app.py](file:///e:/SER%20PROJECT/app.py), [dashboard.py](file:///e:/SER%20PROJECT/dashboard.py), [wellness_engine.py](file:///e:/SER%20PROJECT/wellness_engine.py)):
   - Provides a glassmorphic Streamlit interface allowing real-time microphone recording or WAV/MP3 uploads.
   - Calculates emotional risk (0–100) via [calculate_emotional_risk](file:///e:/SER%20PROJECT/wellness_engine.py#L51) by combining prediction confidence with acoustic arousal features.
   - Offers dynamic coping recommendations based on the detected emotion.
   - Renders interactive Plotly donut charts and speedometer risk gauges.

---

## 🛠️ Prerequisites & Dependencies

- **Python version**: Python 3.10 or higher.
- **Dependencies**: Listed in [requirements.txt](file:///e:/SER%20PROJECT/requirements.txt):
  - `streamlit` (Interactive web UI framework)
  - `tensorflow` (Deep learning framework for Keras MLP)
  - `librosa` & `soundfile` & `sounddevice` (Audio analysis, reading, and recording)
  - `scikit-learn` & `joblib` (Preprocessing, model serialization, and parallel workers)
  - `plotly` & `matplotlib` & `seaborn` (Data visualizations and dashboards)

---

## 💻 Setup & Usage Instructions

### 1. Environment Setup

Clone the repository and run the following in your terminal:

```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# On Windows (CMD)
.\venv\Scripts\activate.bat
# On macOS/Linux
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Usage — Running the Project

#### Run the Training Pipeline
To extract features, apply data augmentations, train the MLP model, and save it under the `models/` directory, run:
```bash
python train_model.py
```

#### Run the Streamlit Dashboard
To launch the interactive web application, run:
```bash
streamlit run app.py
```
*Alternatively, on Windows, you can execute the runner script:*
```powershell
.\run.ps1
```
Then open: **`http://localhost:8501`** in your browser.

#### Running the CLI Version
For a fast command-line inference utility:
```bash
python speech_emotion_recognition.py
```

---

## 📈 Model Performance & Results

| Metric / Attribute | Value / Details |
| :--- | :--- |
| **Model Type** | Multi-Layer Perceptron (MLP) Classifier (Keras) |
| **Accuracy** | ~65–75% (on test split with feature augmentations) |
| **Dataset Size** | 1,441 audio samples (expanded via noise/stretch augmentation) |
| **Emotions Detected** | 8 classes (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised) |
| **Features Input** | 549 dimensions (mean & standard deviation of audio banks) |

### RAVDESS File Naming Convention
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
*Emotion Codes: 01 = Neutral, 02 = Calm, 03 = Happy, 04 = Sad, 05 = Angry, 06 = Fearful, 07 = Disgust, 08 = Surprised.*

---

## 🌟 Standout Features (What Makes This Project Special)

1. **Adaptive Learning (Feedback Loop)**:
   Users can correct misclassifications directly from the web interface. These samples are logged and saved into the `feedback_data/` folder, allowing developers to trigger retraining to resolve model edge-cases.
2. **Explainable Emotional Risk Score**:
   Rather than a simple class output, the app estimates emotional distress dynamically. It blends prediction probability with acoustic arousal (loudness levels and pitch standard deviation) to output a detailed breakdown of the emotional risk score (0-100).
3. **Tailored Coping Interventions**:
   The integrated wellness engine suggests customized mindfulness practices, breathing exercises, or recommendations depending on the state of the user.
4. **Rich Plotly Visualizations**:
   Includes dynamic, responsive gauges showing risk score categories (Low, Moderate, High, Extreme) and donut charts representing overall emotion distribution history.