"""
🎭 Speech Emotion Recognition - Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import os
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
st.set_page_config(page_title="🎭 Speech Emotion Recognition", page_icon="🎭", layout="wide")

DATASET_PATH = "dataset/Audio_Speech_Actors_01-24"
MODEL_DIR = "models"
SAMPLE_RATE = 22050

EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

EMOTION_EMOJIS = {
    'neutral': '😐', 'calm': '😌', 'happy': '😊', 'sad': '😢',
    'angry': '😠', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲'
}

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;
    }
    .main-header h1 { color: white; font-size: 2.5rem; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.9); margin-top: 0.5rem; }
    .emotion-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px; padding: 2rem; text-align: center; margin: 1rem 0;
    }
    .emotion-result h2 { color: white; font-size: 2.5rem; margin: 0; text-transform: uppercase; }
    .emotion-emoji { font-size: 4rem; }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 20px; padding: 0.5rem 2rem;
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
if 'model' not in st.session_state:
    st.session_state.model = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = 0
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []  # Store user feedback for retraining
if 'last_result' not in st.session_state:
    st.session_state.last_result = None  # Store last prediction result
if 'last_audio' not in st.session_state:
    st.session_state.last_audio = None
if 'last_sr' not in st.session_state:
    st.session_state.last_sr = None

FEEDBACK_DIR = "feedback_data"  # Directory to save feedback audio

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
def extract_features(audio_data, sr=SAMPLE_RATE):
    """Extract audio features for emotion recognition."""
    if len(audio_data) < sr * 0.5:
        audio_data = np.pad(audio_data, (0, int(sr * 0.5) - len(audio_data)), mode='constant')
    
    features = []
    
    # MFCC (40) - Most important for emotion
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    features.append(np.mean(mfccs.T, axis=0))
    features.append(np.std(mfccs.T, axis=0))  # Add standard deviation
    
    # Delta MFCC (40) - Captures dynamics
    delta_mfccs = librosa.feature.delta(mfccs)
    features.append(np.mean(delta_mfccs.T, axis=0))
    
    # Delta-Delta MFCC (40) - Acceleration
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    features.append(np.mean(delta2_mfccs.T, axis=0))
    
    # Chroma (12)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    features.append(np.mean(chroma.T, axis=0))
    
    # Spectral Contrast (7)
    spec_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
    features.append(np.mean(spec_contrast.T, axis=0))
    
    # ZCR - mean and std
    zcr = librosa.feature.zero_crossing_rate(audio_data)
    features.append(np.array([np.mean(zcr), np.std(zcr)]))
    
    # RMS Energy - mean and std
    rms = librosa.feature.rms(y=audio_data)
    features.append(np.array([np.mean(rms), np.std(rms)]))
    
    # Mel Spectrogram (128)
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
    features.append(np.mean(mel.T, axis=0))
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)
    
    features.append(np.array([
        np.mean(spectral_centroid), np.std(spectral_centroid),
        np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
        np.mean(spectral_rolloff), np.std(spectral_rolloff),
        np.mean(spectral_flatness), np.std(spectral_flatness)
    ]))
    
    # Pitch/F0 features
    pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    features.append(np.array([pitch_mean, pitch_std]))
    
    return np.hstack(features)

# ============================================================================
# LOAD DATASET
# ============================================================================
def load_dataset():
    """Load RAVDESS dataset."""
    x, y = [], []
    
    if not os.path.exists(DATASET_PATH):
        return np.array(x), np.array(y)
    
    for actor_folder in sorted(os.listdir(DATASET_PATH)):
        actor_path = os.path.join(DATASET_PATH, actor_folder)
        if not os.path.isdir(actor_path):
            continue
        
        for file in os.listdir(actor_path):
            if not file.endswith(".wav"):
                continue
            
            parts = file.split("-")
            if len(parts) < 3:
                continue
            
            emotion = EMOTIONS.get(parts[2])
            if not emotion:
                continue
            
            try:
                audio, sr = librosa.load(os.path.join(actor_path, file), duration=3, offset=0.5, sr=SAMPLE_RATE)
                features = extract_features(audio, sr)
                x.append(features)
                y.append(emotion)
            except:
                continue
    
    return np.array(x), np.array(y)

# ============================================================================
# TRAIN MODEL
# ============================================================================
def train_model(x, y):
    """Train MLP model with enhanced architecture."""
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Enhanced model with deeper layers and regularization
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(512, 256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,  # L2 regularization to prevent overfitting
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=42
        )
    )
    model.fit(x_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(x_test)) * 100
    return model, label_encoder, accuracy

# ============================================================================
# PREDICT
# ============================================================================
def predict_emotion(model, label_encoder, audio_data, sr=SAMPLE_RATE):
    """Predict emotion from audio."""
    features = extract_features(audio_data, sr).reshape(1, -1)
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    
    emotion = label_encoder.inverse_transform([pred])[0]
    confidence = np.max(proba)
    all_probs = dict(zip(label_encoder.classes_, proba))
    
    return emotion, confidence, all_probs

# ============================================================================
# ADAPTIVE LEARNING - Save feedback data
# ============================================================================
def save_feedback_audio(audio_data, sr, correct_emotion):
    """Save audio with correct emotion label for future retraining."""
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    
    # Generate unique filename with emotion label
    import time
    filename = f"{correct_emotion}_{int(time.time())}.wav"
    filepath = os.path.join(FEEDBACK_DIR, filename)
    
    sf.write(filepath, audio_data, sr)
    return filepath

def load_feedback_dataset():
    """Load feedback data for retraining."""
    x, y = [], []
    
    if not os.path.exists(FEEDBACK_DIR):
        return np.array(x), np.array(y)
    
    for file in os.listdir(FEEDBACK_DIR):
        if not file.endswith('.wav'):
            continue
        
        # Extract emotion from filename (emotion_timestamp.wav)
        emotion = file.split('_')[0]
        if emotion not in EMOTION_EMOJIS:
            continue
        
        try:
            audio, sr = librosa.load(os.path.join(FEEDBACK_DIR, file), duration=3, offset=0.5, sr=SAMPLE_RATE)
            features = extract_features(audio, sr)
            x.append(features)
            y.append(emotion)
        except:
            continue
    
    return np.array(x), np.array(y)

def retrain_with_feedback(x_orig, y_orig, x_feedback, y_feedback):
    """Retrain model with original + feedback data."""
    # Combine datasets
    x_combined = np.vstack([x_orig, x_feedback]) if len(x_feedback) > 0 else x_orig
    y_combined = np.hstack([y_orig, y_feedback]) if len(y_feedback) > 0 else y_orig
    
    # Train with combined data
    return train_model(x_combined, y_combined)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎭 Speech Emotion Recognition</h1>
        <p>AI-Powered Emotion Detection from Voice</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        st.markdown("---")
        
        # Model Status
        if st.session_state.trained:
            st.success(f"✅ Model Ready! ({st.session_state.accuracy:.1f}%)")
        else:
            st.warning("⚠️ Model not trained")
        
        st.markdown("---")
        
        # Train Button
        if st.button("🚀 Train Model", use_container_width=True):
            with st.spinner("Loading dataset..."):
                x, y = load_dataset()
            
            if len(x) == 0:
                st.error("❌ Dataset not found!")
            else:
                st.info(f"📊 Loaded {len(x)} samples")
                
                with st.spinner("Training model..."):
                    model, le, acc = train_model(x, y)
                
                st.session_state.model = model
                st.session_state.label_encoder = le
                st.session_state.trained = True
                st.session_state.accuracy = acc
                
                st.success(f"✅ Done! Accuracy: {acc:.1f}%")
                st.rerun()
        
        st.markdown("---")
        
        # Save/Load
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save", use_container_width=True):
                if st.session_state.model:
                    os.makedirs(MODEL_DIR, exist_ok=True)
                    joblib.dump(st.session_state.model, f"{MODEL_DIR}/model.pkl")
                    joblib.dump(st.session_state.label_encoder, f"{MODEL_DIR}/encoder.pkl")
                    st.success("✅ Saved!")
        
        with col2:
            if st.button("📂 Load", use_container_width=True):
                try:
                    st.session_state.model = joblib.load(f"{MODEL_DIR}/model.pkl")
                    st.session_state.label_encoder = joblib.load(f"{MODEL_DIR}/encoder.pkl")
                    st.session_state.trained = True
                    st.success("✅ Loaded!")
                    st.rerun()
                except:
                    st.error("❌ No model!")
        
        st.markdown("---")
        
        # Adaptive Learning - Retrain with Feedback
        st.markdown("### 🧠 Adaptive Learning")
        feedback_count = len(os.listdir(FEEDBACK_DIR)) if os.path.exists(FEEDBACK_DIR) else 0
        st.caption(f"Feedback samples: {feedback_count}")
        
        if st.button("🔄 Retrain with Feedback", use_container_width=True):
            if feedback_count == 0:
                st.warning("⚠️ No feedback data yet!")
            else:
                with st.spinner("Loading original dataset..."):
                    x_orig, y_orig = load_dataset()
                with st.spinner("Loading feedback data..."):
                    x_fb, y_fb = load_feedback_dataset()
                
                st.info(f"📊 Original: {len(x_orig)} + Feedback: {len(x_fb)} = {len(x_orig)+len(x_fb)} samples")
                
                with st.spinner("Retraining model..."):
                    model, le, acc = retrain_with_feedback(x_orig, y_orig, x_fb, y_fb)
                
                st.session_state.model = model
                st.session_state.label_encoder = le
                st.session_state.accuracy = acc
                st.success(f"✅ Retrained! New Accuracy: {acc:.1f}%")
                st.rerun()
    
    # Main Content
    st.markdown("### 📁 Upload Audio File")
    uploaded_file = st.file_uploader("Choose a WAV file", type=['wav', 'mp3', 'ogg', 'flac'])
    
    if uploaded_file:
        # Load audio
        audio_data, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE, duration=5)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.audio(uploaded_file)
            
            # Waveform
            time = np.linspace(0, len(audio_data) / sr, len(audio_data))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time, y=audio_data, mode='lines', 
                                     line=dict(color='#667eea', width=1),
                                     fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.3)'))
            fig.update_layout(title="🎵 Waveform", height=200, 
                              xaxis_title="Time (s)", yaxis_title="Amplitude",
                              margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if st.button("🔮 Analyze Emotion", use_container_width=True):
                if not st.session_state.trained:
                    st.error("⚠️ Please train the model first!")
                else:
                    with st.spinner("Analyzing..."):
                        emotion, confidence, all_probs = predict_emotion(
                            st.session_state.model,
                            st.session_state.label_encoder,
                            audio_data, sr
                        )
                    
                    # Store result in session state so it persists
                    st.session_state.last_audio = audio_data.copy()
                    st.session_state.last_sr = sr
                    st.session_state.last_result = {
                        'emotion': emotion,
                        'confidence': confidence,
                        'all_probs': all_probs
                    }
            
            # Show results if we have them (persists after button click)
            if st.session_state.last_result:
                result = st.session_state.last_result
                emotion = result['emotion']
                confidence = result['confidence']
                all_probs = result['all_probs']
                
                # Result
                st.markdown(f"""
                <div class="emotion-result">
                    <div class="emotion-emoji">{EMOTION_EMOJIS.get(emotion, '🎭')}</div>
                    <h2>{emotion}</h2>
                    <p style="color:white; font-size:1.2rem;">Confidence: {confidence*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Bar chart
                emotions = list(all_probs.keys())
                probs = [all_probs[e] * 100 for e in emotions]
                colors = ['#667eea' if e == emotion else '#764ba2' for e in emotions]
                
                fig = go.Figure(data=[go.Bar(x=emotions, y=probs, marker_color=colors,
                                              text=[f'{p:.1f}%' for p in probs], textposition='auto')])
                fig.update_layout(title="📊 All Emotions", height=300, yaxis_range=[0, 100],
                                  margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Feedback section - ADAPTIVE LEARNING
                st.markdown("---")
                st.markdown("### 🔄 Was this prediction correct?")
                st.caption("If wrong, select the correct emotion and save to improve the model!")
                
                col_fb1, col_fb2 = st.columns([1, 2])
                with col_fb1:
                    correct_emotion = st.selectbox(
                        "Select correct emotion:",
                        list(EMOTION_EMOJIS.keys()),
                        index=list(EMOTION_EMOJIS.keys()).index(emotion) if emotion in EMOTION_EMOJIS else 0,
                        key="feedback_emotion"
                    )
                with col_fb2:
                    if st.button("✅ Save for Retraining", use_container_width=True, key="save_feedback"):
                        if st.session_state.last_audio is not None:
                            filepath = save_feedback_audio(st.session_state.last_audio, st.session_state.last_sr, correct_emotion)
                            st.success(f"✅ Saved as '{correct_emotion}' emotion!")
                            st.info(f"📁 File: {filepath}")
                            st.balloons()
                        else:
                            st.error("❌ No audio to save!")
    
    else:
        # Show instructions when no file uploaded
        st.info("👆 Upload an audio file to analyze emotions. Train the model first using the sidebar.")
        
        # Quick test with sample from dataset
        st.markdown("### 🎵 Or Test with Dataset Sample")
        if st.button("🎲 Test Random Sample", use_container_width=True):
            if not st.session_state.trained:
                st.error("⚠️ Please train the model first!")
            elif not os.path.exists(DATASET_PATH):
                st.error("❌ Dataset not found!")
            else:
                # Get random file
                import random
                actors = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
                actor = random.choice(actors)
                files = [f for f in os.listdir(os.path.join(DATASET_PATH, actor)) if f.endswith('.wav')]
                file = random.choice(files)
                filepath = os.path.join(DATASET_PATH, actor, file)
                
                # Get true emotion
                true_emotion = EMOTIONS.get(file.split('-')[2], 'unknown')
                
                # Load and predict
                audio, sr = librosa.load(filepath, duration=3, offset=0.5, sr=SAMPLE_RATE)
                emotion, confidence, all_probs = predict_emotion(
                    st.session_state.model, st.session_state.label_encoder, audio, sr
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**True Emotion:** {EMOTION_EMOJIS.get(true_emotion, '')} {true_emotion}")
                with col2:
                    st.markdown(f"**Predicted:** {EMOTION_EMOJIS.get(emotion, '')} {emotion} ({confidence*100:.1f}%)")
                
                if true_emotion == emotion:
                    st.success("✅ Correct!")
                else:
                    st.warning("❌ Wrong prediction")

if __name__ == "__main__":
    main()
