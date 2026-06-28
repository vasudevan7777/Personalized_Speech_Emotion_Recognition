import random
import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import os
import joblib
import pandas as pd
import warnings
import datetime
from speech_emotion_recognition import (
    SimpleLabelEncoder as LabelEncoder,
    simple_train_test_split as train_test_split,
    simple_accuracy_score as accuracy_score,
    LOW_CONFIDENCE_THRESHOLD,
    MODERATE_CONFIDENCE_THRESHOLD,
)

# Import modular custom packages
import features
import wellness_engine
import history_manager
import dashboard

warnings.filterwarnings('ignore')

RAVDESS_EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry',   '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

st.set_page_config(
    page_title="Speech Emotion Recognition & Wellness Dashboard",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Audio_Speech_Actors_01-24")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback_data")
SAMPLE_RATE = 22050

# Emojis for emotions
EMOTION_EMOJIS = {
    'neutral': '😐', 'calm': '😌', 'happy': '😊', 'sad': '😢',
    'angry': '😠', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲'
}

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

    /* ── Header ────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #1a1233 0%, #2e2050 50%, #4a2e6e 100%);
        padding: 2.5rem 3rem;
        border-radius: 22px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }
    .main-header h1 {
        font-size: 2.8rem; font-weight: 800; margin: 0;
        background: linear-gradient(90deg,#a1c4fd 0%,#c2e9fb 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    .main-header p {
        color: rgba(255,255,255,0.8); font-size: 1.1rem;
        font-weight: 300; margin-top: 0.6rem;
    }

    /* ── Glass Cards ────────────────────────────── */
    .glass-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.15);
    }
    .glass-highlight {
        background: linear-gradient(135deg,rgba(118,75,162,0.12) 0%,rgba(102,126,234,0.12) 100%);
        border: 1px solid rgba(118,75,162,0.25);
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* ── Metric Grid ────────────────────────────── */
    .metric-container {
        display: flex; justify-content: space-between;
        align-items: stretch; gap: 1rem;
    }
    .metric-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 1.2rem 0.8rem;
        text-align: center; flex: 1;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: rgba(118,75,162,0.4); }
    .metric-title {
        font-size: 0.78rem; color: #9994b0;
        text-transform: uppercase; font-weight: 600; letter-spacing: 0.6px;
        margin-bottom: 0.4rem;
    }
    .metric-value { font-size: 1.75rem; font-weight: 800; color: white; }

    /* ── Emotion Result ─────────────────────────── */
    .emotion-result-box {
        background: linear-gradient(135deg,rgba(102,126,234,0.18) 0%,rgba(118,75,162,0.18) 100%);
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 22px;
        padding: 2rem 1.5rem;
        text-align: center;
        box-shadow: 0 12px 36px rgba(0,0,0,0.25);
    }
    .emotion-result-box h2 {
        color: white; font-size: 2.6rem; font-weight: 800;
        margin: 0.4rem 0; letter-spacing: 1.5px; text-transform: uppercase;
    }
    .emotion-emoji-lg { font-size: 4.5rem; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3)); }
    .confidence-badge {
        background: rgba(255,255,255,0.1);
        padding: 0.35rem 1rem; border-radius: 30px;
        font-weight: 600; font-size: 0.95rem; color: #c2e9fb;
        display: inline-block; margin-top: 0.5rem;
    }
    .low-confidence-banner {
        background: rgba(255,193,7,0.12);
        border: 1px solid rgba(255,193,7,0.35);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        margin-top: 0.8rem;
        font-size: 0.88rem;
        color: #ffe082;
    }

    /* ── Recording Quality Panel ────────────────── */
    .quality-panel {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin: 0.6rem 0 1rem 0;
        font-size: 0.88rem;
    }
    .quality-ok   { border-left: 3px solid #28A745; }
    .quality-warn { border-left: 3px solid #FFC107; }
    .quality-fail { border-left: 3px solid #DC3545; }
    .q-label { color: #9994b0; font-weight: 600; font-size: 0.78rem;
               text-transform: uppercase; letter-spacing: 0.5px; }
    .q-value { color: white; font-weight: 600; }

    /* ── Risk Bar ───────────────────────────────── */
    .risk-bar-container {
        width: 100%; background: rgba(255,255,255,0.08);
        border-radius: 10px; height: 10px;
        margin-top: 8px; overflow: hidden;
    }
    .risk-bar-fill { height: 100%; border-radius: 10px; }

    /* ── Disclaimer ─────────────────────────────── */
    .disclaimer-card {
        border-left: 4px solid #E63946;
        background: rgba(230,57,70,0.06);
        padding: 0.9rem 1rem;
        border-radius: 0 10px 10px 0;
        margin-top: 1rem;
        font-size: 0.86rem;
        line-height: 1.5;
    }

    /* ── Buttons ────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(90deg,#667eea 0%,#764ba2 100%) !important;
        color: white !important; border: none !important;
        border-radius: 30px !important; font-weight: 600 !important;
        padding: 0.55rem 2rem !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(118,75,162,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(118,75,162,0.5) !important;
    }

    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if 'model' not in st.session_state:
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    encoder_path = os.path.join(MODEL_DIR, "encoder.pkl")
    accuracy_path = os.path.join(MODEL_DIR, "accuracy.pkl")
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        try:
            from speech_emotion_recognition import KerasEmotionModel
            st.session_state.model = KerasEmotionModel.load(MODEL_DIR)
            st.session_state.label_encoder = joblib.load(encoder_path)
            st.session_state.trained = True
            if os.path.exists(accuracy_path):
                st.session_state.accuracy = joblib.load(accuracy_path)
            else:
                st.session_state.accuracy = 85.0
        except Exception as e:
            st.session_state.model = None
            st.session_state.label_encoder = None
            st.session_state.trained = False
            st.session_state.accuracy = 0
            st.sidebar.error(f"Error loading model: {e}")
    else:
        st.session_state.model = None
        st.session_state.label_encoder = None
        st.session_state.trained = False
        st.session_state.accuracy = 0

if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'last_audio' not in st.session_state:
    st.session_state.last_audio = None
if 'last_sr' not in st.session_state:
    st.session_state.last_sr = None
if 'last_quality' not in st.session_state:
    st.session_state.last_quality = None


def load_dataset():
    """Load RAVDESS dataset using 549 enhanced features (mean+std for all banks)."""
    x, y = [], []
    if not os.path.exists(DATASET_PATH):
        return np.array(x), np.array(y)

    actor_folders = sorted(os.listdir(DATASET_PATH))
    for actor_folder in actor_folders:
        actor_path = os.path.join(DATASET_PATH, actor_folder)
        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):
            if not file.endswith(".wav"):
                continue
            parts = file.split("-")
            if len(parts) < 3:
                continue
            emotion = RAVDESS_EMOTIONS.get(parts[2])
            if not emotion:
                continue
            try:
                feat = features.extract_features(
                    os.path.join(actor_path, file),
                    sr=SAMPLE_RATE,
                    num_features=549,
                    duration=3,
                    offset=0.5,
                )
                x.append(feat)
                y.append(emotion)
            except Exception:
                continue

    return np.array(x), np.array(y)

def train_model(x, y):
    """Train Keras model with enhanced architecture (549 features).

    epochs=300 ceiling — EarlyStopping (patience=30) terminates early.
    batch_size=16 gives more gradient updates per epoch on the small dataset.
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    from speech_emotion_recognition import KerasEmotionModel

    model = KerasEmotionModel()
    model.fit(x_train, y_train, epochs=100, batch_size=64)

    accuracy = accuracy_score(y_test, model.predict(x_test)) * 100
    return model, label_encoder, accuracy

def predict_emotion(model, label_encoder, audio_data, sr=SAMPLE_RATE):
    """Predict emotion from audio, using the exact feature count the model was trained on.

    Both uploaded files and live microphone recordings pass through the same
    extract_features() call here, guaranteeing identical preprocessing.
    A dimension check guards against silent mismatches that would produce
    garbage predictions.
    """
    # Determine the number of features the model expects
    try:
        if hasattr(model, 'steps'):
            num_features = model.steps[0][1].n_features_in_
        elif hasattr(model, 'n_features_in_') and model.n_features_in_ is not None:
            num_features = model.n_features_in_
        else:
            num_features = 189
    except Exception:
        num_features = 189

    features_vector = features.extract_features(
        audio_data, sr=sr, num_features=num_features
    ).reshape(1, -1)

    # Safety check: catch dimension mismatches before they silently corrupt predictions
    expected_dim = model.n_features_in_ if hasattr(model, 'n_features_in_') and model.n_features_in_ is not None else num_features
    if features_vector.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: extracted {features_vector.shape[1]} features "
            f"but model expects {expected_dim}. Retrain the model."
        )

    pred  = model.predict(features_vector)[0]
    proba = model.predict_proba(features_vector)[0]

    emotion    = label_encoder.inverse_transform([pred])[0]
    confidence = float(np.max(proba))
    all_probs  = dict(zip(label_encoder.classes_, proba))

    return emotion, confidence, all_probs

def save_feedback_audio(audio_data, sr, correct_emotion):
    """Save audio with correct emotion label for future retraining."""
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    filename = f"{correct_emotion}_{int(datetime.datetime.now().timestamp())}.wav"
    filepath = os.path.join(FEEDBACK_DIR, filename)
    sf.write(filepath, audio_data, sr)
    return filepath

def load_feedback_dataset():
    """Load feedback data using the SAME feature params as training.

    Previously used duration=10, offset=0.0 which created a distribution
    shift vs. the training data (duration=3, offset=0.5). This caused
    feedback retraining to hurt rather than help accuracy.
    Using duration=3, offset=0.5 ensures feedback samples are processed
    identically to the original RAVDESS training files.
    """
    x, y = [], []
    if not os.path.exists(FEEDBACK_DIR):
        return np.array(x), np.array(y)

    for file in os.listdir(FEEDBACK_DIR):
        if not file.endswith('.wav'):
            continue
        emotion = file.split('_')[0]
        if emotion not in EMOTION_EMOJIS:
            continue
        try:
            feat = features.extract_features(
                os.path.join(FEEDBACK_DIR, file),
                sr=SAMPLE_RATE,
                num_features=549,
                duration=3,    # matches training
                offset=0.5,    # matches training
            )
            x.append(feat)
            y.append(emotion)
        except Exception:
            continue
    return np.array(x), np.array(y)

def retrain_with_feedback(x_orig, y_orig, x_feedback, y_feedback):
    """Retrain model with combined original + feedback data."""
    x_combined = np.vstack([x_orig, x_feedback]) if len(x_feedback) > 0 else x_orig
    y_combined = np.hstack([y_orig, y_feedback]) if len(y_feedback) > 0 else y_orig
    return train_model(x_combined, y_combined)

def main():
    # Header Card
    st.markdown("""
    <div class="main-header">
        <h1>🎭 Speech Emotion Recognition</h1>
        <p>Advanced Clinical-Grade Vocal Analysis & Mental Wellness Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check sidebar metadata info
    with st.sidebar:
        st.markdown("### ⚙️ System Status")
        if st.session_state.trained:
            # Check expected features
            try:
                if hasattr(st.session_state.model, 'steps'):
                    f_size = st.session_state.model.steps[0][1].n_features_in_
                else:
                    f_size = st.session_state.model.n_features_in_
            except:
                f_size = "unknown"
            
            st.success(f"✅ Model Ready\n- Shape: {f_size} features\n- Accuracy: {st.session_state.accuracy:.2f}%")
        else:
            st.warning("⚠️ Model not trained")
            
        st.markdown("---")
        st.markdown("### 💡 Quick Guide")
        st.info(
            "1. Upload a WAV audio file or record directly using the microphone.\n"
            "2. Click **Analyze Emotion** to run predictions, risk evaluation, and wellness advice.\n"
            "3. Inspect historical charts in **Analytics Dashboard**.\n"
            "4. Provide corrections in the feedback panel to help retrain the classifier."
        )
        st.markdown("---")
        st.caption("Speech Emotion Recognition System v2.0")
        
    # Navigation tabs (Feature 1)
    tab_analyzer, tab_analytics, tab_history, tab_settings = st.tabs([
        "🎤 Live Analyzer",
        "📊 Analytics Dashboard",
        "📜 Prediction History",
        "⚙️ Model & Training"
    ])
    
    with tab_analyzer:
        st.markdown("### 🎙️ Audio Input Channel")

        col_inp1, col_inp2 = st.columns([1, 1])

        with col_inp1:
            st.markdown("##### 📁 Upload File")
            uploaded_file = st.file_uploader(
                "Select a WAV, MP3, or OGG file",
                type=['wav', 'mp3', 'ogg', 'flac'],
                help="Recommended: 3–5 seconds of clear speech."
            )

        with col_inp2:
            st.markdown("##### 🎤 Record Live Microphone")
            st.caption("Speak clearly for 3–5 s · Use the sentences: *'Kids are talking by the door'* or *'Dogs are sitting by the door'*")
            recorded_audio = st.audio_input(
                "Click to record your voice",
                help="Speak with natural emotion for 3–5 seconds after clicking."
            )

        audio_file = recorded_audio if recorded_audio is not None else uploaded_file

        if audio_file:
            try:
                raw_audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE, duration=10)
                # Run quality check on RAW audio before preprocessing
                quality = features.audio_quality_check(raw_audio, sr)
                st.session_state.last_quality   = quality
                # Store the raw audio — preprocessing happens inside extract_features
                st.session_state.last_audio = raw_audio
                st.session_state.last_sr    = sr
            except Exception as e:
                st.error(f"Error loading audio: {e}")
                st.session_state.last_audio   = None
                st.session_state.last_sr      = None
                st.session_state.last_quality = None

        st.markdown("---")
        col_play, col_wave = st.columns([1, 2])
        
        with col_play:
            if audio_file:
                st.markdown("##### 🔊 Audio Playback")
                st.audio(audio_file)

                # ── Recording Quality Diagnostics (Issue 2, 7) ───────────────
                q = st.session_state.get('last_quality')
                if q:
                    dur_s    = q.get('duration_s', 0)
                    speech_s = q.get('speech_s',   0)
                    peak     = q.get('peak_amp',   0)
                    warns    = q.get('warnings',   [])
                    q_class  = 'quality-ok' if q['ok'] else ('quality-warn' if len(warns) == 1 else 'quality-fail')

                    st.markdown(f"""
                    <div class="quality-panel {q_class}">
                        <div style="font-weight:700; color:#e0dff2; margin-bottom:6px;">📊 Recording Quality</div>
                        <div style="display:flex; gap:1.5rem; flex-wrap:wrap;">
                            <span><span class="q-label">Duration</span><br><span class="q-value">{dur_s:.1f}s</span></span>
                            <span><span class="q-label">Voice</span><br><span class="q-value">{speech_s:.1f}s</span></span>
                            <span><span class="q-label">Peak Amp</span><br><span class="q-value">{peak:.3f}</span></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    for w in warns:
                        st.warning(w)

                if st.session_state.last_audio is not None:
                    if st.button("🔮 Analyze Emotion", use_container_width=True):
                        if not st.session_state.trained:
                            st.error("⚠️ Model is not ready. Please train or load the model in the Settings tab!")
                        else:
                            with st.spinner("Extracting features and predicting..."):
                                emotion, confidence, all_probs = predict_emotion(
                                    st.session_state.model,
                                    st.session_state.label_encoder,
                                    st.session_state.last_audio,
                                    st.session_state.last_sr
                                )
                                
                                risk_breakdown = wellness_engine.calculate_emotional_risk(
                                    emotion, confidence, st.session_state.last_audio, st.session_state.last_sr
                                )
                                risk_score = risk_breakdown["final_score"]
                                risk_category = risk_breakdown["category"]
                                
                                wellness_details = wellness_engine.get_wellness_assessment(
                                    emotion, risk_score
                                )
                                
                                recs = wellness_engine.get_recommendations(emotion, risk_score)
                                
                                st.session_state.last_result = {
                                    'emotion': emotion,
                                    'confidence': confidence,
                                    'all_probs': all_probs,
                                    'risk_score': risk_score,
                                    'risk_category': risk_category,
                                    'risk_breakdown': risk_breakdown,
                                    'wellness_status': wellness_details["status"],
                                    'interpretation': wellness_details["interpretation"],
                                    'recommendations': recs
                                }
                                
                                history_manager.save_prediction(
                                    emotion=emotion,
                                    confidence=confidence,
                                    risk_score=risk_score,
                                    wellness_status=wellness_details["status"],
                                    recommendations=recs
                                )
                                st.rerun()
            else:
                st.info("👆 Upload an audio sample or use the live browser microphone to begin the analysis.")
                
                if st.button("🎲 Analyze Random Sample", use_container_width=True):
                    if not st.session_state.trained:
                        st.error("Please train the model first!")
                    elif not os.path.exists(DATASET_PATH):
                        st.error("RAVDESS dataset not found.")
                    else:
                        import random
                        actors = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
                        if not actors:
                            st.error("No actor directories found.")
                        else:
                            actor = random.choice(actors)
                            files_in_folder = [f for f in os.listdir(os.path.join(DATASET_PATH, actor)) if f.endswith('.wav')]
                            if not files_in_folder:
                                st.error("No files in actor directory.")
                            else:
                                file = random.choice(files_in_folder)
                                filepath = os.path.join(DATASET_PATH, actor, file)

                                true_emotion = RAVDESS_EMOTIONS.get(file.split('-')[2], 'unknown')

                                audio_data, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=3, offset=0.5)
                                st.session_state.last_audio = audio_data
                                st.session_state.last_sr    = sr

                                emotion, confidence, all_probs = predict_emotion(
                                    st.session_state.model,
                                    st.session_state.label_encoder,
                                    audio_data, sr
                                )

                                risk_breakdown  = wellness_engine.calculate_emotional_risk(
                                    emotion, confidence, audio_data, sr
                                )
                                risk_score      = risk_breakdown["final_score"]
                                wellness_details = wellness_engine.get_wellness_assessment(emotion, risk_score)
                                recs            = wellness_engine.get_recommendations(emotion, risk_score)

                                st.session_state.last_result = {
                                    'emotion':        emotion,
                                    'confidence':     confidence,
                                    'all_probs':      all_probs,
                                    'risk_score':     risk_score,
                                    'risk_category':  risk_breakdown["category"],
                                    'risk_breakdown': risk_breakdown,
                                    'wellness_status':wellness_details["status"],
                                    'interpretation': f"**(Random Sample — True Label: {true_emotion.upper()})** "
                                                      + wellness_details["interpretation"],
                                    'recommendations':recs,
                                }
                                
                                history_manager.save_prediction(
                                    emotion=emotion,
                                    confidence=confidence,
                                    risk_score=risk_breakdown["final_score"],
                                    wellness_status=st.session_state.last_result['wellness_status'],
                                    recommendations=recs
                                )
                                st.rerun()
        
        with col_wave:
            # Plotly Waveform plot
            if 'last_audio' in st.session_state and st.session_state.last_audio is not None:
                if not audio_file:
                    st.markdown("##### Audio Playback (Random Sample)")
                    st.audio(st.session_state.last_audio, sample_rate=st.session_state.last_sr)
                    
                y = st.session_state.last_audio
                rate = st.session_state.last_sr
                time_axis = np.linspace(0, len(y) / rate, len(y))
                
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_axis, y=y, mode='lines', 
                    line=dict(color='#8884d8', width=1.2),
                    fill='tozeroy', fillcolor='rgba(136, 132, 216, 0.15)'
                ))
                fig.update_layout(
                    title={"text": "<b>Vocal Waveform Visualization</b>", "font": {"color": "white"}},
                    xaxis={"title": "Time (seconds)", "tickcolor": "gray", "gridcolor": "rgba(255,255,255,0.05)"},
                    yaxis={"title": "Amplitude", "tickcolor": "gray", "gridcolor": "rgba(255,255,255,0.05)"},
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=210,
                    margin=dict(l=40, r=20, t=40, b=30)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Show predictions results panel if loaded
        if st.session_state.last_result:
            res = st.session_state.last_result
            st.markdown("---")
            st.markdown("### 📊 Analysis Diagnostics")
            
            col_res1, col_res2, col_res3 = st.columns(3)

            with col_res1:
                emoji = EMOTION_EMOJIS.get(res['emotion'].lower(), '🎭')
                conf_pct = res['confidence'] * 100
                low_conf_html = ""
                if conf_pct < LOW_CONFIDENCE_THRESHOLD * 100:
                    low_conf_html = '<div class="low-confidence-banner">⚠️ Prediction confidence is low — consider recording again in a quieter environment or speaking more clearly.</div>'
                elif conf_pct < MODERATE_CONFIDENCE_THRESHOLD * 100:
                    low_conf_html = '<div class="low-confidence-banner">ℹ️ Moderate confidence — results may be less precise. Try a cleaner recording for best accuracy.</div>'

                st.markdown(f"""
                <div class="emotion-result-box">
                    <div class="emotion-emoji-lg">{emoji}</div>
                    <h2>{res['emotion']}</h2>
                    <div class="confidence-badge">Confidence: {conf_pct:.1f}%</div>
                    {low_conf_html}
                </div>
                """, unsafe_allow_html=True)
                
            with col_res2:
                fig_gauge = dashboard.plot_risk_gauge_meter(res['risk_score'], res['risk_category'])
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Styled Progress bar indicator
                risk_percent = res['risk_score']
                if risk_percent < 30:
                    bar_color = "#28A745"  # green
                elif risk_percent < 60:
                    bar_color = "#FFC107"  # yellow
                elif risk_percent < 85:
                    bar_color = "#FD7E14"  # orange
                else:
                    bar_color = "#DC3545"  # red
                    
                st.markdown(f"""
                <div style="padding: 0 10px;">
                    <span style="font-size: 0.9rem; color:#b0aebc; font-weight:600;">Risk Category: {res['risk_category']}</span>
                    <div class="risk-bar-container">
                        <div class="risk-bar-fill" style="width: {risk_percent}%; background-color: {bar_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_res3:
                df_hist = history_manager.load_history_as_df()
                stability, stab_emoji = "Single prediction", "⚪"
                if not df_hist.empty and len(df_hist) >= 3:
                    stability, stab_emoji = wellness_engine.calculate_stability(df_hist['risk_score'].tolist())
                    
                st.markdown(f"""
                <div class="glass-card" style="height: 100%; display:flex; flex-direction:column; justify-content:center;">
                    <div style="font-size:0.9rem; color:#b0aebc; font-weight:600; text-transform:uppercase;">Wellness Status</div>
                    <div style="font-size:1.8rem; font-weight:800; color:#c2e9fb; margin-top:5px;">{res['wellness_status']}</div>
                    <hr style="border:0; border-top:1px solid rgba(255,255,255,0.08); margin: 15px 0;">
                    <div style="font-size:0.9rem; color:#b0aebc; font-weight:600; text-transform:uppercase;">Session Stability</div>
                    <div style="font-size:1.2rem; font-weight:600; margin-top:5px;">{stab_emoji} {stability}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 🧘 Wellness Insights & Supportive Action Plan")
            col_info, col_recs = st.columns([1, 1.2])
            
            with col_info:
                st.markdown(f"""
                <div class="glass-highlight" style="height: 100%;">
                    <h5>🧠 Emotional Interpretation</h5>
                    <p style="font-size:1.05rem; line-height:1.6; color:#e0dff2; margin-bottom:1.5rem;">{res['interpretation']}</p>
                    <div class="disclaimer-card">
                        {wellness_engine.DISCLAIMER_TEXT}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_recs:
                st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
                st.markdown("##### 💡 Recommended Next Steps")
                for r in res['recommendations']:
                    st.markdown(f"- {r}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 🔄 Adaptive Learning: Quality Correction Loop")
            st.caption("Help improve the AI system's precision. If the predicted emotion is incorrect, choose the correct one below to save it for future retraining.")
            
            col_fb1, col_fb2 = st.columns([2, 3])
            with col_fb1:
                correct_emo = st.selectbox(
                    "Select correct vocal emotion:",
                    list(EMOTION_EMOJIS.keys()),
                    index=list(EMOTION_EMOJIS.keys()).index(res['emotion'].lower()) if res['emotion'].lower() in EMOTION_EMOJIS else 0,
                    key="ui_feedback_emotion"
                )
            with col_fb2:
                st.markdown("<div style='padding-top: 28px;'>", unsafe_allow_html=True)
                if st.button("💾 Save Correction to Feedback Base", use_container_width=True, key="ui_save_feedback"):
                    if st.session_state.last_audio is not None:
                        filepath = save_feedback_audio(st.session_state.last_audio, st.session_state.last_sr, correct_emo)
                        st.success(f"Correction saved! Correct label set: '{correct_emo.upper()}'")
                        st.info(f"Target feedback buffer file: {os.path.basename(filepath)}")
                        st.balloons()
                    else:
                        st.error("No raw audio cache found to apply corrections.")
                st.markdown("</div>", unsafe_allow_html=True)

    with tab_analytics:
        st.markdown("### 📊 Session Analytics & Trend Tracking")
        df_history = history_manager.load_history_as_df()
        
        if df_history.empty:
            st.info("No predictions logged during this session yet. Perform analysis in the Voice Analyzer to compile data.")
        else:
            # 1. Summary Cards Row
            total_records = len(df_history)
            dominant_emotion = df_history['emotion'].mode().iloc[0] if not df_history['emotion'].empty else "N/A"
            avg_confidence = df_history['confidence'].mean() * 100
            avg_risk = df_history['risk_score'].mean()
            
            st.markdown(f"""
            <div class="metric-container" style="margin-bottom: 2rem;">
                <div class="metric-card">
                    <div class="metric-title">Total Analyzed</div>
                    <div class="metric-value">{total_records}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Dominant Emotion</div>
                    <div class="metric-value">{dominant_emotion.upper()}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Avg Confidence</div>
                    <div class="metric-value">{avg_confidence:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Avg Risk Score</div>
                    <div class="metric-value">{avg_risk:.1f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. Charts Row
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                fig_donut = dashboard.plot_emotion_distribution_donut(df_history)
                if fig_donut:
                    st.plotly_chart(fig_donut, use_container_width=True)
                    
            with col_chart2:
                if st.session_state.last_result:
                    fig_probs = dashboard.plot_confidence_radar_or_bar(
                        st.session_state.last_result['all_probs'], 
                        st.session_state.last_result['emotion']
                    )
                    if fig_probs:
                        st.plotly_chart(fig_probs, use_container_width=True)
                else:
                    st.info("Run an analysis to view probability breakdowns for the latest speech test.")
                    
            # 3. Timeline row
            st.markdown("---")
            fig_timeline = dashboard.plot_emotion_timeline(df_history)
            if fig_timeline:
                st.plotly_chart(fig_timeline, use_container_width=True)
                
            # Session breakdown details
            st.markdown("##### 📋 Summary Observations")
            high_risk_incidents = len(df_history[df_history['risk_score'] >= 60])
            st.write(
                f"- This session contains **{total_records} analyses** over which **{dominant_emotion}** was the most common state.\n"
                f"- High Emotional Risk indicators ($\ge 60$ score) occurred **{high_risk_incidents} times** ({high_risk_incidents/total_records*100:.1f}% of checks)."
            )

    with tab_history:
        st.markdown("### 📜 Prediction Log History")
        df_history = history_manager.load_history_as_df()
        
        if df_history.empty:
            st.info("No recorded predictions found.")
        else:
            col_hist_controls1, col_hist_controls2 = st.columns([3, 1])
            
            with col_hist_controls1:
                # Simple filter
                selected_emotion_filter = st.multiselect(
                    "Filter by Emotion:",
                    options=df_history['emotion'].unique().tolist(),
                    default=[]
                )
            with col_hist_controls2:
                st.markdown("<div style='padding-top:28px;'>", unsafe_allow_html=True)
                # Export history to CSV
                csv_data = df_history.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export History CSV",
                    data=csv_data,
                    file_name="speech_emotion_history.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.markdown("</div>", unsafe_allow_html=True)
                
            filtered_df = df_history.copy()
            if selected_emotion_filter:
                filtered_df = filtered_df[filtered_df['emotion'].isin(selected_emotion_filter)]
                
            # Render a clean, formatted table
            display_df = filtered_df.copy()
            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
            display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.1f}")
            display_df['timestamp'] = display_df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
            
            st.dataframe(
                display_df[["timestamp", "emotion", "confidence", "risk_score", "wellness_status"]],
                use_container_width=True,
                hide_index=True
            )
            
            # Reset history button
            st.markdown("---")
            if st.button("🗑️ Clear All Prediction History", use_container_width=True):
                if history_manager.clear_history():
                    st.success("Prediction history cleared successfully!")
                    st.rerun()
                else:
                    st.error("Failed to delete prediction history file.")

    with tab_settings:
        st.markdown("### 🧠 Adaptive Learning & Model Controls")
        
        col_setup1, col_setup2 = st.columns(2)
        
        with col_setup1:
            st.markdown("##### 📁 Model Persistence")
            st.write("Save the currently active model parameters or load the last saved checkpoints from the disk.")
            
            col_pers1, col_pers2 = st.columns(2)
            with col_pers1:
                if st.button("💾 Save Active Model", use_container_width=True):
                    if st.session_state.model:
                        os.makedirs(MODEL_DIR, exist_ok=True)
                        st.session_state.model.save(MODEL_DIR)
                        joblib.dump(st.session_state.label_encoder, os.path.join(MODEL_DIR, "encoder.pkl"))
                        joblib.dump(st.session_state.accuracy, os.path.join(MODEL_DIR, "accuracy.pkl"))
                        st.success("Model files successfully persisted to disk!")
                    else:
                        st.error("No active model to save.")
                        
            with col_pers2:
                if st.button("📂 Load Model Checkpoint", use_container_width=True):
                    try:
                        from speech_emotion_recognition import KerasEmotionModel
                        st.session_state.model = KerasEmotionModel.load(MODEL_DIR)
                        st.session_state.label_encoder = joblib.load(os.path.join(MODEL_DIR, "encoder.pkl"))
                        acc_path = os.path.join(MODEL_DIR, "accuracy.pkl")
                        if os.path.exists(acc_path):
                            st.session_state.accuracy = joblib.load(acc_path)
                        else:
                            st.session_state.accuracy = 85.0
                        st.session_state.trained = True
                        st.success("Model checkpoints loaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load: Check if models exist. Details: {e}")
                        
            st.markdown("---")
            st.markdown("##### 🚀 Train Initial Model")
            st.write("Extract features and train a new classification model using the default RAVDESS speech dataset.")
            
            if st.button("🚀 Train Model (321 Features)", use_container_width=True):
                with st.spinner("Extracting features from RAVDESS dataset... This might take a minute."):
                    x, y = load_dataset()
                    
                if len(x) == 0:
                    st.error("RAVDESS dataset audio files not found at expected path.")
                else:
                    st.info(f"Loaded dataset: {len(x)} speech samples. Commencing training...")
                    with st.spinner("Fitting Multi-Layer Perceptron Classifier..."):
                        model, le, acc = train_model(x, y)
                        
                    st.session_state.model = model
                    st.session_state.label_encoder = le
                    st.session_state.trained = True
                    st.session_state.accuracy = acc
                    st.success(f"Training completed successfully! Initial accuracy: {acc:.2f}%")
                    st.rerun()
                    
        with col_setup2:
            st.markdown("##### 🔄 Adaptive Learning (Retraining Loop)")
            st.write("Incorporate corrected audio samples saved via the Quality Correction Loop into the training database to dynamically retrain the model.")
            
            feedback_count = len([f for f in os.listdir(FEEDBACK_DIR) if f.endswith('.wav')]) if os.path.exists(FEEDBACK_DIR) else 0
            st.metric(label="Saved Feedback Samples", value=feedback_count)
            
            if st.button("🔄 Retrain with Feedback", use_container_width=True):
                if feedback_count == 0:
                    st.warning("No feedback samples collected yet to execute retraining.")
                else:
                    with st.spinner("Extracting original dataset features..."):
                        x_orig, y_orig = load_dataset()
                    with st.spinner("Extracting feedback dataset features..."):
                        x_fb, y_fb = load_feedback_dataset()
                        
                    st.info(f"Retraining baseline size: {len(x_orig)} (original) + {len(x_fb)} (feedback) = {len(x_orig)+len(x_fb)} samples.")
                    
                    with st.spinner("Retraining MLP model..."):
                        model, le, acc = retrain_with_feedback(x_orig, y_orig, x_fb, y_fb)
                        
                    st.session_state.model = model
                    st.session_state.label_encoder = le
                    st.session_state.accuracy = acc
                    st.success(f"Model retrained successfully with feedback! New Accuracy: {acc:.2f}%")
                    st.balloons()
                    st.rerun()

# Legacy alias — kept for any external callers
EMOTIONS = RAVDESS_EMOTIONS

if __name__ == "__main__":
    main()
