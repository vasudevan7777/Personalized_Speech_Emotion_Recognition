import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sounddevice as sd
import soundfile as sf
from colorama import Fore, Style, init
import warnings
warnings.filterwarnings('ignore')

# Initialize colorama
init(autoreset=True)

# Path to RAVDESS dataset
DATASET_PATH = "dataset/Audio_Speech_Actors_01-24"

# Emotion labels based on RAVDESS filenames (3rd position in filename)
# Format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor
EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# Feature extractor: MFCC + Chroma + Spectral Contrast + Zero Crossing Rate + RMS Energy
def extract_features(file_path, duration=5, offset=0.0):
    """Extract comprehensive audio features for emotion recognition."""
    y, sr = librosa.load(file_path, duration=duration, offset=offset)
    
    # Ensure minimum audio length
    if len(y) < sr * 0.5:  # Less than 0.5 seconds
        y = np.pad(y, (0, int(sr * 0.5) - len(y)), mode='constant')
    
    # MFCC features (40 coefficients)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    
    # Chroma features (12 pitch classes)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    
    # Spectral contrast (7 bands)
    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    
    # Zero crossing rate - useful for differentiating speech characteristics
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    
    # RMS Energy - captures loudness/intensity
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    
    # Mel spectrogram features
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    
    return np.hstack([mfccs, chroma, spec_contrast, zcr, rms, mel])

# Load RAVDESS dataset from Actor folders
def load_dataset():
    """Load all audio files from the RAVDESS dataset (Actor folders)."""
    x, y = [], []
    
    if not os.path.exists(DATASET_PATH):
        print(Fore.RED + f"❌ Dataset not found at: {DATASET_PATH}")
        print(Fore.YELLOW + "Please ensure the RAVDESS dataset is in the correct location.")
        return np.array(x), np.array(y)
    
    actor_folders = sorted(os.listdir(DATASET_PATH))
    total_actors = len([f for f in actor_folders if os.path.isdir(os.path.join(DATASET_PATH, f))])
    
    print(Fore.CYAN + f"   Found {total_actors} actor folders")
    
    for i, actor_folder in enumerate(actor_folders):
        actor_path = os.path.join(DATASET_PATH, actor_folder)
        if os.path.isdir(actor_path):
            actor_files = 0
            for file in os.listdir(actor_path):
                if file.endswith(".wav"):
                    parts = file.split("-")
                    if len(parts) >= 3:
                        emotion_code = parts[2]  # 3rd element is emotion
                        emotion = EMOTIONS.get(emotion_code)
                        if emotion:
                            try:
                                features = extract_features(os.path.join(actor_path, file), duration=3, offset=0.5)
                                x.append(features)
                                y.append(emotion)
                                actor_files += 1
                            except Exception as e:
                                pass  # Skip problematic files silently
            
            # Progress indicator
            progress = (i + 1) / total_actors * 100
            print(f"\r   Loading: {progress:.0f}% ({actor_folder}: {actor_files} files)", end="", flush=True)
    
    print()  # New line after progress
    return np.array(x), np.array(y)

# Train MLP model with improved architecture
def train_model(x_train, y_train):
    """Train a Multi-Layer Perceptron classifier with optimized parameters."""
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),  # Deeper network for better learning
            max_iter=1500,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        )
    )
    model.fit(x_train, y_train)
    return model

# Record live audio - 5 seconds
def record_audio(filename="live.wav", duration=5, samplerate=22050):
    """Record audio from microphone for 5 seconds."""
    print(Fore.CYAN + f"🎙 Recording for {duration} seconds... Speak now!")
    print(Fore.YELLOW + "   ", end="")
    
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    
    # Show countdown
    for i in range(duration, 0, -1):
        print(f"{i}...", end="", flush=True)
        sd.sleep(1000)  # Sleep 1 second
    
    sd.wait()
    print(" Done!")
    sf.write(filename, recording, samplerate)
    print(Fore.GREEN + f"✅ Recording saved as {filename}")

# Predict emotion from file
def predict_emotion(model, filename, label_encoder):
    """Predict emotion from an audio file (handles 5-second live recordings)."""
    # Use full duration for live recordings, no offset
    features = extract_features(filename, duration=5, offset=0.0).reshape(1, -1)
    prediction_encoded = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    
    # Get the emotion string from label encoder
    emotion = label_encoder.inverse_transform([prediction_encoded])[0]
    confidence = np.max(proba)
    
    # Get all emotions with their probabilities
    all_probs = dict(zip(label_encoder.inverse_transform(model.classes_), proba))
    
    return emotion, confidence, all_probs

# Adaptive training: update model with new confident sample
def adaptive_train(model, x_train, y_train, new_feature, new_label_encoded, label_encoder):
    """Add new sample to training data and retrain model."""
    x_train = np.vstack([x_train, new_feature])
    y_train = np.append(y_train, new_label_encoded)
    model.fit(x_train, y_train)
    print(Fore.GREEN + "✅ Model updated with new sample!")
    return model, x_train, y_train

# Display emotion distribution in dataset
def show_emotion_distribution(y):
    """Display the distribution of emotions in the dataset."""
    unique, counts = np.unique(y, return_counts=True)
    print(Fore.CYAN + "\n📊 Emotion Distribution in Dataset:")
    print("-" * 40)
    for emotion, count in zip(unique, counts):
        bar = "█" * (count // 5)
        print(f"  {emotion:12s}: {count:3d} samples {bar}")
    print("-" * 40)

# ✅ Entry point
if __name__ == "__main__":
    print(Fore.YELLOW + "\n" + "=" * 50)
    print(Fore.YELLOW + "   🎭 SPEECH EMOTION RECOGNITION SYSTEM 🎭")
    print(Fore.YELLOW + "=" * 50)
    
    print(Fore.CYAN + "\n🔄 Loading RAVDESS dataset...")
    
    # Load full dataset from Actor folders
    x, y = load_dataset()
    
    if len(x) == 0:
        print(Fore.RED + "❌ No audio files found! Please check dataset path.")
        exit()
    
    print(Fore.GREEN + f"✅ Loaded {len(x)} audio samples")
    
    # Show emotion distribution
    show_emotion_distribution(y)
    
    print(Fore.CYAN + "\n🧠 Encoding labels and training model...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split with stratification to ensure all emotions are represented
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"   Training samples: {len(x_train)}")
    print(f"   Testing samples:  {len(x_test)}")

    model = train_model(x_train, y_train)
    print(Fore.GREEN + "\n✅ Model training completed!")
    
    # Detailed evaluation
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    
    print(Fore.YELLOW + "\n" + "=" * 50)
    print(Fore.YELLOW + f"   🎯 MODEL ACCURACY: {accuracy:.2f}%")
    print(Fore.YELLOW + "=" * 50)
    
    # Show per-class accuracy
    print(Fore.CYAN + "\n📋 Classification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    # Interactive loop for continuous prediction
    while True:
        print(Fore.YELLOW + "\n" + "-" * 50)
        print(Fore.YELLOW + "Options:")
        print("  1. Record and predict emotion")
        print("  2. Test with existing file")
        print("  3. Exit")
        
        choice = input(Fore.WHITE + "\nEnter choice (1/2/3): ").strip()
        
        if choice == "1":
            record_audio()
            emotion, conf, all_probs = predict_emotion(model, "live.wav", label_encoder)
            
            print(Fore.MAGENTA + "\n" + "=" * 50)
            print(Fore.MAGENTA + "   🎤 EMOTION PREDICTION RESULT")
            print(Fore.MAGENTA + "=" * 50)
            print(Fore.GREEN + f"   🗣 Detected Emotion : {emotion.upper()}")
            print(Fore.GREEN + f"   📊 Confidence Level : {conf * 100:.2f}%")
            
            print(Fore.CYAN + "\n   All Emotion Probabilities:")
            for emo, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 20)
                marker = " ◄" if emo == emotion else ""
                print(f"   {emo:12s}: {prob*100:5.1f}% {bar}{marker}")
            
            # Adaptive learning with confirmation
            if conf > 0.85:
                print(Fore.CYAN + f"\n   ➕ High confidence! Add '{emotion}' to training data? (y/n): ", end="")
                if input().strip().lower() == 'y':
                    new_feat = extract_features("live.wav", duration=5, offset=0.0).reshape(1, -1)
                    new_label = label_encoder.transform([emotion])[0]
                    model, x_train, y_train = adaptive_train(
                        model, x_train, y_train, new_feat, new_label, label_encoder
                    )
                    
        elif choice == "2":
            filepath = input(Fore.WHITE + "Enter audio file path: ").strip().strip('"')
            if os.path.exists(filepath):
                emotion, conf, all_probs = predict_emotion(model, filepath, label_encoder)
                print(Fore.MAGENTA + f"\n   🎤 Detected Emotion: {emotion.upper()} ({conf*100:.2f}%)")
            else:
                print(Fore.RED + "   ❌ File not found!")
                
        elif choice == "3":
            print(Fore.GREEN + "\n👋 Goodbye!")
            break
        else:
            print(Fore.RED + "   ❌ Invalid choice!")